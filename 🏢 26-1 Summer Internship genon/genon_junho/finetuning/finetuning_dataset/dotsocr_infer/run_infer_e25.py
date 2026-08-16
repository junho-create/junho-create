#!/usr/bin/env python3
"""dots.ocr 로 combined_e25_28791 의 Table 또는 Formula 포함 unified_layout 페이지를
추론한다 (manifest_e25.jsonl, 5,466 페이지, extract_e25.py 로 생성).

run_infer.py 를 base 로 하되:
  - 서버를 여러 개(GPU 여러 장) 띄워 놓고 --api-urls 로 나열하면 key 해시로
    페이지를 나눠 보낸다. dots 는 페이지 1장 = 요청 1개라 데이터 병렬이 곧 처리량이다.
  - pred_formulas 를 pred_tables 와 나란히 기록한다(카테고리 목록은 동일 프롬프트에
    이미 'Formula' 포함).

사용:
    python3 run_infer_e25.py --api-urls http://127.0.0.1:8119/v1/chat/completions,http://127.0.0.1:8120/v1/chat/completions,http://127.0.0.1:8121/v1/chat/completions --concurrency 144
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).parent
MANIFEST = HERE / "manifest_e25.jsonl"
SPLITS = ("train", "valid", "test")

DOTS_LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

IMG_FACTOR = 28
MIN_PIXELS = 3136
MAX_PIXELS = 11289600


def smart_resize(w: int, h: int) -> tuple[int, int]:
    hb = max(IMG_FACTOR, round(h / IMG_FACTOR) * IMG_FACTOR)
    wb = max(IMG_FACTOR, round(w / IMG_FACTOR) * IMG_FACTOR)
    if hb * wb > MAX_PIXELS:
        beta = math.sqrt((h * w) / MAX_PIXELS)
        hb = math.floor(h / beta / IMG_FACTOR) * IMG_FACTOR
        wb = math.floor(w / beta / IMG_FACTOR) * IMG_FACTOR
    elif hb * wb < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        hb = math.ceil(h * beta / IMG_FACTOR) * IMG_FACTOR
        wb = math.ceil(w * beta / IMG_FACTOR) * IMG_FACTOR
    return wb, hb


def load_manifest(splits, limit=None):
    by_split = {s: [] for s in splits}
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] in by_split:
                by_split[r["split"]].append(r)
    if limit:
        by_split = {s: v[:limit] for s, v in by_split.items()}
    return by_split


def done_keys(path: Path) -> set[str]:
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def data_url(path: str) -> str:
    ext = Path(path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def call_dots(session, url, model, img_path, timeout, max_new_tokens):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(img_path)}},
                {"type": "text", "text": DOTS_LAYOUT_PROMPT},
            ],
        }],
        "max_tokens": max_new_tokens,
        "temperature": 0.1,
        "top_p": 0.9,
    }
    r = session.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)


def salvage_objects(s: str) -> list[dict]:
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        o = json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(o, dict):
                            out.append(o)
                    start = None
    return out


def parse_blocks(raw: str) -> tuple[list[dict], str | None]:
    if not raw:
        return [], "empty"
    s = FENCE.sub("", raw.strip())
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        got = salvage_objects(s)
        return got, ("truncated" if got else "unparsable")
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                obj = v
                break
        else:
            return ([obj] if "bbox" in obj else []), (None if "bbox" in obj else "unexpected_object")
    return (obj if isinstance(obj, list) else []), None


def norm_bbox(bbox, w, h):
    if not bbox or len(bbox) != 4 or not w or not h:
        return None
    x0, y0, x1, y1 = bbox
    return [round(x0 * 1000 / w), round(y0 * 1000 / h),
            round(x1 * 1000 / w), round(y1 * 1000 / h)]


def pack(rec, raw, blocks, note, rw, rh, ow, oh, elapsed) -> dict:
    out_blocks = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        bbox = b.get("bbox")
        bbox = [int(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None
        out_blocks.append({
            "label": (b.get("category") or "").lower(),
            "bbox": bbox,
            "bbox_1000": norm_bbox(bbox, rw, rh),
            "order": len(out_blocks) + 1,
            "content": b.get("text"),
        })
    tables = [
        {"index": i, "bbox": b["bbox"], "bbox_1000": b["bbox_1000"], "html": b["content"]}
        for i, b in enumerate((b for b in out_blocks if b["label"] == "table"), start=1)
    ]
    formulas = [
        {"index": i, "bbox": b["bbox"], "bbox_1000": b["bbox_1000"], "latex": b["content"]}
        for i, b in enumerate((b for b in out_blocks if b["label"] == "formula"), start=1)
    ]
    return {
        "key": rec["key"],
        "split": rec["split"],
        "line_no": rec["line_no"],
        "image_path": rec["image_path"],
        "width": rw,
        "height": rh,
        "orig_width": ow,
        "orig_height": oh,
        "gt_n_tables": rec["n_tables"],
        "gt_n_formulas": rec["n_formulas"],
        "pred_n_tables": len(tables),
        "pred_n_formulas": len(formulas),
        "pred_tables": tables,
        "pred_formulas": formulas,
        "pred_blocks": out_blocks,
        "parse_note": note,
        "inference_time": round(elapsed, 2),
        "raw_len": len(raw or ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--split", choices=SPLITS, action="append")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--api-urls", default="http://127.0.0.1:8119/v1/chat/completions",
                     help="comma-separated. key 해시로 나눠 보낸다")
    ap.add_argument("--model", default="dots.ocr")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    urls = [u.strip() for u in args.api_urls.split(",") if u.strip()]
    print(f"API servers ({len(urls)}): {urls}", flush=True)

    splits = args.split or list(SPLITS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split = load_manifest(splits, args.limit)

    lock = threading.Lock()
    local = threading.local()

    def session():
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    def url_for(key: str) -> str:
        return urls[hash(key) % len(urls)]

    grand_t0 = time.time()
    for split in splits:
        recs = by_split[split]
        out_path = out_dir / f"{split}.jsonl"
        skip = done_keys(out_path)
        todo = [r for r in recs if r["key"] not in skip]
        print(f"\n[{split}] 전체 {len(recs)} / 완료 {len(skip)} / 남은 {len(todo)}", flush=True)
        if not todo:
            continue

        t0 = time.time()
        state = {"done": 0, "fail": 0, "notes": {}}
        fout = out_path.open("a", encoding="utf-8")

        def work(rec):
            with Image.open(rec["image_path"]) as im:
                ow, oh = im.size
            rw, rh = smart_resize(ow, oh)
            api_url = url_for(rec["key"])
            raw, err = "", None
            for attempt in range(args.retries):
                try:
                    raw = call_dots(session(), api_url, args.model,
                                    rec["image_path"], args.timeout, args.max_new_tokens)
                    err = None
                    break
                except Exception as e:
                    err = e
                    time.sleep(2 * (attempt + 1))
            if err is not None:
                with lock:
                    state["fail"] += 1
                    print(f"    skip {rec['key']}: {type(err).__name__}: {err}", flush=True)
                return
            blocks, note = parse_blocks(raw)
            row = pack(rec, raw, blocks, note, rw, rh, ow, oh, time.time() - t0)
            with lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                state["done"] += 1
                if note:
                    state["notes"][note] = state["notes"].get(note, 0) + 1
                n = state["done"] + state["fail"]
                if n % 100 == 0 or n == len(todo):
                    fout.flush()
                    os.fsync(fout.fileno())
                    el = time.time() - t0
                    rate = state["done"] / el if el else 0
                    left = (len(todo) - n) / rate if rate else 0
                    print(f"  [{split}] {n}/{len(todo)}  {rate:.2f} page/s  "
                          f"경과 {el/60:.1f}분  ETA {left/60:.1f}분  "
                          f"실패 {state['fail']}  파싱이슈 {state['notes']}", flush=True)

        try:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                list(ex.map(work, todo))
        finally:
            fout.flush()
            os.fsync(fout.fileno())
            fout.close()

        print(f"[{split}] 완료: 성공 {state['done']} / 실패 {state['fail']} "
              f"/ 파싱이슈 {state['notes']} → {out_path}", flush=True)

    print(f"\nALL DONE  총 {(time.time()-grand_t0)/60:.1f}분", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
