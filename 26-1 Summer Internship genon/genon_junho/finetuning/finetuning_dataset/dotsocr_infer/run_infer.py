#!/usr/bin/env python3
"""dots.ocr 로 combined_e24_refined 의 표 포함 layout 페이지를 추론한다.

대상은 paddle 쪽과 완전히 같은 5,030 페이지(gt_table_audit/manifest.jsonl)이고, 출력
스키마도 `../paddlevl16_infer/{split}.jsonl` 과 같게 맞춘다. 같은 `key` 로 조인해서
두 모델을 나란히 볼 수 있게 하려는 것이다.

paddle 은 레이아웃 검출(paddle)+블록별 VL 인식(vLLM)이라 클라이언트가 CPU 병목이었지만,
dots 는 페이지 1장 = 요청 1개라 스레드풀만으로 서버를 채운다. 샤딩이 필요 없다.

프롬프트는 `/home/jhyeo/tsr_eval/run_dots_ocr_genos500.py` 의 DOTS_LAYOUT_PROMPT 를
그대로 쓴다 — 카테고리 목록이 GT 라벨 체계와 정확히 같아서 따로 매핑할 게 없다.

모델은 `rednote-hilab/dots.ocr` 이다. 이름이 비슷한 `dots.mocr` 은 마크다운 전용이라
같은 프롬프트를 줘도 bbox 를 안 준다.

사용:
    ./serve_vllm.sh 2 8119
    python3 run_infer.py                      # 전체
    python3 run_infer.py --split test --limit 32
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
MANIFEST = Path("/home/jhyeo/finetuning/finetuning_dataset/gt_table_audit/manifest.jsonl")
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

# Qwen2VL smart_resize 상수. preprocessor_config.json 의 patch_size(14) * merge_size(2).
IMG_FACTOR = 28
MIN_PIXELS = 3136
MAX_PIXELS = 11289600


def smart_resize(w: int, h: int) -> tuple[int, int]:
    """모델이 실제로 본 이미지 크기.

    dots 가 돌려주는 bbox 는 원본이 아니라 이 크기 기준이다. 우리 페이지(1654x2339)는
    max_pixels 아래라 축소는 안 되지만 28의 배수로 반올림되면서 몇 px 씩 어긋난다.
    정규화를 원본 크기로 하면 그만큼 밀리므로 여기서 맞춘다.
    """
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


def load_manifest(splits, limit=None, manifest=MANIFEST):
    by_split = {s: [] for s in splits}
    with Path(manifest).open(encoding="utf-8") as f:
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
            # 이미지가 먼저, 프롬프트가 나중이어야 한다. 순서를 뒤집으면 모델이
            # 레이아웃 JSON 대신 본문 마크다운만 뱉는다 — 에러가 아니라 그럴듯한
            # 다른 출력이 나오는 거라 파싱 단계에서야 드러난다.
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(img_path)}},
                {"type": "text", "text": DOTS_LAYOUT_PROMPT},
            ],
        }],
        "max_tokens": max_new_tokens,
        # dots.ocr 레퍼런스 데모(demo_hf.py/demo_vllm.py) 값. 이 조합으로 파인튜닝돼
        # 있어서 greedy(0.0)로 바꾸면 레이아웃 품질이 떨어진다.
        "temperature": 0.1,
        "top_p": 0.9,
    }
    r = session.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)


def salvage_objects(s: str) -> list[dict]:
    """중괄호 짝을 세어 완결된 최상위 객체만 건져낸다.

    dots 는 가끔 같은 문구를 무한 반복하다 max_tokens 에 걸린다(finish_reason=length).
    그러면 마지막 블록이 문자열 한복판에서 잘려 배열 전체가 깨지는데, 그 앞의 멀쩡한
    블록들은 버릴 이유가 없다. `rfind("},")` 로는 잘린 지점이 문자열 안이면 못 잡는다.
    """
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
    """모델 출력에서 블록 배열을 뽑는다. 실패 사유를 함께 돌려준다."""
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
            # 블록 하나만 덜렁 온 경우도 있다.
            return ([obj] if "bbox" in obj else []), (None if "bbox" in obj else "unexpected_object")
    return (obj if isinstance(obj, list) else []), None


def norm_bbox(bbox, w, h):
    if not bbox or len(bbox) != 4 or not w or not h:
        return None
    x0, y0, x1, y1 = bbox
    return [round(x0 * 1000 / w), round(y0 * 1000 / h),
            round(x1 * 1000 / w), round(y1 * 1000 / h)]


def pack(rec, raw, blocks, note, rw, rh, ow, oh, elapsed) -> dict:
    """paddle 쪽 `pack()` 과 같은 스키마로 맞춘다."""
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
    return {
        "key": rec["key"],
        "split": rec["split"],
        "line_no": rec["line_no"],
        "image_path": rec["image_path"],
        # bbox 는 smart_resize 된 좌표계 기준이다. crop 할 때 이 크기를 써야 안 밀린다.
        "width": rw,
        "height": rh,
        "orig_width": ow,
        "orig_height": oh,
        "gt_n_tables": rec["n_tables"],
        "pred_n_tables": len(tables),
        "pred_tables": tables,
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
    ap.add_argument("--api-url", default="http://127.0.0.1:8119/v1/chat/completions")
    ap.add_argument("--model", default="dots.ocr")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--manifest", default=str(MANIFEST), help="manifest.jsonl 경로")
    args = ap.parse_args()

    splits = args.split or list(SPLITS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split = load_manifest(splits, args.limit, args.manifest)

    lock = threading.Lock()
    local = threading.local()

    def session():
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

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
            raw, err = "", None
            for attempt in range(args.retries):
                try:
                    raw = call_dots(session(), args.api_url, args.model,
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
