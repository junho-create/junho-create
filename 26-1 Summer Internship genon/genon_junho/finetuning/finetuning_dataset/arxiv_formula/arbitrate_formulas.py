#!/usr/bin/env python3
"""수식 후보 중재 — dots 와 paddle 중 **맞는 쪽을 고르는 게 기본**, 둘 다 틀리면 judge 가
직접 쓰고 그 사실을 표기한다.

기존 hardcase 는 "122B 가 맨바닥에서 라벨 생성 -> 같은 모델이 판정"인데, 여기서는
후보가 이미 둘 있으므로 **중재(택1)** 로 바꾼다. 생성 콜이 없어져 비용이 절반이고,
모델이 자기 출력을 자기가 승인하는 구조도 사라진다.

judge 에게 주는 이미지 3장 (render-then-verify, MinerU2.5-Pro 3.3):
    1) 원본 페이지에서 잘라낸 수식 크롭들을 세로로 이어붙인 것   <- 정답 원본
    2) dots 후보 전체를 KaTeX 로 렌더한 것                        <- 후보 A
    3) paddle 후보 전체를 KaTeX 로 렌더한 것                      <- 후보 B
텍스트로 LaTeX 를 비교시키면 모델이 미묘한 구조 결함을 놓친다. 렌더해서 눈으로 보게
만들면 그게 레이아웃 붕괴로 증폭돼 보인다.

**KaTeX 렌더 실패 후보는 애초에 선택지에서 뺀다.** 학습 프롬프트가 KaTeX 호환 LaTeX 를
요구하고 평가(OmniDocBench)도 그 전제라, KaTeX 로 못 그리는 건 GT 로 쓰면 안 된다.

수식 1개당 1콜이면 1만 콜이 넘으므로 **페이지 단위로 묶어 1콜**로 처리한다.

사용:
    python3 arbitrate_formulas.py --limit 300     # 단계적 확대용
    python3 arbitrate_formulas.py                 # 전량
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

sys.path.insert(0, "/home/jhyeo/ocr_file_filter")
from ocr_filter.hardcase.client import call_vlm            # noqa: E402
from ocr_filter.report.render_parse import render_formulas_katex, _strip_math  # noqa: E402

HERE = Path(__file__).parent
_write_lock = threading.Lock()

JUDGE_CFG = {"name": "Qwen3.5-122B-A10B-FP8", "endpoint": "http://localhost:8004/v1"}

ARBITRATE_PROMPT = """You are given three images.

[Image 1] CROPS — the ground truth. Each row is one formula cropped from the original
document page, labelled [0], [1], [2], ... in order.
[Image 2] CANDIDATE A — candidate LaTeX for those same formulas, rendered with KaTeX.
[Image 3] CANDIDATE B — a second candidate for the same formulas, rendered with KaTeX.

The rows in all three images correspond by index. For each index, decide which candidate
transcribes the crop correctly.

Judge by what you SEE in the rendered images, symbol by symbol: subscripts, superscripts,
fraction structure, integral/sum limits, matrix and alignment structure, greek letters,
operators, delimiters. Ignore differences that render identically (\\frac vs \\dfrac,
spacing macros, \\left( vs (). A candidate marked "MISSING" or shown as a red KaTeX ERROR
is not selectable.

Verdict per index:
  "A"       - candidate A matches the crop (and B does not, or B is worse)
  "B"       - candidate B matches the crop
  "NEITHER" - both are wrong in a way that changes the mathematical content

Prefer A or B. Only answer NEITHER when both genuinely misread the formula.

Output ONLY a JSON object, no commentary, no markdown fences:
{"verdicts": [{"index": 0, "choice": "A", "note": ""}, ...]}
`note` is a short reason (Korean, under 40 chars) and is required only for NEITHER.
Include exactly one entry per index, %(n)d in total (indices 0..%(last)d).
"""

AUTHOR_PROMPT = """You are given two images.

[Image 1] CROPS — formulas cropped from the original document page, labelled [0], [1], ...
[Image 2] ATTEMPTS — the rejected candidates rendered with KaTeX, same indices.

Both automatic transcriptions of the indices listed below were judged wrong. Transcribe
those formulas yourself, directly from [Image 1].

Indices to transcribe: %(idx)s

Rules:
- Output KaTeX-compatible LaTeX only. No \\usepackage, no custom macros, no \\label,
  no \\tag, no \\ref. Use \\begin{aligned} rather than \\begin{align}.
- Do NOT wrap in $, $$, \\( \\) or \\[ \\]. Give the bare formula body.
- Transcribe exactly what is in the image. Do not simplify, complete or correct the math.
- If a formula is genuinely illegible, give an empty string for it.

Output ONLY a JSON object, no commentary, no markdown fences:
{"formulas": [{"index": 0, "latex": "..."}, ...]}
"""


# ── 파싱 ──────────────────────────────────────────────────────────────────────
def parse_json_obj(raw: str) -> dict | None:
    """모델 출력에서 JSON 객체 하나를 건져낸다. 코드펜스/앞뒤 잡담을 견딘다."""
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


def stack_crops(paths: list[str]) -> Image.Image | None:
    """크롭들을 [i] 라벨과 함께 세로로 이어붙인다 — 렌더 이미지와 행이 대응돼야 한다."""
    from PIL import ImageDraw

    ims = []
    for p in paths:
        try:
            ims.append(Image.open(p).convert("RGB"))
        except Exception:  # noqa: BLE001
            ims.append(Image.new("RGB", (200, 40), "white"))
    if not ims:
        return None
    gutter, pad = 46, 14
    w = gutter + max(im.width for im in ims) + pad
    h = pad + sum(im.height + pad for im in ims)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    y = pad
    for i, im in enumerate(ims):
        draw.text((8, y + im.height // 2 - 6), f"[{i}]", fill=(180, 0, 0))
        canvas.paste(im, (gutter, y))
        y += im.height + pad
        im.close()
    return canvas


# ── 페이지 1건 처리 ────────────────────────────────────────────────────────────
def process_page(key: str, rows: list[dict], judge_cfg: dict) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["formula_idx"])
    n = len(rows)
    base = [{"pair_id": r["pair_id"], "key": key, "split": r["split"],
             "formula_idx": r["formula_idx"], "latex": "", "source": "unresolved",
             "katex_ok": False, "judge_note": "", "revised": False} for r in rows]

    crops = stack_crops([r["crop_path"] for r in rows])
    if crops is None:
        for b in base:
            b["judge_note"] = "crop_missing"
        return base

    # 두 모델 다 `$$...$$` 로 감싸서 준다. 여기서 벗겨두지 않으면 최종 GT 를 만들 때
    # `<p>$$...$$...$$</p>` 처럼 $$ 가 이중으로 박힌다.
    a_tex = [_strip_math(r["cand_dots"]) or "MISSING" for r in rows]
    b_tex = [_strip_math(r["cand_paddle"]) or "MISSING" for r in rows]
    img_a, ok_a = render_formulas_katex(a_tex)
    img_b, ok_b = render_formulas_katex(b_tex)
    if img_a is None and img_b is None:
        for b in base:
            b["judge_note"] = "both_render_failed"
        return base

    imgs = [crops] + [im for im in (img_a, img_b) if im is not None]
    try:
        raw = call_vlm(judge_cfg, imgs, ARBITRATE_PROMPT % {"n": n, "last": n - 1},
                       enable_thinking=False)
    except Exception as e:  # noqa: BLE001
        for b in base:
            b["judge_note"] = f"judge_error: {type(e).__name__}"
        return base

    obj = parse_json_obj(raw) or {}
    choice_by_idx = {}
    for v in obj.get("verdicts") or []:
        if isinstance(v, dict) and isinstance(v.get("index"), int):
            choice_by_idx[v["index"]] = (str(v.get("choice", "")).upper().strip(),
                                         str(v.get("note", ""))[:80])

    need_author: list[int] = []
    for i, b in enumerate(base):
        choice, note = choice_by_idx.get(i, ("", "no_verdict"))
        b["judge_note"] = note
        # KaTeX 로 못 그리는 후보는 고를 수 없다 — 골랐어도 무효화한다.
        if choice == "A" and ok_a[i] and a_tex[i] != "MISSING":
            b.update(latex=a_tex[i], source="dots", katex_ok=True)
        elif choice == "B" and ok_b[i] and b_tex[i] != "MISSING":
            b.update(latex=b_tex[i], source="paddle", katex_ok=True)
        else:
            need_author.append(i)

    if not need_author:
        return base

    # ── 2차: judge 가 직접 작성 (1회만, 되먹임 루프 없음) ──
    attempts = img_a if img_a is not None else img_b
    try:
        raw2 = call_vlm(judge_cfg, [crops, attempts],
                        AUTHOR_PROMPT % {"idx": ", ".join(str(i) for i in need_author)},
                        enable_thinking=False)
    except Exception as e:  # noqa: BLE001
        for i in need_author:
            base[i]["judge_note"] = f"author_error: {type(e).__name__}"
        return base

    obj2 = parse_json_obj(raw2) or {}
    authored = {}
    for v in obj2.get("formulas") or []:
        if isinstance(v, dict) and isinstance(v.get("index"), int):
            authored[v["index"]] = _strip_math(str(v.get("latex") or ""))

    idxs = [i for i in need_author if authored.get(i)]
    if idxs:
        _, ok_new = render_formulas_katex([authored[i] for i in idxs])
        for i, ok in zip(idxs, ok_new):
            base[i]["revised"] = True
            if ok:
                base[i].update(latex=authored[i], source="judge_authored", katex_ok=True)
            else:
                base[i]["judge_note"] = "authored_katex_failed"
    for i in need_author:
        if base[i]["source"] == "unresolved" and not base[i]["judge_note"]:
            base[i]["judge_note"] = "author_empty"
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(HERE / "formula_pairs.jsonl"))
    ap.add_argument("--out", default=str(HERE / "formula_verdicts.jsonl"))
    ap.add_argument("--endpoint", default=JUDGE_CFG["endpoint"])
    ap.add_argument("--model", default=JUDGE_CFG["name"])
    # ocr_file_filter 쪽 hardcase judge 는 workers=4 가 필수였다(32 에서 Chromium 경합으로
    # 스크린샷이 조용히 실패). 여기서는 KaTeX 렌더가 콜당 1.1s 인데 VLM 콜이 ~20s 라
    # 브라우저 점유율이 10% 미만이고, 렌더 실패는 flag 로 드러나 조용히 넘어가지도 않는다.
    # 그래서 8 까지는 안전하다 — 더 올리면 위 전례의 영역으로 들어간다.
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="페이지 수 제한 (단계적 확대용)")
    args = ap.parse_args()

    judge_cfg = {"name": args.model, "endpoint": args.endpoint}

    by_key: dict[str, list[dict]] = defaultdict(list)
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_key[r["key"]].append(r)

    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():   # 재개 가능
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["key"])

    keys = [k for k in sorted(by_key) if k not in done]
    if args.limit:
        keys = keys[:args.limit]
    print(f"페이지 {len(keys)}건 (완료 {len(done)}건 건너뜀), workers={args.workers}", flush=True)
    if not keys:
        return 0

    counts: dict[str, int] = defaultdict(int)
    t0, n_done = time.time(), 0
    with out_path.open("a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_page, k, by_key[k], judge_cfg): k for k in keys}
        for fut in as_completed(futs):
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  페이지 실패 {futs[fut]}: {type(e).__name__}: {e}", flush=True)
                continue
            with _write_lock:
                for r in rows:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    counts[r["source"]] += 1
                fout.flush()
            n_done += 1
            if n_done % 10 == 0 or n_done == len(keys):
                el = time.time() - t0
                eta = el / n_done * (len(keys) - n_done) / 60
                print(f"[{n_done}/{len(keys)}] {el / 60:.1f}분 경과 ETA {eta:.0f}분 "
                      f"{dict(counts)}", flush=True)

    print("\n=== 중재 결과 ===")
    tot = sum(counts.values()) or 1
    for k, v in sorted(counts.items(), key=lambda t: -t[1]):
        print(f"  {k}: {v} ({v / tot * 100:.1f}%)")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
