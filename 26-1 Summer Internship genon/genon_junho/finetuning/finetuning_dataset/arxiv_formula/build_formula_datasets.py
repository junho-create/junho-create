#!/usr/bin/env python3
"""중재 결과로 최종 GT 두 벌을 만든다.

(A) crop  — 이미지 = 수식 크롭 1장, GT = Formula div 1개
(B) layout — 이미지 = 원본 페이지, GT = dots 페이지 예측 전체 div 에 **검증된 수식만 주입**

Formula div 포맷은 반드시 `<p>$$<br/>...<br/>$$</p>` 다
-----------------------------------------------------
`data.build_chandra_dataset._content_for` 는 Table/Picture 만 특수 처리하고 Formula 는
그냥 `<p>{escape(text)}</p>` 로 감싼다. 그래서 나온 GT 에는 `$$` 가 없고, 그걸로 학습한
모델도 맨몸 LaTeX 를 뱉는다. 그러면 평가 경로
(`convert_pred_to_markdown._blocks_from_div_html` -> OmniDocBench `md_tex_filter`)에서
그 블록이 **display_formula 가 아니라 text_block 으로 분류**돼, GT 쪽 equation 은
pred 가 빈 채로 채점된다 — "수식으로 뽑았는데 평가가 수식인 줄 모르는" 현상의 정체다.
실제로 combined_e26_28484 의 Formula div 1,143 개 중 `$$` 가 있는 건 723 개(63%)뿐이다.

따라서 Formula 만 `_content_for` 를 우회해 직접 만든다. escape 는 `_esc` 를 그대로 쓴다
(손수 짜면 `&` -> `&amp;` 를 빠뜨린다 — replace_formula_pages_with_dots.py 가 남긴 기록).

사용:
    python3 build_formula_datasets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/jhyeo/finetuning/vlm")
from data.build_chandra_dataset import _content_for, _div, _esc  # noqa: E402

HERE = Path(__file__).parent
SPLITS = ("train", "valid", "test")
PROMPT_STYLE = "chandra_no_ocr"   # 유효한 값은 chandra_no_ocr / chandra_with_ocr 2종뿐


def formula_content(latex: str) -> str:
    """수식 div 의 content. `$$` 래핑이 이 작업의 핵심 산출물이다."""
    return f"<p>$$<br/>{_esc(latex.strip())}<br/>$$</p>"


def norm_bbox(bbox: list[int]) -> list[int] | None:
    """좌표가 뒤집힌 박스를 바로잡고, 넓이 0 인 박스는 버린다.

    dots 가 드물게 x0>x1 이나 y0==y1 인 박스를 낸다(실측 9,100장 중 4건). 그대로 두면
    GT 에 음수 크기 영역이 들어가고, 학습/평가 어느 쪽도 그걸 어떻게 다룰지 모른다."""
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if x0 >= x1 or y0 >= y1:
        return None
    if not (0 <= x0 and x1 <= 1000 and 0 <= y0 and y1 <= 1000):
        return None
    return [x0, y0, x1, y1]


def blocks_to_gt_html(blocks, formula_by_bbox) -> tuple[str, int, int]:
    """dots pred_blocks -> chandra div HTML. Formula 는 검증된 LaTeX 로 갈아끼우고,
    검증되지 않은 Formula 블록은 **버린다**(검증 안 된 수식을 GT 에 남기지 않는다).

    반환: (gt_html, 주입한 수식 수, 미검증이라 버린 수식 div 수)"""
    divs, n_sub, n_drop_unverified = [], 0, 0
    for b in blocks:
        raw = b.get("bbox_1000")
        if not raw or len(raw) != 4:
            continue
        label_raw = (b.get("label") or "").strip()
        if not label_raw:
            continue
        label = label_raw.capitalize()
        # 조회는 dots 원본 좌표로, 출력은 정규화한 좌표로 한다 — 순서를 바꾸면 키가 안 맞는다.
        latex = formula_by_bbox.get(tuple(int(v) for v in raw)) if label == "Formula" else None
        bbox = norm_bbox(raw)
        if bbox is None:
            continue
        if label == "Formula":
            if not latex:
                continue          # 검증 실패/미매칭 수식은 페이지에서 제거
            divs.append(_div(bbox, "Formula", formula_content(latex)))
            n_sub += 1
            continue
        text = (b.get("content") or "").strip()
        if label != "Table" and text.startswith("$$") and text.endswith("$$") and len(text) > 4:
            # dots 가 display 수식을 Text/List-item 으로 잘못 라벨링한 경우(실측 14건).
            # 내용은 수식인데 라벨이 아니라서 합의·중재를 안 거쳤다 — 검증 안 된 LaTeX 를
            # GT 에 남기지 않는다는 원칙은 Formula div 를 버릴 때와 똑같이 적용한다.
            n_drop_unverified += 1
            continue
        content = _content_for(label, text)
        divs.append(_div(bbox, label, content))
    return "\n".join(divs), n_sub, n_drop_unverified


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(HERE / "formula_pairs.jsonl"))
    ap.add_argument("--verdicts", default=str(HERE / "formula_verdicts.jsonl"))
    ap.add_argument("--dots-dir", default=str(HERE / "dots_out"))
    ap.add_argument("--crop-out", default=str(HERE / "dataset_formula_crop"))
    ap.add_argument("--layout-out", default=str(HERE / "dataset_formula_layout"))
    args = ap.parse_args()

    pairs = {}
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pairs[r["pair_id"]] = r

    # 채택 = 렌더까지 통과한 것만. unresolved 는 버린다.
    verdicts = {}
    with open(args.verdicts, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            v = json.loads(line)
            if v["source"] != "unresolved" and v["katex_ok"] and v["latex"].strip():
                verdicts[v["pair_id"]] = v

    print(f"후보 {len(pairs)}건 / 채택 {len(verdicts)}건")

    # ── (A) crop 데이터셋 ──
    crop_dir = Path(args.crop_out)
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_rows: dict[str, list] = defaultdict(list)
    src_counts: dict[str, int] = defaultdict(int)
    for pid, v in verdicts.items():
        p = pairs.get(pid)
        if not p:
            continue
        bbox = p.get("inner_bbox_1000") or [0, 0, 1000, 1000]
        crop_rows[v["split"]].append({
            "image_path": p["crop_path"],
            "gt_html": _div([int(x) for x in bbox], "Formula", formula_content(v["latex"])),
            "prompt_style": PROMPT_STYLE,
            "bbox_scale": 1000,
            "output_format": "html",
            "ocr_info": [],
            "meta": {
                "source": "arxiv_formula_crop",
                "pair_id": pid,
                # 검색용 표기 — judge 가 직접 쓴 건 나중에 grep 으로 뽑아낸다
                "formula_source": v["source"],
                "page_image_path": p["image_path"],
                "iou": p["iou"],
                "crop_scale": p["crop_scale"],
            },
        })
        src_counts[v["source"]] += 1

    for split in SPLITS:
        out = crop_dir / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in crop_rows.get(split, []):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[crop/{split}] {len(crop_rows.get(split, []))}행 -> {out}")

    # ── (B) layout 데이터셋 ──
    by_key: dict[str, dict[tuple, str]] = defaultdict(dict)
    for pid, v in verdicts.items():
        p = pairs.get(pid)
        if p and p.get("dots_bbox_1000"):
            by_key[p["key"]][tuple(p["dots_bbox_1000"])] = v["latex"]

    layout_dir = Path(args.layout_out)
    layout_dir.mkdir(parents=True, exist_ok=True)
    n_layout = n_skip_parse = n_skip_fence = n_drop_unver = 0
    for split in SPLITS:
        src = Path(args.dots_dir) / f"{split}.jsonl"
        out = layout_dir / f"{split}.jsonl"
        rows = 0
        with out.open("w", encoding="utf-8") as g:
            if src.is_file():
                with src.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        fmap = by_key.get(d["key"])
                        if not fmap:
                            continue
                        if d.get("parse_note"):
                            n_skip_parse += 1   # 잘리거나 못 파싱한 페이지는 골격이 못 미덥다
                            continue
                        gt, n_sub, n_du = blocks_to_gt_html(d.get("pred_blocks") or [], fmap)
                        n_drop_unver += n_du
                        if not n_sub:
                            continue
                        if "```" in gt:
                            # 논문 본문의 코드 리스팅을 dots 가 마크다운 펜스째로 전사한 것.
                            # 전사 자체는 맞지만, 시스템 프롬프트가 모델에게 "코드펜스 금지"를
                            # 요구하는데 GT 로 펜스를 보여주면 앞뒤가 안 맞는다. 5건뿐이라 뺀다.
                            n_skip_fence += 1
                            continue
                        g.write(json.dumps({
                            "image_path": d["image_path"],
                            "gt_html": gt,
                            "prompt_style": PROMPT_STYLE,
                            "bbox_scale": 1000,
                            "output_format": "html",
                            "ocr_info": [],
                            "meta": {
                                "source": "arxiv_formula_layout",
                                "key": d["key"],
                                # 수식 외 영역은 검증 안 됨 — 나중에 hardcase 로 승격할 대상
                                "layout_source": "dots_unjudged",
                                "n_formulas_verified": n_sub,
                                "formula_sources": sorted({
                                    verdicts[pid]["source"] for pid in verdicts
                                    if pairs.get(pid, {}).get("key") == d["key"]
                                }),
                            },
                        }, ensure_ascii=False) + "\n")
                        rows += 1
        n_layout += rows
        print(f"[layout/{split}] {rows}행 -> {out}")

    print("\n=== 수식 출처 분포 ===")
    tot = sum(src_counts.values()) or 1
    for k, v in sorted(src_counts.items(), key=lambda t: -t[1]):
        print(f"  {k}: {v} ({v / tot * 100:.1f}%)")
    print(f"crop {sum(len(v) for v in crop_rows.values())}행 / layout {n_layout}행 "
          f"(parse_note 로 제외 {n_skip_parse}, 코드펜스로 제외 {n_skip_fence}, "
          f"미검증 수식 div 제거 {n_drop_unver})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
