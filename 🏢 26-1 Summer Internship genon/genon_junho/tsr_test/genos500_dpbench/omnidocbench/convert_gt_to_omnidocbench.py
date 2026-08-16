#!/usr/bin/env python3
"""dp-bench GT(reference_dp_bench.json) -> OmniDocBench end2end GT JSON 변환.

이슈 doc_parser#318: OmniDocBench 평가기(원본)를 genos 500-set에 적용하기 위한
GT 어댑터. OmniDocBench end2end GT 스키마(페이지 배열, layout_dets)는
upstream `demo_data/omnidocbench_demo/OmniDocBench_demo.json` 참고.

카테고리 매핑(우리 -> OmniDocBench):
- paragraph -> text_block, heading1 -> title, list -> text_block
- header/footer -> header/footer (매칭엔 참여하되 text 점수에선 upstream이 제외)
- caption -> figure_caption, footnote -> page_footnote (동일하게 점수 제외 대상)
- equation -> equation_isolated (latex 필드 = content.text, $$ 포함)
- table -> table (html 필드 = content.html)
- figure/chart -> figure (어느 평가 풀에도 안 들어감)

필드 근거(upstream 코드):
- 매칭이 읽는 내용 필드: text/latex/html (match.py get_gt_pred_lines)
- reading order: 블록의 `order` 정수 (end2end_dataset.get_sorted_text_list)
- 페이지-예측 파일 매칭: page_info.image_path의 basename
- poly는 매칭에 미사용이나 스키마 호환을 위해 채움

사용:
  python convert_gt_to_omnidocbench.py \
    --gt genos-500-set/reference_dp_bench.json \
    --images_dir <500장 이미지 폴더> \
    --output <출력 JSON> [--docs doc_0001.pdf,doc_0032.pdf]
"""

import argparse
import json
import os

CAT_MAP = {
    "paragraph": "text_block",
    "heading1": "title",
    "list": "text_block",
    "header": "header",
    "footer": "footer",
    "caption": "figure_caption",
    "footnote": "page_footnote",
    "equation": "equation_isolated",
    "table": "table",
    "figure": "figure",
    "chart": "figure",
}

# reading order(order 필드)를 부여할 카테고리 = text/formula/table 평가 풀 참여분
ORDERED_CATS = {c for c in CAT_MAP.values() if c != "figure"}


def _image_dims(images_dir, stem):
    from PIL import Image
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            with Image.open(p) as im:
                return os.path.basename(p), im.size
    raise FileNotFoundError(f"image not found for {stem} in {images_dir}")


def _poly(coords):
    """dp-bench coordinates(4점 [{x,y}...]) -> OmniDocBench poly(8 float)."""
    xs = [p["x"] for p in coords]
    ys = [p["y"] for p in coords]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def convert_doc(doc_key, doc, images_dir):
    stem = os.path.splitext(doc_key)[0]
    img_name, (w, h) = _image_dims(images_dir, stem)

    layout_dets = []
    order = 0
    for el in sorted(doc["elements"], key=lambda e: e["id"]):
        cat = CAT_MAP.get(el["category"])
        if cat is None:
            raise ValueError(f"{doc_key}: unknown category {el['category']!r}")
        det = {
            "category_type": cat,
            "poly": _poly(el["coordinates"]),
            "anno_id": el["id"],
            "ignore": False,
        }
        if cat == "equation_isolated":
            det["latex"] = el["content"]["text"]
        elif cat == "table":
            det["html"] = el["content"]["html"]
        elif cat != "figure":
            det["text"] = el["content"]["text"]
        if cat in ORDERED_CATS:
            order += 1
            det["order"] = order
        layout_dets.append(det)

    return {
        "layout_dets": layout_dets,
        "extra": {"relation": []},
        "page_info": {
            "page_no": 0,
            "height": h,
            "width": w,
            "image_path": img_name,
            "page_attribute": {"data_source": "genos_500_set", "language": "korean"},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="reference_dp_bench.json 경로")
    ap.add_argument("--images_dir", required=True, help="500장 이미지 폴더(페이지 크기 읽기용)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--docs", default="", help="쉼표구분 문서 key 부분집합(생략=전체)")
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    keys = [k.strip() for k in args.docs.split(",") if k.strip()] or list(gt)

    pages = [convert_doc(k, gt[k], args.images_dir) for k in keys]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pages, f, ensure_ascii=False)
    n_eq = sum(1 for p in pages for d in p["layout_dets"] if d["category_type"] == "equation_isolated")
    n_tbl = sum(1 for p in pages for d in p["layout_dets"] if d["category_type"] == "table")
    print(f"docs={len(pages)} equations={n_eq} tables={n_tbl} -> {args.output}")


if __name__ == "__main__":
    main()
