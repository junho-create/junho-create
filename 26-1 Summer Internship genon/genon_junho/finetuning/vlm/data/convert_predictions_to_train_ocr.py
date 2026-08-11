"""
Convert eval predictions.jsonl into train_ocr-style JSONL.

Output schema (per row):
  - image_path
  - gt_html  (from prediction `pred_html` by default)
  - thinking
  - complexity
  - prompt_style
  - ocr_info
  - bbox_scale

Usage:
  python -m data.convert_predictions_to_train_ocr \
    --input eval_results/<run>/predictions.jsonl \
    --output _train_data/<set>/data_ocr/train_ocr_from_pred.jsonl

  # Reuse OCR info by image basename from an existing OCR dataset:
  python -m data.convert_predictions_to_train_ocr \
    --input eval_results/<run>/predictions.jsonl \
    --output _train_data/<set>/data_ocr/train_ocr_from_pred.jsonl \
    --ocr_lookup_jsonl _train_data/<set>/data_ocr/train_ocr.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _basename(path_like: str) -> str:
    s = str(path_like or "").strip().replace("\\", "/")
    if not s:
        return ""
    s = re.split(r"[?#]", s, maxsplit=1)[0]
    return s.rsplit("/", 1)[-1]


def _load_ocr_lookup(path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        image_path = str(row.get("image_path", "")).strip()
        base = _basename(image_path)
        if not base:
            continue
        value = {
            "ocr_info": row.get("ocr_info", []),
            "bbox_scale": row.get("bbox_scale", 1024),
            "prompt_style": row.get("prompt_style", "chandra_table_with_ocr"),
            "thinking": row.get("thinking", ""),
        }
        prev = lookup.get(base)
        if prev is not None and prev != value:
            raise ValueError(
                f"basename collision in lookup with different payload: {base}"
            )
        lookup[base] = value
    return lookup


def _resolve_image(pred: dict, image_field: str) -> str:
    if image_field == "auto":
        v = str(pred.get("image_path_raw", "")).strip()
        if v:
            return v
        return str(pred.get("image_path", "")).strip()
    return str(pred.get(image_field, "")).strip()


def _resolve_gt_html(pred: dict, html_field: str) -> str:
    if html_field == "auto":
        # Backward compatibility: treat "auto" as "pred_html".
        html_field = "pred_html"
    return str(pred.get(html_field, "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert eval predictions JSONL into train_ocr-style JSONL"
    )
    parser.add_argument("--input", required=True, help="predictions.jsonl path")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument(
        "--html_field",
        default="pred_html",
        help="source html field (default: pred_html, auto is alias of pred_html)",
    )
    parser.add_argument(
        "--image_field",
        default="auto",
        help="source image field (default: auto -> image_path_raw then image_path)",
    )
    parser.add_argument(
        "--image_mode",
        choices=("basename", "raw"),
        default="basename",
        help="basename -> images/<file>, raw -> keep source image path",
    )
    parser.add_argument(
        "--prompt_style",
        default="chandra_table_with_ocr",
        help="default prompt_style when lookup is unavailable",
    )
    parser.add_argument(
        "--bbox_scale",
        type=int,
        default=1024,
        help="default bbox_scale when lookup is unavailable",
    )
    parser.add_argument(
        "--default_complexity",
        default="simple",
        help="default complexity if prediction row has no complexity",
    )
    parser.add_argument(
        "--ocr_lookup_jsonl",
        default="",
        help=(
            "optional JSONL with image_path/ocr_info/bbox_scale/prompt_style/thinking "
            "(matched by image basename)"
        ),
    )
    parser.add_argument(
        "--drop_empty_html",
        action="store_true",
        help="skip rows where selected html field is empty",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.bbox_scale <= 0:
        raise ValueError("--bbox_scale must be positive")
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    lookup = {}
    if args.ocr_lookup_jsonl:
        lookup_path = Path(args.ocr_lookup_jsonl)
        if not lookup_path.exists():
            raise FileNotFoundError(f"ocr_lookup_jsonl not found: {lookup_path}")
        lookup = _load_ocr_lookup(lookup_path)

    total = 0
    written = 0
    skipped_empty_html = 0
    lookup_hit = 0

    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            total += 1

            pred = json.loads(line)
            src_image = _resolve_image(pred, args.image_field)
            base = _basename(src_image)
            if not base:
                raise ValueError(f"missing image path at row index={pred.get('index')}")

            gt_html = _resolve_gt_html(pred, args.html_field)
            if args.drop_empty_html and not gt_html:
                skipped_empty_html += 1
                continue

            if args.image_mode == "basename":
                image_path = f"images/{base}"
            else:
                image_path = src_image

            ocr_payload = lookup.get(base)
            if ocr_payload is not None:
                lookup_hit += 1
                ocr_info = ocr_payload.get("ocr_info", [])
                bbox_scale = int(ocr_payload.get("bbox_scale", args.bbox_scale))
                prompt_style = str(
                    ocr_payload.get("prompt_style", args.prompt_style)
                )
                thinking = str(ocr_payload.get("thinking", ""))
            else:
                ocr_info = []
                bbox_scale = int(args.bbox_scale)
                prompt_style = str(args.prompt_style)
                thinking = ""

            out = {
                "image_path": image_path,
                "gt_html": gt_html,
                "thinking": thinking,
                "complexity": str(pred.get("complexity", args.default_complexity)),
                "prompt_style": prompt_style,
                "ocr_info": ocr_info,
                "bbox_scale": bbox_scale,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"total={total}, written={written}, skipped_empty_html={skipped_empty_html}")
    print(f"ocr_lookup_size={len(lookup)}, ocr_lookup_hit={lookup_hit}")


if __name__ == "__main__":
    main()
