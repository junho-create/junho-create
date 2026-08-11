"""
Convert TFLOP-style JSONL to eval-ready JSONL for `python -m eval.evaluate`.

Input row example:
{
  "file_name": "table_001.png",
  "org_html": ["<thead>", ...],
  "dr_coord": {"0": [[[x1,y1,x2,y2]], 0, "text"], ...},
  ...
}

Output row example:
{
  "image_path": "train/tflop/dataset/.../images/train/table_001.png",
  "gt_html": "<table>...</table>",
  "complexity": "complex",
  "prompt_style": "chandra_table_with_ocr",
  "ocr_info": [{"text": "...", "bbox": [x0,y0,x1,y1]}, ...],
  "bbox_scale": 1024
}
"""

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def _clamp_int(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))


def _normalize_bbox(
    bbox: list[float],
    width: int,
    height: int,
    bbox_scale: int,
) -> list[int]:
    x0, y0, x1, y1 = bbox
    x_min, x_max = sorted([float(x0), float(x1)])
    y_min, y_max = sorted([float(y0), float(y1)])

    if width <= 0 or height <= 0:
        return [
            _clamp_int(x_min, 0, bbox_scale),
            _clamp_int(y_min, 0, bbox_scale),
            _clamp_int(x_max, 0, bbox_scale),
            _clamp_int(y_max, 0, bbox_scale),
        ]

    return [
        _clamp_int((x_min / width) * bbox_scale, 0, bbox_scale),
        _clamp_int((y_min / height) * bbox_scale, 0, bbox_scale),
        _clamp_int((x_max / width) * bbox_scale, 0, bbox_scale),
        _clamp_int((y_max / height) * bbox_scale, 0, bbox_scale),
    ]


def _ensure_table_wrapper(html: str) -> str:
    text = (html or "").strip()
    if not text:
        return ""
    if "<table" in text.lower():
        return text
    return f"<table>{text}</table>"


def _coerce_html(org_html: Any) -> str:
    if isinstance(org_html, list):
        html = "".join(str(t) for t in org_html)
    elif isinstance(org_html, str):
        html = org_html
    else:
        html = ""
    return _ensure_table_wrapper(html)


def _sorted_items_by_key(data: dict) -> list[tuple[str, Any]]:
    def _sort_key(k: str) -> tuple[int, str]:
        try:
            return (0, str(int(k)))
        except Exception:
            return (1, k)

    return sorted(data.items(), key=lambda kv: _sort_key(str(kv[0])))


def _parse_dr_coord_item(raw: Any) -> tuple[list[float], str] | None:
    if not isinstance(raw, list) or len(raw) < 3:
        return None

    bbox_container = raw[0]
    text = str(raw[2] or "")

    bbox = None
    if (
        isinstance(bbox_container, list)
        and len(bbox_container) > 0
        and isinstance(bbox_container[0], list)
        and len(bbox_container[0]) >= 4
    ):
        bbox = bbox_container[0][:4]
    elif isinstance(bbox_container, list) and len(bbox_container) >= 4:
        bbox = bbox_container[:4]

    if bbox is None:
        return None

    try:
        return [float(v) for v in bbox], text
    except Exception:
        return None


def _extract_ocr_from_dr_coord(
    dr_coord: Any,
    width: int,
    height: int,
    bbox_scale: int,
) -> list[dict]:
    if not isinstance(dr_coord, dict):
        return []

    out: list[dict] = []
    for _, raw in _sorted_items_by_key(dr_coord):
        parsed = _parse_dr_coord_item(raw)
        if parsed is None:
            continue
        bbox, text = parsed
        out.append(
            {
                "text": text,
                "bbox": _normalize_bbox(bbox, width=width, height=height, bbox_scale=bbox_scale),
            }
        )
    return out


def _extract_ocr_from_gold_coord(
    gold_coord: Any,
    width: int,
    height: int,
    bbox_scale: int,
) -> list[dict]:
    if not isinstance(gold_coord, list):
        return []

    out: list[dict] = []
    for raw in gold_coord:
        if not isinstance(raw, str):
            continue
        parts = raw.strip().split(" ", 5)
        if len(parts) < 6:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in parts[:4]]
        except Exception:
            continue
        if x0 == -1.0 and y0 == -1.0 and x1 == -1.0 and y1 == -1.0:
            continue
        out.append(
            {
                "text": parts[5],
                "bbox": _normalize_bbox([x0, y0, x1, y1], width=width, height=height, bbox_scale=bbox_scale),
            }
        )
    return out


def _resolve_image_path(
    image_path: Path,
    repo_root: Path,
    mode: str,
) -> str:
    if mode == "absolute":
        return str(image_path.resolve())
    if mode == "repo_relative":
        return str(image_path.resolve().relative_to(repo_root.resolve()))
    raise ValueError(f"Unsupported image_path_mode: {mode}")


def _load_html_map(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"html_map_json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("html_map_json must be an object mapping file_name -> payload")
    out: dict[str, dict] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def convert(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    image_dir = Path(args.image_dir).resolve()
    html_map = _load_html_map(args.html_map_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    missing_images = 0
    empty_gt = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            total += 1

            record = json.loads(line)
            file_name = str(record.get("file_name", "")).strip()
            if not file_name:
                print(f"Warning: missing file_name at line {line_no}, skipping")
                continue

            image_path = image_dir / file_name
            if not image_path.exists():
                missing_images += 1
                print(f"Warning: image not found ({image_path}), skipping")
                continue

            mapped = html_map.get(file_name, {})
            mapped_html = mapped.get("html", "") if isinstance(mapped, dict) else ""
            gt_html = _ensure_table_wrapper(str(mapped_html or "").strip())
            if not gt_html:
                gt_html = _coerce_html(record.get("org_html", []))
            if not gt_html:
                empty_gt += 1

            with Image.open(image_path) as img:
                width, height = img.size

            if args.coord_source == "dr_coord":
                ocr_info = _extract_ocr_from_dr_coord(
                    record.get("dr_coord", {}),
                    width=width,
                    height=height,
                    bbox_scale=args.bbox_scale,
                )
            elif args.coord_source == "gold_coord":
                ocr_info = _extract_ocr_from_gold_coord(
                    record.get("gold_coord", []),
                    width=width,
                    height=height,
                    bbox_scale=args.bbox_scale,
                )
            else:
                ocr_info = []

            out_rec = {
                "image_path": _resolve_image_path(
                    image_path=image_path,
                    repo_root=repo_root,
                    mode=args.image_path_mode,
                ),
                "gt_html": gt_html,
                "complexity": str(mapped.get("type", args.default_complexity)),
                "prompt_style": args.prompt_style,
                "ocr_info": ocr_info,
                "bbox_scale": int(args.bbox_scale),
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            written += 1

    print("Conversion complete")
    print(f"- input:   {input_path}")
    print(f"- output:  {output_path}")
    print(f"- total:   {total}")
    print(f"- written: {written}")
    print(f"- missing_images_skipped: {missing_images}")
    print(f"- empty_gt_html_rows:     {empty_gt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert TFLOP-style JSONL to eval-ready JSONL")
    parser.add_argument("--input", required=True, help="Input TFLOP JSONL path")
    parser.add_argument("--output", required=True, help="Output eval JSONL path")
    parser.add_argument("--image_dir", required=True, help="Image directory for file_name lookup")
    parser.add_argument(
        "--html_map_json",
        default="",
        help="Optional JSON map file_name -> {html, type}. Example: eval_51.json",
    )
    parser.add_argument(
        "--coord_source",
        choices=["dr_coord", "gold_coord", "none"],
        default="dr_coord",
        help="OCR source field (default: dr_coord)",
    )
    parser.add_argument("--bbox_scale", type=int, default=1024, help="Normalized bbox scale")
    parser.add_argument(
        "--image_path_mode",
        choices=["repo_relative", "absolute"],
        default="repo_relative",
        help="How to store image_path in output",
    )
    parser.add_argument(
        "--default_complexity",
        default="complex",
        help="Complexity label written to every row (default: complex)",
    )
    parser.add_argument(
        "--prompt_style",
        default="chandra_table_with_ocr",
        help="Prompt style metadata for output rows",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.bbox_scale <= 0:
        raise ValueError("--bbox_scale must be positive")

    convert(args)
