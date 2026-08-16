"""
Repair evaluation JSONL GT using original AI-Hub HTML sidecar files.

This script replaces assistant GT in each record with canonical AI-Hub table HTML
derived from metadata.image_path (or image_map fallback).

Usage:
    python -m data.repair_eval_gt_from_aihub \
        --input data/processed/from_server/eval_used_data/eval_used_data_20260213_141329/eval_localized.jsonl \
        --image_map data/processed/from_server/eval_used_data/eval_used_data_20260213_141329/image_map.tsv \
        --output data/processed/from_server/eval_used_data/eval_used_data_20260213_141329/eval_localized_aihub_gt.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _extract_user_image(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                return str(item.get("image", ""))
    return ""


def _extract_table(full_html: str) -> str:
    match = re.search(r"(<table[\s\S]*?</table>)", full_html, re.IGNORECASE)
    if not match:
        return ""
    table_html = match.group(1).strip()
    table_html = table_html.replace(' rowspan="1"', "")
    table_html = table_html.replace(' colspan="1"', "")
    return table_html


def _resolve_existing_path(raw_path: str, input_jsonl: str) -> str:
    if not raw_path:
        return ""

    p = Path(raw_path).expanduser()
    candidates = [
        p,
        Path(input_jsonl).resolve().parent / p,
        Path(__file__).resolve().parent.parent / p,  # train/vlm
        Path(__file__).resolve().parent.parent.parent.parent / p,  # repo root
    ]

    seen = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return key
    return ""


def _load_image_map(path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path:
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            orig = str(row.get("original_image_path", "")).strip()
            local = str(row.get("localized_image_path", "")).strip()
            if orig and local:
                mapping[local] = orig
                mapping[Path(local).name] = orig
    return mapping


def _find_assistant_index(messages: list[dict]) -> int:
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            return i
    return -1


def repair_eval_gt(
    input_path: str,
    output_path: str,
    image_map_path: str = "",
    strict: bool = False,
) -> dict:
    image_map = _load_image_map(image_map_path)

    repaired = []
    stats = {
        "total": 0,
        "updated": 0,
        "missing_image_path": 0,
        "image_not_found": 0,
        "html_not_found": 0,
        "table_extract_failed": 0,
        "assistant_missing": 0,
        "strict_failed": 0,
    }
    failures: list[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            stats["total"] += 1
            rec = json.loads(line)

            messages = rec.get("messages", [])
            metadata = rec.get("metadata", {}) or {}
            md_image = str(metadata.get("image_path", "")).strip()
            user_image = _extract_user_image(messages)

            candidate_image = md_image
            if not candidate_image and user_image:
                candidate_image = image_map.get(user_image, "")
            if not candidate_image and user_image:
                candidate_image = image_map.get(Path(user_image).name, "")

            if not candidate_image:
                stats["missing_image_path"] += 1
                failures.append(f"idx={idx}: missing image path")
                if strict:
                    stats["strict_failed"] += 1
                    continue
                repaired.append(rec)
                continue

            image_path = _resolve_existing_path(candidate_image, input_path)
            if not image_path:
                stats["image_not_found"] += 1
                failures.append(f"idx={idx}: image not found ({candidate_image})")
                if strict:
                    stats["strict_failed"] += 1
                    continue
                repaired.append(rec)
                continue

            html_path = str(Path(image_path).with_suffix(".html"))
            if not Path(html_path).exists():
                stats["html_not_found"] += 1
                failures.append(f"idx={idx}: html not found ({html_path})")
                if strict:
                    stats["strict_failed"] += 1
                    continue
                repaired.append(rec)
                continue

            with open(html_path, "r", encoding="utf-8", errors="ignore") as hf:
                full_html = hf.read()
            table_html = _extract_table(full_html)
            if not table_html:
                stats["table_extract_failed"] += 1
                failures.append(f"idx={idx}: table extract failed ({html_path})")
                if strict:
                    stats["strict_failed"] += 1
                    continue
                repaired.append(rec)
                continue

            ai = _find_assistant_index(messages)
            if ai < 0:
                stats["assistant_missing"] += 1
                messages = list(messages)
                messages.append({"role": "assistant", "content": table_html})
            else:
                messages = list(messages)
                msg = dict(messages[ai])
                msg["content"] = table_html
                messages[ai] = msg

            rec["messages"] = messages
            rec["metadata"] = dict(metadata)
            rec["metadata"]["gt_source"] = "aihub_html"
            rec["metadata"]["gt_html_path"] = html_path
            stats["updated"] += 1
            repaired.append(rec)

    with open(output_path, "w", encoding="utf-8") as wf:
        for rec in repaired:
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if failures:
        print("Repair warnings (first 20):")
        for line in failures[:20]:
            print(f"  - {line}")

    print("Repair summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  output: {output_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace eval JSONL GT with AI-Hub original HTML."
    )
    parser.add_argument("--input", required=True, help="Input eval JSONL path")
    parser.add_argument("--output", default="", help="Output JSONL path")
    parser.add_argument(
        "--image_map",
        default="",
        help="Optional image_map.tsv (localized_image_path -> original_image_path)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail each problematic sample instead of fallback pass-through",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input file in place",
    )
    args = parser.parse_args()

    if args.inplace and args.output:
        raise ValueError("--inplace and --output cannot be used together.")

    input_path = str(Path(args.input).resolve())
    if args.inplace:
        output_path = input_path
    elif args.output:
        output_path = str(Path(args.output).resolve())
    else:
        p = Path(input_path)
        output_path = str(p.with_name(f"{p.stem}_aihub_gt{p.suffix}"))

    repair_eval_gt(
        input_path=input_path,
        output_path=output_path,
        image_map_path=args.image_map,
        strict=bool(args.strict),
    )


if __name__ == "__main__":
    main()
