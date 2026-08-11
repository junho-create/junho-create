"""
dots.ocr(dots-mocr) 추론 + 6000_test(표 crop 200장) TEDS 평가.

chandra_table / e18 evaluate.py 와 동일한 메트릭 파이프라인(_build_prediction_record)을 사용한다.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate import (  # noqa: E402
    DEFAULT_EMPTY_CELL_TOKEN,
    _build_metrics_dict_for_subset,
    _build_prediction_record,
    _load_eval_samples,
    _write_complexity_artifacts,
    _write_predictions_jsonl,
    extract_html_from_response,
)

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

_JSON_DECODER = json.JSONDecoder()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dots.ocr on table-only test jsonl.")
    parser.add_argument(
        "--test_data",
        required=True,
        help="Path to test.jsonl (e.g. 20260317_4_6000/data_split/test.jsonl)",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for predictions/metrics.")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument(
        "--api_url",
        default="http://192.168.75.174:26001/v1/chat/completions",
        help="dots-mocr OpenAI-compatible endpoint.",
    )
    parser.add_argument("--api_model", default="dots-mocr")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--max_new_tokens", type=int, default=16000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_backoff", type=float, default=1.0)
    parser.add_argument("--nested_teds_mode", default="split_mean")
    parser.add_argument("--normalize_empty_cells", action="store_true", default=True)
    parser.add_argument("--no_normalize_empty_cells", action="store_false", dest="normalize_empty_cells")
    parser.add_argument("--empty_cell_token", default=DEFAULT_EMPTY_CELL_TOKEN)
    return parser.parse_args()


def _image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _parse_response_json(raw_response: str) -> list[dict] | None:
    text = (raw_response or "").strip()
    if not text:
        return None

    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            for key in ("elements", "layout", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
    return None


def _salvage_json_objects(raw_response: str) -> list[dict]:
    text = (raw_response or "").lstrip()
    if text.startswith("["):
        text = text[1:]
    out: list[dict] = []
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= n or text[idx] == "]":
            break
        try:
            obj, end = _JSON_DECODER.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            out.append(obj)
        idx = end
    return out


def _normalize_category(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def _extract_table_html_from_dots(raw_response: str) -> tuple[str, str]:
    """dots layout JSON에서 Table HTML을 추출한다. (html, extraction_note)"""
    items = _parse_response_json(raw_response)
    if items is None:
        items = _salvage_json_objects(raw_response)

    table_htmls: list[str] = []
    if items:
        for item in items:
            cat = _normalize_category(item.get("category"))
            if cat != "table":
                continue
            text = item.get("text")
            if text is None:
                text = ""
            text = str(text).strip()
            if text:
                table_htmls.append(text)

    if table_htmls:
        if len(table_htmls) == 1:
            return table_htmls[0], "table_category"
        merged = max(table_htmls, key=len)
        return merged, f"table_category_max_of_{len(table_htmls)}"

    fallback = extract_html_from_response(raw_response)
    if fallback:
        return fallback, "response_html_fallback"
    return "", "empty"


def _call_dots_api(
    *,
    api_url: str,
    api_model: str,
    api_key: str,
    image_path: str,
    timeout: int,
    max_new_tokens: int,
) -> str:
    payload = {
        "model": api_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DOTS_LAYOUT_PROMPT},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                ],
            }
        ],
        "max_completion_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    if response.status_code == 400 and "max_completion_tokens" in payload:
        fallback_payload = dict(payload)
        token_value = fallback_payload.pop("max_completion_tokens", None)
        if token_value is not None:
            fallback_payload["max_tokens"] = token_value
            response = requests.post(api_url, json=fallback_payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def _infer_one_sample(
    sample: dict,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    started = time.time()
    last_error: Exception | None = None
    raw_response = ""
    for attempt in range(max(1, int(args.retries))):
        try:
            raw_response = _call_dots_api(
                api_url=args.api_url,
                api_model=args.api_model,
                api_key=args.api_key,
                image_path=sample["image_path"],
                timeout=args.timeout,
                max_new_tokens=args.max_new_tokens,
            )
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < args.retries:
                time.sleep(args.retry_backoff * (attempt + 1))
    elapsed = time.time() - started
    if last_error is not None:
        raise last_error

    pred_html, extract_note = _extract_table_html_from_dots(raw_response)
    record = _build_prediction_record(
        sample=sample,
        response=pred_html or raw_response,
        elapsed=elapsed,
        nested_teds_mode=args.nested_teds_mode,
        html_postprocess_mode="off",
        normalize_empty_cells=args.normalize_empty_cells,
        empty_cell_token=args.empty_cell_token,
    )
    record["backend"] = "dots_ocr"
    record["dots_extract_note"] = extract_note
    record["dots_response_raw"] = raw_response
    if pred_html:
        record["pred_html_raw"] = pred_html
        record["pred_html"] = pred_html
    return record, {"extract_note": extract_note}


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_eval_samples(
        test_data_path=args.test_data,
        max_samples=args.max_samples,
        gt_source="dataset",
        strict_aihub_gt=False,
        normalize_empty_cells=args.normalize_empty_cells,
        empty_cell_token=args.empty_cell_token,
    )
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        return 1

    print(f"samples={len(samples)} endpoint={args.api_url} model={args.api_model}")
    predictions: list[dict] = []
    extract_stats: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        futures = {executor.submit(_infer_one_sample, sample, args): sample for sample in samples}
        done = 0
        for future in as_completed(futures):
            sample = futures[future]
            done += 1
            record, meta = future.result()
            predictions.append(record)
            note = str(meta.get("extract_note", "unknown"))
            extract_stats[note] = extract_stats.get(note, 0) + 1
            print(
                f"[{done:03}/{len(samples)}] idx={record['index']} "
                f"teds={record['teds']:.4f} extract={note} "
                f"{Path(sample['image_path']).name}"
            )

    predictions.sort(key=lambda x: int(x["index"]))
    pred_path = output_dir / "predictions.jsonl"
    _write_predictions_jsonl(str(pred_path), predictions)

    metrics_dict = _build_metrics_dict_for_subset(
        predictions=predictions,
        backend="dots_ocr",
        batch_size=args.concurrency,
        max_new_tokens=args.max_new_tokens,
        gt_source="dataset",
        strict_aihub_gt=False,
        nested_teds_mode=args.nested_teds_mode,
        normalize_empty_cells=args.normalize_empty_cells,
        empty_cell_token=args.empty_cell_token,
    )
    metrics_dict["api_url"] = args.api_url
    metrics_dict["api_model"] = args.api_model
    metrics_dict["dots_extract_stats"] = extract_stats

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_complexity_artifacts(
        output_dir=str(output_dir),
        predictions=predictions,
        backend="dots_ocr",
        batch_size=args.concurrency,
        max_new_tokens=args.max_new_tokens,
        gt_source="dataset",
        strict_aihub_gt=False,
        nested_teds_mode=args.nested_teds_mode,
        normalize_empty_cells=args.normalize_empty_cells,
        empty_cell_token=args.empty_cell_token,
    )

    print("-" * 72)
    print(json.dumps(metrics_dict, ensure_ascii=False, indent=2))
    print(f"Saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
