"""
Training monitoring helpers.

Provides:
1) OCR score audit/sanitization for metadata.ocr_info
2) Long-table HTML capacity checks
3) Lightweight training progress HTML chart generation
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from utils.html_utils import extract_first_balanced_table

TR_RE = re.compile(r"<tr\b", re.IGNORECASE)


def extract_assistant_text(record: dict, include_thinking: bool = True) -> str:
    """레코드에서 assistant 텍스트(thinking + HTML)를 추출한다."""
    thinking = record.get("thinking", "")
    gt_html = record.get("gt_html", "")
    if include_thinking and thinking:
        return f"<think>\n{thinking}\n</think>\n{gt_html}"
    return gt_html


def extract_table_html_with_flags(text: str) -> tuple[str, bool, bool]:
    content = str(text or "")
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    lowered = content.lower()
    has_table_open = "<table" in lowered
    has_table_close = "</table>" in lowered
    table_html = extract_first_balanced_table(content)
    if table_html:
        return table_html, has_table_open, has_table_close
    return content.strip(), has_table_open, has_table_close


def audit_and_sanitize_ocr_scores(records: list[dict], remove_score: bool = True) -> dict:
    """OCR score 감사 및 정리."""
    stats = {
        "records_total": len(records),
        "records_with_ocr_info": 0,
        "records_with_ocr_score": 0,
        "ocr_items_total": 0,
        "ocr_items_with_score": 0,
        "ocr_items_score_removed": 0,
        "remove_score_enabled": bool(remove_score),
    }

    for rec in records:
        ocr_info = rec.get("ocr_info")
        if not isinstance(ocr_info, list):
            continue

        stats["records_with_ocr_info"] += 1
        has_score_in_record = False
        for item in ocr_info:
            if not isinstance(item, dict):
                continue
            stats["ocr_items_total"] += 1
            if "score" in item:
                has_score_in_record = True
                stats["ocr_items_with_score"] += 1
                if remove_score:
                    item.pop("score", None)
                    stats["ocr_items_score_removed"] += 1
        if has_score_in_record:
            stats["records_with_ocr_score"] += 1

    return stats


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    if p <= 0:
        return values[0]
    if p >= 1:
        return values[-1]
    idx = int(math.ceil((len(values) - 1) * p))
    return values[idx]


def _basic_stats(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}
    sorted_vals = sorted(values)
    return {
        "count": len(sorted_vals),
        "min": sorted_vals[0],
        "p50": _percentile(sorted_vals, 0.50),
        "p95": _percentile(sorted_vals, 0.95),
        "p99": _percentile(sorted_vals, 0.99),
        "max": sorted_vals[-1],
        "mean": float(mean(sorted_vals)),
    }


def _pick_sample_indices(total: int, limit: Optional[int]) -> list[int]:
    if total <= 0:
        return []
    if limit is None or limit <= 0 or limit >= total:
        return list(range(total))
    step = float(total) / float(limit)
    idxs = {min(total - 1, int(i * step)) for i in range(limit)}
    return sorted(idxs)


def _sample_key(record: dict, fallback_idx: int) -> str:
    """레코드 식별 키를 반환한다."""
    ip = record.get("image_path", "")
    if ip:
        return Path(ip).name
    return f"idx_{fallback_idx}"


def audit_long_table_capacity(
    records: list[dict],
    tokenizer: Any,
    dataset_name: str,
    max_seq_length: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    sample_limit: Optional[int] = 2000,
    include_thinking: bool = True,
) -> dict:
    indices = _pick_sample_indices(len(records), sample_limit)

    token_lengths: list[int] = []
    char_lengths: list[int] = []
    row_counts: list[int] = []
    truncated_like = 0
    top_long: list[tuple[int, dict]] = []

    for idx in indices:
        rec = records[idx]
        assistant_text = extract_assistant_text(
            rec,
            include_thinking=include_thinking,
        )
        table_html, has_open, has_close = extract_table_html_with_flags(assistant_text)
        if has_open and not has_close:
            truncated_like += 1

        char_len = len(table_html)
        char_lengths.append(char_len)

        row_count = len(TR_RE.findall(table_html))
        row_counts.append(row_count)

        token_len = 0
        if tokenizer is not None:
            try:
                token_len = len(tokenizer.encode(table_html, add_special_tokens=False))
            except Exception:
                token_len = 0
        token_lengths.append(token_len)

        top_long.append(
            (
                token_len,
                {
                    "sample": _sample_key(rec, idx),
                    "token_length": token_len,
                    "char_length": char_len,
                    "row_count": row_count,
                    "table_has_close_tag": bool(has_close),
                },
            )
        )

    top_long_sorted = [item for _, item in sorted(top_long, key=lambda x: x[0], reverse=True)[:5]]
    token_stats = _basic_stats(token_lengths)
    char_stats = _basic_stats(char_lengths)
    row_stats = _basic_stats(row_counts)

    warnings: list[str] = []
    if truncated_like > 0:
        warnings.append(
            f"{dataset_name}: {truncated_like}/{len(indices)} samples contain '<table' without '</table>'."
        )
    if max_seq_length and token_stats["max"] > int(max_seq_length):
        warnings.append(
            f"{dataset_name}: max HTML tokens ({token_stats['max']}) exceed max_seq_length ({max_seq_length})."
        )
    if max_seq_length and token_stats["p99"] > int(max_seq_length):
        warnings.append(
            f"{dataset_name}: p99 HTML tokens ({token_stats['p99']}) exceed max_seq_length ({max_seq_length})."
        )
    if max_new_tokens and token_stats["p99"] > int(max_new_tokens):
        warnings.append(
            f"{dataset_name}: p99 HTML tokens ({token_stats['p99']}) exceed max_new_tokens ({max_new_tokens})."
        )

    recommended_max_new_tokens = int(
        max(
            512,
            math.ceil(token_stats["p99"] * 1.25),
            math.ceil(token_stats["max"] * 1.10),
        )
    )

    return {
        "dataset": dataset_name,
        "records_total": len(records),
        "records_sampled": len(indices),
        "sample_limit": sample_limit,
        "include_thinking": bool(include_thinking),
        "table_open_without_close_count": truncated_like,
        "token_length_stats": token_stats,
        "char_length_stats": char_stats,
        "row_count_stats": row_stats,
        "recommended_max_new_tokens": recommended_max_new_tokens,
        "top_long_samples": top_long_sorted,
        "warnings": warnings,
    }


def write_json_report(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _chart_svg(points: list[tuple[float, float]], title: str, color: str) -> str:
    if not points:
        return f"<h3>{title}</h3><p>No data</p>"

    width = 860
    height = 260
    left = 56
    right = 18
    top = 20
    bottom = 36
    inner_w = width - left - right
    inner_h = height - top - bottom

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(max_x, min_x):
        max_x = min_x + 1.0
    if math.isclose(max_y, min_y):
        max_y = min_y + 1e-6

    def to_x(v: float) -> float:
        return left + (v - min_x) * inner_w / (max_x - min_x)

    def to_y(v: float) -> float:
        return top + inner_h - (v - min_y) * inner_h / (max_y - min_y)

    poly = " ".join(f"{to_x(x):.2f},{to_y(y):.2f}" for x, y in points)
    y_ticks = []
    for i in range(5):
        frac = i / 4.0
        yv = min_y + (max_y - min_y) * frac
        yy = to_y(yv)
        y_ticks.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
            f'<text x="{left-6}" y="{yy+4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{yv:.4g}</text>'
        )

    return (
        f"<h3>{title}</h3>"
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="{title} chart">'
        + "".join(y_ticks)
        + f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" '
          'stroke="#9ca3af" stroke-width="1.2"/>'
        + f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
          'stroke="#9ca3af" stroke-width="1.2"/>'
        + f'<polyline fill="none" stroke="{color}" stroke-width="2.0" points="{poly}" />'
        + f'<text x="{left}" y="{height-10}" font-size="11" fill="#6b7280">step: {min_x:.0f} → {max_x:.0f}</text>'
        + "</svg>"
    )


def write_training_progress_html(
    log_history: list[dict],
    output_path: str | Path,
    title: str,
) -> dict:
    train_loss: list[tuple[float, float]] = []
    eval_loss: list[tuple[float, float]] = []
    learning_rate: list[tuple[float, float]] = []

    fallback_step = 0.0
    for row in log_history:
        if not isinstance(row, dict):
            continue
        step = row.get("step")
        if step is None:
            fallback_step += 1.0
            step = fallback_step
        try:
            step_val = float(step)
        except Exception:
            continue

        if "loss" in row:
            try:
                train_loss.append((step_val, float(row["loss"])))
            except Exception:
                pass
        if "eval_loss" in row:
            try:
                eval_loss.append((step_val, float(row["eval_loss"])))
            except Exception:
                pass
        if "learning_rate" in row:
            try:
                learning_rate.append((step_val, float(row["learning_rate"])))
            except Exception:
                pass

    train_loss.sort(key=lambda x: x[0])
    eval_loss.sort(key=lambda x: x[0])
    learning_rate.sort(key=lambda x: x[0])

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    html = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        f"<title>{title} - Training Progress</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;color:#111827;background:#f9fafb;}"
        "h1{margin:0 0 6px 0;font-size:24px;} .meta{color:#6b7280;font-size:13px;margin-bottom:20px;}"
        ".card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin-bottom:16px;}"
        ".grid{display:grid;grid-template-columns:1fr;gap:12px;}"
        ".summary{font-size:13px;color:#374151;white-space:pre-wrap;}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        f"<div class='meta'>generated_at={generated_at}</div>"
        "<div class='grid'>"
        f"<div class='card'>{_chart_svg(train_loss, 'Train Loss', '#0ea5e9')}</div>"
        f"<div class='card'>{_chart_svg(eval_loss, 'Eval Loss', '#f97316')}</div>"
        f"<div class='card'>{_chart_svg(learning_rate, 'Learning Rate', '#22c55e')}</div>"
        "</div>"
        "<div class='card summary'>"
        f"train_points={len(train_loss)}\n"
        f"eval_points={len(eval_loss)}\n"
        f"lr_points={len(learning_rate)}"
        "</div>"
        "</body></html>"
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    return {
        "output_path": str(out),
        "train_points": len(train_loss),
        "eval_points": len(eval_loss),
        "lr_points": len(learning_rate),
    }
