"""
Filter predictions JSONL by metric and write:
1) filtered JSONL
2) interactive HTML report for filtered rows

The HTML report supports row checkboxes and can download checked-only results
as JSONL and HTML.

Usage:
  python -m data.filter_predictions_by_teds_s \
    --input eval_results/<run>/predictions.jsonl \
    --output eval_results/<run>/predictions_teds_s_not_1.jsonl \
    --report_html eval_results/<run>/report_teds_s_not_1.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _should_keep(value: float, target: float, op: str, atol: float) -> bool:
    if op == "ne":
        return abs(value - target) > atol
    if op == "eq":
        return abs(value - target) <= atol
    if op == "lt":
        return value < target
    if op == "le":
        return value <= target
    if op == "gt":
        return value > target
    return value >= target


def _resolve_report_image_src(row: dict, report_dir: Path) -> str:
    image_path = str(row.get("image_path") or "").strip()
    image_path_raw = str(row.get("image_path_raw") or "").strip()

    candidates = []
    if image_path:
        candidates.append(image_path)
    if image_path_raw:
        candidates.append(image_path_raw)

    basename = ""
    for candidate in candidates:
        try:
            name = Path(candidate).name
        except Exception:
            name = ""
        if name:
            basename = name
            break

    if basename:
        shared_rel = Path("..") / ".." / "images" / basename
        shared_abs = (report_dir / shared_rel).resolve()
        if shared_abs.exists():
            return shared_rel.as_posix()

    for candidate in candidates:
        p = Path(candidate)
        if p.is_absolute():
            if p.exists():
                return str(p)
            continue
        if (report_dir / p).exists():
            return candidate

    return candidates[0] if candidates else ""


def _escape_text(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _fmt_metric(value: object, digits: int = 4) -> str:
    num = _to_float(value)
    if num is not None:
        return f"{num:.{digits}f}"
    text = str(value).strip() if value is not None else ""
    return "-" if not text else html.escape(text)


def _avg_metric(rows: list[dict], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        num = _to_float(row.get(key))
        if num is not None:
            values.append(num)
    if not values:
        return None
    return sum(values) / len(values)


def _metric_class(value: object) -> str:
    num = _to_float(value)
    if num is None:
        return ""
    if num >= 0.9:
        return "metric-good"
    if num >= 0.7:
        return "metric-warn"
    return "metric-bad"


def _normalize_complexity(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    return text


def _badge_class(complexity: str) -> str:
    if complexity == "simple":
        return "badge-simple"
    if complexity == "medium":
        return "badge-medium"
    if complexity.startswith("complex"):
        return "badge-complex"
    return "badge-other"


def _render_iframe_srcdoc(table_html: object) -> str:
    raw = str(table_html or "").strip()
    if not raw:
        return ""

    preview_doc = (
        "<!doctype html><html><head><meta charset='UTF-8'>"
        "<style>"
        "body { margin: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }"
        "table { border-collapse: collapse; width: 100%; font-size: 12px; }"
        "td, th { border: 1px solid #cbd5e0; padding: 4px 6px; text-align: center; }"
        "th { background: #edf2f7; }"
        "</style>"
        "</head><body>"
        f"{raw}"
        "</body></html>"
    )
    return html.escape(preview_doc, quote=True)


def _build_summary_rows_html(rows: list[dict]) -> str:
    summary_items = [
        ("filtered_samples", str(len(rows))),
        ("avg_teds", _fmt_metric(_avg_metric(rows, "teds"), 4)),
        ("avg_teds_structure", _fmt_metric(_avg_metric(rows, "teds_structure"), 4)),
        ("avg_span_f1", _fmt_metric(_avg_metric(rows, "span_f1"), 4)),
        ("avg_attribute_accuracy", _fmt_metric(_avg_metric(rows, "attribute_accuracy"), 4)),
    ]
    return "".join(
        f"<tr><td>{_escape_text(key)}</td><td><b>{value}</b></td></tr>"
        for key, value in summary_items
    )


def _build_metric_items_html(row: dict) -> str:
    specs = [
        ("TEDS", "teds", 3),
        ("TEDS-S", "teds_structure", 3),
        ("Span F1", "span_f1", 3),
        ("Attr Acc", "attribute_accuracy", 3),
    ]
    if "teds_norm" in row:
        specs.insert(2, ("TEDS-N", "teds_norm", 3))
    if "teds_norm_structure" in row:
        specs.insert(3, ("TEDS-NS", "teds_norm_structure", 3))

    parts: list[str] = []
    for label, key, digits in specs:
        value = row.get(key)
        css = _metric_class(value)
        css_attr = f" metric-value {css}".strip()
        parts.append(
            "<div class=\"metric\">"
            f"<span class=\"metric-label\">{_escape_text(label)}:</span>"
            f"<span class=\"{css_attr}\">{_fmt_metric(value, digits)}</span>"
            "</div>"
        )

    inference_time = row.get("inference_time")
    if inference_time is not None:
        num = _to_float(inference_time)
        time_text = f"{num:.1f}s" if num is not None else _escape_text(inference_time)
        parts.append(
            "<div class=\"metric\">"
            "<span class=\"metric-label\">Time:</span>"
            f"<span class=\"metric-value\">{time_text}</span>"
            "</div>"
        )

    return "".join(parts)


def _build_cards_html(rows: list[dict]) -> str:
    parts: list[str] = []
    for i, row in enumerate(rows):
        image_src = str(
            row.get("image_src") or row.get("image_path") or row.get("image_path_raw") or ""
        ).strip()
        image_src_attr = html.escape(image_src, quote=True)
        image_name = Path(image_src).name if image_src else ""
        if not image_name:
            image_name = Path(str(row.get("image_path") or row.get("image_path_raw") or "")).name

        gt_html = str(row.get("gt_html") or "")
        pred_html = str(row.get("pred_html") or "")

        gt_iframe_srcdoc = _render_iframe_srcdoc(gt_html)
        pred_iframe_srcdoc = _render_iframe_srcdoc(pred_html)
        gt_preview_html = (
            f'<iframe class="table-preview" loading="lazy" sandbox srcdoc="{gt_iframe_srcdoc}"></iframe>'
            if gt_iframe_srcdoc
            else '<pre class="empty-preview">(empty gt_html)</pre>'
        )
        pred_preview_html = (
            f'<iframe class="table-preview" loading="lazy" sandbox srcdoc="{pred_iframe_srcdoc}"></iframe>'
            if pred_iframe_srcdoc
            else '<pre class="empty-preview">(empty pred_html)</pre>'
        )

        image_html = (
            f'<img src="{image_src_attr}" alt="table image" loading="lazy" />'
            if image_src
            else '<div class="image-empty">image_path 없음</div>'
        )
        image_path_html = (
            f'<div class="image-path">{_escape_text(image_src)}</div>' if image_src else ""
        )

        complexity = _normalize_complexity(row.get("complexity"))
        badge_css = _badge_class(complexity)
        index_text = _escape_text(row.get("index", i))
        sample_title = _escape_text(image_name or "unknown")
        metric_items = _build_metric_items_html(row)

        sample_html = f"""<div class="sample" data-row-idx="{i}" data-complexity="{_escape_text(complexity)}">
  <div class="sample-header">
    <div class="sample-head-left">
      <label class="check-label"><input type="checkbox" class="row-check" data-row-idx="{i}" /> 선택</label>
      <span class="sample-title">#{index_text} — {sample_title}</span>
    </div>
    <span class="badge {badge_css}">{_escape_text(complexity).upper()}</span>
  </div>
  <div class="sample-body">
    <div class="image-col">
      {image_html}
      {image_path_html}
    </div>
    <div class="table-col">
      <h3>✅ Ground Truth</h3>
      {gt_preview_html}
    </div>
    <div class="table-col">
      <h3>🤖 Prediction</h3>
      {pred_preview_html}
    </div>
  </div>
  <div class="code-wrap">
    <details>
      <summary>Show HTML code</summary>
      <div class="code-grid">
        <div class="code-box">
          <h4>GT HTML</h4>
          <pre>{_escape_text(gt_html)}</pre>
        </div>
        <div class="code-box">
          <h4>Prediction HTML</h4>
          <pre>{_escape_text(pred_html)}</pre>
        </div>
      </div>
    </details>
  </div>
  <div class="metrics-bar">
    {metric_items}
  </div>
</div>"""
        parts.append(sample_html)
    return "\n".join(parts)


def _build_interactive_report_html(
    rows: list[dict],
    *,
    report_title: str,
    input_path: Path,
    output_jsonl_path: Path,
    report_html_path: Path,
    filter_description: str,
) -> str:
    escaped_title = html.escape(report_title)
    escaped_input = html.escape(str(input_path))
    escaped_output = html.escape(str(output_jsonl_path))
    escaped_filter = html.escape(filter_description)

    rows_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    cards_html = _build_cards_html(rows)
    summary_rows_html = _build_summary_rows_html(rows)
    meta = {
        "report_title": report_title,
        "input_path": str(input_path),
        "output_jsonl_path": str(output_jsonl_path),
        "report_html_path": str(report_html_path),
        "output_jsonl_name": output_jsonl_path.name,
        "report_html_name": report_html_path.name,
    }
    meta_json = json.dumps(meta, ensure_ascii=False).replace("</", "<\\/")

    template = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }
    h1 { color: #333; margin: 0 0 14px 0; }
    .summary {
      background: #fff;
      padding: 20px;
      border-radius: 8px;
      margin-bottom: 16px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .summary table { border-collapse: collapse; }
    .summary td, .summary th { padding: 6px 16px; text-align: left; }
    .meta-line {
      margin: 6px 0 0 0;
      color: #4a5568;
      font-size: 13px;
      word-break: break-all;
    }
    .filter-bar {
      margin-bottom: 16px;
      position: sticky;
      top: 8px;
      z-index: 10;
      background: rgba(245, 245, 245, 0.95);
      backdrop-filter: blur(4px);
      padding: 8px;
      border-radius: 8px;
      border: 1px solid #d7dde7;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .filter-bar button {
      padding: 8px 14px;
      border: 1px solid #cbd5e0;
      border-radius: 4px;
      background: #fff;
      cursor: pointer;
      font-size: 13px;
    }
    .filter-bar button.primary {
      background: #2b6cb0;
      border-color: #2b6cb0;
      color: #fff;
    }
    .counter {
      margin-left: auto;
      color: #4a5568;
      font-size: 14px;
      white-space: nowrap;
    }
    .sample {
      background: #fff;
      margin-bottom: 24px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      overflow: hidden;
      border: 1px solid transparent;
    }
    .sample.selected {
      border-color: #2b6cb0;
      box-shadow: 0 0 0 2px rgba(43, 108, 176, 0.15);
    }
    .sample-header {
      background: #2d3748;
      color: #fff;
      padding: 12px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .sample-head-left {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }
    .check-label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #edf2f7;
      font-size: 13px;
      white-space: nowrap;
    }
    .check-label input {
      width: 15px;
      height: 15px;
      accent-color: #2b6cb0;
    }
    .sample-title {
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: min(62vw, 860px);
    }
    .sample-header .badge {
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: bold;
      text-transform: uppercase;
      color: #fff;
      letter-spacing: 0.02em;
    }
    .badge-simple { background: #48bb78; }
    .badge-medium { background: #ed8936; }
    .badge-complex { background: #e53e3e; }
    .badge-other { background: #718096; }
    .sample-body { display: grid; grid-template-columns: auto 1fr 1fr; gap: 0; }
    .image-col {
      padding: 16px;
      border-right: 1px solid #e2e8f0;
      max-width: 420px;
      min-width: 260px;
      background: #fff;
    }
    .image-col img {
      max-width: 100%;
      height: auto;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      background: #fff;
    }
    .image-empty {
      min-height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px dashed #cbd5e0;
      border-radius: 6px;
      color: #718096;
      font-size: 13px;
      padding: 16px;
    }
    .image-path {
      margin-top: 8px;
      color: #718096;
      font-size: 12px;
      word-break: break-all;
    }
    .table-col { padding: 16px; overflow-x: auto; }
    .table-col h3 { margin-top: 0; color: #555; font-size: 14px; }
    .table-preview {
      width: 100%;
      min-height: 260px;
      height: 360px;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      background: #fff;
    }
    .empty-preview {
      white-space: pre-wrap;
      word-break: break-word;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px;
      color: #718096;
      min-height: 120px;
      margin: 0;
    }
    .code-wrap {
      padding: 0 16px 14px 16px;
      border-top: 1px dashed #e2e8f0;
      background: #fcfcfd;
    }
    .code-wrap details {
      margin-top: 10px;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
    }
    .code-wrap summary {
      cursor: pointer;
      color: #4a5568;
      font-size: 13px;
      font-weight: 600;
      user-select: none;
    }
    .code-grid {
      margin-top: 10px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .code-box h4 { margin: 0 0 6px 0; color: #555; font-size: 12px; }
    .code-box pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px;
      max-height: 220px;
      overflow: auto;
      font-size: 12px;
      margin: 0;
    }
    .metrics-bar {
      display: flex;
      gap: 16px;
      padding: 12px 20px;
      background: #f7fafc;
      border-top: 1px solid #e2e8f0;
      font-size: 13px;
      flex-wrap: wrap;
    }
    .metric { display: flex; gap: 4px; }
    .metric-label { color: #718096; }
    .metric-value { font-weight: bold; color: #2d3748; }
    .metric-good { color: #48bb78; }
    .metric-warn { color: #ed8936; }
    .metric-bad { color: #e53e3e; }
    @media (max-width: 1200px) {
      .sample-body { grid-template-columns: 1fr; }
      .image-col { max-width: none; min-width: 0; border-right: none; border-bottom: 1px solid #e2e8f0; }
    }
    @media (max-width: 900px) {
      .code-grid { grid-template-columns: 1fr; }
      .counter { width: 100%; margin-left: 0; }
      .sample-title { max-width: 56vw; }
    }
  </style>
</head>
<body>
  <h1>🔍 __TITLE__</h1>

  <div class="summary">
    <h2>Summary</h2>
    <table id="summary-table">__SUMMARY_ROWS__</table>
    <p class="meta-line">Input: __INPUT__</p>
    <p class="meta-line">Filtered JSONL: __OUTPUT__</p>
    <p class="meta-line">Filter: __FILTER__</p>
  </div>

  <div id="selection-controls" class="filter-bar">
      <button id="select-all">전체 선택</button>
      <button id="clear-all">전체 해제</button>
      <button id="download-both" class="primary">체크된 결과 저장 (JSONL+HTML)</button>
      <span class="counter">
        selected <strong id="selected-count">0</strong> / total <strong id="total-count">__TOTAL_COUNT__</strong>
      </span>
  </div>

  <div id="samples">__CARDS_HTML__</div>

  <script id="rows-data" type="application/json">__ROWS_JSON__</script>
  <script id="meta-data" type="application/json">__META_JSON__</script>
  <script class="interactive-script">
  <!--
    var rows = JSON.parse(document.getElementById("rows-data").textContent || "[]");
    var meta = JSON.parse(document.getElementById("meta-data").textContent || "{}");
    var selectedCountEl = document.getElementById("selected-count");
    var totalCountEl = document.getElementById("total-count");
    totalCountEl.textContent = String(rows.length);

    function allChecks() {
      return Array.prototype.slice.call(document.querySelectorAll(".row-check"));
    }

    function updateSelectedCount() {
      var count = 0;
      allChecks().forEach(function (el) {
        var isChecked = !!el.checked;
        if (isChecked) count += 1;
        var sample = el.closest(".sample");
        if (sample) {
          sample.classList.toggle("selected", isChecked);
        }
      });
      selectedCountEl.textContent = String(count);
    }

    function selectedRowsAndIndexes() {
      var selectedRows = [];
      var selectedIndexes = [];
      allChecks().forEach(function (el) {
        if (!el.checked) return;
        var idx = Number(el.dataset.rowIdx || "");
        if (Number.isInteger(idx) && rows[idx]) {
          selectedRows.push(rows[idx]);
          selectedIndexes.push(idx);
        }
      });
      return { rows: selectedRows, indexes: selectedIndexes };
    }

    function filenameWithSuffix(name, suffix) {
      if (!name) return "selected" + suffix;
      var dot = name.lastIndexOf(".");
      if (dot <= 0) return name + suffix;
      return name.slice(0, dot) + suffix + name.slice(dot);
    }

    function triggerDownload(filename, content, mimeType) {
      var blob = new Blob([content], { type: mimeType });
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function buildSelectedHtml(selectedIndexes) {
      var selectedMap = {};
      selectedIndexes.forEach(function (idx) {
        selectedMap[String(idx)] = true;
      });

      var root = document.documentElement.cloneNode(true);
      Array.prototype.slice.call(root.querySelectorAll("script, #rows-data, #meta-data, #selection-controls"))
        .forEach(function (el) { el.remove(); });

      Array.prototype.slice.call(root.querySelectorAll(".sample")).forEach(function (sample) {
        var idx = sample.getAttribute("data-row-idx") || "";
        if (!selectedMap[idx]) {
          sample.remove();
          return;
        }
        var check = sample.querySelector(".row-check");
        if (check) check.checked = false;
        sample.classList.remove("selected");
      });

      var selectedCount = selectedIndexes.length;
      var selEl = root.querySelector("#selected-count");
      if (selEl) selEl.textContent = String(selectedCount);
      var totalEl = root.querySelector("#total-count");
      if (totalEl) totalEl.textContent = String(selectedCount);
      var summaryTable = root.querySelector("#summary-table");
      if (summaryTable) {
        var firstValueCell = summaryTable.querySelector("tr td:nth-child(2) b");
        if (firstValueCell) firstValueCell.textContent = String(selectedCount);
      }

      return "<!doctype html>\\n" + root.outerHTML;
    }

    allChecks().forEach(function (el) {
      el.addEventListener("change", updateSelectedCount);
    });
    updateSelectedCount();

    document.getElementById("select-all").addEventListener("click", function () {
      allChecks().forEach(function (el) { el.checked = true; });
      updateSelectedCount();
    });
    document.getElementById("clear-all").addEventListener("click", function () {
      allChecks().forEach(function (el) { el.checked = false; });
      updateSelectedCount();
    });
    document.getElementById("download-both").addEventListener("click", function () {
      var bundle = selectedRowsAndIndexes();
      if (bundle.rows.length === 0) {
        alert("선택된 항목이 없습니다.");
        return;
      }
      var jsonlContent = bundle.rows.map(function (row) { return JSON.stringify(row); }).join("\\n") + "\\n";
      var jsonlFilename = filenameWithSuffix(meta.output_jsonl_name || "selected.jsonl", "_checked");
      triggerDownload(jsonlFilename, jsonlContent, "application/x-ndjson;charset=utf-8");

      var htmlContent = buildSelectedHtml(bundle.indexes);
      var htmlFilename = filenameWithSuffix(meta.report_html_name || "selected_report.html", "_checked");
      triggerDownload(htmlFilename, htmlContent, "text/html;charset=utf-8");
    });
  //-->
  </script>
</body>
</html>
"""

    return (
        template.replace("__TITLE__", escaped_title)
        .replace("__INPUT__", escaped_input)
        .replace("__OUTPUT__", escaped_output)
        .replace("__FILTER__", escaped_filter)
        .replace("__SUMMARY_ROWS__", summary_rows_html)
        .replace("__TOTAL_COUNT__", str(len(rows)))
        .replace("__CARDS_HTML__", cards_html)
        .replace("__ROWS_JSON__", rows_json)
        .replace("__META_JSON__", meta_json)
    )


def _chunked_path(path: Path, part_index: int) -> Path:
    return path.with_name(f"{path.stem}_part{part_index:03d}{path.suffix}")


def _write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter predictions JSONL by metric value and write rows that pass the"
            " condition."
        )
    )
    parser.add_argument("--input", required=True, help="input predictions.jsonl path")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument(
        "--report_html",
        default="",
        help=(
            "interactive report html path "
            "(default: <output filename without ext>.html)"
        ),
    )
    parser.add_argument(
        "--report_title",
        default="Filtered Predictions Report",
        help="report title",
    )
    parser.add_argument(
        "--metric_field",
        default="teds_structure",
        help="metric field to check (default: teds_structure)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=1.0,
        help="target value to compare against (default: 1.0)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="absolute tolerance for floating-point equality (default: 1e-12)",
    )
    parser.add_argument(
        "--op",
        choices=("ne", "eq", "lt", "le", "gt", "ge"),
        default="ne",
        help="comparison operator vs target (default: ne)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help=(
            "optional chunk size for additional split outputs "
            "(0 disables chunked files; default: 0)"
        ),
    )
    args = parser.parse_args()

    if args.atol < 0:
        raise ValueError("--atol must be >= 0")
    if args.chunk_size < 0:
        raise ValueError("--chunk_size must be >= 0")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.report_html:
        report_html_path = Path(args.report_html)
    else:
        report_html_path = output_path.with_suffix(".html")

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_html_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped_missing_or_invalid = 0
    matched_output_rows: list[dict] = []
    matched_report_rows: list[dict] = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)

            value = _to_float(row.get(args.metric_field))
            if value is None:
                skipped_missing_or_invalid += 1
                continue

            keep = _should_keep(value, args.target, args.op, args.atol)
            if keep:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                matched_output_rows.append(row)
                row_for_report = dict(row)
                row_for_report["image_src"] = _resolve_report_image_src(
                    row_for_report, report_html_path.parent
                )
                matched_report_rows.append(row_for_report)

    filter_description = (
        f"{args.metric_field} {args.op} {args.target} (atol={args.atol})"
    )
    report_html = _build_interactive_report_html(
        matched_report_rows,
        report_title=args.report_title,
        input_path=input_path,
        output_jsonl_path=output_path,
        report_html_path=report_html_path,
        filter_description=filter_description,
    )
    report_html_path.write_text(report_html, encoding="utf-8")

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"report_html={report_html_path}")
    print(
        "total="
        f"{total}, written={written}, skipped_missing_or_invalid={skipped_missing_or_invalid}"
    )
    print(f"filter=({filter_description})")

    if args.chunk_size > 0 and written > args.chunk_size:
        chunk_count = math.ceil(written / args.chunk_size)
        print(
            f"chunking=enabled, chunk_size={args.chunk_size}, chunk_count={chunk_count}"
        )
        for part_idx in range(1, chunk_count + 1):
            start = (part_idx - 1) * args.chunk_size
            end = min(part_idx * args.chunk_size, written)
            chunk_output_rows = matched_output_rows[start:end]
            chunk_report_rows = matched_report_rows[start:end]

            chunk_output_path = _chunked_path(output_path, part_idx)
            chunk_report_path = _chunked_path(report_html_path, part_idx)

            _write_jsonl_rows(chunk_output_path, chunk_output_rows)
            chunk_report_html = _build_interactive_report_html(
                chunk_report_rows,
                report_title=f"{args.report_title} (part {part_idx}/{chunk_count})",
                input_path=input_path,
                output_jsonl_path=chunk_output_path,
                report_html_path=chunk_report_path,
                filter_description=(
                    f"{filter_description}, part {part_idx}/{chunk_count} "
                    f"(rows {start + 1}-{end})"
                ),
            )
            chunk_report_path.write_text(chunk_report_html, encoding="utf-8")
            print(f"chunk_output[{part_idx}]={chunk_output_path} rows={len(chunk_output_rows)}")
            print(f"chunk_report[{part_idx}]={chunk_report_path}")
    elif args.chunk_size > 0:
        print(
            "chunking=skipped "
            f"(written={written} <= chunk_size={args.chunk_size})"
        )


if __name__ == "__main__":
    main()
