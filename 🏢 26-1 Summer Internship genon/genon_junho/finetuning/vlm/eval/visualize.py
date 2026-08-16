"""
결과 시각화 - HTML 렌더링 비교

예측과 정답을 나란히 렌더링하여 시각적으로 비교할 수 있는 HTML 리포트를 생성한다.

Usage:
    python -m eval.visualize \
        --predictions eval_results/predictions.jsonl \
        --output eval_results/comparison.html \
        --max_samples 50 \
        --sort_by teds
"""

import argparse
import base64
import html
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>TSR Evaluation - Prediction vs Ground Truth</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .summary {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .summary table {{ border-collapse: collapse; }}
  .summary td, .summary th {{ padding: 6px 16px; text-align: left; }}
  .sample {{ background: #fff; margin-bottom: 24px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }}
  .sample-header {{ background: #2d3748; color: #fff; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; }}
  .sample-header .badge {{ padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
  .badge-simple {{ background: #48bb78; }}
  .badge-medium {{ background: #ed8936; }}
  .badge-complex {{ background: #e53e3e; }}
  .sample-body {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 0; }}
  .image-col {{ padding: 16px; border-right: 1px solid #e2e8f0; max-width: 400px; }}
  .image-col img {{ max-width: 100%; height: auto; }}
  .table-col {{ padding: 16px; overflow-x: auto; }}
  .table-col h3 {{ margin-top: 0; color: #555; font-size: 14px; }}
  .table-col table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  .table-col td, .table-col th {{ border: 1px solid #cbd5e0; padding: 6px 10px; text-align: center; }}
  .table-col th {{ background: #edf2f7; }}
  .table-preview {{ width: 100%; min-height: 260px; height: 360px; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; }}
  .table-col pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; max-height: 220px; overflow: auto; }}
  .metrics-bar {{ display: flex; gap: 16px; padding: 12px 20px; background: #f7fafc; border-top: 1px solid #e2e8f0; font-size: 13px; }}
  .metric {{ display: flex; gap: 4px; }}
  .metric-label {{ color: #718096; }}
  .metric-value {{ font-weight: bold; }}
  .metric-good {{ color: #48bb78; }}
  .metric-warn {{ color: #ed8936; }}
  .metric-bad {{ color: #e53e3e; }}
  .filter-bar {{ margin-bottom: 16px; }}
  .filter-bar button {{ padding: 8px 16px; margin-right: 8px; border: 1px solid #cbd5e0; border-radius: 4px; background: #fff; cursor: pointer; }}
  .filter-bar button.active {{ background: #2d3748; color: #fff; border-color: #2d3748; }}
  .code-wrap {{ padding: 0 16px 14px 16px; border-top: 1px dashed #e2e8f0; background: #fcfcfd; }}
  .code-wrap details {{ margin-top: 10px; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 10px; background: #fff; }}
  .code-wrap summary {{ cursor: pointer; color: #4a5568; font-size: 13px; font-weight: 600; user-select: none; }}
  .code-grid {{ margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .code-box h4 {{ margin: 0 0 6px 0; color: #555; font-size: 12px; }}
  .code-box pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; max-height: 220px; overflow: auto; font-size: 12px; }}
</style>
</head>
<body>
<h1>🔍 TSR Evaluation Report</h1>

<div class="summary">
  <h2>Summary</h2>
  {summary_html}
</div>

<div class="filter-bar">
  <button class="active" onclick="filterSamples(event, 'all')">All ({total})</button>
  <button onclick="filterSamples(event, 'complex')">Complex ({complex_count})</button>
  <button onclick="filterSamples(event, 'medium')">Medium ({medium_count})</button>
  <button onclick="filterSamples(event, 'simple')">Simple ({simple_count})</button>
  <button onclick="filterSamples(event, 'errors')">Errors (Span F1 &lt; 0.5)</button>
</div>

<div id="samples">
  {samples_html}
</div>

<script>
function filterSamples(event, type) {{
  document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
  if (event && event.target) event.target.classList.add('active');

  document.querySelectorAll('.sample').forEach(el => {{
    const comp = el.dataset.complexity;
    const f1 = parseFloat(el.dataset.spanf1);
    if (type === 'all') el.style.display = '';
    else if (type === 'errors') el.style.display = f1 < 0.5 ? '' : 'none';
    else el.style.display = comp === type ? '' : 'none';
  }});
}}
</script>
</body>
</html>
"""


TABLE_IFRAME_STYLE = """
<style>
  body { margin: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; }
  td, th { border: 1px solid #cbd5e0; padding: 4px 6px; text-align: center; }
  th { background: #edf2f7; }
</style>
"""

_TABLE_OPEN_TAG_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_TABLE_ANY_TAG_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)


def image_to_base64(image_path: str) -> str:
    """이미지를 base64로 인코딩."""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        ext = Path(image_path).suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
            ext.lstrip("."), "image/png"
        )
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ""


def _normalize_image_extension(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        return "jpg"
    if ext in {"png", "webp", "gif", "bmp"}:
        return ext
    return "jpg"


def _ensure_shared_image_file(image_path: str, shared_image_dir: str) -> str:
    """공유 이미지 디렉터리에 파일을 저장(동일 basename 우선, 충돌 시 해시명)하고 경로를 반환한다."""
    src = Path(image_path)
    if not src.is_file():
        return ""

    shared_dir = Path(shared_image_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)

    data = src.read_bytes()
    basename_dst = shared_dir / src.name
    if basename_dst.exists():
        try:
            if basename_dst.read_bytes() == data:
                return str(basename_dst)
        except Exception:
            pass
    else:
        basename_dst.write_bytes(data)
        return str(basename_dst)

    # basename 충돌 시 해시명으로 저장
    digest = hashlib.sha256(data).hexdigest()[:24]
    ext = _normalize_image_extension(image_path)
    hash_dst = shared_dir / f"img_{digest}.{ext}"
    if not hash_dst.exists():
        hash_dst.write_bytes(data)
    return str(hash_dst)


def _resolve_image_path(pred: dict, image_root: Optional[str] = None) -> str:
    """예측 레코드의 이미지 경로를 로컬 경로로 보정한다."""
    train_vlm_root = Path(__file__).resolve().parent.parent
    raw_value = (pred.get("image_path_raw") or "").strip()
    image_value = (pred.get("image_path") or "").strip()

    candidates: list[Path] = []

    def add_candidate(path_value: str):
        if not path_value:
            return
        p = Path(path_value)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(train_vlm_root / p)
            candidates.append(Path.cwd() / p)
            if image_root:
                candidates.append(Path(image_root) / p)

    add_candidate(image_value)
    add_candidate(raw_value)

    # 서버 절대경로(/root/.../train/vlm/...)를 로컬 train/vlm 루트로 매핑
    if image_value:
        norm = image_value.replace("\\", "/")
        marker = "/train/vlm/"
        if marker in norm:
            suffix = norm.split(marker, 1)[1]
            candidates.append(train_vlm_root / suffix)
            if image_root:
                candidates.append(Path(image_root) / suffix)

    # 중복 제거 + 존재하는 첫 경로 반환
    seen = set()
    for c in candidates:
        try:
            key = str(c.resolve(strict=False))
        except Exception:
            key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.is_file():
            return str(c)

    return ""


def metric_class(value: float, thresholds=(0.8, 0.5)) -> str:
    """메트릭 값에 따른 CSS 클래스."""
    if value >= thresholds[0]:
        return "metric-good"
    elif value >= thresholds[1]:
        return "metric-warn"
    return "metric-bad"


def _strip_markdown_fence(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_balanced_table(text: str) -> str:
    """문자열에서 첫 번째 balanced <table>...</table> 조각을 추출한다."""
    s = str(text or "").strip()
    if not s:
        return ""

    # 공용 유틸이 있으면 우선 사용 (중복 구현 방지)
    try:
        from utils.html_utils import extract_first_balanced_table as _extract

        fragment = _extract(s)
        if fragment:
            return str(fragment).strip()
    except Exception:
        pass

    start_match = _TABLE_OPEN_TAG_RE.search(s)
    if start_match is None:
        return ""

    start = start_match.start()
    depth = 0
    for m in _TABLE_ANY_TAG_RE.finditer(s, start):
        tag = m.group(0).lower()
        is_close = tag.startswith("</")
        is_self_closing = tag.endswith("/>")

        if not is_close:
            if is_self_closing:
                if depth == 0:
                    return s[start:m.end()].strip()
                continue
            depth += 1
            continue

        if depth <= 0:
            continue
        depth -= 1
        if depth == 0:
            return s[start:m.end()].strip()

    # 비정상 응답(닫힘 누락)일 때 table 시작 이후를 반환
    return s[start:].strip()


def _extract_table_fragment(text: str) -> str:
    s = _strip_markdown_fence(text)
    fragment = _extract_first_balanced_table(s)
    if fragment:
        return fragment
    match = re.search(r"<table\b", s, flags=re.IGNORECASE)
    if not match:
        return ""
    start = match.start()
    close_match = re.search(r"</table\s*>", s[start:], flags=re.IGNORECASE)
    if close_match:
        end = start + close_match.end()
        return s[start:end]
    # 끊긴 HTML로 인해 DOM이 깨지지 않도록 최소한의 종결 태그를 보강
    return s[start:] + "</table>"


def _render_table_preview(raw_html: str) -> str:
    fragment = _extract_table_fragment(raw_html)
    raw = (raw_html or "").strip()
    if not fragment:
        if not raw:
            return '<p style="color:#999">No HTML</p>'
        return f"<pre>{html.escape(raw)}</pre>"

    doc = (
        "<!doctype html><html><head><meta charset='UTF-8'>"
        + TABLE_IFRAME_STYLE
        + "</head><body>"
        + fragment
        + "</body></html>"
    )
    srcdoc = html.escape(doc, quote=True)
    return f'<iframe class="table-preview" loading="lazy" sandbox srcdoc="{srcdoc}"></iframe>'


def _render_html_code(raw_html: str) -> str:
    code = _extract_table_fragment(raw_html)
    if not code:
        code = _strip_markdown_fence(raw_html).strip()
    if not code:
        return "<pre>No HTML</pre>"
    return f"<pre>{html.escape(code)}</pre>"


def render_sample(
    pred: dict,
    idx: int,
    embed_images: bool = True,
    image_mode: str = "embed",
    image_root: Optional[str] = None,
    shared_image_dir: Optional[str] = None,
    report_dir: Optional[str] = None,
) -> str:
    """단일 샘플의 HTML을 생성."""
    complexity = pred.get("complexity", "unknown")
    span_f1 = pred.get("span_f1", 0.0)
    attr_acc = pred.get("attribute_accuracy", 0.0)
    teds = pred.get("teds", 0.0)
    teds_structure = pred.get("teds_structure", 0.0)
    teds_norm = pred.get("teds_norm", 0.0)
    teds_norm_struct = pred.get("teds_norm_structure", 0.0)

    resolved_mode = image_mode
    if not embed_images and image_mode == "embed":
        resolved_mode = "none"

    # 이미지
    image_html = ""
    if resolved_mode != "none" and (pred.get("image_path") or pred.get("image_path_raw")):
        resolved_image_path = _resolve_image_path(pred, image_root=image_root)
        if resolved_image_path and resolved_mode == "external":
            if not shared_image_dir:
                display_path = pred.get("image_path_raw") or pred.get("image_path")
                image_html = f'<p style="color:#999">Shared image dir is not set: {display_path}</p>'
            else:
                shared_file = _ensure_shared_image_file(resolved_image_path, shared_image_dir)
                if shared_file:
                    if report_dir:
                        rel_src = os.path.relpath(shared_file, start=report_dir)
                    else:
                        rel_src = shared_file
                    image_html = f'<img src="{html.escape(rel_src, quote=True)}" alt="table image">'
                else:
                    display_path = pred.get("image_path_raw") or pred.get("image_path")
                    image_html = f'<p style="color:#999">Image not found: {display_path}</p>'
        elif resolved_mode == "external" and shared_image_dir:
            # 원본 경로를 찾지 못한 경우에도 shared_image_dir에 basename 파일이 있으면 재사용
            raw_name = os.path.basename(pred.get("image_path") or pred.get("image_path_raw") or "")
            candidate = Path(shared_image_dir) / raw_name
            if raw_name and candidate.is_file():
                if report_dir:
                    rel_src = os.path.relpath(str(candidate), start=report_dir)
                else:
                    rel_src = str(candidate)
                image_html = f'<img src="{html.escape(rel_src, quote=True)}" alt="table image">'
            else:
                display_path = pred.get("image_path_raw") or pred.get("image_path")
                image_html = f'<p style="color:#999">Image not found: {display_path}</p>'
        else:
            b64 = image_to_base64(resolved_image_path) if resolved_image_path else ""
            if b64:
                image_html = f'<img src="{b64}" alt="table image">'
            else:
                display_path = pred.get("image_path_raw") or pred.get("image_path")
                image_html = f'<p style="color:#999">Image not found: {display_path}</p>'

    # GT/Pred 테이블 렌더링
    gt_html = _render_table_preview(pred.get("gt_html", ""))
    pred_html = _render_table_preview(pred.get("pred_html", ""))
    gt_code = _render_html_code(pred.get("gt_html", ""))
    pred_code = _render_html_code(pred.get("pred_html", ""))

    return f"""
    <div class="sample" data-complexity="{complexity}" data-spanf1="{span_f1:.3f}">
      <div class="sample-header">
        <span>#{idx} — {os.path.basename(pred.get('image_path', ''))}</span>
        <span class="badge badge-{complexity}">{complexity.upper()}</span>
      </div>
      <div class="sample-body">
        <div class="image-col">{image_html}</div>
        <div class="table-col">
          <h3>✅ Ground Truth</h3>
          {gt_html if gt_html else '<p style="color:#999">No GT</p>'}
        </div>
        <div class="table-col">
          <h3>🤖 Prediction</h3>
          {pred_html if pred_html else '<p style="color:#999">No prediction</p>'}
        </div>
      </div>
      <div class="code-wrap">
        <details>
          <summary>Show HTML code</summary>
          <div class="code-grid">
            <div class="code-box">
              <h4>GT HTML</h4>
              {gt_code}
            </div>
            <div class="code-box">
              <h4>Prediction HTML</h4>
              {pred_code}
            </div>
          </div>
        </details>
      </div>
      <div class="metrics-bar">
        <div class="metric">
          <span class="metric-label">TEDS:</span>
          <span class="metric-value {metric_class(teds)}">{teds:.3f}</span>
        </div>
        <div class="metric">
          <span class="metric-label">TEDS-S:</span>
          <span class="metric-value {metric_class(teds_structure)}">{teds_structure:.3f}</span>
        </div>
        <div class="metric">
          <span class="metric-label">TEDS-N:</span>
          <span class="metric-value {metric_class(teds_norm)}">{teds_norm:.3f}</span>
        </div>
        <div class="metric">
          <span class="metric-label">TEDS-NS:</span>
          <span class="metric-value {metric_class(teds_norm_struct)}">{teds_norm_struct:.3f}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Span F1:</span>
          <span class="metric-value {metric_class(span_f1)}">{span_f1:.3f}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Attr Acc:</span>
          <span class="metric-value {metric_class(attr_acc)}">{attr_acc:.3f}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Time:</span>
          <span class="metric-value">{pred.get('inference_time', 0):.1f}s</span>
        </div>
      </div>
    </div>
    """


def generate_report(
    predictions_path: str,
    output_path: str,
    metrics_path: Optional[str] = None,
    max_samples: int = 100,
    embed_images: bool = True,
    image_mode: str = "embed",
    sort_by: str = "span_f1",
    sort_desc: bool = False,
    image_root: Optional[str] = None,
    shared_image_dir: Optional[str] = None,
):
    """평가 리포트 HTML을 생성한다."""
    # 예측 로드
    predictions = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    # 메트릭 로드
    metrics = {}
    metrics_candidates = []
    if metrics_path:
        metrics_candidates.append(Path(metrics_path))
    else:
        pred_path = Path(predictions_path)
        if pred_path.name == "predictions.jsonl":
            metrics_candidates.append(pred_path.with_name("metrics.json"))
        metrics_candidates.append(pred_path.parent / "metrics.json")
        metrics_candidates.append(pred_path.parent.parent / "metrics.json")

    seen = set()
    for candidate in metrics_candidates:
        candidate_resolved = candidate.resolve()
        if candidate_resolved in seen:
            continue
        seen.add(candidate_resolved)
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            break
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse metrics JSON: {candidate}")

    # Summary
    summary_rows = ""
    for key, value in metrics.items():
        if isinstance(value, float):
            summary_rows += f"<tr><td>{key}</td><td><b>{value:.4f}</b></td></tr>"
        else:
            summary_rows += f"<tr><td>{key}</td><td><b>{value}</b></td></tr>"
    summary_html = f"<table>{summary_rows}</table>"

    # Counts
    complexities = [p.get("complexity", "unknown") for p in predictions]
    comp_counts = Counter(complexities)

    # 기본: span_f1 오름차순 (오류 우선), 옵션으로 정렬키 변경 가능
    predictions.sort(key=lambda x: x.get(sort_by, 0.0), reverse=sort_desc)

    # 샘플 렌더링
    samples_html = ""
    report_dir = str(Path(output_path).resolve().parent)
    for i, pred in enumerate(predictions[:max_samples]):
        samples_html += render_sample(
            pred,
            i,
            embed_images=embed_images,
            image_mode=image_mode,
            image_root=image_root,
            shared_image_dir=shared_image_dir,
            report_dir=report_dir,
        )

    # 최종 HTML
    html = HTML_TEMPLATE.format(
        summary_html=summary_html,
        samples_html=samples_html,
        total=len(predictions),
        complex_count=comp_counts.get("complex", 0),
        medium_count=comp_counts.get("medium", 0),
        simple_count=comp_counts.get("simple", 0),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="결과 시각화")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="eval_results/comparison.html")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument(
        "--sort_by",
        default="span_f1",
        choices=["span_f1", "teds", "teds_structure", "teds_norm", "teds_norm_structure", "attribute_accuracy", "inference_time", "index"],
        help="샘플 정렬 기준",
    )
    parser.add_argument(
        "--sort_desc",
        action="store_true",
        help="정렬 기준 내림차순 사용",
    )
    parser.add_argument("--no-embed-images", action="store_true")
    parser.add_argument(
        "--image_mode",
        choices=["embed", "external", "none"],
        default="embed",
        help="이미지 렌더링 방식",
    )
    parser.add_argument(
        "--shared_image_dir",
        default=None,
        help="--image_mode external에서 사용할 공용 이미지 저장 디렉터리",
    )
    parser.add_argument(
        "--image_root",
        default=os.environ.get("TSR_IMAGE_ROOT"),
        help="image_path_raw 기준 상대경로를 해석할 이미지 루트 디렉터리",
    )
    args = parser.parse_args()

    image_mode = args.image_mode
    if args.no_embed_images and image_mode == "embed":
        image_mode = "none"

    generate_report(
        args.predictions,
        args.output,
        args.metrics,
        args.max_samples,
        embed_images=not args.no_embed_images,
        image_mode=image_mode,
        sort_by=args.sort_by,
        sort_desc=args.sort_desc,
        image_root=args.image_root,
        shared_image_dir=args.shared_image_dir,
    )


if __name__ == "__main__":
    main()
