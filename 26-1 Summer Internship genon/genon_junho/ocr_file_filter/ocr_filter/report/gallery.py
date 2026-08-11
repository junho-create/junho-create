"""cmcv 결과를 Easy/Medium/Hard 티어별로 묶어 GT|target|dots.ocr|paddle 4열 HTML 갤러리 생성.

레이아웃 레코드(source_type=="layout")만 대상 — bbox+category 오버레이가 의미 있는 건
이쪽뿐이고, table 레코드는 crop 전체가 표 하나라 박스 비교의 의미가 약함.

    python -m ocr_filter.cli report gallery --per-tier 5 --out gallery.html
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from ocr_filter.cmcv.normalize import full_text
from ocr_filter.io.schema import Record, read_jsonl
from ocr_filter.report.render import draw_boxes, to_data_uri

TIER_ORDER = ["Easy", "Medium", "Hard"]
TIER_COLOR = {"Easy": "#27ae60", "Medium": "#f39c12", "Hard": "#c0392b"}
MODEL_LABELS = {"target": "target (Qwen3.5+LoRA)", "external_a": "dots.ocr", "external_b": "PaddleOCR-VL"}


def _load_records_by_id(unified_path: str | Path) -> dict[str, Record]:
    return {r.id: r for r in read_jsonl(unified_path)}


def pick_ids_by_tier(
    cmcv_results_path: str | Path, per_tier: int, source_type: str = "layout",
) -> dict[str, list[dict]]:
    picked: dict[str, list[dict]] = {t: [] for t in TIER_ORDER}
    with open(cmcv_results_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("source_type") != source_type:
                continue
            t = d.get("tier")
            if t in picked and len(picked[t]) < per_tier:
                picked[t].append(d)
    return picked


def _render_row(record: Record, cmcv_row: dict) -> dict:
    # cmcv_row["elements"] 는 cmcv 가 실제 채점에 쓴 3모델 출력 그대로 — 모델을 다시
    # 호출하지 않는다(thinking 모델 결정성 미보장으로 갤러리와 실제 tier/score 가
    # 어긋날 수 있음, ocr_filter/cmcv/run.py `_process_one` docstring 참고).
    elements = cmcv_row["elements"]

    gt_elements = record.gt if isinstance(record.gt, list) else []
    panels = {
        "gt": to_data_uri(draw_boxes(record.image_path, gt_elements, "pixel")),
        "target": to_data_uri(draw_boxes(record.image_path, elements["target"], "norm1000")),
        "external_a": to_data_uri(draw_boxes(record.image_path, elements["external_a"], "pixel")),
        "external_b": to_data_uri(draw_boxes(record.image_path, elements["external_b"], "pixel")),
    }
    n_elements = {k: len(v) for k, v in elements.items()}

    return {
        "id": record.id,
        "panels": panels,
        "paddle_text": full_text(elements["external_b"])[:200] or "(빈 응답)",
        "gt_score": cmcv_row.get("gt_score"),
        "agreement_score": cmcv_row.get("agreement_score"),
        "n_elements": n_elements,
    }


def _row_html(row: dict) -> str:
    def figure(title: str, src: str, caption: str = "") -> str:
        cap = f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ""
        return (
            f'<figure><div class="fig-title">{html.escape(title)}</div>'
            f'<img src="{src}" loading="lazy">{cap}</figure>'
        )

    n = row["n_elements"]
    gt_score = row["gt_score"]
    agreement = row["agreement_score"]
    # gt_score 는 GT 없는 레코드(신규 원본 PDF 등)에서 None — "N/A" 로 표시.
    gt_score_s = f"{gt_score:.3f}" if gt_score is not None else "N/A"
    agreement_s = f"{agreement:.3f}" if agreement is not None else "N/A"
    return f"""
    <div class="row">
      <div class="row-header">
        <code>{html.escape(row['id'])}</code>
        <span class="scores">gt_score={gt_score_s}  agreement={agreement_s}</span>
      </div>
      <div class="panels">
        {figure("GT", row['panels']['gt'])}
        {figure(MODEL_LABELS['target'], row['panels']['target'], f"elements={n.get('target')}")}
        {figure(MODEL_LABELS['external_a'], row['panels']['external_a'], f"elements={n.get('external_a')}")}
        {figure(MODEL_LABELS['external_b'], row['panels']['external_b'], row['paddle_text'][:200])}
      </div>
    </div>"""


_CSS = """
body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 24px; }
h1 { margin-top: 0; }
h2 { border-left: 6px solid; padding-left: 10px; margin-top: 40px; }
.row { margin-bottom: 28px; border: 1px solid #333; border-radius: 8px; padding: 12px; background: #1a1a1a; }
.row-header { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; color: #aaa; }
.scores { font-family: monospace; }
.panels { display: flex; gap: 10px; overflow-x: auto; }
figure { margin: 0; flex: 1; min-width: 260px; max-width: 320px; }
.fig-title { font-size: 12px; font-weight: 600; margin-bottom: 4px; color: #9cf; }
figure img { width: 100%; border-radius: 4px; border: 1px solid #333; background: #fff; }
figcaption { font-size: 11px; color: #999; margin-top: 4px; word-break: break-all; }
.tabs { display: flex; gap: 10px; margin-bottom: 20px; }
.tab-btn { padding: 8px 16px; cursor: pointer; background: #333; border: 1px solid #555; color: #eee; border-radius: 4px; font-weight: bold; }
.tab-btn:hover { background: #444; }
.tab-btn.active { background: #3498db; border-color: #2980b9; color: #fff; }
.tier-section { display: block; }
"""


def build_html(unified_path: str | Path, cmcv_results_path: str | Path,
                per_tier: int = 5) -> str:
    """모델 재호출 없이 `cmcv_results_path` 에 저장된 `elements`(실제 채점에 쓴 3모델 출력)만
    으로 렌더링한다 (`ocr_filter/cmcv/run.py` `_process_one` docstring 참고)."""
    records_by_id = _load_records_by_id(unified_path)
    picked = pick_ids_by_tier(cmcv_results_path, per_tier)

    sections = []
    for tier in TIER_ORDER:
        rows_html = []
        for cmcv_row in picked[tier]:
            record = records_by_id.get(cmcv_row["id"])
            if record is None or "elements" not in cmcv_row:
                continue
            row = _render_row(record, cmcv_row)
            rows_html.append(_row_html(row))
        color = TIER_COLOR[tier]
        sections.append(
            f'<div class="tier-section" data-tier="{tier}">'
            f'<h2 style="border-color:{color}">{tier} ({len(rows_html)}건 샘플)</h2>'
            + "\n".join(rows_html)
            + '</div>'
        )

    tabs_html = '<div class="tabs">'
    tabs_html += '<button class="tab-btn active" onclick="showTier(\'All\')">All</button>'
    for tier in TIER_ORDER:
        tabs_html += f'<button class="tab-btn" onclick="showTier(\'{tier}\')">{tier}</button>'
    tabs_html += '</div>'
    
    js = """<script>
function showTier(tier) {
    document.querySelectorAll('.tier-section').forEach(el => {
        if (tier === 'All' || el.dataset.tier === tier) {
            el.style.display = 'block';
        } else {
            el.style.display = 'none';
        }
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.textContent === tier) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}
</script>"""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>CMCV 갤러리</title><style>{_CSS}</style></head><body>
<h1>CMCV 3모델 비교 — GT / target / dots.ocr / PaddleOCR-VL</h1>
<p style="color:#999">티어별 샘플. bbox 색은 카테고리별 고정(빨강=Title, 파랑=Text, 초록=Table 등).
PaddleOCR-VL 패널은 실제 PaddleOCRVL 파이프라인(PP-DocLayoutV3 레이아웃검출 + vLLM 인식) 결과.</p>
{tabs_html}
{"".join(sections)}
{js}
</body></html>"""
