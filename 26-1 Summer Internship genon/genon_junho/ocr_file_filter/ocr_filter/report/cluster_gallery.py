"""taster CMCV 결과(클러스터당 소량 샘플) 전체를 클러스터별로 묶어 GT|target|dots.ocr|paddle
4열 HTML 갤러리로 — Easy/Medium/Hard 버튼으로 필터링 가능.

레코드 수가 많을 수 있어(클러스터 수 × taster_per_cluster) 이미지 크기/화질을 낮춰서
파일 용량을 억제한다 (report.gallery 의 기본 티어 갤러리보다 작게).

    python -m ocr_filter.cli report cluster-gallery --out gallery_cluster.html
"""

from __future__ import annotations

import html
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ocr_filter.cmcv.normalize import full_text
from ocr_filter.io.schema import Record, read_jsonl
from ocr_filter.report.diff import DIFF_CSS, word_diff_html
from ocr_filter.report.gallery import MODEL_LABELS, TIER_COLOR
from ocr_filter.report.render import draw_boxes, to_data_uri

MAX_SIDE = 700
JPEG_QUALITY = 68


def _load_records_by_id(unified_path: str | Path) -> dict[str, Record]:
    return {r.id: r for r in read_jsonl(unified_path)}


def _page_cluster_of(clusters_path: str | Path) -> dict[str, int]:
    mapping = {}
    with open(clusters_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("level") == "page":
                mapping[d["id"]] = d["cluster"]
    return mapping


def _group_by_cluster(
    taster_results_path: str | Path, id_to_cluster: dict[str, int],
) -> dict[int, list[dict]]:
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    with open(taster_results_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cluster = id_to_cluster.get(d["id"])
            if cluster is not None:
                by_cluster[cluster].append(d)
    return dict(by_cluster)


def _render_row(record: Record, cmcv_row: dict) -> dict:
    # cmcv_row["elements"] 는 taster CMCV 가 실제 채점에 쓴 3모델 출력 그대로 — 모델을
    # 다시 호출하면(특히 thinking 모델) 결정성이 보장 안 돼 갤러리가 실제 티어/점수를
    # 만든 출력과 다른 걸 보여줄 위험이 있어서, 재호출 없이 저장된 걸 그대로 쓴다.
    elements = cmcv_row["elements"]

    gt_elements = record.gt if isinstance(record.gt, list) else []
    panels = {
        "gt": to_data_uri(draw_boxes(record.image_path, gt_elements, "pixel", max_side=MAX_SIDE),
                           quality=JPEG_QUALITY),
        "target": to_data_uri(draw_boxes(record.image_path, elements["target"], "norm1000",
                                          max_side=MAX_SIDE), quality=JPEG_QUALITY),
        "external_a": to_data_uri(draw_boxes(record.image_path, elements["external_a"], "pixel",
                                              max_side=MAX_SIDE), quality=JPEG_QUALITY),
        "external_b": to_data_uri(draw_boxes(record.image_path, elements["external_b"], "pixel",
                                              max_side=MAX_SIDE), quality=JPEG_QUALITY),
    }
    n_elements = {k: len(v) for k, v in elements.items()}

    gt_text = full_text(gt_elements) if gt_elements else str(record.gt or "")
    target_text = full_text(elements["target"])
    dots_text = full_text(elements["external_a"])

    return {
        "id": record.id,
        "tier": cmcv_row.get("tier"),
        "panels": panels,
        "paddle_text": full_text(elements["external_b"])[:150] or "(빈 응답)",
        "gt_score": cmcv_row.get("gt_score"),
        "agreement_score": cmcv_row.get("agreement_score"),
        "n_elements": n_elements,
        "diff_gt_target": word_diff_html(gt_text, target_text, "GT", "target"),
        "diff_target_dots": word_diff_html(target_text, dots_text, "target", "dots.ocr"),
    }


def _row_html(row: dict) -> str:
    def figure(title: str, src: str, caption: str = "") -> str:
        cap = f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ""
        return (
            f'<figure><div class="fig-title">{html.escape(title)}</div>'
            f'<img src="{src}" loading="lazy">{cap}</figure>'
        )

    n = row["n_elements"]
    tier = row["tier"]
    color = TIER_COLOR.get(tier, "#888")
    gt_score = row["gt_score"]
    agreement = row["agreement_score"]
    gt_score_s = f"{gt_score:.3f}" if gt_score is not None else "N/A"
    agreement_s = f"{agreement:.3f}" if agreement is not None else "N/A"
    return f"""
    <div class="row" data-tier="{tier}">
      <div class="row-header">
        <span class="tier-badge" style="background:{color}">{tier}</span>
        <code>{html.escape(row['id'])}</code>
        <span class="scores">gt_score={gt_score_s}  agreement={agreement_s}</span>
      </div>
      <div class="panels">
        {figure("GT", row['panels']['gt'])}
        {figure(MODEL_LABELS['target'], row['panels']['target'], f"elements={n.get('target')}")}
        {figure(MODEL_LABELS['external_a'], row['panels']['external_a'], f"elements={n.get('external_a')}")}
        {figure(MODEL_LABELS['external_b'], row['panels']['external_b'], row['paddle_text'])}
      </div>
      {row['diff_gt_target']}
      {row['diff_target_dots']}
    </div>"""


_CSS = """
body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 24px; }
h1 { margin-top: 0; }
h2 { border-left: 6px solid #555; padding-left: 10px; margin-top: 36px; font-size: 16px; color: #ccc; }
.toolbar { position: sticky; top: 0; background: #111; padding: 10px 0; z-index: 10;
           border-bottom: 1px solid #333; display: flex; gap: 8px; align-items: center; }
.toolbar button { cursor: pointer; border: none; border-radius: 6px; padding: 6px 14px;
                   font-size: 13px; font-weight: 600; color: #fff; opacity: 0.55; }
.toolbar button.active { opacity: 1; box-shadow: 0 0 0 2px #fff inset; }
.toolbar .count { color: #999; font-size: 13px; margin-left: 8px; }
.row { margin-bottom: 22px; border: 1px solid #333; border-radius: 8px; padding: 12px; background: #1a1a1a; }
.row-header { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; font-size: 13px; color: #aaa; }
.tier-badge { font-size: 11px; font-weight: 700; color: #111; padding: 2px 8px; border-radius: 10px; }
.scores { font-family: monospace; margin-left: auto; }
.panels { display: flex; gap: 10px; overflow-x: auto; }
figure { margin: 0; flex: 1; min-width: 190px; max-width: 230px; }
.fig-title { font-size: 12px; font-weight: 600; margin-bottom: 4px; color: #9cf; }
figure img { width: 100%; border-radius: 4px; border: 1px solid #333; background: #fff; }
figcaption { font-size: 11px; color: #999; margin-top: 4px; word-break: break-all; }
#summary-view table { border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 13px; }
#summary-view th, #summary-view td { border: 1px solid #333; padding: 6px 10px; text-align: right; }
#summary-view th { background: #1a1a1a; position: sticky; top: 54px; }
#summary-view td:first-child, #summary-view th:first-child { text-align: left; }
#summary-view .bar-cell { text-align: left; min-width: 160px; }
#summary-view .bar { display: flex; height: 14px; border-radius: 3px; overflow: hidden; }
#summary-view tr:hover td { background: #1e1e1e; }
"""

_JS = """
<script>
function filterTier(tier) {
  var detail = document.getElementById('detail-view');
  var summary = document.getElementById('summary-view');
  if (tier === 'Summary') {
    detail.style.display = 'none';
    summary.style.display = '';
  } else {
    detail.style.display = '';
    summary.style.display = 'none';
    document.querySelectorAll('.row').forEach(function(row) {
      row.style.display = (tier === 'All' || row.dataset.tier === tier) ? '' : 'none';
    });
  }
  document.querySelectorAll('.toolbar button').forEach(function(btn) {
    btn.classList.toggle('active', btn.dataset.tier === tier);
  });
}
</script>
"""


def _load_taster_report(taster_report_path: str | Path | None) -> dict | None:
    if not taster_report_path or not Path(taster_report_path).exists():
        return None
    return json.loads(Path(taster_report_path).read_text(encoding="utf-8"))


def _summary_html(report: dict | None) -> str:
    if report is None:
        return '<div id="summary-view" style="display:none"><p>taster_report.json 없음</p></div>'

    scores, sizes = report["scores"], report["sizes"]
    counts, alloc = report["tier_counts"], report["alloc"]
    cluster_ids = sorted(scores, key=lambda c: -scores[c])

    rows = []
    for cid in cluster_ids:
        c = counts[cid]
        total = c["easy"] + c["medium"] + c["hard"]
        bar = ""
        if total > 0:
            for tier, key in (("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")):
                pct = 100 * c[key] / total
                if pct > 0:
                    bar += f'<div style="width:{pct:.1f}%;background:{TIER_COLOR[tier]}"></div>'
        rows.append(f"""
        <tr>
          <td>{cid}</td>
          <td>{sizes[cid]}</td>
          <td>{scores[cid]:.3f}</td>
          <td>{c['easy']}</td>
          <td>{c['medium']}</td>
          <td>{c['hard']}</td>
          <td>{alloc[cid]}</td>
          <td class="bar-cell"><div class="bar">{bar}</div></td>
        </tr>""")

    return f"""
    <div id="summary-view" style="display:none">
      <p style="color:#999">클러스터 {len(cluster_ids)}개, S_i(난이도점수) 내림차순. taster 샘플
      기준 E/M/H 개수 + DDAS 할당량(N_i). 막대는 그 클러스터의 E/M/H 비율(초록/주황/빨강).</p>
      <table>
        <thead><tr><th>Cluster</th><th>|C_i|</th><th>S_i</th><th>Easy</th><th>Medium</th>
        <th>Hard</th><th>N_i</th><th>비율</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""


def build_cluster_gallery_html(
    unified_path: str | Path, taster_results_path: str | Path, clusters_path: str | Path,
    workers: int = 32, taster_report_path: str | Path | None = None,
) -> str:
    """모델 재호출 없이 taster CMCV 결과(`taster_cmcv_results.jsonl`)에 저장된
    `elements`(실제 채점에 쓴 3모델 출력)만으로 렌더링한다 — 갤러리가 실제 tier/score 를
    만든 출력과 다른 걸 보여주는 걸 막기 위함(`ocr_filter/cmcv/run.py` 의 `_process_one`
    docstring 참고). 남는 작업은 이미지 로드+박스 그리기뿐이라 그것만 병렬화."""
    records_by_id = _load_records_by_id(unified_path)
    id_to_cluster = _page_cluster_of(clusters_path)
    by_cluster = _group_by_cluster(taster_results_path, id_to_cluster)

    jobs = [
        (cluster_id, cmcv_row, records_by_id[cmcv_row["id"]])
        for cluster_id, rows in by_cluster.items()
        for cmcv_row in rows if cmcv_row["id"] in records_by_id and "elements" in cmcv_row
    ]
    rendered_by_id: dict[str, dict] = {}
    n_total, n_done, t0 = len(jobs), 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_render_row, record, cmcv_row): cmcv_row["id"]
            for _, cmcv_row, record in jobs
        }
        for fut in as_completed(futures):
            rendered_by_id[futures[fut]] = fut.result()
            n_done += 1
            if n_done % 20 == 0 or n_done == n_total:
                print(f"[{n_done}/{n_total}] {time.time() - t0:.0f}s 경과")

    tier_counts = {"Easy": 0, "Medium": 0, "Hard": 0}
    sections = []
    for cluster_id in sorted(by_cluster):
        cluster_rows = by_cluster[cluster_id]
        rows_html = []
        local_counts = {"Easy": 0, "Medium": 0, "Hard": 0}
        for cmcv_row in cluster_rows:
            row = rendered_by_id.get(cmcv_row["id"])
            if row is None:
                continue
            tier_counts[row["tier"]] = tier_counts.get(row["tier"], 0) + 1
            local_counts[row["tier"]] = local_counts.get(row["tier"], 0) + 1
            rows_html.append(_row_html(row))
        sections.append(
            f'<h2>Cluster {cluster_id} '
            f'(Easy {local_counts["Easy"]} / Medium {local_counts["Medium"]} / '
            f'Hard {local_counts["Hard"]})</h2>' + "\n".join(rows_html)
        )

    buttons = "".join(
        f'<button data-tier="{t}" onclick="filterTier(\'{t}\')" '
        f'style="background:{TIER_COLOR.get(t, "#555")}">{t}'
        f'{f" ({tier_counts[t]})" if t not in ("All", "Summary") else ""}</button>'
        for t in ["All", "Easy", "Medium", "Hard", "Summary"]
    )
    summary_section = _summary_html(_load_taster_report(taster_report_path))

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>클러스터별 taster 갤러리</title><style>{_CSS}{DIFF_CSS}</style></head><body>
<h1>클러스터별 taster CMCV 갤러리 — GT / target / dots.ocr / PaddleOCR-VL</h1>
<p style="color:#999">클러스터 64개, 클러스터당 taster 샘플(최대 8개) 전부. 버튼으로 티어 필터링,
"Summary" 로 클러스터별 난이도 분포 표 보기.</p>
<div class="toolbar" id="toolbar">{buttons}<span class="count">
총 {sum(tier_counts.values())}건</span></div>
<div id="detail-view">
{"".join(sections)}
</div>
{summary_section}
{_JS}
<script>filterTier('All');</script>
</body></html>"""
