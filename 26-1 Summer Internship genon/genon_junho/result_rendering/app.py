#!/usr/bin/env python3
"""추론 결과 2-grid 뷰어.

좌측: 이미지 + bbox + 카테고리 오버레이
우측: 렌더링된 HTML (pred_layout_html / pred_table_html / GT html 등)

지원 입력:
  - predictions_unified.jsonl (dp500, layout HTML 예측)
  - predictions_unified.jsonl (tsr200, table HTML 예측)
  - reference_dp_bench.json (GT, elements[].coordinates 폴리곤 + content.html)

GT도 "한 번 예측된 결과"로 취급한다 - 같은 뷰어에 경로만 바꿔 넣으면 동일하게 렌더링된다.

사용:
    python3 app.py            # http://localhost:8091
    python3 app.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from PIL import Image

app = FastAPI()

# 탭/창마다 독립적으로 결과를 로드할 수 있도록 세션별로 상태를 분리한다.
# session_id 는 클라이언트가 페이지 로드 시 무작위로 생성해 모든 API 호출에 실어 보낸다.
SESSIONS_LOCK = threading.Lock()
SESSIONS: dict[str, dict[str, Any]] = {}


def new_session_state() -> dict[str, Any]:
    return {
        "loaded": False,
        "data_path": "",
        "image_dir": "",
        "cases": {},          # key -> Case dict
        "order": [],          # sorted list of keys
        "warnings": [],
    }


def get_session(session_id: str) -> dict[str, Any]:
    with SESSIONS_LOCK:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = new_session_state()
        return SESSIONS[session_id]


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]

# ---------------------------------------------------------------------------
# bbox / html 파싱 (viz/app.py, review_e21grid.py 에서 재사용한 패턴)
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<div([^>]*)>")
BBOX_ATTR_RE = re.compile(r'data-bbox="([\d\s.]+)"')
LABEL_ATTR_RE = re.compile(r'data-label="([^"]*)"')

LABEL_COLORS = {
    "Table": "#e6194b", "Text": "#3cb44b", "Title": "#4363d8",
    "Section-header": "#f58231", "List-item": "#911eb4", "Picture": "#46f0f0",
    "Caption": "#f032e6", "Formula": "#bcf60c", "Footnote": "#fabebe",
    "Page-header": "#008080", "Page-footer": "#9a6324",
    "table": "#e6194b", "header": "#008080", "footer": "#9a6324",
    "figure": "#46f0f0", "caption": "#f032e6", "paragraph": "#3cb44b",
    "heading1": "#4363d8", "list": "#911eb4", "footnote": "#fabebe",
    "equation": "#bcf60c",
}
DEFAULT_COLOR = "#808080"


def color_for_label(label: str) -> str:
    return LABEL_COLORS.get(label, DEFAULT_COLOR)


def extract_layout_boxes(html_str: str) -> list[dict]:
    """`<div data-bbox="x0 y0 x1 y1" data-label="...">` 파싱."""
    boxes = []
    for m in TAG_RE.finditer(html_str or ""):
        attrs = m.group(1)
        bbox_m = BBOX_ATTR_RE.search(attrs)
        if not bbox_m:
            continue
        try:
            nums = [float(x) for x in bbox_m.group(1).split()]
        except ValueError:
            continue
        if len(nums) != 4:
            continue
        label_m = LABEL_ATTR_RE.search(attrs)
        label = label_m.group(1) if label_m else ""
        x0, y0, x1, y1 = nums
        boxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "label": label})
    return boxes


def stem_of(name: str) -> str:
    return Path(name).stem


def find_image(image_dir: Optional[str], key: str, fallback_path: str, base_dir: Optional[Path] = None) -> Optional[str]:
    if image_dir:
        d = Path(image_dir)
        for ext in IMAGE_EXTS:
            p = d / f"{key}{ext}"
            if p.exists():
                return str(p)
        # 확장자 못 맞추면 디렉토리 안에서 stem 일치 검색
        try:
            for p in d.iterdir():
                if p.suffix.lower() in IMAGE_EXTS and p.stem == key:
                    return str(p)
        except OSError:
            pass
    if fallback_path:
        fp = Path(fallback_path)
        if fp.is_absolute():
            if fp.exists():
                return str(fp)
        else:
            # jsonl 안의 상대경로는 jsonl 파일 위치 기준으로 해석 (CWD 기준이 아님)
            if base_dir is not None:
                resolved = (base_dir / fp).resolve()
                if resolved.exists():
                    return str(resolved)
            if fp.exists():
                return str(fp)
    return None


# ---------------------------------------------------------------------------
# 소스별 로더
# ---------------------------------------------------------------------------

RENDER_FIELD_CANDIDATES = ["pred_layout_html", "pred_table_html", "full_response", "gt_html"]
BBOX_FIELD_CANDIDATES = ["pred_layout_html", "full_response", "pred_table_html", "gt_html"]
META_SCALAR_FIELDS = [
    "teds", "teds_structure", "layout_teds", "layout_teds_structure",
    "span_f1", "attribute_accuracy", "html_fallback_used", "json_parse_success",
    "html_parse_success", "inference_time", "prompt_style",
]


def load_jsonl(data_path: str, image_dir: Optional[str], bbox_scale: float):
    cases = {}
    warnings = []
    base_dir = Path(data_path).parent
    with open(data_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.append(f"line {lineno}: JSON parse error ({e})")
                continue

            img_path = rec.get("image_path", "")
            key = stem_of(img_path) if img_path else f"row{lineno}"
            resolved_img = find_image(image_dir, key, img_path, base_dir=base_dir)
            if not resolved_img:
                warnings.append(f"{key}: image not found (image_path={img_path!r})")
                continue

            render_field = next((f for f in RENDER_FIELD_CANDIDATES if rec.get(f)), None)
            bbox_field = next((f for f in BBOX_FIELD_CANDIDATES if rec.get(f)), None)
            html_str = rec.get(render_field, "") if render_field else ""
            bbox_src = rec.get(bbox_field, "") if bbox_field else ""
            boxes = extract_layout_boxes(bbox_src)

            meta = {k: rec[k] for k in META_SCALAR_FIELDS if k in rec}

            if key in cases:
                warnings.append(f"{key}: duplicate key, keeping last occurrence")

            cases[key] = {
                "key": key,
                "image_path": resolved_img,
                "boxes": boxes,
                "coord_space": "grid",
                "bbox_scale": bbox_scale,
                "html": html_str,
                "meta": meta,
                "render_field": render_field or "",
            }
    return cases, warnings


def poly_to_bbox(coords: list[dict]) -> tuple[float, float, float, float]:
    xs = [c["x"] for c in coords]
    ys = [c["y"] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def load_reference_json(data_path: str, image_dir: Optional[str]):
    cases = {}
    warnings = []
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        warnings.append("reference json: top-level is not a dict, unsupported shape")
        return cases, warnings

    for doc_name, doc in data.items():
        key = stem_of(doc_name)
        elements = doc.get("elements", []) if isinstance(doc, dict) else []
        resolved_img = find_image(image_dir, key, "")
        if not resolved_img:
            warnings.append(f"{key}: image not found in image_dir (reference json has no image path)")
            continue

        boxes = []
        html_parts = []
        for el in elements:
            coords = el.get("coordinates") or []
            if len(coords) < 2:
                continue
            x0, y0, x1, y1 = poly_to_bbox(coords)
            label = el.get("category", "")
            boxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "label": label})

            content = el.get("content", {}) or {}
            piece = content.get("html") or content.get("markdown") or content.get("text") or ""
            if piece:
                if content.get("html"):
                    html_parts.append(piece)
                else:
                    html_parts.append(f"<p>{piece}</p>")

        cases[key] = {
            "key": key,
            "image_path": resolved_img,
            "boxes": boxes,
            "coord_space": "pixel",
            "bbox_scale": 1.0,
            "html": "\n".join(html_parts),
            "meta": {"num_elements": len(elements)},
            "render_field": "reference_dp_bench.elements",
        }
    return cases, warnings


def load_source(data_path: str, image_dir: Optional[str], coord_space: str, bbox_scale: float):
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {data_path}")

    if p.suffix == ".jsonl":
        cases, warnings = load_jsonl(data_path, image_dir, bbox_scale)
    elif p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            head = f.read(2048)
        stripped = head.lstrip()
        if stripped.startswith("{"):
            cases, warnings = load_reference_json(data_path, image_dir)
        else:
            # list-of-records json (jsonl과 유사한 케이스 대비)
            with open(p, "r", encoding="utf-8") as f:
                records = json.load(f)
            cases, warnings = {}, []
            base_dir = p.parent
            for i, rec in enumerate(records):
                img_path = rec.get("image_path", "")
                key = stem_of(img_path) if img_path else f"row{i}"
                resolved_img = find_image(image_dir, key, img_path, base_dir=base_dir)
                if not resolved_img:
                    warnings.append(f"{key}: image not found")
                    continue
                render_field = next((f for f in RENDER_FIELD_CANDIDATES if rec.get(f)), None)
                bbox_field = next((f for f in BBOX_FIELD_CANDIDATES if rec.get(f)), None)
                boxes = extract_layout_boxes(rec.get(bbox_field, "")) if bbox_field else []
                cases[key] = {
                    "key": key,
                    "image_path": resolved_img,
                    "boxes": boxes,
                    "coord_space": "grid",
                    "bbox_scale": bbox_scale,
                    "html": rec.get(render_field, "") if render_field else "",
                    "meta": {k: rec[k] for k in META_SCALAR_FIELDS if k in rec},
                    "render_field": render_field or "",
                }
    else:
        raise ValueError(f"지원하지 않는 확장자: {p.suffix} (.json/.jsonl만 지원)")

    # coord_space override (auto가 아니면 사용자 지정값으로 덮어씀)
    if coord_space != "auto":
        for c in cases.values():
            c["coord_space"] = coord_space

    return cases, warnings


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/load")
def api_load(payload: dict):
    session_id = (payload.get("session_id") or "").strip()
    if not session_id:
        return JSONResponse({"error": "session_id가 없습니다."}, status_code=400)

    data_path = (payload.get("data_path") or "").strip()
    image_dir = (payload.get("image_dir") or "").strip() or None
    coord_space = payload.get("coord_space") or "auto"
    try:
        bbox_scale = float(payload.get("bbox_scale") or 1000)
    except (TypeError, ValueError):
        bbox_scale = 1000.0

    if not data_path:
        return JSONResponse({"error": "결과 파일 경로를 입력하세요."}, status_code=400)

    try:
        cases, warnings = load_source(data_path, image_dir, coord_space, bbox_scale)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if not cases:
        return JSONResponse(
            {"error": "로드된 케이스가 없습니다. 경로 및 이미지 디렉토리를 확인하세요.", "warnings": warnings},
            status_code=400,
        )

    order = sorted(cases.keys())
    table_keys = sorted(
        k for k, c in cases.items()
        if any((b["label"] or "").strip().lower() == "table" for b in c["boxes"])
    )
    formula_keys = sorted(
        k for k, c in cases.items()
        if any((b["label"] or "").strip().lower() in ("formula", "equation") for b in c["boxes"])
    )

    state = get_session(session_id)
    state["loaded"] = True
    state["data_path"] = data_path
    state["image_dir"] = image_dir or ""
    state["cases"] = cases
    state["order"] = order
    state["warnings"] = warnings

    labels = sorted({b["label"] for c in cases.values() for b in c["boxes"] if b["label"]})
    return {
        "count": len(cases),
        "keys": order,
        "table_keys": table_keys,
        "formula_keys": formula_keys,
        "labels": labels,
        "warnings": warnings,
    }


@app.post("/api/reset")
def api_reset(payload: dict):
    session_id = (payload.get("session_id") or "").strip()
    if session_id:
        with SESSIONS_LOCK:
            SESSIONS[session_id] = new_session_state()
    return {"ok": True}


@app.get("/api/case")
def api_case(key: str = Query(...), session_id: str = Query(...)):
    state = get_session(session_id)
    case = state["cases"].get(key)
    if not case:
        return JSONResponse({"error": f"case not found: {key}"}, status_code=404)

    try:
        with Image.open(case["image_path"]) as img:
            w, h = img.size
    except Exception as e:
        return JSONResponse({"error": f"image open failed: {e}"}, status_code=500)

    scale = case["bbox_scale"] if case["coord_space"] == "grid" else 1.0
    boxes = []
    for b in case["boxes"]:
        if case["coord_space"] == "grid":
            px0, py0 = b["x0"] / scale, b["y0"] / scale
            px1, py1 = b["x1"] / scale, b["y1"] / scale
        else:  # pixel
            px0, py0 = b["x0"] / w, b["y0"] / h
            px1, py1 = b["x1"] / w, b["y1"] / h
        boxes.append({
            "x0": max(0.0, min(1.0, px0)), "y0": max(0.0, min(1.0, py0)),
            "x1": max(0.0, min(1.0, px1)), "y1": max(0.0, min(1.0, py1)),
            "label": b["label"], "color": color_for_label(b["label"]),
        })

    return {
        "key": key,
        "image_url": f"/api/image?key={quote(key, safe='')}&session_id={quote(session_id, safe='')}",
        "width": w,
        "height": h,
        "boxes": boxes,
        "html": case["html"],
        "meta": case["meta"],
        "render_field": case["render_field"],
        "coord_space": case["coord_space"],
    }


@app.get("/api/image")
def api_image(key: str = Query(...), session_id: str = Query(...)):
    state = get_session(session_id)
    case = state["cases"].get(key)
    if not case:
        return JSONResponse({"error": f"case not found: {key}"}, status_code=404)
    return FileResponse(case["image_path"])


# ---------------------------------------------------------------------------
# UI (단일 HTML 페이지)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>추론 결과 2-grid 뷰어</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", sans-serif; font-size: 13px; color: #1a1a1a; background: #f5f5f7; }
  #toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px 14px; background: #fff; border-bottom: 1px solid #ddd; }
  #toolbar label { font-size: 12px; color: #555; margin-right: 4px; }
  #toolbar input[type=text] { padding: 5px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }
  #data_path { width: 480px; }
  #image_dir { width: 320px; }
  #bbox_scale { width: 70px; }
  #toolbar select { padding: 5px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }
  #toolbar button { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; }
  #btn_load { background: #2563eb; color: #fff; }
  #btn_reset { background: #e5e7eb; color: #333; }
  #status { font-size: 12px; color: #666; margin-left: auto; }
  #warnings { padding: 6px 14px; background: #fff7ed; border-bottom: 1px solid #fed7aa; font-size: 11px; color: #9a3412; max-height: 80px; overflow-y: auto; display: none; }

  #main { display: flex; height: calc(100vh - 46px); }
  #sidebar { width: 240px; border-right: 1px solid #ddd; background: #fff; display: flex; flex-direction: column; }
  #search { margin: 8px 8px 0 8px; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; }
  #table_filter_row, #formula_filter_row { margin: 6px 8px 0 8px; display: flex; align-items: center; gap: 5px; font-size: 12px; color: #444; }
  #formula_filter_row { margin-bottom: 8px; }
  #list { flex: 1; overflow-y: auto; }
  #list .item { padding: 7px 12px; cursor: pointer; border-bottom: 1px solid #f0f0f0; font-size: 12px; word-break: break-all; }
  #list .item:hover { background: #f0f4ff; }
  #list .item.active { background: #dbeafe; font-weight: 600; }
  #count { padding: 6px 12px; font-size: 11px; color: #888; border-top: 1px solid #eee; }

  #panels { flex: 1; display: flex; overflow: hidden; }
  .panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; border-right: 1px solid #ddd; }
  .panel:last-child { border-right: none; }
  .panel-head { padding: 8px 12px; background: #fff; border-bottom: 1px solid #eee; font-size: 11px; color: #666; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .panel-body { flex: 1; overflow: auto; padding: 12px; }

  #imgwrap { position: relative; display: inline-block; }
  #imgwrap img { max-width: 100%; display: block; }
  .box { position: absolute; border: 2px solid; pointer-events: auto; }
  .box .tag { position: absolute; top: -16px; left: -2px; font-size: 10px; padding: 0 3px; color: #fff; white-space: nowrap; }

  #legend { display: flex; flex-wrap: wrap; gap: 6px; }
  #legend .chip { display: flex; align-items: center; gap: 4px; padding: 2px 6px; border-radius: 10px; background: #f0f0f0; cursor: pointer; font-size: 11px; user-select: none; }
  #legend .chip.off { opacity: 0.35; }
  #legend .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

  #meta { display: flex; flex-wrap: wrap; gap: 8px; }
  #meta .badge { background: #eef2ff; color: #3730a3; padding: 2px 8px; border-radius: 10px; font-size: 11px; }

  #html_render { background: #fff; padding: 10px; }
  #html_render table { border-collapse: collapse; margin-bottom: 8px; }
  #html_render td, #html_render th { border: 1px solid #999; padding: 4px 8px; font-size: 12px; }
  #html_render p { margin: 4px 0; }

  #empty { display: flex; align-items: center; justify-content: center; height: 100%; color: #999; font-size: 13px; }
</style>
</head>
<body>

<div id="toolbar">
  <label>이미지 경로(선택)</label>
  <input type="text" id="image_dir" placeholder="/home/.../qwen_infer_images">
  <label>결과 파일</label>
  <input type="text" id="data_path" placeholder="/home/.../predictions_unified.jsonl 또는 reference_dp_bench.json">
  <label>좌표계</label>
  <select id="coord_space">
    <option value="auto">자동</option>
    <option value="grid">grid(0~1000)</option>
    <option value="pixel">pixel</option>
  </select>
  <label>scale</label>
  <input type="text" id="bbox_scale" value="1000">
  <button id="btn_load">확인</button>
  <button id="btn_reset">초기화</button>
  <span id="status"></span>
</div>
<div id="warnings"></div>

<div id="main">
  <div id="sidebar">
    <input type="text" id="search" placeholder="파일명 검색...">
    <div id="table_filter_row">
      <input type="checkbox" id="table_only">
      <label for="table_only">table 섹션 있는 페이지만</label>
    </div>
    <div id="formula_filter_row">
      <input type="checkbox" id="formula_only">
      <label for="formula_only">수식 섹션 있는 페이지만</label>
    </div>
    <div id="list"></div>
    <div id="count"></div>
  </div>
  <div id="panels">
    <div id="empty" style="width:100%">왼쪽 상단에 결과 파일 경로를 입력하고 [확인]을 누르세요.</div>
  </div>
</div>

<script>
// 탭마다 독립된 세션으로 결과를 로드하기 위한 id. sessionStorage 라 새로고침에도
// 유지되지만 새 탭/창에서는 각각 새로 발급된다.
function genSessionId() {
  return 'sess_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2);
}
const SESSION_ID = sessionStorage.getItem('viewer_session_id') || (() => {
  const id = genSessionId();
  sessionStorage.setItem('viewer_session_id', id);
  return id;
})();

let STATE = { keys: [], filtered: [], activeKey: null, labelOn: {}, tableKeys: new Set(), formulaKeys: new Set() };

const $ = (id) => document.getElementById(id);

function setStatus(msg) { $('status').textContent = msg; }

function showWarnings(warnings) {
  const el = $('warnings');
  if (!warnings || warnings.length === 0) { el.style.display = 'none'; el.textContent = ''; return; }
  el.style.display = 'block';
  el.textContent = `경고 ${warnings.length}건: ` + warnings.slice(0, 20).join(' | ') + (warnings.length > 20 ? ' ...' : '');
}

async function doLoad() {
  const payload = {
    session_id: SESSION_ID,
    image_dir: $('image_dir').value.trim(),
    data_path: $('data_path').value.trim(),
    coord_space: $('coord_space').value,
    bbox_scale: $('bbox_scale').value,
  };
  setStatus('로딩 중...');
  const res = await fetch('/api/load', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) {
    setStatus('에러: ' + data.error);
    showWarnings(data.warnings || []);
    return;
  }
  STATE.keys = data.keys;
  STATE.tableKeys = new Set(data.table_keys || []);
  STATE.formulaKeys = new Set(data.formula_keys || []);
  STATE.labelOn = {};
  for (const l of data.labels) STATE.labelOn[l] = true;
  $('table_only').checked = false;
  $('formula_only').checked = false;
  applyFilters();
  setStatus(`${data.count}건 로드됨 (table 포함 ${STATE.tableKeys.size}건, 수식 포함 ${STATE.formulaKeys.size}건)`);
  showWarnings(data.warnings);
  if (STATE.filtered.length > 0) selectCase(STATE.filtered[0]);
}

function applyFilters() {
  const q = $('search').value.trim().toLowerCase();
  const tableOnly = $('table_only').checked;
  const formulaOnly = $('formula_only').checked;
  let keys = STATE.keys;
  if (tableOnly) keys = keys.filter(k => STATE.tableKeys.has(k));
  if (formulaOnly) keys = keys.filter(k => STATE.formulaKeys.has(k));
  if (q) keys = keys.filter(k => k.toLowerCase().includes(q));
  STATE.filtered = keys;
  renderList();
}

async function doReset() {
  await fetch('/api/reset', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({session_id: SESSION_ID})
  });
  STATE = { keys: [], filtered: [], activeKey: null, labelOn: {}, tableKeys: new Set(), formulaKeys: new Set() };
  $('search').value = '';
  $('table_only').checked = false;
  $('formula_only').checked = false;
  $('list').innerHTML = '';
  $('count').textContent = '';
  $('panels').innerHTML = '<div id="empty" style="width:100%">왼쪽 상단에 결과 파일 경로를 입력하고 [확인]을 누르세요.</div>';
  setStatus('초기화됨');
  showWarnings([]);
}

function renderList() {
  const listEl = $('list');
  listEl.innerHTML = '';
  for (const key of STATE.filtered) {
    const div = document.createElement('div');
    div.className = 'item' + (key === STATE.activeKey ? ' active' : '');
    div.textContent = key;
    div.onclick = () => selectCase(key);
    listEl.appendChild(div);
  }
  $('count').textContent = `${STATE.filtered.length} / ${STATE.keys.length}`;
}

$('search').addEventListener('input', applyFilters);
$('table_only').addEventListener('change', applyFilters);
$('formula_only').addEventListener('change', applyFilters);

async function selectCase(key) {
  STATE.activeKey = key;
  renderList();
  const res = await fetch(`/api/case?key=${encodeURIComponent(key)}&session_id=${encodeURIComponent(SESSION_ID)}`);
  const data = await res.json();
  if (!res.ok) {
    $('panels').innerHTML = `<div id="empty" style="width:100%">에러: ${data.error}</div>`;
    return;
  }
  renderPanels(data);
}

function renderPanels(data) {
  const uniqueLabels = [...new Set(data.boxes.map(b => b.label || '(none)'))];
  for (const l of uniqueLabels) if (!(l in STATE.labelOn)) STATE.labelOn[l] = true;

  const legendHtml = uniqueLabels.map(l => {
    const color = data.boxes.find(b => (b.label || '(none)') === l)?.color || '#808080';
    const on = STATE.labelOn[l] !== false;
    return `<span class="chip ${on ? '' : 'off'}" data-label="${escapeAttr(l)}">
      <span class="swatch" style="background:${color}"></span>${escapeHtml(l)} (${data.boxes.filter(b=>(b.label||'(none)')===l).length})
    </span>`;
  }).join('');

  const metaHtml = Object.entries(data.meta || {}).map(([k, v]) => {
    let vv = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v;
    return `<span class="badge">${escapeHtml(k)}: ${escapeHtml(String(vv))}</span>`;
  }).join('');

  const boxesHtml = data.boxes.map(b => {
    const label = b.label || '(none)';
    const style = `left:${b.x0*100}%; top:${b.y0*100}%; width:${(b.x1-b.x0)*100}%; height:${(b.y1-b.y0)*100}%; border-color:${b.color}; display:${STATE.labelOn[label]!==false?'block':'none'}`;
    return `<div class="box" data-label="${escapeAttr(label)}" style="${style}" title="${escapeAttr(label)}">
      <span class="tag" style="background:${b.color}">${escapeHtml(label)}</span>
    </div>`;
  }).join('');

  $('panels').innerHTML = `
    <div class="panel">
      <div class="panel-head">
        <div id="legend">${legendHtml}</div>
      </div>
      <div class="panel-body">
        <div id="imgwrap">
          <img src="${data.image_url}" alt="${escapeAttr(data.key)}">
          ${boxesHtml}
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <div id="meta">${metaHtml}<span class="badge">field: ${escapeHtml(data.render_field || '(none)')}</span><span class="badge">coord: ${escapeHtml(data.coord_space)}</span></div>
      </div>
      <div class="panel-body">
        <div id="html_render">${data.html || '<em>(렌더링할 HTML 없음)</em>'}</div>
      </div>
    </div>
  `;

  document.querySelectorAll('#legend .chip').forEach(chip => {
    chip.onclick = () => {
      const label = chip.dataset.label;
      STATE.labelOn[label] = !(STATE.labelOn[label] !== false);
      chip.classList.toggle('off');
      document.querySelectorAll(`.box[data-label="${cssEscape(label)}"]`).forEach(b => {
        b.style.display = STATE.labelOn[label] !== false ? 'block' : 'none';
      });
    };
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function escapeAttr(s) { return escapeHtml(s); }
function cssEscape(s) { return String(s).replace(/["\\]/g, '\\$&'); }

$('btn_load').addEventListener('click', doLoad);
$('btn_reset').addEventListener('click', doReset);
$('data_path').addEventListener('keydown', (e) => { if (e.key === 'Enter') doLoad(); });
$('image_dir').addEventListener('keydown', (e) => { if (e.key === 'Enter') doLoad(); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
