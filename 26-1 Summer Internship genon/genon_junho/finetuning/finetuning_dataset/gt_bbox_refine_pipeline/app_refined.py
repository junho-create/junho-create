"""
train.jsonl(원본 GT) vs train_refined.jsonl(PaddleOCR 스냅 보정) 비교 뷰어.

- 위에서 출처(reference_16886=16천장 / new_63541=6만장)만 선택하고 조회 버튼을 누르면
  아래에 이미지가 2열 그리드로 쭉 나온다(페이지네이션으로 계속 넘겨보기).
- 각 이미지 위에 원래 GT 전체 박스(회색, 옅게) + 보정된 전체 박스(라벨별 색상)를
  같이 그린다. Text/List-item/Section-header/Table/Picture 등 모든 라벨 포함.

app.py(8090, 원본 4컬럼 뷰어)는 건드리지 않고 별도 포트(8092)로 띄운다.

Usage:
    python3 app_refined.py
    -> http://localhost:8092
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageDraw

COMBINED_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined")
ORIG_FILE = COMBINED_DIR / "train.jsonl"
REFINED_FILE = COMBINED_DIR / "train_refined.jsonl"

GROUPS = {
    "reference_16886": "16천장짜리 (reference_16886)",
    "new_63541": "6만장짜리 (new_63541)",
}


def classify(image_path: str) -> str:
    if "reference_16886" in image_path:
        return "reference_16886"
    if "ocr_filter_result" in image_path or "new_63541" in image_path:
        return "new_63541"
    return "other"


# 라벨별 색상(모든 data-label 종류). 범례는 /api/labels 로 그대로 노출.
LABEL_COLORS = {
    "Text": "#4fc3f7",
    "List-item": "#ffd54f",
    "Section-header": "#ff8a65",
    "Table": "#ba68c8",
    "Picture": "#90caf9",
    "Page-header": "#81c784",
    "Page-footer": "#4db6ac",
    "Caption": "#f06292",
    "Formula": "#fff176",
    "Title": "#e57373",
    "Footnote": "#a1887f",
}
DEFAULT_COLOR = "#ffffff"
GRAY = "#9e9e9e"

# 인덱스: idx -> {"orig_offset", "refined_offset", "path", "group"}
INDEX: list[dict] = []


def build_index() -> None:
    with ORIG_FILE.open("r", encoding="utf-8") as fo, REFINED_FILE.open("r", encoding="utf-8") as fr:
        while True:
            oo = fo.tell()
            lo = fo.readline()
            ro = fr.tell()
            lr = fr.readline()
            if not lo or not lr:
                break
            if not lo.strip():
                continue
            try:
                d = json.loads(lo)
            except json.JSONDecodeError:
                continue
            image_path = d.get("image_path", "")
            INDEX.append({
                "orig_offset": oo, "refined_offset": ro,
                "path": image_path, "group": classify(image_path),
            })
    print(f"[index] train: {len(INDEX)} rows")


def read_at(path: Path, offset: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        line = f.readline()
    return json.loads(line)


TAG_RE = re.compile(r"<div([^>]*)>")
BBOX_ATTR_RE = re.compile(r'data-bbox="([\d\s]+)"')
LABEL_ATTR_RE = re.compile(r'data-label="([^"]*)"')


def extract_layout_boxes(gt_html: str) -> list[tuple[list[int], str]]:
    boxes = []
    for m in TAG_RE.finditer(gt_html or ""):
        attrs = m.group(1)
        bbox_m = BBOX_ATTR_RE.search(attrs)
        if not bbox_m:
            continue
        nums = [int(x) for x in bbox_m.group(1).split()]
        if len(nums) != 4:
            continue
        label_m = LABEL_ATTR_RE.search(attrs)
        label = label_m.group(1) if label_m else ""
        boxes.append((nums, label))
    return boxes


def draw_combined(img: Image.Image, orig_boxes, refined_boxes, scale: int) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size

    def denorm(nums):
        x0, y0, x1, y1 = nums
        return [x0 / scale * w, y0 / scale * h, x1 / scale * w, y1 / scale * h]

    # 원래 GT 전체 박스 = 회색(옅게), 먼저 그려서 아래 깔림
    for nums, _label in orig_boxes:
        d.rectangle(denorm(nums), outline=GRAY, width=2)
    # 보정된 전체 박스 = 라벨별 색상, 위에 덮어 그림
    for nums, label in refined_boxes:
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)
        d.rectangle(denorm(nums), outline=color, width=2)
    return out


def img_to_response(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


app = FastAPI()


@app.on_event("startup")
def _startup():
    build_index()


@app.get("/api/groups")
def api_groups():
    result = []
    for gid, label in GROUPS.items():
        count = sum(1 for r in INDEX if r["group"] == gid)
        result.append({"id": gid, "label": label, "count": count})
    return JSONResponse(result)


@app.get("/api/labels")
def api_labels():
    return JSONResponse({**LABEL_COLORS, "__gray_original__": GRAY})


@app.get("/api/images")
def api_images(group: str, q: str = "", page: int = 1, page_size: int = 30):
    filtered = [
        (i, r) for i, r in enumerate(INDEX)
        if r["group"] == group and (not q or q.lower() in r["path"].lower())
    ]
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = [
        {"idx": i, "basename": Path(r["path"]).name}
        for i, r in filtered[start:end]
    ]
    return JSONResponse({"total": total, "page": page, "page_size": page_size, "items": page_items})


@app.get("/img/combined")
def img_combined(idx: int):
    row = INDEX[idx]
    orig_rec = read_at(ORIG_FILE, row["orig_offset"])
    refined_rec = read_at(REFINED_FILE, row["refined_offset"])
    image_path = orig_rec.get("image_path", "")
    scale = int(orig_rec.get("bbox_scale", 1000))

    try:
        img = Image.open(image_path)
        img.load()
    except Exception:
        placeholder = Image.new("RGB", (600, 200), (40, 40, 40))
        d = ImageDraw.Draw(placeholder)
        d.text((10, 90), f"image not found:\n{image_path}", fill=(255, 80, 80))
        return img_to_response(placeholder)

    orig_boxes = extract_layout_boxes(orig_rec.get("gt_html", ""))
    refined_boxes = extract_layout_boxes(refined_rec.get("gt_html", ""))
    return img_to_response(draw_combined(img, orig_boxes, refined_boxes, scale))


INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>GT bbox 보정 비교 뷰어 (전체 라벨)</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; padding: 16px; background: #111; color: #eee; }
  select, input, button { font-size: 14px; padding: 6px 10px; margin: 4px 4px 4px 0; }
  #controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 10px; }
  #legend { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; font-size: 12px; color: #ccc; }
  #legend .chip { display: flex; align-items: center; gap: 5px; }
  #legend .swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
  #pager { margin: 8px 0; }
  #status { color: #999; font-size: 12px; margin-left: 8px; }
  #grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .cell { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 6px; }
  .cell .fn { font-size: 11px; color: #9cf; word-break: break-all; margin-bottom: 6px; }
  .cell img { max-width: 100%; display: block; border-radius: 4px; }
</style>
</head>
<body>
<h2>GT bbox 보정 비교 — 회색(기존 GT) vs 라벨색(보정본)</h2>

<div id="legend"></div>

<div id="controls">
  <label>출처:
    <select id="group"></select>
  </label>
  <input id="q" placeholder="파일명 검색" />
  <button id="loadBtn">조회</button>
  <span id="status"></span>
</div>

<div id="pager"></div>
<div id="grid"></div>
<div id="pager2" style="margin-top:14px;"></div>

<script>
let currentPage = 1;
const pageSize = 30;

async function loadLegend() {
  const res = await fetch('/api/labels');
  const labels = await res.json();
  const el = document.getElementById('legend');
  el.innerHTML = Object.entries(labels).map(([name, color]) => {
    const label = name === '__gray_original__' ? '기존 GT(전체, 회색)' : name;
    return `<span class="chip"><span class="swatch" style="background:${color}"></span>${label}</span>`;
  }).join('');
}

async function loadGroups() {
  const res = await fetch('/api/groups');
  const groups = await res.json();
  const sel = document.getElementById('group');
  sel.innerHTML = groups.map(g => `<option value="${g.id}">${g.label} (${g.count}건)</option>`).join('');
}

function renderPager(container, page, total) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  container.innerHTML = `
    <button class="prevBtn" ${page<=1?'disabled':''}>이전</button>
    <span> page ${page} / ${totalPages} </span>
    <button class="nextBtn" ${page>=totalPages?'disabled':''}>다음</button>
  `;
  container.querySelector('.prevBtn').onclick = () => loadImages(page-1);
  container.querySelector('.nextBtn').onclick = () => loadImages(page+1);
}

async function loadImages(page=1) {
  currentPage = page;
  const group = document.getElementById('group').value;
  const q = document.getElementById('q').value;
  document.getElementById('status').innerText = '로딩중...';
  const res = await fetch(`/api/images?group=${encodeURIComponent(group)}&q=${encodeURIComponent(q)}&page=${page}&page_size=${pageSize}`);
  const data = await res.json();
  document.getElementById('status').innerText = `총 ${data.total}건 중 ${data.items.length}건 표시 (page ${data.page})`;

  const grid = document.getElementById('grid');
  grid.innerHTML = data.items.map(it => `
    <div class="cell">
      <div class="fn">${it.basename}</div>
      <img src="/img/combined?idx=${it.idx}" loading="lazy">
    </div>
  `).join('');

  renderPager(document.getElementById('pager'), data.page, data.total);
  renderPager(document.getElementById('pager2'), data.page, data.total);
}

document.getElementById('loadBtn').onclick = () => loadImages(1);
document.getElementById('group').addEventListener('change', () => loadImages(1));

loadLegend();
loadGroups().then(() => loadImages(1));
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8092)
