"""
combined/{train,valid,test}.jsonl 시각화 도구.

- 출처(reference_16886 / new_63541) + split 선택 -> 이미지 목록
- 선택한 이미지들을 한 줄에 4컬럼(원본 / layout bbox / ocr bbox / gt_html)으로 시각화

Usage:
    python3 app.py
    -> http://localhost:8090
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageDraw, ImageFont

COMBINED_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined")
SPLIT_FILES = {
    "train": COMBINED_DIR / "train.jsonl",
    "valid": COMBINED_DIR / "valid.jsonl",
    "test": COMBINED_DIR / "test.jsonl",
}

GROUPS = {
    "reference_16886": "16000장짜리 (reference_16886)",
    "new_63541": "20000장짜리 (new_63541)",
}


def classify(image_path: str) -> str:
    if "reference_16886" in image_path:
        return "reference_16886"
    if "ocr_filter_result" in image_path or "new_63541" in image_path:
        return "new_63541"
    return "other"


# split -> list[{"offset": int, "path": str, "group": str}]
INDEX: dict[str, list[dict]] = {}


def build_index() -> None:
    for split, path in SPLIT_FILES.items():
        items = []
        with path.open("r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_path = d.get("image_path", "")
                items.append({"offset": offset, "path": image_path, "group": classify(image_path)})
        INDEX[split] = items
        print(f"[index] {split}: {len(items)} rows")


def read_record(split: str, idx: int) -> dict:
    offset = INDEX[split][idx]["offset"]
    with SPLIT_FILES[split].open("r", encoding="utf-8") as f:
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


def draw_boxes(
    img: Image.Image,
    boxes: list[tuple[list[int], str]],
    scale: int,
    color: str,
    show_label: bool = False,
) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for nums, label in boxes:
        x0, y0, x1, y1 = nums
        px0, py0 = x0 / scale * w, y0 / scale * h
        px1, py1 = x1 / scale * w, y1 / scale * h
        draw.rectangle([px0, py0, px1, py1], outline=color, width=2)
        if show_label and label:
            draw.text((px0 + 2, max(0, py0 - 12)), label, fill=color)
    return out


def img_to_response(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


app = FastAPI()


@app.on_event("startup")
def _startup():
    build_index()


@app.get("/api/groups")
def api_groups():
    result = []
    for gid, label in GROUPS.items():
        counts = {}
        for split in SPLIT_FILES:
            counts[split] = sum(1 for r in INDEX[split] if r["group"] == gid)
        result.append({"id": gid, "label": label, "counts": counts})
    return JSONResponse(result)


@app.get("/api/images")
def api_images(
    group: str,
    split: str,
    q: str = "",
    page: int = 1,
    page_size: int = 60,
):
    items = INDEX.get(split, [])
    filtered = [
        (i, r) for i, r in enumerate(items)
        if r["group"] == group and (not q or q.lower() in r["path"].lower())
    ]
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = [
        {"idx": i, "basename": Path(r["path"]).name, "path": r["path"]}
        for i, r in filtered[start:end]
    ]
    return JSONResponse({"total": total, "page": page, "page_size": page_size, "items": page_items})


@app.get("/api/record")
def api_record(split: str, idx: int):
    rec = read_record(split, idx)
    return JSONResponse(
        {
            "image_path": rec.get("image_path", ""),
            "gt_html": rec.get("gt_html", ""),
            "prompt_style": rec.get("prompt_style", ""),
            "bbox_scale": rec.get("bbox_scale", 1000),
            "ocr_info_count": len(rec.get("ocr_info") or []),
        }
    )


@app.get("/img/{kind}")
def img_kind(kind: str, split: str, idx: int):
    rec = read_record(split, idx)
    image_path = rec.get("image_path", "")
    scale = int(rec.get("bbox_scale", 1000))

    try:
        img = Image.open(image_path)
        img.load()
    except Exception:
        placeholder = Image.new("RGB", (600, 200), (40, 40, 40))
        d = ImageDraw.Draw(placeholder)
        d.text((10, 90), f"image not found:\n{image_path}", fill=(255, 80, 80))
        return img_to_response(placeholder)

    if kind == "original":
        return img_to_response(img)

    if kind == "layout":
        boxes = extract_layout_boxes(rec.get("gt_html", ""))
        return img_to_response(draw_boxes(img, boxes, scale, "red", show_label=True))

    if kind == "ocr":
        ocr_info = rec.get("ocr_info") or []
        boxes = [(o["bbox"], "") for o in ocr_info if o.get("bbox")]
        return img_to_response(draw_boxes(img, boxes, scale, "lime", show_label=False))

    return Response(status_code=404)


INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>OCR/Layout GT Viewer</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; padding: 16px; background: #111; color: #eee; }
  select, input, button { font-size: 14px; padding: 6px 10px; margin: 4px 4px 4px 0; }
  #controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 12px; }
  #imgList { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 6px; max-height: 320px; overflow-y: auto; border: 1px solid #333; padding: 8px; background: #1a1a1a; }
  .imgItem { font-size: 11px; word-break: break-all; padding: 4px; border: 1px solid #333; border-radius: 4px; cursor: pointer; }
  .imgItem.selected { background: #274; border-color: #4a8; }
  #pager { margin: 6px 0; }
  #result { margin-top: 20px; }
  .row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px; margin-bottom: 24px; border-bottom: 1px solid #333; padding-bottom: 16px; }
  .cell { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 6px; overflow: hidden; }
  .cell h4 { margin: 0 0 6px 0; font-size: 12px; color: #9cf; }
  .cell img { max-width: 100%; display: block; }
  .cell pre.htmlSource { width: 100%; height: 360px; margin: 0; overflow: auto; background: #0b0b0b; color: #d4d4d4; border-radius: 4px; padding: 8px; box-sizing: border-box; font-family: "SFMono-Regular", Consolas, Menlo, monospace; font-size: 11px; white-space: pre-wrap; word-break: break-all; }
  .cell pre.htmlSource .tag { color: #569cd6; }
  .cell pre.htmlSource .attr { color: #9cdcfe; }
  .cell pre.htmlSource .val { color: #ce9178; }
  #status { color: #999; font-size: 12px; margin-left: 8px; }
</style>
</head>
<body>
<h2>OCR / Layout GT Viewer</h2>

<div id="controls">
  <label>출처:
    <select id="group"></select>
  </label>
  <label>split:
    <select id="split">
      <option value="train">train</option>
      <option value="valid">valid</option>
      <option value="test">test</option>
    </select>
  </label>
  <input id="q" placeholder="파일명 검색" />
  <button id="searchBtn">검색</button>
  <span id="status"></span>
</div>

<div id="pager"></div>
<div id="imgList"></div>
<div style="margin-top:10px;">
  <button id="visualizeBtn">선택한 이미지 시각화 (<span id="selCount">0</span>)</button>
  <button id="clearBtn">선택 초기화</button>
</div>

<div id="result"></div>

<script>
let selected = new Map(); // idx -> basename
let currentPage = 1;
const pageSize = 60;

async function loadGroups() {
  const res = await fetch('/api/groups');
  const groups = await res.json();
  const sel = document.getElementById('group');
  sel.innerHTML = groups.map(g =>
    `<option value="${g.id}">${g.label} (train:${g.counts.train}, valid:${g.counts.valid}, test:${g.counts.test})</option>`
  ).join('');
}

function updateSelCount() {
  document.getElementById('selCount').innerText = selected.size;
}

async function loadImages(page=1) {
  currentPage = page;
  const group = document.getElementById('group').value;
  const split = document.getElementById('split').value;
  const q = document.getElementById('q').value;
  document.getElementById('status').innerText = '로딩중...';
  const res = await fetch(`/api/images?group=${encodeURIComponent(group)}&split=${split}&q=${encodeURIComponent(q)}&page=${page}&page_size=${pageSize}`);
  const data = await res.json();
  document.getElementById('status').innerText = `총 ${data.total}건 중 ${data.items.length}건 표시 (page ${data.page})`;

  const listEl = document.getElementById('imgList');
  listEl.innerHTML = data.items.map(it => {
    const key = `${split}::${it.idx}`;
    const isSel = selected.has(key) ? 'selected' : '';
    return `<div class="imgItem ${isSel}" data-idx="${it.idx}" data-split="${split}" data-basename="${it.basename.replace(/"/g,'&quot;')}">${it.basename}</div>`;
  }).join('');

  listEl.querySelectorAll('.imgItem').forEach(el => {
    el.addEventListener('click', () => {
      const idx = parseInt(el.dataset.idx);
      const key = `${el.dataset.split}::${idx}`;
      if (selected.has(key)) {
        selected.delete(key);
        el.classList.remove('selected');
      } else {
        selected.set(key, {split: el.dataset.split, idx: idx, basename: el.dataset.basename});
        el.classList.add('selected');
      }
      updateSelCount();
    });
  });

  const totalPages = Math.max(1, Math.ceil(data.total / pageSize));
  const pager = document.getElementById('pager');
  pager.innerHTML = `
    <button id="prevBtn" ${page<=1?'disabled':''}>이전</button>
    <span> page ${page} / ${totalPages} </span>
    <button id="nextBtn" ${page>=totalPages?'disabled':''}>다음</button>
  `;
  document.getElementById('prevBtn').onclick = () => loadImages(page-1);
  document.getElementById('nextBtn').onclick = () => loadImages(page+1);
}

document.getElementById('searchBtn').onclick = () => loadImages(1);
document.getElementById('group').addEventListener('change', () => loadImages(1));
document.getElementById('split').addEventListener('change', () => loadImages(1));
document.getElementById('clearBtn').onclick = () => {
  selected.clear();
  updateSelCount();
  document.querySelectorAll('.imgItem.selected').forEach(el => el.classList.remove('selected'));
};

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function highlightHtmlSource(raw) {
  // raw 태그를 그대로 보여주되(escape), 태그/속성/속성값에 색만 입힌다.
  const escaped = escapeHtml(raw);
  return escaped.replace(
    /(&lt;\/?[a-zA-Z0-9]+)((?:\s+[a-zA-Z-]+=&quot;[^&]*&quot;)*)(\s*\/?&gt;)/g,
    (whole, tagOpen, attrs, tagClose) => {
      const coloredAttrs = attrs.replace(
        /([a-zA-Z-]+)(=)(&quot;[^&]*&quot;)/g,
        '<span class="attr">$1</span>$2<span class="val">$3</span>'
      );
      return `<span class="tag">${tagOpen}</span>${coloredAttrs}<span class="tag">${tagClose}</span>`;
    }
  );
}

document.getElementById('visualizeBtn').onclick = async () => {
  const resultEl = document.getElementById('result');
  resultEl.innerHTML = '';
  let ok = 0, failed = 0;
  for (const [key, info] of selected.entries()) {
    const split = info.split;
    const idx = info.idx;
    try {
      const res = await fetch(`/api/record?split=${split}&idx=${idx}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const rec = await res.json();
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `
        <div class="cell"><h4>원본 (${info.basename})</h4><img src="/img/original?split=${split}&idx=${idx}" loading="lazy"></div>
        <div class="cell"><h4>layout bbox (gt_html)</h4><img src="/img/layout?split=${split}&idx=${idx}" loading="lazy"></div>
        <div class="cell"><h4>ocr bbox (ocr_info, ${rec.ocr_info_count}건)</h4><img src="/img/ocr?split=${split}&idx=${idx}" loading="lazy"></div>
        <div class="cell"><h4>gt_html (prompt_style=${rec.prompt_style})</h4><pre class="htmlSource">${highlightHtmlSource(rec.gt_html)}</pre></div>
      `;
      resultEl.appendChild(row);
      ok++;
    } catch (e) {
      console.error('failed to render', key, e);
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `<div class="cell" style="grid-column: 1 / -1; color:#f66;">렌더링 실패: ${split} idx=${idx} (${info.basename}) - ${e}</div>`;
      resultEl.appendChild(row);
      failed++;
    }
  }
  document.getElementById('status').innerText = `시각화 완료: 성공 ${ok}건, 실패 ${failed}건 (선택 총 ${selected.size}건)`;
};

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

    uvicorn.run(app, host="0.0.0.0", port=8090)
