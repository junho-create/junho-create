#!/usr/bin/env python3
"""combined_e21_grid 학습 데이터 육안 검수 도구.

레코드마다 [이미지+bbox 오버레이 | gt_html 실제 렌더링] 2그리드를 만들고,
"라벨링이 수상해 보이는 순"으로 정렬한 정적 HTML 갤러리를 만든다.
갤러리에서 O / △ / X 로 판정하면 브라우저 localStorage 에 저장되고
JSON 으로 내려받을 수 있다(정적 서버라 POST 를 못 받으므로 export 방식).

수상함 점수(각 0~1 로 정규화 후 가중합, 높을수록 의심):
  ocr_outside   OCR 텍스트 박스 중심이 어떤 레이아웃 박스에도 안 들어가는 비율
                -> 라벨이 빠진 영역이 있다는 뜻. 가장 강한 신호.
  overlap       레이아웃 박스끼리 IoU>0.3 으로 겹치는 쌍의 비율 (박스가 뭉개짐)
  giant         페이지의 80% 이상을 한 박스가 덮는데 요소가 여러 개 (통짜 박스)
  tiny          면적이 극단적으로 작은 박스(1000x1000 기준 area<200) 비율
  degenerate    폭/높이가 0 이하이거나 좌표가 역전된 박스
  order         div 순서와 기하학적 읽기순서(위->아래, 좌->우)의 역전 비율
  monotony      요소가 8개 이상인데 label 이 전부 같음
  density       박스 면적 대비 텍스트 길이가 극단(글자가 박스에 안 들어감)
  dup_conflict  같은 이미지가 다른 경로로 또 있고 정답 HTML 이 다름 (소스 병합 충돌)

사용:
  python review_e21grid.py --top 300            # 소스별 상위 300개 렌더
  python review_e21grid.py --top 300 --port 8904
  python review_e21grid.py --top 300 --no_serve
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html as htmllib
import json
import os
import re
import socketserver
import http.server

BB = re.compile(r'<div\s+data-bbox="([^"]+)"\s+data-label="([^"]+)"\s*>(.*?)</div>', re.S | re.I)
BB_LOOSE = re.compile(r'data-bbox="([^"]+)"[^>]*data-label="([^"]+)"', re.I)
TAG = re.compile(r"<[^>]+>")
SCRIPT = re.compile(r"<script.*?</script>", re.S | re.I)
ONATTR = re.compile(r'\son\w+\s*=\s*"[^"]*"', re.I)

LABEL_COLORS = {
    "Table": "#e6194b", "Text": "#3cb44b", "Title": "#4363d8",
    "Section-header": "#f58231", "List-item": "#911eb4", "Picture": "#46f0f0",
    "Caption": "#f032e6", "Formula": "#bcf60c", "Footnote": "#fabebe",
    "Page-header": "#008080", "Page-footer": "#9a6324",
}
DEFAULT_COLOR = "#808080"

SCALE = 1000.0  # 데이터 좌표계(0~1000). review 는 이 스케일만 다룬다.


def source_of(path: str) -> str:
    if "reference_16886" in path:
        return "reference_16886"
    if "ocr_filter_result" in path or "new_63541" in path:
        return "new_63541"
    return "other"


def parse_elements(gt_html: str):
    els = []
    for m in BB.finditer(gt_html or ""):
        try:
            v = [float(x) for x in m.group(1).split()]
        except ValueError:
            continue
        if len(v) != 4:
            continue
        els.append({"bbox": v, "label": m.group(2), "inner": m.group(3)})
    if not els:  # div 중첩 등으로 위 정규식이 실패하면 느슨하게 bbox/label 만 회수
        for m in BB_LOOSE.finditer(gt_html or ""):
            try:
                v = [float(x) for x in m.group(1).split()]
            except ValueError:
                continue
            if len(v) == 4:
                els.append({"bbox": v, "label": m.group(2), "inner": ""})
    return els


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score_record(rec, dup_conflict: bool):
    els = parse_elements(rec.get("gt_html", ""))
    n = len(els)
    sig = collections.OrderedDict()
    if n == 0:
        sig["no_element"] = 1.0
        return 1.0, sig, els

    boxes = [e["bbox"] for e in els]
    page_area = SCALE * SCALE

    # degenerate / tiny
    deg = sum(1 for b in boxes if b[2] <= b[0] or b[3] <= b[1] or min(b) < 0 or max(b) > SCALE)
    tiny = sum(1 for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) < 200)
    sig["degenerate"] = deg / n
    sig["tiny"] = tiny / n

    # giant: 한 박스가 페이지 80% 이상인데 요소가 2개 이상
    giant = 0.0
    if n >= 2:
        big = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / page_area
        giant = 1.0 if big > 0.8 else 0.0
    sig["giant"] = giant

    # overlap: IoU>0.3 쌍 비율 (쌍이 너무 많으면 앞쪽 60개만)
    bs = boxes[:60]
    pairs = 0
    hit = 0
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            pairs += 1
            if iou(bs[i], bs[j]) > 0.3:
                hit += 1
    sig["overlap"] = (hit / pairs) if pairs else 0.0

    # OCR 이 라벨 밖에 있는 비율
    ocr = [o.get("bbox") for o in (rec.get("ocr_info") or []) if o.get("bbox") and len(o["bbox"]) == 4]
    outside = 0
    tol = SCALE * 0.02
    for o in ocr:
        cx, cy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
        if not any(b[0] - tol <= cx <= b[2] + tol and b[1] - tol <= cy <= b[3] + tol for b in boxes):
            outside += 1
    sig["ocr_outside"] = (outside / len(ocr)) if ocr else 0.0

    # 읽기순서 역전
    order = sorted(range(n), key=lambda i: (round(boxes[i][1] / 20), boxes[i][0]))
    inv = sum(1 for a in range(n) for b in range(a + 1, n) if order.index(a) > order.index(b))
    sig["order"] = min(1.0, inv / max(1, n * (n - 1) / 2) * 2)

    # 라벨 단조성
    labs = {e["label"] for e in els}
    sig["monotony"] = 1.0 if (n >= 8 and len(labs) == 1) else 0.0

    # 텍스트 밀도 이상: 박스 면적 대비 글자수 극단
    worst = 0.0
    for e in els:
        b = e["bbox"]
        area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        chars = len(TAG.sub("", e["inner"]).strip())
        if chars > 40:
            d = chars / (area / 1000.0)  # 면적 1000 단위당 글자수
            worst = max(worst, min(1.0, d / 60.0))
    sig["density"] = worst

    sig["dup_conflict"] = 1.0 if dup_conflict else 0.0

    W = {
        "no_element": 3.0, "ocr_outside": 3.0, "overlap": 2.0, "giant": 1.5,
        "degenerate": 2.0, "tiny": 1.0, "order": 0.8, "monotony": 0.7,
        "density": 1.0, "dup_conflict": 1.2,
    }
    total = sum(sig.get(k, 0.0) * w for k, w in W.items())
    return total, sig, els


def dup_conflict_set(records):
    """이미지 내용이 같은데 gt_html 이 다른 레코드 인덱스 집합."""
    by_size = collections.defaultdict(list)
    for i, r in enumerate(records):
        try:
            by_size[os.path.getsize(r["image_path"])].append(i)
        except OSError:
            pass
    out = set()
    for _, idxs in by_size.items():
        if len(idxs) < 2:
            continue
        h = collections.defaultdict(list)
        for i in idxs:
            try:
                with open(records[i]["image_path"], "rb") as f:
                    h[hashlib.md5(f.read()).hexdigest()].append(i)
            except OSError:
                pass
        for _, group in h.items():
            if len(group) > 1 and len({records[i].get("gt_html") for i in group}) > 1:
                out.update(group)
    return out


def render_overlay(rec, els, out_path, width=760):
    from PIL import Image, ImageDraw

    try:
        img = Image.open(rec["image_path"]).convert("RGB")
    except Exception:
        return None
    scale = width / img.width
    img = img.resize((width, max(1, int(img.height * scale))))
    d = ImageDraw.Draw(img)
    W, H = img.size
    for i, e in enumerate(els):
        b = e["bbox"]
        x0, y0 = b[0] / SCALE * W, b[1] / SCALE * H
        x1, y1 = b[2] / SCALE * W, b[3] / SCALE * H
        c = LABEL_COLORS.get(e["label"], DEFAULT_COLOR)
        d.rectangle([x0, y0, x1, y1], outline=c, width=3)
        d.text((x0 + 3, max(0, y0 - 12)), f"{i}:{e['label']}", fill=c)
    img.save(out_path, "JPEG", quality=80)
    return os.path.basename(out_path)


def sanitize(h):
    h = SCRIPT.sub("", h or "")
    return ONATTR.sub("", h)


PAGE_HEAD = """<meta charset="utf-8">
<title>combined_e21_grid 라벨 검수</title>
<style>
 body{font-family:system-ui,-apple-system,'Noto Sans KR',sans-serif;margin:0;background:#f6f7f9;color:#111}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:10;
        display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 select,button{font-size:14px;padding:5px 9px;border-radius:6px;border:1px solid #ccc;background:#fff;cursor:pointer}
 .card{background:#fff;margin:14px;border:1px solid #ddd;border-radius:10px;overflow:hidden}
 .card.done-o{border-color:#2a9d3f;border-width:2px}
 .card.done-t{border-color:#e08b00;border-width:2px}
 .card.done-x{border-color:#d33;border-width:2px}
 .hd{padding:8px 12px;background:#fafafa;border-bottom:1px solid #eee;display:flex;gap:12px;
     align-items:center;flex-wrap:wrap;font-size:13px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:0}
 @media(max-width:1100px){.grid{grid-template-columns:1fr}}
 .pane{padding:10px;overflow:auto;max-height:78vh}
 .pane:first-child{border-right:1px solid #eee}
 .pane img{width:100%;height:auto;display:block}
 .pane{min-height:420px}
 .render{font-size:13px;line-height:1.5}
 .render table{border-collapse:collapse;margin:6px 0}
 .render td,.render th{border:1px solid #999;padding:3px 6px}
 .render div[data-label]{border-left:3px solid #bbb;padding:2px 0 2px 8px;margin:5px 0}
 .render div[data-label]::before{content:attr(data-label);font-size:10px;color:#888;display:block}
 .score{font-weight:700}
 .sig{color:#666;font-size:12px}
 .mark{font-size:16px;padding:4px 12px}
 .mark.on{background:#111;color:#fff}
 .counts{margin-left:auto;font-size:13px;color:#333}
 .hide{display:none}
</style>"""


def build_html(items, out_dir, total_by_src):
    """카드 '골격'만 HTML 로 굽고, 무거운 렌더 패널 내용은 content.json 으로 분리한다.

    600개 카드의 gt_html 을 그대로 DOM 에 넣으면 표 셀만 5만 개가 넘어 브라우저가
    첫 페인트 전에 멈춘다. 그래서 내용은 JSON 으로 두고 IntersectionObserver 로
    화면에 들어온 카드만 주입하고, 멀어지면 다시 비운다(가상 스크롤).
    """
    content = {}
    rows = []
    for it in items:
        sig = " · ".join(f"{k}={v:.2f}" for k, v in it["sig"].items() if v > 0.01)
        content[it["id"]] = {"img": it["img"], "html": it["html"]}
        rows.append(
            f'<div class="card" data-src="{it["src"]}" data-id="{it["id"]}">'
            f'<div class="hd"><span class="score">#{it["rank"]} 점수 {it["score"]:.2f}</span>'
            f'<span>[{it["src"]}]</span><span>{htmllib.escape(it["name"])}</span>'
            f'<span>요소 {it["n"]}개</span><span class="sig">{sig}</span>'
            f'<span style="margin-left:auto">'
            f'<button class="mark" data-v="o">O</button>'
            f'<button class="mark" data-v="t">△</button>'
            f'<button class="mark" data-v="x">X</button></span></div>'
            f'<div class="grid"><div class="pane pane-img"></div>'
            f'<div class="pane render"></div></div></div>'
        )
    with open(os.path.join(out_dir, "content.json"), "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False)

    counts = " / ".join(f"{k} {v}" for k, v in total_by_src.items())
    body = """<header>
 <b>라벨 검수</b>
 <label>소스 <select id="src">
   <option value="all">전체</option>
   <option value="reference_16886">reference_16886</option>
   <option value="new_63541">new_63541</option>
 </select></label>
 <label>판정 <select id="flt">
   <option value="all">전체</option><option value="none">미판정</option>
   <option value="o">O</option><option value="t">△</option><option value="x">X</option>
 </select></label>
 <button id="exp">판정 JSON 내보내기</button>
 <button id="clr">판정 초기화</button>
 <span class="counts" id="cnt">로딩중…</span>
 <span class="sig">렌더 대상: __COUNTS__ · 단축키 1=O 2=△ 3=X</span>
</header>
__ROWS__
<script>
const KEY='e21grid_review_v1';
let marks=JSON.parse(localStorage.getItem(KEY)||'{}');
let DATA=null;

// 화면에 들어온 카드에만 이미지/HTML 을 주입하고, 벗어나면 비워서 DOM 을 가볍게 유지
const io=new IntersectionObserver(es=>{
  for(const e of es){
    const card=e.target, id=card.dataset.id;
    const pi=card.querySelector('.pane-img'), pr=card.querySelector('.render');
    if(e.isIntersecting){
      if(DATA && !card.dataset.mounted){
        const d=DATA[id]; if(!d) continue;
        pi.innerHTML='<img src="images/'+d.img+'">';
        pr.innerHTML=d.html;
        card.dataset.mounted='1';
      }
    } else if(card.dataset.mounted){
      pi.innerHTML=''; pr.innerHTML=''; delete card.dataset.mounted;
    }
  }
},{rootMargin:'1200px 0px'});
document.querySelectorAll('.card').forEach(c=>io.observe(c));

fetch('content.json').then(r=>r.json()).then(d=>{
  DATA=d;
  // 이미 화면에 있는 카드 강제 주입
  document.querySelectorAll('.card').forEach(card=>{
    const r=card.getBoundingClientRect();
    if(r.top<window.innerHeight+1200 && r.bottom>-1200 && !card.dataset.mounted){
      const x=DATA[card.dataset.id]; if(!x) return;
      card.querySelector('.pane-img').innerHTML='<img src="images/'+x.img+'">';
      card.querySelector('.render').innerHTML=x.html;
      card.dataset.mounted='1';
    }
  });
  paint();
});

function paint(){
  let c={o:0,t:0,x:0};
  document.querySelectorAll('.card').forEach(card=>{
    const id=card.dataset.id, v=marks[id];
    card.classList.remove('done-o','done-t','done-x');
    if(v){card.classList.add('done-'+v); c[v]++;}
    card.querySelectorAll('.mark').forEach(b=>b.classList.toggle('on', b.dataset.v===v));
  });
  document.getElementById('cnt').textContent='O '+c.o+' · △ '+c.t+' · X '+c.x;
  filter();
}
function filter(){
  const s=document.getElementById('src').value, f=document.getElementById('flt').value;
  document.querySelectorAll('.card').forEach(card=>{
    const v=marks[card.dataset.id];
    let ok=(s==='all'||card.dataset.src===s);
    if(f==='none') ok=ok&&!v; else if(f!=='all') ok=ok&&v===f;
    card.classList.toggle('hide',!ok);
  });
}
document.addEventListener('click',e=>{
  const b=e.target.closest('.mark'); if(!b) return;
  const card=b.closest('.card'), id=card.dataset.id;
  marks[id] = (marks[id]===b.dataset.v) ? undefined : b.dataset.v;
  if(!marks[id]) delete marks[id];
  localStorage.setItem(KEY,JSON.stringify(marks)); paint();
});
document.addEventListener('keydown',e=>{
  if(!['1','2','3'].includes(e.key)) return;
  const cards=[...document.querySelectorAll('.card:not(.hide)')];
  const cur=cards.find(c=>c.getBoundingClientRect().top>-100); if(!cur) return;
  const v={'1':'o','2':'t','3':'x'}[e.key];
  marks[cur.dataset.id]= marks[cur.dataset.id]===v?undefined:v;
  if(!marks[cur.dataset.id]) delete marks[cur.dataset.id];
  localStorage.setItem(KEY,JSON.stringify(marks)); paint();
});
document.getElementById('src').onchange=filter;
document.getElementById('flt').onchange=filter;
document.getElementById('exp').onclick=()=>{
  const blob=new Blob([JSON.stringify(marks,null,1)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='review_marks.json'; a.click();
};
document.getElementById('clr').onclick=()=>{
  if(confirm('판정을 모두 지웁니다')){marks={};localStorage.removeItem(KEY);paint();}
};
paint();
</script>"""
    body = body.replace("__COUNTS__", counts).replace("__ROWS__", "".join(rows))
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(PAGE_HEAD + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="combined_e21_grid/train.jsonl")
    ap.add_argument("--out_dir", default="e21grid_review_out")
    ap.add_argument("--top", type=int, default=300, help="소스별 렌더 개수(수상함 상위)")
    ap.add_argument("--port", type=int, default=8904)
    ap.add_argument("--no_serve", action="store_true")
    ap.add_argument("--skip_dup", action="store_true", help="이미지 md5 중복 검사 생략(빠름)")
    ap.add_argument("--reuse_images", action="store_true", help="이미 만들어둔 오버레이 이미지를 재사용(재렌더 생략)")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    print(f"[load] {len(recs)}건")

    dup = set()
    if not a.skip_dup:
        tagname = a.data.replace("/", "_").replace(".jsonl", "")
        cache = os.path.join(os.path.dirname(a.data) or ".", f".dup_cache_{tagname}.json")
        if os.path.exists(cache):
            dup = set(json.load(open(cache)))
            print(f"[dup] 캐시 사용: 충돌 {len(dup)}건")
        else:
            print("[dup] 이미지 md5 중복/충돌 검사 중...")
            dup = dup_conflict_set(recs)
            json.dump(sorted(dup), open(cache, "w"))
            print(f"[dup] 충돌 {len(dup)}건")

    scored = []
    for i, r in enumerate(recs):
        s, sig, els = score_record(r, i in dup)
        scored.append((s, i, sig, els))
        if (i + 1) % 5000 == 0:
            print(f"  scored {i+1}/{len(recs)}")
    scored.sort(key=lambda t: -t[0])

    out_dir = a.out_dir
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    per_src = collections.Counter()
    items = []
    rank = 0
    for s, i, sig, els in scored:
        r = recs[i]
        src = source_of(r["image_path"])
        if per_src[src] >= a.top:
            continue
        stem = f"{src}_{i}"
        dst = os.path.join(img_dir, stem + ".jpg")
        if a.reuse_images and os.path.exists(dst):
            img = os.path.basename(dst)
        else:
            img = render_overlay(r, els, dst)
        if img is None:
            continue
        per_src[src] += 1
        rank += 1
        items.append({
            "rank": rank, "id": str(i), "src": src, "score": s, "sig": sig,
            "n": len(els), "name": os.path.basename(r["image_path"]),
            "img": img, "html": sanitize(r.get("gt_html", "")),
        })
        if rank % 100 == 0:
            print(f"  rendered {rank}")
    build_html(items, out_dir, dict(per_src))
    print(f"[done] {len(items)}건 -> {os.path.join(out_dir,'index.html')}")

    if not a.no_serve:
        os.chdir(out_dir)
        # 단일 스레드 TCPServer 는 이미지 요청 하나가 끝날 때까지 다음 요청을 막는다.
        # 카드 600개가 이미지를 동시에 당겨오므로 반드시 스레드 서버를 써야 한다.
        http.server.ThreadingHTTPServer.allow_reuse_address = True
        with http.server.ThreadingHTTPServer(("0.0.0.0", a.port), http.server.SimpleHTTPRequestHandler) as httpd:
            print(f"serving on :{a.port}")
            httpd.serve_forever()


if __name__ == "__main__":
    main()
