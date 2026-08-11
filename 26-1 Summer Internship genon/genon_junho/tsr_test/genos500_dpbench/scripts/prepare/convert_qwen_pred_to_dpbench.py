"""
qwen3.5(VLM) 추론 결과(predictions_unified.jsonl) -> dp-bench pred JSON.

이미 추론이 끝난 예측 파일을 dp-bench 평가 포맷으로 변환한다.
(Genos doc_parser 로 직접 추론하는 대신, 외부 VLM 추론 결과를 채점에 쓰는 경로)

지원하는 두 출력 형식:
- JSON 출력: full_response = [{bbox, category, text}]  (dots-ocr 배열)
- HTML 출력: pred_layout_html = 플랫 HTML 태그열(<p>,<h2>,<img/>,<table>,<ul>...)  (※ HTML 형식만 beautifulsoup4 필요)

출력 key 는 GT(reference_dp_bench.json)와 맞추기 위해 이미지 파일명을 <doc_id>.pdf 로 바꾼다.
NID(layout)은 좌표 무시·순서대로 content.text 비교(figure/table/chart 제외),
TEDS(table)은 table 의 content.html 만 사용.

사용:
  python convert_qwen_pred_to_dpbench.py \
    --input  genos-500-set/qwen_3.5_finetune/genos500_unified_base_20260609/predictions_unified.jsonl \
    --output dp_out/qwen_3.5_finetune/pred_dp_bench.json
"""

import argparse
import json
import os
from pathlib import Path

JSON_CAT = {
    "text": "paragraph", "section-header": "heading1", "title": "heading1",
    "page-header": "header", "page-footer": "footer", "list-item": "list",
    "footnote": "footnote", "caption": "caption", "formula": "equation",
    "picture": "figure", "table": "table", "chart": "chart",
}
TAG_CAT = {
    "p": "paragraph", "h1": "heading1", "h2": "heading1", "h3": "heading1",
    "h4": "heading1", "h5": "heading1", "h6": "heading1",
    "img": "figure", "table": "table", "ul": "list", "ol": "list",
    "nav": "paragraph", "li": "list", "figure": "figure",
}
_DEC = json.JSONDecoder()


def _label_to_cat(label):
    """Chandra div-HTML data-label 값을 dp-bench category 로 변환."""
    key = str(label or "").strip().lower().replace("_", "-").replace(" ", "-")
    return JSON_CAT.get(key, "paragraph")


def _parse_bbox_str(s):
    """data-bbox="x0 y0 x1 y1" 또는 "x0,y0,x1,y1" 문자열을 bbox list 로 변환."""
    if s is None:
        return None
    raw = str(s).replace(",", " ").split()
    if len(raw) != 4:
        return None
    try:
        return [int(round(float(v))) for v in raw]
    except Exception:
        return None


def _salvage(raw):
    """잘린 JSON 배열에서 완전한 객체만 복구."""
    s = raw.lstrip()
    if s.startswith("["):
        s = s[1:]
    out, i, n = [], 0, len(s)
    while i < n:
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n or s[i] == "]":
            break
        try:
            o, e = _DEC.raw_decode(s, i)
        except Exception:
            break
        out.append(o)
        i = e
    return out


def _coords(b):
    if not isinstance(b, list) or len(b) != 4:
        return [{"x": 0, "y": 0}] * 4
    x1, y1, x2, y2 = b
    return [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]


def _elem(category, idx=0, text="", html="", bbox=None):
    is_t = category == "table"
    return {
        "coordinates": _coords(bbox),
        "category": category,
        "id": idx,
        "page": 1,
        "content": {"text": "" if is_t else text, "html": html if is_t else "", "markdown": ""},
    }


def elements_from_json(full_response):
    items = []
    if full_response:
        try:
            v = json.loads(full_response)
            items = v if isinstance(v, list) else []
        except Exception:
            items = _salvage(full_response)
    els = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        cat = JSON_CAT.get(str(it.get("category", "")).strip().lower().replace("_", "-").replace(" ", "-"), "paragraph")
        t = it.get("text")
        t = "" if t is None else str(t)
        els.append(_elem(cat, idx=idx, text=t, html=t, bbox=it.get("bbox")))
    return els


def elements_from_html(html):
    from bs4 import BeautifulSoup  # HTML 출력 변환 시에만 필요
    els = []
    if not html or not html.strip():
        return els
    soup = BeautifulSoup(html, "html.parser")

    # Chandra 학습/평가 출력:
    # <div data-bbox="..." data-label="Text">...</div>
    # <div data-bbox="..." data-label="Table"><table>...</table></div>
    # bbox 는 모델 출력 grid 좌표(보통 0~1000)로 보존하고,
    # evaluate.py 호출 시 --pred_coord_space grid --pred_grid 로 image 좌표에 맞춘다.
    layout_divs = soup.find_all("div", attrs={"data-label": True})
    if layout_divs:
        for idx, el in enumerate(layout_divs):
            cat = _label_to_cat(el.get("data-label"))
            bbox = _parse_bbox_str(el.get("data-bbox"))
            if cat == "table":
                tbl = el.find("table")
                els.append(_elem("table", idx=idx, html=str(tbl) if tbl else "", bbox=bbox))
            else:
                els.append(_elem(cat, idx=idx, text=el.get_text(" ", strip=True), bbox=bbox))
        return els

    tops = soup.find_all(recursive=False) or [soup]
    for idx, el in enumerate(tops):
        name = getattr(el, "name", None)
        if name is None:
            continue
        cat = TAG_CAT.get(name, "paragraph")
        if cat == "table":
            els.append(_elem("table", idx=idx, html=str(el)))
        else:
            els.append(_elem(cat, idx=idx, text=el.get_text(" ", strip=True)))
    return els


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="predictions_unified.jsonl 경로")
    ap.add_argument("--output", required=True, help="출력 dp-bench pred JSON 경로")
    args = ap.parse_args()

    pred, empty, fmt = {}, 0, None
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = r["image_path"].split("/")[-1].rsplit(".", 1)[0] + ".pdf"  # doc_0001.png -> doc_0001.pdf
            of = r.get("output_format")
            if of == "html" or (of is None and r.get("pred_layout_html") is not None):
                fmt = "html"
                els = elements_from_html(r.get("pred_layout_html") or r.get("full_response") or "")
            else:
                fmt = "json"
                els = elements_from_json(r.get("full_response") or "")
            if not els:
                empty += 1
            pred[key] = {"elements": els}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False)
    print(f"[{fmt}] 변환 완료: {len(pred)} docs (빈 문서 {empty}) -> {args.output}")


if __name__ == "__main__":
    main()
