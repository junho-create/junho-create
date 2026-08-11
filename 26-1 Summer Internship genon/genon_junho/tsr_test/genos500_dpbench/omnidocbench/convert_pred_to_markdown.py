#!/usr/bin/env python3
"""모델 pred -> OmniDocBench end2end pred(페이지당 markdown 1파일) 변환.

이슈 doc_parser#318. OmniDocBench는 pred 폴더에서 GT `page_info.image_path`
basename과 같은 stem의 `<stem>.md`를 찾아 통짜 markdown을 스스로 분해하므로
(md_tex_filter), 블록 구조 없이 order 순으로 내용을 이어붙이면 된다.

입력 2형식:
- dpbench : dp-bench pred JSON (dots.ocr 계열, {doc_key: {elements: [...]}}).
            element id 순으로 text/수식($$..$$)/표 HTML을 이어붙임.
- chandra : predictions_unified.jsonl (div-HTML 출력).
            <div data-bbox data-label>를 순서대로 훑어 Table은 내부 <table>
            HTML, Picture는 스킵, 그 외(Formula 포함)는 내부 텍스트를 추출.
            ※ 모델이 attribute를 작은따옴표로 출력하므로 파싱은 BeautifulSoup 사용.

블록 사이는 빈 줄(\n\n)로 구분 — quick_match의 문단 분할 단위.

사용:
  python convert_pred_to_markdown.py --format dpbench \
    --input dp_out/.../pred_dp_bench_full.json --output_dir <md 폴더>
  python convert_pred_to_markdown.py --format chandra \
    --input .../predictions_unified.jsonl --output_dir <md 폴더>
"""

import argparse
import json
import os
import re
import sys

# bs4의 HTML 문자열화(decode_contents)는 태그 중첩 깊이만큼 재귀함.
# 모델 폭주 출력(수천 단계 중첩)이 기본 한도(1000)를 넘겨 RecursionError를
# 내므로 한도를 올리고, 그래도 넘는 문서는 태그 제거 폴백으로 처리한다.
sys.setrecursionlimit(30000)

# 하류의 OmniDocBench 평가기는 기본 재귀 한도(1000)로 markdown 내 HTML을
# 다시 파싱하므로, 변환을 통과했더라도 깊은 중첩을 그대로 내보내면 평가기가
# 죽는다. 정상 표 HTML의 중첩은 수십 단계면 충분하므로, 이를 크게 넘는
# 블록은 모델 폭주 출력으로 보고 태그제거 폴백을 적용한다.
MAX_TABLE_DEPTH = 200


def _html_nesting_depth(html_text, cap=100000):
    """raw HTML 문자열의 최대 태그 중첩 깊이(재귀 없이 스캔). cap 초과 시 조기 반환."""
    depth = max_depth = 0
    for m in re.finditer(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)[^>]*?(/?)>", html_text or ""):
        closing, _tag, self_closing = m.group(1), m.group(2), m.group(3)
        if self_closing:
            continue
        if closing:
            depth = max(0, depth - 1)
        else:
            depth += 1
            if depth > max_depth:
                max_depth = depth
                if max_depth > cap:
                    return max_depth
    return max_depth


def _blocks_from_dpbench_doc(doc):
    blocks = []
    n_degraded = 0
    for el in sorted(doc.get("elements", []), key=lambda e: e.get("id", 0)):
        cat = el.get("category")
        content = el.get("content", {}) or {}
        if cat in ("figure", "chart"):
            continue
        if cat == "table":
            html = (content.get("html") or "").strip()
            if html:
                if _html_nesting_depth(html, cap=MAX_TABLE_DEPTH) > MAX_TABLE_DEPTH:
                    n_degraded += 1
                    blocks.extend(_blocks_fallback_strip_tags(html))
                else:
                    blocks.append(html)
            continue
        text = (content.get("text") or "").strip()
        if text:
            blocks.append(text)
    return blocks, n_degraded


def convert_dpbench(input_path, output_dir):
    data = json.load(open(input_path))
    n = 0
    degraded_docs = []
    for doc_key, doc in data.items():
        stem = os.path.splitext(doc_key)[0]
        blocks, n_degraded = _blocks_from_dpbench_doc(doc)
        if n_degraded:
            degraded_docs.append(f"{stem}({n_degraded})")
        with open(os.path.join(output_dir, stem + ".md"), "w") as f:
            f.write("\n\n".join(blocks))
        n += 1
    _warn_degraded(degraded_docs)
    return n


def _blocks_from_div_html(html_text):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text or "", "html.parser")
    divs = soup.find_all("div", attrs={"data-label": True})
    blocks = []
    n_degraded = 0
    if not divs:
        # div 래퍼가 없는 출력(플랫 HTML)은 통짜 텍스트로 반환
        text = soup.get_text("\n").strip()
        return ([text] if text else []), 0
    for div in divs:
        label = str(div.get("data-label", "")).strip().lower()
        if label == "picture":
            continue
        if label == "table":
            table = div.find("table")
            if table is not None:
                table_str = str(table)  # 깊은 중첩이면 RecursionError -> 상위에서 폴백
                if _html_nesting_depth(table_str, cap=MAX_TABLE_DEPTH) > MAX_TABLE_DEPTH:
                    n_degraded += 1
                    text = div.get_text("\n").strip()
                    if text:
                        blocks.append(text)
                else:
                    blocks.append(table_str)
                continue
        text = div.get_text("\n").strip()
        if text:
            blocks.append(text)
    return blocks, n_degraded


_TAG_RE = re.compile(r"<[^>]+>")


def _blocks_fallback_strip_tags(html_text):
    """폭주 출력용 폴백: 태그를 전부 제거한 평문. 표 구조는 유실된다."""
    text = _TAG_RE.sub("\n", html_text or "")
    blocks = [b.strip() for b in re.split(r"\n{2,}", text)]
    return [b for b in blocks if b]


def _warn_degraded(degraded_docs):
    if degraded_docs:
        print(
            f"[warn] 중첩 과다(>{MAX_TABLE_DEPTH})로 태그제거 폴백 적용(표 구조 유실) "
            f"{len(degraded_docs)}개: " + ", ".join(degraded_docs)
        )


def convert_chandra(input_path, output_dir):
    n = 0
    degraded_docs = []
    for line in open(input_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        img = r.get("image_path") or r.get("image") or ""
        stem = os.path.splitext(os.path.basename(str(img)))[0]
        if not stem:
            raise ValueError(f"image_path 없는 행: {line[:120]}")
        txt = str(r.get("pred_layout_html") or r.get("full_response") or "")
        try:
            blocks, n_degraded = _blocks_from_div_html(txt)
        except RecursionError:
            n_degraded = 1
            blocks = _blocks_fallback_strip_tags(txt)
        if n_degraded:
            degraded_docs.append(f"{stem}({n_degraded})")
        with open(os.path.join(output_dir, stem + ".md"), "w") as f:
            f.write("\n\n".join(blocks))
        n += 1
    _warn_degraded(degraded_docs)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=["dpbench", "chandra"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.format == "dpbench":
        n = convert_dpbench(args.input, args.output_dir)
    else:
        n = convert_chandra(args.input, args.output_dir)
    print(f"{n} docs -> {args.output_dir}")


if __name__ == "__main__":
    main()
