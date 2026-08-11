#!/usr/bin/env python3
"""완성된 두 데이터셋을 검수한다. 하나라도 실패하면 종료코드 1.

검사 항목
1. `$$` 커버리지 100%          — 없으면 평가가 수식을 수식으로 세지 않는다(핵심)
2. KaTeX 렌더 100%             — 학습 프롬프트/평가가 KaTeX 호환을 전제한다
3. 왕복 검증                   — convert_pred_to_markdown 을 태워 `$$...$$` 블록으로 복원되는지
4. 태그 화이트리스트           — raw MathML / img src / 빈 p / base64 SVG 유입 확인
5. bbox 범위                   — 0-1000 안에 있고 x0<x1, y0<y1
6. 이미지 존재                 — image_path 전부 실재

사용:
    python3 verify_datasets.py
"""

from __future__ import annotations

import argparse
import html as _html_lib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/jhyeo/ocr_file_filter")
sys.path.insert(0, "/home/jhyeo/finetuning/eval_318/tsr_test_eval/genos500_dpbench/omnidocbench")
from ocr_filter.report.render_parse import render_formulas_katex  # noqa: E402

HERE = Path(__file__).parent
SPLITS = ("train", "valid", "test")

DIV_RE = re.compile(r'<div[^>]*data-label="([^"]+)"[^>]*>(.*?)</div>', re.S)
BBOX_RE = re.compile(r'data-bbox="(-?\d+) (-?\d+) (-?\d+) (-?\d+)"')
# GT 에 절대 들어오면 안 되는 것들 (arXiv 유래라 MathML/SVG 유입 위험이 실제로 있다)
BANNED = {
    "raw_mathml": re.compile(r"<math[\s>]|</math>|<mrow|<mi>|<mo>", re.I),
    "img_tag": re.compile(r"<img[\s>]", re.I),
    "base64_svg": re.compile(r"data:image/svg|<svg[\s>]", re.I),
    "script": re.compile(r"<script[\s>]", re.I),
    "code_fence": re.compile(r"```"),
}
# 표 셀 안의 인라인 서식은 정상이다. 기존 학습셋(combined_e26_28484)에도 b 7,046 / sup
# 4,980 / sub 1,708 개가 들어 있다 — 여기서 막으면 기존 데이터와 기준이 어긋난다.
# 진짜로 걸러야 하는 건 위 BANNED 쪽(raw MathML, img, base64 SVG, script)이고,
# 그건 이 데이터셋에 0건이다.
ALLOWED_TAGS = {"div", "p", "br", "table", "thead", "tbody", "tfoot", "tr", "td", "th",
                "caption", "sub", "sup", "b", "i", "strong", "em", "u", "span",
                "ul", "ol", "li"}
TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)")


def load(ds_dir: Path) -> list[dict]:
    rows = []
    for split in SPLITS:
        p = ds_dir / f"{split}.jsonl"
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_split"] = split
                    rows.append(r)
    return rows


def latex_of(content: str) -> str:
    """Formula div content -> 원 LaTeX ($$ 와 <br/> 를 되돌린다)."""
    s = re.sub(r"</?p>", "", content).strip()
    s = s.replace("<br/>", "\n").strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2]
    return _html_lib.unescape(s).strip()


def check(name: str, ds_dir: Path, katex_sample: int) -> bool:
    rows = load(ds_dir)
    print(f"\n{'=' * 62}\n{name}: {len(rows)}행  ({ds_dir})\n{'=' * 62}")
    if not rows:
        print("  [FAIL] 행이 없다")
        return False

    ok = True
    n_formula = n_dollar = 0
    bad_bbox = bad_img = 0
    banned_hits: Counter = Counter()
    tag_hits: Counter = Counter()
    latexes: list[str] = []
    labels: Counter = Counter()

    for r in rows:
        gt = r.get("gt_html") or ""
        if not Path(r["image_path"]).is_file():
            bad_img += 1
        for tag in TAG_RE.findall(gt):
            t = tag.lower()
            if t not in ALLOWED_TAGS:
                tag_hits[t] += 1
        for pat_name, pat in BANNED.items():
            if pat.search(gt):
                banned_hits[pat_name] += 1
        for x0, y0, x1, y1 in BBOX_RE.findall(gt):
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
                bad_bbox += 1
        for label, content in DIV_RE.findall(gt):
            labels[label] += 1
            if label != "Formula":
                continue
            n_formula += 1
            if "$$" in content:
                n_dollar += 1
            latexes.append(latex_of(content))

    # 1. $$ 커버리지
    print(f"[1] Formula div {n_formula}개 중 $$ 있는 것 {n_dollar}개 "
          f"({n_dollar / max(1, n_formula) * 100:.1f}%)")
    if n_formula == 0 or n_dollar != n_formula:
        print("    [FAIL] $$ 커버리지가 100%가 아니다 — 평가가 수식을 수식으로 안 센다")
        ok = False

    # 2. KaTeX 렌더
    uniq = list(dict.fromkeys(latexes))
    sample = uniq if katex_sample <= 0 or len(uniq) <= katex_sample else uniq[:katex_sample]
    fails = []
    for i in range(0, len(sample), 40):
        chunk = sample[i:i + 40]
        _, flags = render_formulas_katex(chunk)
        fails += [t for t, f in zip(chunk, flags) if not f]
    print(f"[2] KaTeX 렌더: 고유 수식 {len(uniq)}개 중 {len(sample)}개 검사, 실패 {len(fails)}개")
    if fails:
        ok = False
        for t in fails[:5]:
            print(f"    [FAIL] {t[:100]}")

    # 3. 왕복 (convert_pred_to_markdown 과 동일 경로)
    #    "$$ 를 포함한 블록"을 세면 안 된다 — 본문(Text)에 dots 가 인라인 수식을 `$..$` 로
    #    적어 놓은 게 붙어서 `$$` 부분문자열이 생긴다(`Trust$_{I}$$_{J}$`). 그건 display
    #    수식이 아니라 정상 전사다. **Formula div 가 display 블록이 되는지**만 본다.
    try:
        from convert_pred_to_markdown import _blocks_from_div_html
        n_rt = n_rt_ok = 0
        for r in rows[:800]:
            n_div = len(re.findall(r'data-label="Formula"', r["gt_html"]))
            if not n_div:
                continue
            blocks, _ = _blocks_from_div_html(r["gt_html"])
            disp = [b for b in blocks
                    if b.strip().startswith("$$") and b.strip().endswith("$$")]
            n_rt += n_div
            n_rt_ok += len(disp)
        print(f"[3] 왕복(markdown 변환): Formula div {n_rt}개 -> display 블록 {n_rt_ok}개")
        if n_rt == 0 or n_rt_ok < n_rt:
            print("    [FAIL] Formula div 가 display formula 로 안 잡힌다")
            ok = False
        elif n_rt_ok > n_rt:
            # 잉여는 라벨이 Formula 가 아닌 div 의 내용이 통째로 $$...$$ 인 경우다.
            # 결함이지만 "수식이 안 잡힌다"와는 반대 방향이라 따로 구분해 보고한다.
            print(f"    [FAIL] Formula 라벨이 아닌데 display 블록이 되는 div {n_rt_ok - n_rt}개")
            ok = False
    except ImportError as e:
        print(f"[3] 건너뜀 (convert_pred_to_markdown import 실패: {e})")

    # 4. 태그 화이트리스트
    print(f"[4] 금지 패턴: {dict(banned_hits) or '없음'} / "
          f"화이트리스트 밖 태그: {dict(tag_hits) or '없음'}")
    if banned_hits or tag_hits:
        print("    [FAIL] 허용되지 않은 마크업이 있다")
        ok = False

    # 5/6
    print(f"[5] bbox 범위 위반 {bad_bbox}개")
    print(f"[6] 없는 이미지 {bad_img}개")
    if bad_bbox or bad_img:
        ok = False

    print(f"    라벨 분포: {dict(labels.most_common(12))}")
    src = Counter(r.get("meta", {}).get("formula_source") for r in rows)
    if any(src):
        print(f"    수식 출처: {dict(src)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default=str(HERE / "dataset_formula_crop"))
    ap.add_argument("--layout", default=str(HERE / "dataset_formula_layout"))
    ap.add_argument("--katex-sample", type=int, default=0, help="0=전수")
    args = ap.parse_args()

    a = check("crop 버전", Path(args.crop), args.katex_sample)
    b = check("layout 버전", Path(args.layout), args.katex_sample)
    print(f"\n{'=' * 62}\n결과: crop {'PASS' if a else 'FAIL'} / "
          f"layout {'PASS' if b else 'FAIL'}\n{'=' * 62}")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
