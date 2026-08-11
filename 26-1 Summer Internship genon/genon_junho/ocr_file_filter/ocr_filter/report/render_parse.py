"""파싱 **결과 자체**를 이미지로 렌더 (render-then-verify, MinerU2.5-Pro §3.3).

judge 에게 "원본 이미지 + 파싱결과를 렌더한 이미지"를 짝으로 보여주기 위한 모듈이다.
`report/render.py` 의 `draw_boxes` 와 목적이 완전히 다르다 —

    draw_boxes()      원본 이미지 위에 bbox 사각형을 그림  → **박스 위치**만 검증 가능
    render_elements() 파싱된 HTML/LaTeX 를 새로 렌더      → **내용/구조**까지 검증 가능

논문이 지적한 요지: 모델은 "이미지 → 구조화 시퀀스"는 잘하지만 그 역방향(시퀀스가 어떻게
보일지)을 추론하지 못해서, 텍스트만 주고 자기 출력을 검토시키면 오류를 놓치고 자기 출력을
승인해버린다. 렌더해서 눈으로 비교하게 만들면 닫히지 않은 태그·빠진 정렬기호 같은
미묘한 구조 결함이 **레이아웃 붕괴라는 눈에 띄는 시각적 이상**으로 증폭된다.

렌더 백엔드:
    표(HTML)  → playwright chromium 스크린샷 (설치돼 있음, 2026-07-30 확인)
    수식(LaTeX) → pdflatex + magick (CDM 이 쓰는 것과 동일한 시스템 도구)
둘 다 실패하면 None 을 돌려주고 호출측이 그 이미지를 빼고 진행한다 — 렌더 실패로 judge
자체를 못 돌게 만들지는 않는다.
"""

from __future__ import annotations

import html as _html_lib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

_MAX_SIDE = 1400          # judge 프롬프트에 넣을 렌더 이미지 최대 변
_TABLE_VIEWPORT = (1200, 1600)


# ── 표: HTML → PNG (playwright) ────────────────────────────────────────────────
_TABLE_PAGE_CSS = """
body { margin: 16px; font-family: sans-serif; background: #fff; }
table { border-collapse: collapse; }
td, th { border: 1px solid #333; padding: 4px 8px; font-size: 15px; }
"""


def render_tables_html(table_htmls: list[str]) -> Image.Image | None:
    """표 HTML 들을 한 장의 PNG 로 렌더. 실패하면 None.

    표가 여러 개면 순서대로 세로로 쌓고 인덱스를 붙인다 — judge 가 element_index 로
    지적할 수 있어야 하므로 어느 표가 몇 번인지 보여야 한다."""
    if not table_htmls:
        return None
    blocks = []
    for i, h in enumerate(table_htmls):
        blocks.append(f'<div style="margin-bottom:18px">'
                      f'<div style="font:bold 13px sans-serif;color:#b00">[{i}]</div>{h}</div>')
    doc = f"<style>{_TABLE_PAGE_CSS}</style>" + "".join(blocks)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("render_parse: playwright 미설치 — 표 렌더 스킵")
        return None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tables.png"
            with sync_playwright() as p:
                browser = p.chromium.launch(args=["--no-sandbox"])
                try:
                    page = browser.new_page(
                        viewport={"width": _TABLE_VIEWPORT[0], "height": _TABLE_VIEWPORT[1]})
                    page.set_content(doc, wait_until="load")
                    page.screenshot(path=str(out), full_page=True)
                finally:
                    browser.close()
            with Image.open(out) as im:
                return _fit(im.convert("RGB"))
    except Exception as e:  # noqa: BLE001 — 렌더 실패는 치명적이지 않다(해당 이미지만 생략)
        logger.warning("render_parse: 표 렌더 실패 (%s: %s)", type(e).__name__, e)
        return None


# ── 수식: LaTeX → PNG (pdflatex + magick) ─────────────────────────────────────
_FORMULA_TEX = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,xcolor,geometry}
\geometry{paperwidth=180mm,paperheight=%(h)dmm,margin=8mm}
\pagestyle{empty}
\begin{document}
%(body)s
\end{document}
"""


# ── 수식: LaTeX → PNG (KaTeX + chromium) ──────────────────────────────────────
_KATEX_DIR = Path(__file__).parent / "_vendor_katex"

_KATEX_PAGE_CSS = """
body { margin: 18px; background: #fff; font-family: sans-serif; }
.row { display: flex; align-items: baseline; gap: 10px; margin-bottom: 16px; }
.idx { font: bold 13px sans-serif; color: #b00; flex: 0 0 auto; }
.eq  { flex: 1 1 auto; overflow-x: hidden; }
.err { font: 13px monospace; color: #b00; }
"""


def render_formulas_katex(latexes: list[str]) -> tuple[Image.Image | None, list[bool]]:
    """LaTeX 수식들을 **KaTeX + chromium** 으로 한 장의 PNG 에 렌더.

    `render_formulas_latex`(pdflatex) 와 목적이 다르다. pdflatex 는 커버리지가 넓어서
    "TeX 으로 컴파일만 되면 통과"인데, 우리 학습 프롬프트가 모델에게 요구하는 건
    **KaTeX 호환** LaTeX 이고 평가(OmniDocBench)도 그 전제로 돈다. 그래서 여기서는
    `throwOnError: true` 로 렌더해서 **KaTeX 가 못 그리는 수식은 실패로 표시**한다 —
    이 실패 자체가 "GT 로 쓰면 안 되는 수식" 신호라서 버그가 아니라 기능이다.

    반환: (합성 이미지 | None, 수식별 성공 여부 리스트).
    이미지가 None 이어도 플래그 리스트는 항상 len(latexes) 길이다."""
    n = len(latexes)
    if not n:
        return None, []
    css, js = _KATEX_DIR / "katex.min.css", _KATEX_DIR / "katex.min.js"
    if not (css.is_file() and js.is_file()):
        logger.warning("render_parse: KaTeX 벤더 파일 없음 (%s) — 수식 렌더 스킵", _KATEX_DIR)
        return None, [False] * n

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("render_parse: playwright 미설치 — KaTeX 수식 렌더 스킵")
        return None, [False] * n

    # 임시 HTML 을 **_vendor_katex/ 안에** 쓰고 goto 로 연다. set_content 는 about:blank
    # origin 이라 chromium 이 file:// 서브리소스를 통째로 막고("Not allowed to load local
    # resource") katex 가 undefined 로 뜬다. 같은 디렉터리에서 열면 css/js/fonts 가 전부
    # 상대경로로 풀린다. doctype 도 필수 — 없으면 quirks mode 라 KaTeX 가 경고를 낸다.
    doc = (
        "<!doctype html><html><head>"
        '<link rel="stylesheet" href="katex.min.css">'
        '<script src="katex.min.js"></script>'
        f"<style>{_KATEX_PAGE_CSS}</style>"
        '</head><body><div id="root"></div></body></html>'
    )
    script = """
    (texs) => {
        const root = document.getElementById('root');
        const ok = [];
        texs.forEach((t, i) => {
            const row = document.createElement('div');
            row.className = 'row';
            const idx = document.createElement('div');
            idx.className = 'idx';
            idx.textContent = '[' + i + ']';
            const eq = document.createElement('div');
            eq.className = 'eq';
            try {
                eq.innerHTML = katex.renderToString(t, {
                    displayMode: true, throwOnError: true, strict: false,
                });
                ok.push(true);
            } catch (e) {
                eq.className = 'eq err';
                eq.textContent = 'KaTeX ERROR: ' + e.message;
                ok.push(false);
            }
            row.appendChild(idx); row.appendChild(eq); root.appendChild(row);
        });
        return ok;
    }
    """
    page_file = _KATEX_DIR / f"_render_{os.getpid()}_{threading.get_ident()}.html"
    try:
        page_file.write_text(doc, encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "formulas.png"
            with sync_playwright() as p:
                browser = p.chromium.launch(args=["--no-sandbox"])
                try:
                    page = browser.new_page(viewport={"width": 1100, "height": 900})
                    page.goto(page_file.as_uri(), wait_until="load")
                    flags = page.evaluate(script, [_strip_math(t) for t in latexes])
                    page.wait_for_timeout(120)  # 웹폰트 적용 대기
                    # full_page 대신 #root 만 — viewport 높이만큼 흰 여백이 붙으면
                    # judge 프롬프트에 빈 픽셀만 실려 나간다.
                    page.locator("#root").screenshot(path=str(out))
                finally:
                    browser.close()
            flags = [bool(x) for x in (flags or [])]
            flags += [False] * (n - len(flags))
            with Image.open(out) as im:
                return _fit(im.convert("RGB")), flags[:n]
    except Exception as e:  # noqa: BLE001 — 렌더 실패로 상위 파이프라인을 죽이지 않는다
        logger.warning("render_parse: KaTeX 수식 렌더 실패 (%s: %s)", type(e).__name__, e)
        return None, [False] * n
    finally:
        page_file.unlink(missing_ok=True)


def render_formulas_latex(latexes: list[str]) -> Image.Image | None:
    """LaTeX 수식들을 한 장의 PNG 로 렌더. pdflatex/magick 이 없거나 컴파일 실패면 None.

    수식 하나가 깨져서 전체 컴파일이 실패하는 걸 막으려고 `\fbox` 없이 각 수식을
    독립 문단으로 넣고, nonstopmode 로 돌려 부분 실패를 감수한다."""
    if not latexes:
        return None
    pdflatex, magick = shutil.which("pdflatex"), (shutil.which("magick") or shutil.which("convert"))
    if not (pdflatex and magick):
        logger.warning("render_parse: pdflatex/magick 없음 — 수식 렌더 스킵")
        return None

    body = "\n\n".join(
        r"\noindent\textcolor{red}{\small [%d]}\quad $\displaystyle %s$" % (i, _strip_math(t))
        for i, t in enumerate(latexes)
    )
    tex = _FORMULA_TEX % {"body": body, "h": max(40, 22 * len(latexes) + 20)}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "f.tex").write_text(tex, encoding="utf-8")
            proc = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "f.tex"],
                cwd=d, capture_output=True, timeout=60,
            )
            pdf = d / "f.pdf"
            if not pdf.is_file():
                logger.warning("render_parse: pdflatex 실패 (%s)",
                               proc.stdout.decode("utf-8", "replace")[-300:])
                return None
            png = d / "f.png"
            subprocess.run([magick, "-density", "180", str(pdf), "-background", "white",
                            "-alpha", "remove", str(png)], capture_output=True, timeout=60)
            cand = png if png.is_file() else (d / "f-0.png")
            if not cand.is_file():
                return None
            with Image.open(cand) as im:
                return _fit(im.convert("RGB"))
    except Exception as e:  # noqa: BLE001
        logger.warning("render_parse: 수식 렌더 실패 (%s: %s)", type(e).__name__, e)
        return None


def _strip_math(t: str) -> str:
    """모델이 $...$ 나 \\(...\\) 로 감싸 낸 경우 겉껍질만 제거 (안쪽은 그대로)."""
    s = (t or "").strip()
    for a, b in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if s.startswith(a) and s.endswith(b) and len(s) > len(a) + len(b):
            return s[len(a):-len(b)].strip()
    return s


def _fit(im: Image.Image) -> Image.Image:
    w, h = im.size
    if max(w, h) <= _MAX_SIDE:
        return im
    scale = _MAX_SIDE / max(w, h)
    return im.resize((max(1, int(w * scale)), max(1, int(h * scale))))


# ── 요소 리스트 → 렌더 이미지들 ────────────────────────────────────────────────
def render_parsed_elements(elements: list[dict]) -> dict[str, Image.Image]:
    """파싱 결과에서 렌더 가능한 것(표/수식)만 골라 렌더.

    반환: {"tables": Image, "formulas": Image} — 해당 요소가 없거나 렌더 실패면 키가 없다.
    본문 텍스트는 렌더해봐야 원본과 대조 가치가 낮아(글자를 다시 글자로 그릴 뿐) 제외한다.
    구조가 있는 표/수식이 바로 논문이 말한 '렌더링의 오류 증폭 효과'가 작동하는 대상이다."""
    out: dict[str, Image.Image] = {}
    tables = [e.get("text") or "" for e in elements if e.get("category") == "Table"]
    tables = [t for t in tables if "<" in t]  # 태그가 없으면 표 구조가 아님(렌더 의미 없음)
    if tables:
        img = render_tables_html(tables)
        if img is not None:
            out["tables"] = img

    formulas = [e.get("text") or "" for e in elements if e.get("category") == "Formula"]
    formulas = [f for f in formulas if f.strip()]
    if formulas:
        img = render_formulas_latex(formulas)
        if img is not None:
            out["formulas"] = img
    return out


def escape_text(t: str) -> str:
    return _html_lib.escape(t or "", quote=False)
