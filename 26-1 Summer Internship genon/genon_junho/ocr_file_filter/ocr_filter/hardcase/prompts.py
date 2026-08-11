"""[6] hardcase 프롬프트: judge(판정) / refine(자동교정) / prelabel(사전 라벨링 초안).

judge 는 원본+렌더링(bbox 오버레이) 2장을 보고 JSON 판정만, refine/prelabel 은 원본 1장 +
현재 예측을 보고 새 예측(우리 캐노니컬 스키마: bbox 는 원본 픽셀 좌표)을 통째로 다시 낸다.
"""

from __future__ import annotations

import json

from ocr_filter.cmcv.prompts import CHANDRA_SYSTEM, CHANDRA_USER_NO_OCR

# ── 라벨 생성(단일 콜): 비판적 병합 맥락 + Chandra 레이아웃 지시문을 한 콜에 융합 ──────────
# Hard 케이스(두 외부 모델 dots/paddle 이 불일치)에서, 큰 judge 모델(397B)에게 원본 이미지 +
# A/B 두 예측을 레퍼런스로 주고, 우리 캐노니컬 포맷(Chandra div-HTML, bbox 0-1000)으로 최종
# 레이아웃을 **한 번에** 뽑게 한다. 별도 병합 콜 → 별도 정제 콜(각각 397B) 을 하나로 합친 것.
# 핵심 설계 의도: A/B 는 "영역 회수(recall)용 스캐폴딩" — 모델이 영역을 놓치지 않게만 돕고,
# 박스 위치/카테고리/텍스트는 원본 이미지를 1차 소스로 스스로 판단하게 한다(= 틀린 A/B 박스를
# 그대로 베끼는 앵커링 회피). A/B 박스는 원본 픽셀좌표라, 출력 좌표계(0-1000)와 맞추려고
# 프롬프트에 넣을 땐 이미지 크기로 0-1000 정규화해서 보여준다.
CHANDRA_MERGE_PREAMBLE = """This document page was flagged as a HARD CASE: two independent \
layout models disagree on it, so neither can be trusted blindly. Below are their two predictions \
as REFERENCES ONLY (bboxes normalized to 0-1000, same space you must output in). Use them only to \
avoid missing regions — for every box position, category, and text, trust the ORIGINAL IMAGE over \
the references and fix whatever the references got wrong. Do NOT simply union or copy them.

Reference prediction A:
{a}

Reference prediction B:
{b}

Now follow the instructions below and output the single best layout for the ORIGINAL image.

"""

_REF_TEXT_CAP = 200  # 레퍼런스 텍스트는 내용 힌트용이라 길면 잘라 프롬프트 크기를 묶는다.


def _norm_ref_block(elements: list[dict], img_w: int, img_h: int) -> str:
    """A/B 예측(원본 픽셀 bbox)을 0-1000 정규화해 category+bbox+짧은 텍스트로 나열.
    bbox 가 없는 요소(예: 구조 없는 순수 인식 텍스트)는 스캐폴딩 가치가 없어 제외한다."""
    lines = []
    for e in elements:
        bbox = e.get("bbox")
        if not bbox or len(bbox) != 4 or not img_w or not img_h:
            continue
        x0, y0, x1, y1 = bbox
        nb = [round(x0 * 1000 / img_w), round(y0 * 1000 / img_h),
              round(x1 * 1000 / img_w), round(y1 * 1000 / img_h)]
        text = (e.get("text") or "").replace("\n", " ")
        if len(text) > _REF_TEXT_CAP:
            text = text[:_REF_TEXT_CAP] + "…"
        lines.append(f"- {e.get('category')} bbox={nb} text={text!r}")
    return "\n".join(lines) if lines else "(none)"


def build_merge_chandra_prompt(
    a_elements: list[dict], b_elements: list[dict], img_w: int, img_h: int,
) -> str:
    """단일 콜 라벨 생성용 user 프롬프트 (system 은 CHANDRA_SYSTEM 을 따로 넘긴다)."""
    return CHANDRA_MERGE_PREAMBLE.format(
        a=_norm_ref_block(a_elements, img_w, img_h),
        b=_norm_ref_block(b_elements, img_w, img_h),
    ) + CHANDRA_USER_NO_OCR


# ── 판정: 검증된 단일 종합판정 프롬프트 + labeler 식 이슈 필드만 보강 ─────────────────────
# 1차 시도(labeler 프롬프트 그대로 이식, 4메트릭 분해)는 실측에서 판별력이 죽었다
# (2026-07-16: 박스를 20% 오른쪽으로 밀거나 절반을 지운 합성 결함도 5점/PASS로 통과).
# 원인을 좁혀보니 프롬프트 톤(관대함 강조)이 아니라 **4개 메트릭으로 쪼개는 구조 자체**였다 —
# 검증된 단순 문구를 그대로 유지한 채 4메트릭 스키마만 얹은 2차 시도에서도, bbox 밀림은
# 잡았지만(score=2) 표 전체가 빠진 coverage 결함은 여전히 못 잡았다(같은 모델·같은
# thinking-off 로 예전 단일판정 프롬프트는 정확히 잡음 — "메트릭별로 각각 훑는" 구조가
# "원본과 전체를 대조"해야 하는 coverage 체크를 피상적으로 만드는 것으로 보임). 그래서
# **단일 종합 판정(pass/overall_score/issues) 구조를 그대로 유지**하고, issue 항목에만
# labeler 식 `expected_category`/`severity` 필드를 보강해서 revise 단계에 쓸 진단 정보를 얻는다.
LAYOUT_ERROR_TYPE_ENUM = {
    "missing", "extra", "wrong_category", "bbox_loose", "bbox_tight",
    "bbox_shifted", "duplicate", "fragmented", "merged", "text_error",
}

LAYOUT_JUDGE_PROMPT = """You are reviewing an OCR/layout-detection result. You are given two images:
1. The ORIGINAL document image.
2. The SAME image with the model's predicted layout boxes drawn on top (colored rectangles +
   category labels), representing what the model detected.

Check whether the predicted boxes are visually correct: every visible text/table/figure region
should have a box, box positions should tightly match the actual content, and category labels
should be correct (Title/Text/Table/Picture/etc). Ignore minor pixel-level box looseness — flag
only clear failures: missing regions, badly misplaced/oversized boxes, wrong categories, or
duplicated/hallucinated boxes.

ALSO verify the TRANSCRIBED TEXT: each element below includes the text the model claims to have
read inside that box. Compare it against the actual characters visible in the ORIGINAL image at
that location. Flag issue "text_error" when the transcription is clearly wrong: misread or
invented words/numbers, a different sentence than what is printed, truncated content replaced
with "...", or non-empty visible text transcribed as empty. Do NOT flag trivial differences
(whitespace, line-break positions, minor punctuation). A page whose boxes are perfect but whose
text is misread must still FAIL — this label becomes training ground truth, so text accuracy
matters as much as box accuracy.

The predicted bboxes below are given as [x0, y0, x1, y1] NORMALIZED to a 0-1000 coordinate space
(NOT pixels), relative to the page: x=0 is the left edge, x=1000 the right edge, y=0 the top,
y=1000 the bottom. The two images may be rendered at different pixel sizes — always interpret the
bbox numbers in this normalized space, never against either image's pixel width/height.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{
  "pass": true or false,
  "overall_score": 0-5 (5 = perfect),
  "issues": [
    {"element_index": <0-based index into the list below, or null if a region is missing
      entirely>, "category": "<predicted category, or your best guess of what the missing
      region should be>", "expected_category": "<only when issue is wrong_category — the
      category it should have been, else null>", "issue": "<one of: missing, extra,
      wrong_category, bbox_loose, bbox_tight, bbox_shifted, duplicate, fragmented, merged,
      text_error>", "severity": "MINOR|MAJOR|CRITICAL", "description": "<short reason, Korean,
      under 60 chars — keep category names/issue types/coordinates in English>"}
  ]
}
Parser output elements (0-based index, bbox normalized 0-1000, for element_index above):
{{ELEMENTS_INDEX}}
"""


# render-then-verify(MinerU2.5-Pro §3.3): 파싱된 표/수식을 실제로 렌더한 이미지를 추가로
# 붙일 때, judge 에게 그 이미지가 무엇이고 어떻게 쓰라는지 알려주는 블록. 렌더된 게 없으면
# 이 블록은 통째로 빠지고 기존 2장(원본+오버레이) 프롬프트와 동일해진다.
_RENDER_BLOCK = {
    "tables": """
IMAGE {n}: the parsed TABLES re-rendered from the model's own HTML output (each labeled [i]
matching the element index below). Compare each rendered table against the corresponding table
in the ORIGINAL image, cell by cell. Structural defects show up here as visible collapse:
wrong row/column counts, cells merged or split incorrectly, shifted content, missing rows.
If a rendered table does not match the original's structure or cell contents, report it with
issue "text_error" (content wrong) or "merged"/"fragmented" (structure wrong).""",
    "formulas": """
IMAGE {n}: the parsed FORMULAS re-rendered from the model's own LaTeX output (each labeled [i]
matching the element index below). Compare each rendered formula against the corresponding
formula in the ORIGINAL image. Missing alignment marks, wrong sub/superscripts, dropped
operators, and unbalanced delimiters are clearly visible here. Report mismatches as
"text_error".""",
}

_RENDER_PREAMBLE = """
You are ALSO given re-rendered images of the model's own structured output. This matters: a
model can generate a structured sequence that looks plausible as text but renders into a broken
table or formula. Judge the STRUCTURE AND CONTENT by looking at these renders next to the
original — do not assume the output is correct just because the boxes are in the right places.
{blocks}
"""


def build_layout_judge_prompt(
    elements: list[dict], rendered_kinds: list[str] | None = None,
) -> str:
    """elements 는 0-1000 정규화 bbox 여야 한다(호출측에서 이미지 크기로 정규화해서 넘김).

    rendered_kinds: judge 에게 추가로 붙인 렌더 이미지 종류 순서대로 ("tables", "formulas").
    호출측(judge_layout_once)이 images 리스트에 넣은 순서와 정확히 같아야 IMAGE 번호가 맞는다."""
    prompt = LAYOUT_JUDGE_PROMPT.replace("{{ELEMENTS_INDEX}}", _elements_index(elements))
    if not rendered_kinds:
        return prompt
    blocks = []
    for i, kind in enumerate(rendered_kinds):
        tmpl = _RENDER_BLOCK.get(kind)
        if tmpl:
            blocks.append(tmpl.format(n=i + 3))  # 1=원본, 2=오버레이 다음부터
    if not blocks:
        return prompt
    return prompt + _RENDER_PREAMBLE.format(blocks="\n".join(blocks))


# ── 단일 revise (루프 아님, FAIL 시 딱 한 번만) ───────────────────────────────────────────
# judge_layout_once 가 낸 element_issues(정확한 element_index + expected_category + 설명)를
# 그대로 피드백으로 줘서 flagged 된 요소만 고치게 한다. 예전 REFINE_PROMPT/judge_and_refine
# 은 "Box 10a" 식 자유 텍스트 지적이라 refine 이 뭘 고칠지 못 찾고 원본을 거의 그대로
# 에코했었는데(2026-07 관찰), 이번엔 인덱스가 정확해서 타겟팅된 수정이 가능하다.
REVISE_CHANDRA_PREAMBLE = """You previously produced the layout below for this document image, \
and a reviewer found specific problems with it (bboxes normalized to 0-1000, same space you must \
output in). Fix ONLY the flagged elements — keep everything else exactly as it was.

Previous layout (0-based index):
{prev}

Reviewer-flagged issues (fix these, and only these):
{issues}

Now follow the instructions below and output the CORRECTED full layout for the ORIGINAL image.

"""


def _flat_issues_text(issues: list[dict]) -> str:
    if not issues:
        return "(none listed)"
    lines = []
    for i in issues:
        idx = i.get("element_index")
        idx_label = f"[{idx}]" if idx is not None else "[missing]"
        category = i.get("category")
        expected = i.get("expected_category")
        if i.get("issue") == "wrong_category" and expected:
            category = f"{category}→{expected}"
        lines.append(
            f"- {idx_label} {category} {i.get('issue')} ({i.get('severity')}): "
            f"{i.get('description')}"
        )
    return "\n".join(lines)


def build_revise_chandra_prompt(prev_elements_1000: list[dict], element_issues: list[dict]) -> str:
    """prev_elements_1000: 0-1000 정규화된 이전 라벨. element_issues: judge_layout_once 결과의
    (여러 메트릭에서 모은) flat 리스트."""
    return REVISE_CHANDRA_PREAMBLE.format(
        prev=_elements_index(prev_elements_1000),
        issues=_flat_issues_text(element_issues),
    ) + CHANDRA_USER_NO_OCR


JUDGE_PROMPT = """You are reviewing an OCR/layout-detection result. You are given two images:
1. The ORIGINAL document image.
2. The SAME image with the model's predicted layout boxes drawn on top (colored rectangles +
   category labels), representing what the model detected.

Check whether the predicted boxes are visually correct: every visible text/table/figure region
should have a box, box positions should tightly match the actual content, and category labels
should be correct (Title/Text/Table/Picture/etc). Ignore minor pixel-level box looseness — flag
only clear failures: missing regions, badly misplaced/oversized boxes, wrong categories, or
duplicated/hallucinated boxes.

The predicted bboxes below are given as [x0, y0, x1, y1] NORMALIZED to a 0-1000 coordinate space
(NOT pixels), relative to the page: x=0 is the left edge, x=1000 the right edge, y=0 the top,
y=1000 the bottom. The two images may be rendered at different pixel sizes — always interpret the
bbox numbers in this normalized space, never against either image's pixel width/height.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{{
  "pass": true or false,
  "overall_score": 0-5 (5 = perfect),
  "issues": [
    {{"element_index": <int index into the predicted element list, or null if a region is
      missing entirely>, "category": "<predicted category or 'missing'>",
      "issue": "<one of: missing_region, wrong_category, bbox_misaligned, bbox_too_large,
      bbox_too_small, duplicate, hallucinated>", "description": "<short human-readable reason>"}}
  ]
}}
"predicted element list" (0-based index, in the order given, for element_index above):
{elements_index}
"""

REFINE_PROMPT = """You previously produced a layout prediction for this document image that a
reviewer found issues with. Look at the ORIGINAL image again and produce a corrected prediction.

Previous prediction (0-based index):
{elements_index}

Issues found by the reviewer:
{issues_text}

Output ONLY a JSON array (no markdown fences, no commentary) of layout elements, each:
{{"bbox": [x0, y0, x1, y1], "category": "<Title|Text|Table|Picture|Caption|Footnote|Formula|
List-item|Page-footer|Page-header|Section-header>", "text": "<content, HTML for Table, plain
text otherwise>"}}
bbox coordinates are in the ORIGINAL image's pixel space (not normalized). Fix every issue
listed above; keep the parts that were already correct."""

PRELABEL_PROMPT = """This document image was flagged as a "hard case" during automatic quality
control — an OCR/layout model's prediction had visual errors that automatic retries could not
fix. Your job is to produce a clean, accurate layout prediction from scratch by looking directly
at the ORIGINAL image, to give a human annotator a good starting draft to correct.

The best automatic prediction so far (for reference only, it has known errors — do not just
copy it):
{elements_index}

Known issues with it:
{issues_text}

Output ONLY a JSON array (no markdown fences, no commentary) of layout elements, each:
{{"bbox": [x0, y0, x1, y1], "category": "<Title|Text|Table|Picture|Caption|Footnote|Formula|
List-item|Page-footer|Page-header|Section-header>", "text": "<content, HTML for Table, plain
text otherwise>"}}
bbox coordinates are in the ORIGINAL image's pixel space (not normalized)."""


_IDX_TEXT_CAP = 300  # 판정 대조용 — 요소당 이 길이면 오독/환각/누락 판단에 충분하고 프롬프트 크기를 묶는다.


def _elements_index(elements: list[dict]) -> str:
    """judge/revise 프롬프트에 넣는 요소 인덱스. **text 를 반드시 포함한다** — 예전엔
    category+bbox 만 넣어서 judge 가 대조할 텍스트 자체가 없었고, 그 결과 text_error 이슈가
    실질적으로 발동 불가였다(Hard 티어 24,548건의 OCR 텍스트가 한 번도 검증되지 않고 GT 로
    확정된 원인, 2026-07-30 진단). 이미지 속 실제 글자와 여기 적힌 전사(transcription)를
    대조해야 텍스트 정확도 판정이 가능하다."""
    lines = []
    for i, e in enumerate(elements):
        text = (e.get("text") or "").replace("\n", " ")
        if len(text) > _IDX_TEXT_CAP:
            text = text[:_IDX_TEXT_CAP] + "…"
        lines.append(f"[{i}] category={e.get('category')}, bbox={e.get('bbox')}, text={text!r}")
    return "\n".join(lines) if lines else "(no elements predicted)"


def build_judge_prompt(elements: list[dict]) -> str:
    return JUDGE_PROMPT.format(elements_index=_elements_index(elements))


def _issues_text(issues: list[dict]) -> str:
    if not issues:
        return "(none listed)"
    lines = []
    for issue in issues:
        idx = issue.get("element_index")
        lines.append(
            f"- [{idx if idx is not None else 'missing'}] {issue.get('category')}: "
            f"{issue.get('issue')} — {issue.get('description')}"
        )
    return "\n".join(lines)


def build_refine_prompt(elements: list[dict], issues: list[dict]) -> str:
    return REFINE_PROMPT.format(
        elements_index=_elements_index(elements), issues_text=_issues_text(issues),
    )


def build_prelabel_prompt(elements: list[dict], issues: list[dict]) -> str:
    return PRELABEL_PROMPT.format(
        elements_index=_elements_index(elements), issues_text=_issues_text(issues),
    )
