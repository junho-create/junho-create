"""judge/refine/prelabel VLM 응답 파싱 — 코드펜스/잡담 섞여 나와도 관대하게."""

from __future__ import annotations

import json
import re

from ocr_filter.cmcv.normalize import (
    _parse_bbox_str,
    _strip_tags,
    _strip_thinking,
    normalize_category,
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# ── 라벨 생성용 Chandra div-HTML 파서 (hardcase 전용, 관대판) ─────────────────────────────
# cmcv 의 parse_chandra_output 은 target(9B)을 위한 엄격판 — 그 모델은 Chandra 포맷으로
# 파인튜닝돼서 `data-bbox="..." data-label="..."` 를 정확히 그 순서로 낸다. 반면 라벨 생성에
# 쓰는 heavy 모델(397B)은 이 포맷으로 학습된 적이 없어서, 프롬프트로 지시해도 실측상
# `<div class="Section-header" data-bbox="...">` 처럼 **data-label 대신 class 를 쓰고 속성
# 순서도 뒤집어** 낸다 (2026-07-16 실측: 엄격판으로는 전부 0개 파싱 → 빈 라벨). 내용 자체는
# 멀쩡하므로(표 구조/좌표 정상) 여기서만 관대하게 받아준다 — cmcv 채점의 target 파서는
# 그대로 둬야 한다(돌고 있는 라운드가 그걸 쓴다).
_DIV_ANY_RE = re.compile(r"<div\s+([^>]*?)>(.*?)</div>", re.DOTALL)
_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')
_HTML_FENCE_RE = re.compile(r"^```(?:html|json)?\s*|\s*```$", re.MULTILINE)


def parse_chandra_label(raw: str) -> list[dict]:
    """Chandra div-HTML → [{"category","text","bbox"}] (bbox 는 0-1000 정규화 그대로).
    속성 순서 무관, 라벨은 data-label 우선·없으면 class 로 폴백. category 는
    normalize_category() 로 표준 11종으로 정규화한다 — heavy judge 모델(397B)은 9B target 과
    달리 이 스키마로 SFT된 적이 없어서 프롬프트로 11종을 지시해도 실측상 13%가 케이스 변형
    (list-item, text, ...)이나 자유형식 레이블(reference_content, aside_text, Button, ...)을
    낸다(2026-07-20 확인). 정규화를 먼저 해야 "table" 소문자처럼 표준과 케이스만 다른 라벨도
    Table 취급돼서 원본 <table> HTML 이 태그 벗겨진 텍스트로 깨지지 않는다."""
    text = _strip_thinking(raw or "")
    text = _HTML_FENCE_RE.sub("", text).strip()
    elements = []
    for attr_str, inner in _DIV_ANY_RE.findall(text):
        attrs = dict(_ATTR_RE.findall(attr_str))
        bbox_str = attrs.get("data-bbox")
        label = attrs.get("data-label") or attrs.get("class")
        if not bbox_str or not label:
            continue
        inner = inner.strip()
        category = normalize_category(label)
        text_val = inner if category == "Table" else _strip_tags(inner)
        elements.append({"category": category, "text": text_val,
                         "bbox": _parse_bbox_str(bbox_str)})
    return elements


def parse_judge_verdict(raw: str) -> dict:
    """실패(파싱 불가)하면 보수적으로 pass=False, issue 없음(=원인 불명) 처리."""
    text = _CODE_FENCE_RE.sub("", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"pass": False, "overall_score": 0, "issues": [],
                "parse_error": True, "raw": raw}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"pass": False, "overall_score": 0, "issues": [],
                "parse_error": True, "raw": raw}
    return {
        "pass": bool(data.get("pass", False)),
        "overall_score": data.get("overall_score", 0),
        "issues": data.get("issues", []) or [],
        "parse_error": False,
    }


# ── labeler 스키마(4메트릭+element_issues) 판정 파서 ─────────────────────────────────────
# 예전 parse_judge_verdict(단일 pass/score/issues, 판별력 검증됨)에 labeler 식
# expected_category/severity 필드만 보강한 버전. **4메트릭으로 쪼개는 구조는 실측에서
# coverage(누락 탐지) 판별력을 죽여서 폐기했다** — 자세한 경위는 prompts.py 의
# LAYOUT_JUDGE_PROMPT 주석 참고.
LAYOUT_ERROR_TYPE_ENUM = {
    "missing", "extra", "wrong_category", "bbox_loose", "bbox_tight",
    "bbox_shifted", "duplicate", "fragmented", "merged", "text_error",
}
_SEVERITY_ENUM = {"MINOR", "MAJOR", "CRITICAL"}


def _validate_layout_issue(entry: object) -> dict | None:
    """malformed 항목은 조용히 None(=드롭). bbox 는 요구하지 않는다(이전 라벨의 bbox 는
    호출측이 이미 알고 있어 revise 프롬프트에서 직접 채움 — 판정 응답에 다시 안 실어도 됨)."""
    if not isinstance(entry, dict):
        return None
    issue = entry.get("issue")
    if issue not in LAYOUT_ERROR_TYPE_ENUM:
        return None
    ei = entry.get("element_index")
    if ei is None:
        if issue != "missing":
            return None
    else:
        try:
            ei = int(ei)
        except (TypeError, ValueError):
            return None
        if ei < 0:
            return None
    category = entry.get("category")
    if not isinstance(category, str) or not category.strip():
        return None
    expected = entry.get("expected_category")
    expected = expected.strip() if isinstance(expected, str) and expected.strip() else None
    severity = entry.get("severity", "MINOR")
    if severity not in _SEVERITY_ENUM:
        severity = "MINOR"
    description = entry.get("description")
    description = description.strip() if isinstance(description, str) else ""
    if not description:
        return None
    return {
        "element_index": ei, "category": category.strip(), "expected_category": expected,
        "issue": issue, "severity": severity, "description": description,
    }


def parse_layout_verdict(raw: str) -> dict:
    """단일 종합 판정(pass/overall_score) + 보강된 issues(element_index/expected_category/
    severity/description). 파싱 실패시 보수적으로 resolved=False."""
    text = _CODE_FENCE_RE.sub("", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"resolved": False, "overall_score": 0.0,
                "element_issues": [], "parse_error": True, "raw": raw}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"resolved": False, "overall_score": 0.0,
                "element_issues": [], "parse_error": True, "raw": raw}

    cleaned = [v for entry in (data.get("issues") or [])
               if (v := _validate_layout_issue(entry)) is not None]
    try:
        overall_score = float(data.get("overall_score", 0))
    except (TypeError, ValueError):
        overall_score = 0.0

    return {
        "resolved": bool(data.get("pass", False)),
        "overall_score": overall_score,
        "element_issues": cleaned,
        "parse_error": False,
    }


def parse_elements_response(raw: str) -> list[dict]:
    """refine/prelabel 이 낸 JSON 배열 → [{"category","text","bbox"}]. 파싱 실패면 빈 리스트."""
    text = _CODE_FENCE_RE.sub("", raw or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [
        {"category": item.get("category", ""), "text": item.get("text", ""),
         "bbox": item.get("bbox")}
        for item in items if isinstance(item, dict)
    ]
