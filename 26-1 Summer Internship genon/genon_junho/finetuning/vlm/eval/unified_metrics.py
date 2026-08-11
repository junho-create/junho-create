"""통합(Table + Layout) 출력 평가 지표.

통합 모델의 출력은 layout element 의 JSON 배열이다::

    [
      {"bbox": [x0, y0, x1, y1], "category": "<Label>", "text": "<content>"},
      ...
    ]

테이블 샘플은 category="Table" 단일 원소이며 text 가 표 HTML 이다.
레이아웃 샘플은 여러 element 의 배열이다.

지표 설계
--------
1) 공통
   - json_parse_success: 예측이 위 스키마(JSON 배열)로 파싱되는지 (0/1)

2) 테이블 샘플 (task_type == "table")
   - 예측/정답에서 Table element 의 HTML 을 꺼내 기존 TEDS 계열 지표 사용
   - avg_teds, avg_teds_structure, avg_span_f1 (eval.metrics 재사용)
   - 기존 e18 테이블 전용 평가와 직접 비교 가능

3) 레이아웃 샘플 (task_type == "layout")
   - element-level 매칭: bbox IoU >= iou_threshold AND category 일치 → TP
   - layout_f1 / layout_precision / layout_recall
   - layout_category_accuracy: 위치(IoU)만 맞춘 매칭에서 category 일치율
   - layout_mean_iou: 매칭쌍 평균 IoU
   - layout_text_sim: 매칭쌍 텍스트 정규화 유사도(1 - 편집거리/최대길이)

이 모듈은 추론 없이 (pred_text, gt_json, task_type) 만으로 동작하도록 설계되어
독립 단위 테스트가 가능하다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import editdistance  # eval.metrics 와 동일 의존성
except Exception:  # pragma: no cover
    editdistance = None


# =============================================================================
# 출력 파싱
# =============================================================================

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_TABLE_HTML_RE = re.compile(r"<table\b.*?</table\s*>", re.DOTALL | re.IGNORECASE)
_TABLE_OPEN_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_TABLE_ANY_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)


def _extract_balanced_table(text: str) -> str:
    """첫 `<table>` 부터 짝이 맞는 `</table>` 까지를 추출한다(중첩표 고려).

    non-greedy 정규식(`<table>...</table>`)은 **첫 번째** `</table>` 에서 끊겨
    중첩표(셀 안에 표가 또 있는 경우) 바깥표의 나머지 행을 잃어버린다.
    여기서는 table 태그 깊이(depth)를 세어 바깥표 전체를 회수한다.
    닫는 태그가 부족하면(출력이 잘린 경우) `<table>` 시작점부터 끝까지 반환한다.
    """
    if not text:
        return ""
    start_match = _TABLE_OPEN_RE.search(text)
    if start_match is None:
        return ""
    start = start_match.start()
    depth = 0
    for tag_match in _TABLE_ANY_RE.finditer(text, start):
        if tag_match.group(0).lower().startswith("</"):
            depth -= 1
            if depth <= 0:
                return text[start:tag_match.end()]
        else:
            depth += 1
    # 닫는 태그 부족(잘림) → 시작점 이후 전체 반환
    return text[start:]


def _unescape_json_fragment(s: str) -> str:
    """JSON 문자열 값 안에 들어있던 HTML 조각의 escape 를 되돌린다.

    파싱 실패한 원본 출력에서 정규식으로 `<table>...</table>` 를 직접 긁어내면
    JSON 문자열 내부의 escape(`\\"`, `\\n`, `\\/`, `\\\\`) 가 그대로 남아 있다.
    `json.loads` 를 거치지 못했으므로 여기서 수동 복원한다.
    (unicode_escape 디코딩은 한글을 깨뜨리므로 흔한 escape 만 치환한다.)
    """
    if not s:
        return s
    s = s.replace('\\"', '"')
    s = s.replace("\\/", "/")
    s = s.replace("\\n", "").replace("\\t", "").replace("\\r", "")
    s = s.replace("\\\\", "\\")
    return s


def extract_table_html_from_raw(text: str) -> str:
    """원본 출력 문자열에서 `<table>...</table>` 를 직접 추출한다(HTML fallback).

    JSON 파싱이 실패해도 모델이 표 HTML 자체는 만들어 둔 경우가 많으므로,
    표 블록을 긁어 escape 를 복원해 반환한다. 없으면 빈 문자열.
    중첩표는 깊이를 세어 바깥표 전체를 회수한다(첫 `</table>` 에서 끊지 않음).
    """
    body = strip_thinking(text)
    if not body:
        return ""
    fragment = _extract_balanced_table(body)
    if not fragment:
        return ""
    return _unescape_json_fragment(fragment).strip()


def strip_thinking(text: str) -> str:
    """<think>...</think> 블록 제거 후 본문 반환."""
    s = str(text or "")
    if "</think>" in s:
        s = s.split("</think>")[-1]
    return s.strip()


def parse_unified_output(text: str) -> tuple[Optional[list], bool]:
    """모델 출력 문자열을 element 리스트로 파싱한다.

    Returns:
        (elements, ok)
        - elements: dict 의 list (실패 시 None)
        - ok: 파싱 성공 여부
    """
    body = strip_thinking(text)
    if not body:
        return None, False

    # 1) 그대로 JSON 파싱 시도
    candidates = [body]
    # 2) 본문에서 첫 JSON 배열 추출 시도 (앞뒤 잡음 제거)
    m = _JSON_ARRAY_RE.search(body)
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, list):
            elements = [e for e in obj if isinstance(e, dict)]
            return elements, True
        if isinstance(obj, dict):
            # 단일 객체로 나온 경우도 배열로 취급
            return [obj], True
    return None, False


def parse_gt_elements(gt_html: str) -> list[dict]:
    """통합 정답(gt_html 에 저장된 JSON 문자열)을 element 리스트로 파싱한다."""
    elements, ok = parse_unified_output(gt_html)
    return elements if (ok and elements is not None) else []


def extract_table_html(elements: Optional[list]) -> str:
    """element 리스트에서 Table element 의 HTML(text)을 추출한다.

    여러 Table element 가 있으면 첫 번째를 사용한다.
    element 가 없거나 Table 이 없으면 빈 문자열.
    """
    if not elements:
        return ""
    for el in elements:
        if not isinstance(el, dict):
            continue
        cat = str(el.get("category", "")).strip().lower()
        if cat == "table":
            return str(el.get("text", "") or "")
    return ""


# =============================================================================
# 레이아웃 element 매칭 지표
# =============================================================================


def _iou(box_a: list, box_b: list) -> float:
    """두 bbox([x0,y0,x1,y1])의 IoU."""
    try:
        ax0, ay0, ax1, ay1 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
        bx0, by0, bx1, by1 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
    except (TypeError, ValueError, IndexError):
        return 0.0
    if ax1 < ax0:
        ax0, ax1 = ax1, ax0
    if ay1 < ay0:
        ay0, ay1 = ay1, ay0
    if bx1 < bx0:
        bx0, bx1 = bx1, bx0
    if by1 < by0:
        by0, by1 = by1, by0

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0.0, inter_x1 - inter_x0)
    ih = max(0.0, inter_y1 - inter_y0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _text_similarity(a: str, b: str) -> float:
    """정규화 편집거리 기반 유사도 (1 - dist/maxlen)."""
    na, nb = _norm_text(a), _norm_text(b)
    if not na and not nb:
        return 1.0
    if editdistance is None:
        return 1.0 if na == nb else 0.0
    max_len = max(len(na), len(nb))
    if max_len == 0:
        return 1.0
    return 1.0 - (editdistance.eval(na, nb) / max_len)


@dataclass
class LayoutMatchResult:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    category_accuracy: float = 0.0  # 위치 매칭쌍 중 category 일치율
    mean_iou: float = 0.0           # 위치 매칭쌍 평균 IoU
    text_sim: float = 0.0           # 위치 매칭쌍 평균 텍스트 유사도
    num_gt: int = 0
    num_pred: int = 0
    num_pos_matched: int = 0        # IoU 기준 매칭쌍 수 (category 무관)
    num_tp: int = 0                 # IoU + category 모두 일치


def _greedy_match(
    gt: list[dict],
    pred: list[dict],
    iou_threshold: float,
    require_category: bool,
) -> list[tuple[int, int, float]]:
    """IoU 내림차순 greedy 매칭. (gt_idx, pred_idx, iou) 리스트 반환."""
    pairs = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if require_category:
                gc = str(g.get("category", "")).strip().lower()
                pc = str(p.get("category", "")).strip().lower()
                if gc != pc:
                    continue
            iou = _iou(g.get("bbox", []), p.get("bbox", []))
            if iou >= iou_threshold:
                pairs.append((gi, pi, iou))

    pairs.sort(key=lambda x: x[2], reverse=True)
    used_gt, used_pred = set(), set()
    matched = []
    for gi, pi, iou in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched.append((gi, pi, iou))
    return matched


def compute_layout_metrics(
    gt: list[dict],
    pred: list[dict],
    iou_threshold: float = 0.5,
) -> LayoutMatchResult:
    """레이아웃 element-level 지표."""
    res = LayoutMatchResult(num_gt=len(gt), num_pred=len(pred))

    if not gt and not pred:
        res.precision = res.recall = res.f1 = 1.0
        res.category_accuracy = res.mean_iou = res.text_sim = 1.0
        return res
    if not gt or not pred:
        return res  # 전부 0

    # TP: IoU + category 모두 일치
    tp_pairs = _greedy_match(gt, pred, iou_threshold, require_category=True)
    tp = len(tp_pairs)
    res.num_tp = tp
    res.precision = tp / len(pred) if pred else 0.0
    res.recall = tp / len(gt) if gt else 0.0
    res.f1 = (
        2 * res.precision * res.recall / (res.precision + res.recall)
        if (res.precision + res.recall) > 0
        else 0.0
    )

    # 위치(IoU)만 기준으로 매칭 → category accuracy / mean IoU / text sim
    pos_pairs = _greedy_match(gt, pred, iou_threshold, require_category=False)
    res.num_pos_matched = len(pos_pairs)
    if pos_pairs:
        cat_ok = 0
        iou_sum = 0.0
        text_sum = 0.0
        for gi, pi, iou in pos_pairs:
            gcat = str(gt[gi].get("category", "")).strip().lower()
            pcat = str(pred[pi].get("category", "")).strip().lower()
            if gcat == pcat:
                cat_ok += 1
            iou_sum += iou
            text_sum += _text_similarity(gt[gi].get("text", ""), pred[pi].get("text", ""))
        n = len(pos_pairs)
        res.category_accuracy = cat_ok / n
        res.mean_iou = iou_sum / n
        res.text_sim = text_sum / n
    return res


# =============================================================================
# 샘플 단위 평가
# =============================================================================


def _evaluate_html_sample(pred_text: str, gt_html: str, task: str) -> dict:
    """HTML 파이프라인 샘플 평가.

    - task == "table": 표 HTML 끼리 TEDS (테이블 전용 TEDSCalculator)
    - 그 외(layout / 빈 값=통합 추론): 페이지 전체 HTML 트리 TEDS.
      카테고리를 구분하지 않고 모델이 낸 HTML 본문 전체를 그대로 보존한다
      (bbox 없음 → IoU 불가, 트리 편집거리로 대체).
    """
    from eval.metrics import (
        TEDSCalculator,
        compute_span_metrics,
        compute_generic_html_teds,
    )
    from utils.html_unified import extract_html_body

    pred_html = extract_html_body(pred_text)
    gt_body = extract_html_body(gt_html)

    record: dict[str, Any] = {
        "task_type": task,
        "output_format": "html",
        "html_parse_success": 1.0 if pred_html else 0.0,
    }

    if task == "table":
        teds_calc = TEDSCalculator(structure_only=False)
        teds_struct_calc = TEDSCalculator(structure_only=True)
        record["teds"] = teds_calc.compute(pred_html, gt_body)
        record["teds_structure"] = teds_struct_calc.compute(pred_html, gt_body)
        span = compute_span_metrics(pred_html, gt_body)
        record["span_f1"] = span.span_f1
        record["span_precision"] = span.span_precision
        record["span_recall"] = span.span_recall
        record["attribute_accuracy"] = span.attribute_accuracy
        record["pred_table_html"] = pred_html
        record["gt_table_html"] = gt_body
    else:
        teds, teds_struct = compute_generic_html_teds(pred_html, gt_body)
        record["layout_teds"] = teds
        record["layout_teds_structure"] = teds_struct
        record["pred_layout_html"] = pred_html
        record["gt_layout_html"] = gt_body

    return record


def evaluate_unified_sample(
    pred_text: str,
    gt_html: str,
    task_type: str,
    iou_threshold: float = 0.5,
    output_format: str = "json",
) -> dict:
    """한 샘플을 평가해 지표 dict 를 반환한다.

    table 샘플과 layout 샘플 모두 동일 인터페이스로 처리한다.
    output_format == "html" 이면 HTML 파이프라인 평가로 분기한다.

    task_type 이 비어 있으면(통합 추론) 카테고리를 강제하지 않고, 전체 HTML
    본문을 그대로 보존하는 공통 경로(= table 이 아닌 경로)로 처리한다.
    """
    task = str(task_type or "").strip().lower()

    if str(output_format).strip().lower() == "html":
        return _evaluate_html_sample(pred_text, gt_html, task)

    pred_elements, parse_ok = parse_unified_output(pred_text)
    gt_elements = parse_gt_elements(gt_html)

    record: dict[str, Any] = {
        "task_type": task,
        "json_parse_success": 1.0 if parse_ok else 0.0,
        "num_pred_elements": len(pred_elements) if pred_elements else 0,
        "num_gt_elements": len(gt_elements),
        # 추론 결과에 bbox/category/text 를 그대로 보존(라벨러 UI 확인용).
        "pred_elements": pred_elements or [],
        "gt_elements": gt_elements,
    }

    if task == "table":
        from eval.metrics import (
            TEDSCalculator,
            compute_span_metrics,
        )

        pred_html = extract_table_html(pred_elements)
        # HTML fallback: JSON 파싱/추출이 실패해도 원본 출력에 표 HTML 이 있으면
        # 그것으로 TEDS 를 계산한다("text 에 html 이 있는데 0점" 방지).
        html_fallback_used = False
        if not pred_html:
            pred_html = extract_table_html_from_raw(pred_text)
            html_fallback_used = bool(pred_html)
        record["html_fallback_used"] = 1.0 if html_fallback_used else 0.0

        gt_table_html = extract_table_html(gt_elements)
        if not gt_table_html:
            gt_table_html = extract_table_html_from_raw(gt_html)
        teds_calc = TEDSCalculator(structure_only=False)
        teds_struct_calc = TEDSCalculator(structure_only=True)
        record["teds"] = teds_calc.compute(pred_html, gt_table_html)
        record["teds_structure"] = teds_struct_calc.compute(pred_html, gt_table_html)
        span = compute_span_metrics(pred_html, gt_table_html)
        record["span_f1"] = span.span_f1
        record["span_precision"] = span.span_precision
        record["span_recall"] = span.span_recall
        record["attribute_accuracy"] = span.attribute_accuracy
        record["pred_table_html"] = pred_html
        record["gt_table_html"] = gt_table_html
    else:
        layout = compute_layout_metrics(
            gt_elements,
            pred_elements or [],
            iou_threshold=iou_threshold,
        )
        record["layout_f1"] = layout.f1
        record["layout_precision"] = layout.precision
        record["layout_recall"] = layout.recall
        record["layout_category_accuracy"] = layout.category_accuracy
        record["layout_mean_iou"] = layout.mean_iou
        record["layout_text_sim"] = layout.text_sim
        record["layout_num_tp"] = layout.num_tp

    return record


# =============================================================================
# 집계
# =============================================================================


def _avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def aggregate_unified_metrics(records: list[dict]) -> dict:
    """샘플 레코드 리스트를 집계한다."""
    table_recs = [r for r in records if r.get("task_type") == "table"]
    # task_type 이 비어 있으면(통합 추론) layout 경로로 집계한다.
    layout_recs = [r for r in records if r.get("task_type") in ("layout", "")]

    is_html = any(r.get("output_format") == "html" for r in records)

    out: dict[str, Any] = {
        "total_samples": len(records),
        "table_samples": len(table_recs),
        "layout_samples": len(layout_recs),
        "output_format": "html" if is_html else "json",
    }
    if is_html:
        out["html_parse_success_rate"] = _avg(
            [r.get("html_parse_success", 0.0) for r in records]
        )
    else:
        out["json_parse_success_rate"] = _avg(
            [r.get("json_parse_success", 0.0) for r in records]
        )

    if table_recs:
        table_out = {
            "avg_teds": _avg([r.get("teds", 0.0) for r in table_recs]),
            "avg_teds_structure": _avg([r.get("teds_structure", 0.0) for r in table_recs]),
            "avg_span_f1": _avg([r.get("span_f1", 0.0) for r in table_recs]),
            "avg_span_precision": _avg([r.get("span_precision", 0.0) for r in table_recs]),
            "avg_span_recall": _avg([r.get("span_recall", 0.0) for r in table_recs]),
            "avg_attribute_accuracy": _avg([r.get("attribute_accuracy", 0.0) for r in table_recs]),
            "teds_nonzero_rate": _avg([1.0 if r.get("teds", 0.0) > 0 else 0.0 for r in table_recs]),
        }
        if is_html:
            table_out["html_parse_success_rate"] = _avg(
                [r.get("html_parse_success", 0.0) for r in table_recs]
            )
        else:
            table_out["json_parse_success_rate"] = _avg(
                [r.get("json_parse_success", 0.0) for r in table_recs]
            )
            table_out["html_fallback_rate"] = _avg(
                [r.get("html_fallback_used", 0.0) for r in table_recs]
            )
        out["table"] = table_out

    if layout_recs:
        if is_html:
            # HTML 모드: bbox 가 없어 IoU 불가 → 페이지 전체 HTML 트리 TEDS 로 평가
            out["layout"] = {
                "avg_layout_teds": _avg([r.get("layout_teds", 0.0) for r in layout_recs]),
                "avg_layout_teds_structure": _avg(
                    [r.get("layout_teds_structure", 0.0) for r in layout_recs]
                ),
                "layout_teds_nonzero_rate": _avg(
                    [1.0 if r.get("layout_teds", 0.0) > 0 else 0.0 for r in layout_recs]
                ),
                "html_parse_success_rate": _avg(
                    [r.get("html_parse_success", 0.0) for r in layout_recs]
                ),
            }
        else:
            out["layout"] = {
                "avg_layout_f1": _avg([r.get("layout_f1", 0.0) for r in layout_recs]),
                "avg_layout_precision": _avg([r.get("layout_precision", 0.0) for r in layout_recs]),
                "avg_layout_recall": _avg([r.get("layout_recall", 0.0) for r in layout_recs]),
                "avg_layout_category_accuracy": _avg([r.get("layout_category_accuracy", 0.0) for r in layout_recs]),
                "avg_layout_mean_iou": _avg([r.get("layout_mean_iou", 0.0) for r in layout_recs]),
                "avg_layout_text_sim": _avg([r.get("layout_text_sim", 0.0) for r in layout_recs]),
                "json_parse_success_rate": _avg([r.get("json_parse_success", 0.0) for r in layout_recs]),
            }

    return out
