"""metrics: gt vs pred 유사도 스코어러 모음. [cmcv] 스테이지가 이 레지스트리로
match_thresholds(`configs/default.yaml` 의 `cmcv.match_thresholds`)를 채점한다.

    SCORERS["edit_distance"](gt_text, pred_text)        -> float [0,1]
    SCORERS["teds"](gt_html, pred_html)                  -> float [0,1] (표 전용, label 완전일치)
    SCORERS["teds_s"](gt_html, pred_html)                -> float [0,1] (구조만)
    SCORERS["generic_teds"](a_html, b_html)              -> float [0,1] (임의 HTML, 텍스트는 부분점수)
    SCORERS["generic_teds_s"](a_html, b_html)            -> float [0,1] (위의 구조만 버전)
    SCORERS["page_iou"](a_boxes, b_boxes)                -> float [0,1] (박스 리스트, 페이지 커버리지 IoU)
    SCORERS["cdm"](gt_formula, pred_formula)             -> float | None (수식 전용, TeX Live 미설치/렌더 실패시 None=스킵)

전부 `(gt, pred) -> float | None` 시그니처만 지키면 되므로, 개별 구현은 자유롭게
교체 가능 (특히 teds/cdm 은 다른 라이브러리/서비스로 바뀔 걸 염두에 두고 짬).
"""

from functools import partial

from ocr_filter.metrics.cdm import cdm_score
from ocr_filter.metrics.edit_distance import edit_distance_score, levenshtein
from ocr_filter.metrics.generic_teds import generic_teds_score
from ocr_filter.metrics.page_iou import page_iou_score
from ocr_filter.metrics.teds import teds_score

SCORERS = {
    "edit_distance": edit_distance_score,
    "teds": teds_score,
    "teds_s": partial(teds_score, structure_only=True),
    "generic_teds": generic_teds_score,
    "generic_teds_s": partial(generic_teds_score, structure_only=True),
    "page_iou": page_iou_score,
    "cdm": cdm_score,
}

__all__ = [
    "SCORERS",
    "edit_distance_score",
    "levenshtein",
    "teds_score",
    "generic_teds_score",
    "page_iou_score",
    "cdm_score",
]
