"""[6] Hard case 정제: Judge-and-Refine(1단계) + Targeted Expert Annotation 사전라벨(2단계).

    judge_and_refine  → 렌더링 오버레이 vs 원본을 heavy judge VLM 이 판정, FAIL 이면 재교정 반복
    prelabel_once     → 1단계에서 안 풀린 것만, 독립 모델이 원본만 보고 초안 재생성
    run_judge/run_prelabel/run_both → cmcv Hard 티어 레코드에 위 로직 일괄 적용 (resumable)
"""

from ocr_filter.hardcase.pipeline import (
    generate_label,
    judge_and_refine,
    judge_layout_once,
    judge_once,
    prelabel_once,
    refine_once,
    revise_label,
)
from ocr_filter.hardcase.run import run_both, run_judge, run_prelabel

__all__ = [
    "generate_label", "judge_and_refine", "judge_once", "judge_layout_once", "revise_label",
    "refine_once", "prelabel_once", "run_judge", "run_prelabel", "run_both",
]
