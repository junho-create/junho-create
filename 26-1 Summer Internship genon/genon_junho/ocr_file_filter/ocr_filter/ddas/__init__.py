"""DDAS: Diversity-and-Difficulty-Aware Sampling.

    score_clusters   → S_i (클러스터 난이도 점수)
    allocate_budget  → N_i (클러스터별 샘플링 할당량, 잉여 재분배 포함)
    sample_from_clusters → 실제 샘플 추출
    select_final_ids → taster_report.json + clusters.jsonl(전체 페이지) → 실제 최종 id 목록
"""

from ocr_filter.ddas.allocate import allocate_budget
from ocr_filter.ddas.sample import flatten, sample_from_clusters
from ocr_filter.ddas.score import (
    DifficultyWeights,
    difficulty_score,
    score_clusters,
)
from ocr_filter.ddas.select import (
    select_additional_ids,
    select_final_ids,
    select_top_clusters_by_real_rate,
    write_selected_ids,
)

__all__ = [
    "DifficultyWeights",
    "difficulty_score",
    "score_clusters",
    "allocate_budget",
    "sample_from_clusters",
    "flatten",
    "select_final_ids",
    "select_additional_ids",
    "select_top_clusters_by_real_rate",
    "write_selected_ids",
]
