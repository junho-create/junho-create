"""[2]+[3] Taster CMCV: 클러스터별 소량 샘플 → 난이도 추정 → DDAS 예산배분."""

from ocr_filter.taster.aggregate import aggregate_tier_counts, cluster_sizes
from ocr_filter.taster.run import run_taster
from ocr_filter.taster.sample import flatten, load_page_clusters, sample_taster_ids

__all__ = [
    "run_taster", "sample_taster_ids", "load_page_clusters", "flatten",
    "aggregate_tier_counts", "cluster_sizes",
]
