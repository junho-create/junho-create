"""[2]+[3] Taster CMCV → DDAS 연결: 클러스터별 소량 샘플만 채점해서 클러스터 난이도 추정,
그걸로 DDAS 예산 배분까지 한 번에.

    python -m ocr_filter.cli taster run --n-per-cluster 8

cluster build 로 만든 `_work/clusters.jsonl` 이 미리 있어야 한다.
"""

from __future__ import annotations

from pathlib import Path

from ocr_filter.cmcv.run import run_cmcv
from ocr_filter.ddas import allocate_budget, score_clusters
from ocr_filter.taster.aggregate import aggregate_tier_counts, cluster_sizes
from ocr_filter.taster.sample import flatten, load_page_clusters, sample_taster_ids


def run_taster(
    clusters_path: str | Path,
    unified_path: str | Path,
    cmcv_out_path: str | Path,
    models_cfg: dict,
    n_per_cluster: int = 8,
    workers: int = 16,
    agree_min: float = 0.85,
    text_min: float = 0.90,
    ddas_alpha: float = 1.0,
    ddas_beta: float = 2.0,
    ddas_n_total: int | None = None,
    seed: int = 0,
) -> dict:
    picked = sample_taster_ids(clusters_path, n_per_cluster=n_per_cluster, seed=seed)
    ids = flatten(picked)

    cmcv_stats = run_cmcv(
        unified_path, cmcv_out_path, models_cfg, ids=ids, workers=workers,
        agree_min=agree_min, text_min=text_min,
    )

    id_to_cluster = load_page_clusters(clusters_path)
    sizes = {str(c): n for c, n in cluster_sizes(clusters_path).items()}
    raw_counts = aggregate_tier_counts(cmcv_out_path, id_to_cluster)
    # allocate_budget 은 scores/sizes 의 cluster_id 집합이 정확히 일치해야 함 —
    # taster 샘플이 하나도 안 걸린(에러 등) 클러스터는 0/0/0 으로 채워 넣는다.
    counts = {
        cid: raw_counts.get(int(cid), {"easy": 0, "medium": 0, "hard": 0})
        for cid in sizes
    }

    scores = score_clusters(counts)
    n_total = ddas_n_total if ddas_n_total is not None else sum(sizes.values())
    alloc = allocate_budget(scores, sizes, n_total=n_total, alpha=ddas_alpha, beta=ddas_beta)

    return {
        "cmcv_stats": cmcv_stats,
        "n_clusters": len(sizes),
        "n_taster_samples": len(ids),
        "tier_counts": counts,
        "scores": scores,
        "sizes": sizes,
        "alloc": alloc,
    }
