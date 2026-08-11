"""[2] Taster CMCV 2단계: taster 샘플의 cmcv 결과(E/M/H 티어) → 클러스터별 개수 집계.

이 출력(`{cluster_id: {"easy":n,"medium":n,"hard":n}}`)이 그대로 DDAS 의
`score_clusters`/`allocate_budget` 입력 형식이다 (`ocr_filter.ddas`, 이미 구현됨).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def aggregate_tier_counts(
    cmcv_results_path: str | Path, id_to_cluster: dict[str, int],
) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"easy": 0, "medium": 0, "hard": 0})
    with open(cmcv_results_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cluster = id_to_cluster.get(d["id"])
            if cluster is None:
                continue
            tier = d["tier"].lower()
            counts[cluster][tier] += 1
    return dict(counts)


def cluster_sizes(clusters_path: str | Path) -> dict[int, int]:
    """페이지클러스터별 전체 크기 |C_i| (DDAS allocate_budget 의 sizes 인자)."""
    sizes: dict[int, int] = defaultdict(int)
    with open(clusters_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("level") == "page":
                sizes[d["cluster"]] += 1
    return dict(sizes)
