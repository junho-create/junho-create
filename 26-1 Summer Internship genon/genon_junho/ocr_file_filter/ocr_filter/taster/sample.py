"""[2] Taster CMCV 1단계: 클러스터별로 소량만 뽑기 (`_work/clusters.jsonl` 의 page 레벨 기준).

전체(수천~수만 건)에 CMCV 를 다 돌리는 대신, 페이지클러스터마다 `taster_per_cluster`
(configs/default.yaml 의 `cluster.taster_per_cluster`, 기본 8) 개만 뽑아서 그걸로
클러스터 난이도(E/M/H 비율)를 추정한다 — DDAS(allocate_budget)가 원하는 입력이 바로 이거.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def load_page_clusters(clusters_path: str | Path) -> dict[str, int]:
    """{record_id: page_cluster} — level=="page" 인 라인만."""
    mapping = {}
    with open(clusters_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("level") == "page":
                mapping[d["id"]] = d["cluster"]
    return mapping


def sample_taster_ids(
    clusters_path: str | Path, n_per_cluster: int = 8, seed: int = 0,
) -> dict[int, list[str]]:
    """cluster -> 그 클러스터에서 무작위로 뽑은 최대 n_per_cluster 개 record id."""
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for record_id, cluster in load_page_clusters(clusters_path).items():
        by_cluster[cluster].append(record_id)

    rng = random.Random(seed)
    picked = {}
    for cluster, ids in by_cluster.items():
        rng.shuffle(ids)
        picked[cluster] = ids[:n_per_cluster]
    return picked


def flatten(picked: dict[int, list[str]]) -> set[str]:
    return {i for ids in picked.values() for i in ids}
