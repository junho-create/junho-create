"""2단계 K-Means: 1단계 페이지 수준(n_clusters_page) → 2단계 각 페이지클러스터 내부에서
요소 수준(n_clusters_element) 재세분화. `configs/default.yaml` 의 `cluster:` 섹션이 이 값들.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def cluster_ids(embeddings: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """embeddings: (N, dim). N < n_clusters 면 자동으로 N개 클러스터로 낮춘다."""
    n_clusters = max(1, min(n_clusters, len(embeddings)))
    if n_clusters == 1:
        return np.zeros(len(embeddings), dtype=int)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    return km.fit_predict(embeddings)


def two_stage_cluster(
    page_embeddings: np.ndarray,
    element_embeddings: np.ndarray,
    element_page_index: np.ndarray,
    n_clusters_page: int,
    n_clusters_element: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """1단계: 페이지 임베딩으로 페이지클러스터 배정.
    2단계: 각 페이지클러스터에 속한 요소들만 모아 그 안에서 다시 element 클러스터 배정
    (element_page_index[i] 는 element_embeddings[i] 가 어느 page_embeddings 인덱스 소속인지).

    반환: (page_cluster_ids, element_cluster_ids). element_cluster_ids 는
    "<page_cluster>-<local_cluster>" 형태가 아니라 전역 정수 id 로 재부여해서 반환한다
    (페이지클러스터 A의 0번과 B의 0번이 섞이지 않게).
    """
    page_clusters = cluster_ids(page_embeddings, n_clusters_page, seed)
    element_page_clusters = page_clusters[element_page_index]

    element_clusters = np.full(len(element_embeddings), -1, dtype=int)
    next_global_id = 0
    for pc in np.unique(element_page_clusters):
        mask = element_page_clusters == pc
        local_ids = cluster_ids(element_embeddings[mask], n_clusters_element, seed)
        n_local = local_ids.max() + 1 if len(local_ids) else 0
        element_clusters[mask] = local_ids + next_global_id
        next_global_id += n_local

    return page_clusters, element_clusters
