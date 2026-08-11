"""cluster.kmeans 단위 테스트: ViT/torch 안 쓰고 합성 임베딩으로 2단계 K-Means 로직만 검증.

    pytest tests/test_cluster_kmeans.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.cluster.kmeans import cluster_ids, two_stage_cluster  # noqa: E402


def test_cluster_ids_caps_at_sample_count():
    embeddings = np.random.RandomState(0).randn(3, 8)
    ids = cluster_ids(embeddings, n_clusters=64)  # 샘플 3개뿐인데 64개로 나누라고 하면
    assert len(set(ids.tolist())) <= 3
    assert len(ids) == 3


def test_cluster_ids_single_cluster_when_requested():
    embeddings = np.random.RandomState(0).randn(10, 8)
    ids = cluster_ids(embeddings, n_clusters=1)
    assert set(ids.tolist()) == {0}


def test_two_stage_cluster_assigns_disjoint_element_ids_per_page_cluster():
    rng = np.random.RandomState(0)
    # 페이지 두 그룹(멀리 떨어진 두 뭉치) → 페이지 클러스터 2개로 잘 나뉘어야.
    page_embeddings = np.vstack([
        rng.randn(4, 8) + np.array([10] * 8),
        rng.randn(4, 8) + np.array([-10] * 8),
    ])
    # 각 페이지(총 8개)마다 요소 2개씩 = 16개, 소속 페이지 인덱스
    element_page_index = np.repeat(np.arange(8), 2)
    element_embeddings = rng.randn(16, 8)

    page_clusters, element_clusters = two_stage_cluster(
        page_embeddings, element_embeddings, element_page_index,
        n_clusters_page=2, n_clusters_element=2, seed=0,
    )

    assert len(page_clusters) == 8
    assert len(set(page_clusters.tolist())) == 2
    # 앞 4페이지 vs 뒤 4페이지가 서로 다른 클러스터에 뭉쳐 있어야 (뚜렷이 분리된 임베딩이라)
    assert len(set(page_clusters[:4].tolist())) == 1
    assert len(set(page_clusters[4:].tolist())) == 1
    assert page_clusters[0] != page_clusters[4]

    # element_clusters 는 전역적으로 재부여돼서, 서로 다른 페이지클러스터에 속한 요소는
    # 절대 같은 element cluster id 를 공유하지 않아야 함.
    page_cluster_of_element = page_clusters[element_page_index]
    for pc in set(page_cluster_of_element.tolist()):
        ids_here = set(element_clusters[page_cluster_of_element == pc].tolist())
        ids_elsewhere = set(element_clusters[page_cluster_of_element != pc].tolist())
        assert ids_here.isdisjoint(ids_elsewhere)
