"""DDAS 3단계: 할당량 N_i 에 따라 실제 샘플을 뽑는다.

allocate.py 가 정한 N_i 는 이미 |C_i| 로 캡이 걸려 있으므로 기본은 비복원 추출.
단, 문서의 '소형 클러스터 업샘플링' 전략을 쓰고 싶으면 with_replacement=True 로
가치 높은 소형 클러스터를 복원 추출(반복)할 수 있다.
"""

from __future__ import annotations

import random


def sample_from_clusters(
    members: dict[str, list],
    allocation: dict[str, int],
    seed: int = 0,
    with_replacement: bool = False,
) -> dict[str, list]:
    """클러스터별 멤버 리스트에서 N_i 개씩 추출한다.

    members:    {cluster_id: [sample_id, ...]}
    allocation: {cluster_id: N_i}
    반환:       {cluster_id: [뽑힌 sample_id, ...]}
    """
    rng = random.Random(seed)
    selected: dict[str, list] = {}
    for cid, n in allocation.items():
        pool = members.get(cid, [])
        if n <= 0 or not pool:
            selected[cid] = []
            continue
        if with_replacement:
            selected[cid] = [rng.choice(pool) for _ in range(n)]
        else:
            k = min(n, len(pool))
            selected[cid] = rng.sample(pool, k)
    return selected


def flatten(selected: dict[str, list]) -> list:
    """{cluster_id: [ids]} → 단일 id 리스트 (학습셋 export 용)."""
    out: list = []
    for ids in selected.values():
        out.extend(ids)
    return out
