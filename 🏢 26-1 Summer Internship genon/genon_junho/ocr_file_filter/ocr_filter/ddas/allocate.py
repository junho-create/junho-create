"""DDAS 2단계: 클러스터별 샘플링 할당량 N_i.

PaddleOCR 다항식 가중치 구조를 차용하되 입력을 S_i 로 대체:

    N_i = min( (S_i + alpha)^beta / sum_j (S_j + alpha)^beta * N_total , |C_i| )

- alpha (평탄화, 기본 1.0): S_i 가 낮은 클러스터도 최소 베이스라인은 확보.
- beta  (증폭, 기본 2.0):   점수 차이를 제곱으로 벌려 가치 높은 클러스터에 예산 집중.
- min(.., |C_i|):           소형 롱테일 클러스터에 과배정되는 걸 막는 안전장치.

★ 안전장치의 함의: 캡(|C_i|)에 걸린 클러스터의 '남는 예산'은 버리면 안 되고
   캡에 안 걸린 클러스터들로 재분배해야 한다. 단순 1패스 계산은 총합이 N_total 보다
   작아지므로 여기서는 water-filling 방식으로 반복 재분배한다.
"""

from __future__ import annotations

import math


def allocate_budget(
    scores: dict[str, float],
    sizes: dict[str, int],
    n_total: int,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> dict[str, int]:
    """클러스터별 정수 할당량 N_i 를 계산한다.

    scores: {cluster_id: S_i}
    sizes:  {cluster_id: |C_i|}  (해당 클러스터가 물리적으로 보유한 샘플 수)
    n_total: 전체 샘플링 예산

    반환: {cluster_id: N_i}  (모든 N_i >= 0, N_i <= |C_i|,
          sum(N_i) == min(n_total, sum(sizes)))

    캡에 걸린 클러스터의 잉여 예산은 남은 클러스터로 반복 재분배된다.
    """
    if scores.keys() != sizes.keys():
        raise ValueError("scores 와 sizes 의 cluster_id 집합이 일치해야 합니다.")

    total_available = sum(sizes.values())
    budget = min(int(n_total), total_available)

    # 1) 연속값 water-filling: 어떤 클러스터가 캡에 걸리는지 확정한다.
    alloc: dict[str, float] = {cid: 0.0 for cid in scores}
    capped: set[str] = set()

    while True:
        active = [cid for cid in scores if cid not in capped]
        if not active:
            break
        remaining = budget - sum(alloc[cid] for cid in capped)
        if remaining <= 0:
            for cid in active:
                alloc[cid] = 0.0
            break

        weights = {cid: (scores[cid] + alpha) ** beta for cid in active}
        wsum = sum(weights.values())
        if wsum <= 0:  # 모든 활성 점수가 사실상 0 → 균등 분배
            weights = {cid: 1.0 for cid in active}
            wsum = float(len(active))

        # 이번 라운드에서 캡을 넘겨버리는 클러스터를 찾는다.
        overflow = [
            cid for cid in active if weights[cid] / wsum * remaining > sizes[cid]
        ]
        if not overflow:
            for cid in active:
                alloc[cid] = weights[cid] / wsum * remaining
            break

        # 넘친 것들은 |C_i| 로 고정하고 남은 예산을 다음 라운드로 넘긴다.
        for cid in overflow:
            alloc[cid] = float(sizes[cid])
            capped.add(cid)

    # 2) 정수화(최대 잉여법, Hamilton): 반올림 오차로 총합이 어긋나지 않게 맞춘다.
    floored = {cid: int(math.floor(v)) for cid, v in alloc.items()}
    deficit = budget - sum(floored.values())

    # 잔여 예산을 소수부가 큰 순서로 1개씩, 단 |C_i| 를 넘지 않게 배분.
    order = sorted(scores, key=lambda cid: alloc[cid] - floored[cid], reverse=True)
    idx = 0
    while deficit > 0 and idx < len(order) * 4:  # 안전한 상한
        cid = order[idx % len(order)]
        if floored[cid] < sizes[cid]:
            floored[cid] += 1
            deficit -= 1
        idx += 1

    return floored
