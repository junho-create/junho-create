"""DDAS 1단계: 클러스터별 난이도 점수 S_i.

CMCV 결과(Easy/Medium/Hard 분포)를 클러스터의 '학습 가치 점수'로 치환한다.

    S_i = w_medium * P_i(Medium) + w_hard * P_i(Hard) + w_easy * P_i(Easy)

설계 의도: Medium(약점 보완)이 가장 가치 높고, Hard(돌파구, 노이즈 위험)가 다음,
Easy(가치 낮음)는 거의 배제. 그래서 Medium/Hard 비율이 높은 롱테일 클러스터일수록
S_i가 가파르게 상승한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyWeights:
    """S_i 계산에 쓰는 난이도별 가중치 (문서 추천값)."""

    easy: float = 0.1
    medium: float = 1.5
    hard: float = 1.0


def difficulty_score(
    n_easy: float,
    n_medium: float,
    n_hard: float,
    weights: DifficultyWeights | None = None,
) -> float:
    """클러스터 하나의 난이도 점수 S_i.

    입력은 개수(count)여도 되고 비율이어도 된다 — 내부에서 비율로 정규화한다.
    클러스터가 비어있으면(합계 0) 0.0을 돌려준다.
    """
    weights = weights or DifficultyWeights()
    total = n_easy + n_medium + n_hard
    if total <= 0:
        return 0.0
    p_easy = n_easy / total
    p_medium = n_medium / total
    p_hard = n_hard / total
    return weights.medium * p_medium + weights.hard * p_hard + weights.easy * p_easy


def score_clusters(
    counts: dict[str, dict[str, float]],
    weights: DifficultyWeights | None = None,
) -> dict[str, float]:
    """여러 클러스터의 S_i 를 한 번에.

    counts: {cluster_id: {"easy": n, "medium": n, "hard": n}}
    반환:   {cluster_id: S_i}
    """
    weights = weights or DifficultyWeights()
    return {
        cid: difficulty_score(
            c.get("easy", 0.0),
            c.get("medium", 0.0),
            c.get("hard", 0.0),
            weights,
        )
        for cid, c in counts.items()
    }
