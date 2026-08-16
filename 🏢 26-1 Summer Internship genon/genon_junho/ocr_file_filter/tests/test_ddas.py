"""DDAS 단위 테스트 + 데모.

    pytest tests/test_ddas.py      # 검증
    python tests/test_ddas.py      # 합성 데이터로 동작 시연
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.ddas import (  # noqa: E402
    DifficultyWeights,
    allocate_budget,
    difficulty_score,
    flatten,
    sample_from_clusters,
    score_clusters,
)


def test_score_medium_beats_hard_beats_easy():
    w = DifficultyWeights()
    s_medium = difficulty_score(0, 10, 0, w)   # 전부 Medium
    s_hard = difficulty_score(0, 0, 10, w)     # 전부 Hard
    s_easy = difficulty_score(10, 0, 0, w)     # 전부 Easy
    assert s_medium > s_hard > s_easy
    assert abs(s_medium - 1.5) < 1e-9
    assert abs(s_hard - 1.0) < 1e-9
    assert abs(s_easy - 0.1) < 1e-9


def test_score_accepts_counts_and_proportions():
    w = DifficultyWeights()
    by_count = difficulty_score(2, 6, 2, w)
    by_prop = difficulty_score(0.2, 0.6, 0.2, w)
    assert abs(by_count - by_prop) < 1e-9


def test_empty_cluster_scores_zero():
    assert difficulty_score(0, 0, 0) == 0.0


def test_allocate_sums_to_budget():
    scores = {"a": 1.5, "b": 1.0, "c": 0.1, "d": 0.0}
    sizes = {"a": 1000, "b": 1000, "c": 1000, "d": 1000}
    alloc = allocate_budget(scores, sizes, n_total=1000)
    assert sum(alloc.values()) == 1000
    # 고가치(a: 전부 Medium)가 저가치(d: 전부 Easy)보다 많이 배정돼야.
    assert alloc["a"] > alloc["b"] > alloc["c"] > alloc["d"]


def test_allocate_respects_cap_and_redistributes():
    # a는 점수 최고지만 물리적으로 5개뿐 → 캡. 남는 예산은 b/c로 재분배돼야.
    scores = {"a": 5.0, "b": 1.0, "c": 1.0}
    sizes = {"a": 5, "b": 1000, "c": 1000}
    alloc = allocate_budget(scores, sizes, n_total=500)
    assert alloc["a"] == 5                     # 캡에 걸림
    assert sum(alloc.values()) == 500          # 잉여가 버려지지 않음
    assert alloc["b"] > 0 and alloc["c"] > 0
    for cid in scores:
        assert alloc[cid] <= sizes[cid]        # 어떤 것도 |C_i| 초과 금지


def test_allocate_budget_exceeds_available():
    # 예산이 전체 보유량보다 크면 전량(=sum sizes)만 뽑는다.
    scores = {"a": 1.0, "b": 1.0}
    sizes = {"a": 3, "b": 4}
    alloc = allocate_budget(scores, sizes, n_total=100)
    assert alloc == {"a": 3, "b": 4}


def test_sample_no_replacement_bounded_by_pool():
    members = {"a": list(range(10)), "b": list(range(5))}
    alloc = {"a": 4, "b": 99}
    picked = sample_from_clusters(members, alloc, seed=1)
    assert len(picked["a"]) == 4
    assert len(picked["b"]) == 5               # 풀보다 많이 요청 → 풀 크기로 제한
    assert len(set(picked["a"])) == 4          # 비복원 → 중복 없음
    assert len(flatten(picked)) == 9


def test_sample_with_replacement_can_upsample():
    members = {"a": [1, 2]}
    alloc = {"a": 6}
    picked = sample_from_clusters(members, alloc, seed=1, with_replacement=True)
    assert len(picked["a"]) == 6               # 복원 추출 → 업샘플링 가능


def _demo():
    # 합성 클러스터 5개: 각기 다른 E/M/H 분포 + 크기
    counts = {
        "papers_easy":   {"easy": 90, "medium": 8,  "hard": 2},   # 흔한 학술논문
        "mixed":         {"easy": 30, "medium": 50, "hard": 20},
        "longtail_hard": {"easy": 5,  "medium": 40, "hard": 55},  # 희귀/어려움
        "rare_small":    {"easy": 2,  "medium": 6,  "hard": 4},   # 소형 고가치
        "boilerplate":   {"easy": 98, "medium": 2,  "hard": 0},
    }
    sizes = {
        "papers_easy": 8000, "mixed": 4000, "longtail_hard": 2500,
        "rare_small": 60, "boilerplate": 3000,
    }
    scores = score_clusters(counts)
    alloc = allocate_budget(scores, sizes, n_total=6000, alpha=1.0, beta=2.0)

    print(f"{'cluster':<16}{'S_i':>7}{'|C_i|':>8}{'N_i':>7}{'비율%':>8}")
    print("-" * 46)
    for cid in counts:
        pct = 100 * alloc[cid] / sizes[cid]
        print(f"{cid:<16}{scores[cid]:>7.3f}{sizes[cid]:>8}{alloc[cid]:>7}{pct:>7.1f}%")
    print("-" * 46)
    print(f"{'TOTAL':<16}{'':>7}{sum(sizes.values()):>8}{sum(alloc.values()):>7}")
    print("\n관찰: 흔한 easy 클러스터는 억눌리고, longtail_hard/mixed 로 예산이 몰림.")
    print("      rare_small 은 |C_i|=60 캡에 걸려 소진율 100% 에 근접.")


if __name__ == "__main__":
    _demo()
