"""[3]→[4] 연결: taster_report.json 의 클러스터별 할당량(N_i)으로 **실제 최종 샘플 id 목록**을
뽑는다. `sample_from_clusters`(이미 구현됨, `ocr_filter/ddas/sample.py`) 는 "클러스터별 멤버
풀에서 N_i개 뽑기"만 하는 순수 로직이고, 이 파일은 그 멤버 풀을 clusters.jsonl 전체 페이지
집합(taster 8개가 아니라 클러스터의 진짜 전체 구성원)에서 만들어 연결해주는 접착부.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ocr_filter.ddas.allocate import allocate_budget
from ocr_filter.ddas.sample import flatten, sample_from_clusters
from ocr_filter.ddas.score import DifficultyWeights, score_clusters


def _cluster_members(clusters_path: str | Path) -> dict[str, list[str]]:
    """{cluster_id(str): [해당 클러스터의 전체 page id, ...]} — clusters.jsonl 의 level=="page" 전부."""
    members: dict[str, list[str]] = defaultdict(list)
    with open(clusters_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("level") == "page":
                members[str(d["cluster"])].append(d["id"])
    return dict(members)


def select_final_ids(
    taster_report_path: str | Path, clusters_path: str | Path, seed: int = 0,
    with_replacement: bool = False, n_per_cluster: int | None = None,
) -> dict[str, list[str]]:
    """taster_report.json 의 alloc(N_i, 클러스터별) 으로 clusters.jsonl 전체 페이지 풀에서
    실제로 뽑힌 id 를 골라 {cluster_id: [id, ...]} 로 반환.

    n_per_cluster 를 주면 난이도 기반 water-filling(N_i) 대신 **클러스터마다 균일하게
    n_per_cluster 개**(|C_i| 로 캡) 뽑는다 — 전체 파이프라인 스모크 테스트용(예:
    클러스터 96개 × 1개 = 96건만 빠르게 cmcv 끝까지 돌려보고 싶을 때)."""
    report = json.loads(Path(taster_report_path).read_text(encoding="utf-8"))
    members = _cluster_members(clusters_path)
    if n_per_cluster is not None:
        alloc = {cid: min(n_per_cluster, len(ids)) for cid, ids in members.items()}
    else:
        alloc = report["alloc"]
    return sample_from_clusters(members, alloc, seed=seed,
                                 with_replacement=with_replacement)


def select_additional_ids(
    cmcv_results_path: str | Path,
    clusters_path: str | Path,
    already_selected: set[str],
    n_total: int,
    weights: DifficultyWeights | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    seed: int = 1,
    with_replacement: bool = False,
) -> dict[str, list[str]]:
    """1라운드 본채점(cmcv_results.jsonl, 실측)을 기준으로 클러스터별 **진짜** 난이도를
    다시 계산해서(taster 의 클러스터당 8개짜리 노이즈 낀 추정치가 아니라), 이미 뽑힌
    already_selected 를 제외한 나머지 풀에서 n_total 개를 추가로 뽑는다.

    taster 8샘플 추정은 클러스터를 노이즈 있는 순위로 매겨서 상위권일수록 "승자의 저주"
    (regression to the mean)로 실제보다 어렵게 보이는 경향이 있음 — 실측 수백~수천건
    기준으로 재계산하면 이 왜곡이 사라진다."""
    from ocr_filter.taster.aggregate import aggregate_tier_counts  # 지연 임포트: 순환참조 방지
    # (ocr_filter.taster 패키지 __init__ 이 taster.run 을 통해 ocr_filter.ddas 를 다시
    # 임포트하므로, 모듈 최상단에 두면 ddas.__init__ 로딩 도중 순환참조가 됨)

    members = _cluster_members(clusters_path)
    id_to_cluster = {pid: int(cid) for cid, ids in members.items() for pid in ids}
    raw_counts = aggregate_tier_counts(cmcv_results_path, id_to_cluster)
    counts = {
        cid: raw_counts.get(int(cid), {"easy": 0, "medium": 0, "hard": 0})
        for cid in members
    }

    remaining = {cid: [i for i in ids if i not in already_selected] for cid, ids in members.items()}
    sizes = {cid: len(ids) for cid, ids in remaining.items()}

    scores = score_clusters(counts, weights)
    alloc = allocate_budget(scores, sizes, n_total=n_total, alpha=alpha, beta=beta)
    return sample_from_clusters(remaining, alloc, seed=seed, with_replacement=with_replacement)


def select_top_clusters_by_real_rate(
    cmcv_results_path: str | Path,
    clusters_path: str | Path,
    already_selected: set[str],
    top_n: int,
    min_samples: int = 5,
) -> dict[str, list[str]]:
    """`select_additional_ids`(스무스 가중 N_i)와 달리, 실측 medium+hard 비율이 높은
    클러스터 top_n 개를 골라 그 클러스터의 **남은 풀 전체**를 가져온다(부분 샘플링 없음).
    Easy 를 최대한 배제하고 medium/hard 밀도를 높이고 싶을 때 씀 — 스무스 가중은 낮은
    클러스터에도 조금씩 예산을 나눠줘서 easy 가 섞이지만, 이건 아예 하위 클러스터를
    건드리지 않는다.

    min_samples 미만으로 실측된 클러스터(표본 부족)는 순위 계산에서 제외한다(신뢰 못할
    비율로 상위권에 잘못 오르는 걸 방지)."""
    from ocr_filter.taster.aggregate import aggregate_tier_counts  # 지연 임포트: 순환참조 방지

    members = _cluster_members(clusters_path)
    id_to_cluster = {pid: int(cid) for cid, ids in members.items() for pid in ids}
    raw_counts = aggregate_tier_counts(cmcv_results_path, id_to_cluster)

    ranked = []
    for cid, ids in members.items():
        c = raw_counts.get(int(cid), {"easy": 0, "medium": 0, "hard": 0})
        total = c["easy"] + c["medium"] + c["hard"]
        if total < min_samples:
            continue
        rate = (c["medium"] + c["hard"]) / total
        ranked.append((rate, cid))
    ranked.sort(reverse=True)

    selected: dict[str, list[str]] = {}
    for _rate, cid in ranked[:top_n]:
        remaining = [i for i in members[cid] if i not in already_selected]
        if remaining:
            selected[cid] = remaining
    return selected


def write_selected_ids(selected: dict[str, list[str]], out_path: str | Path) -> int:
    """flatten 해서 한 줄에 id 하나씩 텍스트로 저장 (`cmcv run --ids-file` 이 읽는 형식)."""
    ids = flatten(selected)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return len(ids)
