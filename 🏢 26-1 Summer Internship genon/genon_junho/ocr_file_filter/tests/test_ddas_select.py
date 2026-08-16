"""ddas.select 단위 테스트: taster_report.json(alloc) + clusters.jsonl(전체 페이지) →
실제 최종 id 선택.

    pytest tests/test_ddas_select.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.ddas.select import (  # noqa: E402
    select_additional_ids,
    select_final_ids,
    select_top_clusters_by_real_rate,
    write_selected_ids,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_select_final_ids_respects_allocation_and_cap():
    with tempfile.TemporaryDirectory() as tmp:
        clusters_path = Path(tmp) / "clusters.jsonl"
        # cluster "0" 페이지 10개, cluster "1" 페이지 3개
        rows = [{"id": f"p0-{i}", "level": "page", "cluster": 0} for i in range(10)]
        rows += [{"id": f"p1-{i}", "level": "page", "cluster": 1} for i in range(3)]
        rows.append({"id": "p0-0#0", "level": "element", "cluster": 5, "page_id": "p0-0"})
        _write_jsonl(clusters_path, rows)

        report_path = Path(tmp) / "taster_report.json"
        report_path.write_text(json.dumps({
            "scores": {"0": 1.5, "1": 0.5},
            "sizes": {"0": 10, "1": 3},
            "tier_counts": {"0": {"easy": 1, "medium": 3, "hard": 4},
                             "1": {"easy": 8, "medium": 0, "hard": 0}},
            "alloc": {"0": 4, "1": 5},  # cluster "1" 은 요청 5개지만 실제 3개뿐 → 캡
        }), encoding="utf-8")

        selected = select_final_ids(report_path, clusters_path, seed=0)
        assert len(selected["0"]) == 4
        assert set(selected["0"]) <= {f"p0-{i}" for i in range(10)}
        assert len(selected["1"]) == 3  # 3개뿐이라 5개 요청해도 3개까지만
        assert set(selected["1"]) == {"p1-0", "p1-1", "p1-2"}


def test_select_final_ids_n_per_cluster_overrides_alloc():
    # n_per_cluster 를 주면 taster_report.json 의 alloc(N_i) 은 무시하고 클러스터마다
    # 균일하게 뽑는다 — 스모크 테스트용 "클러스터당 1개" 같은 케이스.
    with tempfile.TemporaryDirectory() as tmp:
        clusters_path = Path(tmp) / "clusters.jsonl"
        rows = [{"id": f"p0-{i}", "level": "page", "cluster": 0} for i in range(10)]
        rows += [{"id": f"p1-{i}", "level": "page", "cluster": 1} for i in range(3)]
        _write_jsonl(clusters_path, rows)

        report_path = Path(tmp) / "taster_report.json"
        report_path.write_text(json.dumps({
            "scores": {"0": 1.5, "1": 0.5},
            "sizes": {"0": 10, "1": 3},
            "tier_counts": {"0": {"easy": 1, "medium": 3, "hard": 4},
                             "1": {"easy": 8, "medium": 0, "hard": 0}},
            "alloc": {"0": 4, "1": 3},  # n_per_cluster 가 있으면 이건 무시돼야 함
        }), encoding="utf-8")

        selected = select_final_ids(report_path, clusters_path, seed=0, n_per_cluster=1)
        assert len(selected["0"]) == 1
        assert len(selected["1"]) == 1

        # 클러스터 크기(3)보다 큰 n_per_cluster 는 |C_i| 로 캡.
        selected = select_final_ids(report_path, clusters_path, seed=0, n_per_cluster=5)
        assert len(selected["0"]) == 5
        assert len(selected["1"]) == 3


def test_select_additional_ids_excludes_already_selected_and_favors_real_hard_rate():
    with tempfile.TemporaryDirectory() as tmp:
        clusters_path = Path(tmp) / "clusters.jsonl"
        # cluster 0: 실측으로 hard 비율이 높음. cluster 1: 실측으로 전부 easy.
        rows = [{"id": f"p0-{i}", "level": "page", "cluster": 0} for i in range(20)]
        rows += [{"id": f"p1-{i}", "level": "page", "cluster": 1} for i in range(20)]
        _write_jsonl(clusters_path, rows)

        cmcv_results_path = Path(tmp) / "cmcv_results.jsonl"
        # cluster 0: 이미 채점된 5건 중 4건 hard. cluster 1: 이미 채점된 5건 전부 easy.
        results = [{"id": f"p0-{i}", "tier": "Hard"} for i in range(4)]
        results += [{"id": "p0-4", "tier": "Easy"}]
        results += [{"id": f"p1-{i}", "tier": "Easy"} for i in range(5)]
        _write_jsonl(cmcv_results_path, results)

        already_selected = {f"p0-{i}" for i in range(5)} | {f"p1-{i}" for i in range(5)}

        selected = select_additional_ids(
            cmcv_results_path, clusters_path, already_selected, n_total=10, seed=1,
        )
        all_ids = {i for ids in selected.values() for i in ids}

        # 이미 뽑힌 건 다시 안 뽑힘.
        assert all_ids.isdisjoint(already_selected)
        # hard 비율이 높은 cluster 0 이 easy 뿐인 cluster 1 보다 더 많이 배정됨.
        assert len(selected["0"]) > len(selected["1"])
        assert len(all_ids) == 10


def test_select_additional_ids_caps_at_remaining_pool_size():
    with tempfile.TemporaryDirectory() as tmp:
        clusters_path = Path(tmp) / "clusters.jsonl"
        rows = [{"id": f"p0-{i}", "level": "page", "cluster": 0} for i in range(3)]
        _write_jsonl(clusters_path, rows)

        cmcv_results_path = Path(tmp) / "cmcv_results.jsonl"
        _write_jsonl(cmcv_results_path, [{"id": "p0-0", "tier": "Hard"}])

        selected = select_additional_ids(
            cmcv_results_path, clusters_path, already_selected={"p0-0"}, n_total=100, seed=1,
        )
        # 클러스터에 남은 게 2개뿐이라 100개 요청해도 2개까지만.
        assert len(selected["0"]) == 2
        assert set(selected["0"]) == {"p0-1", "p0-2"}


def test_select_top_clusters_by_real_rate_takes_whole_remaining_pool_of_top_clusters():
    with tempfile.TemporaryDirectory() as tmp:
        clusters_path = Path(tmp) / "clusters.jsonl"
        # cluster 0: 실측 hard 비율 높음(4/5). cluster 1: 중간(2/5). cluster 2: 낮음(0/5).
        rows = [{"id": f"p0-{i}", "level": "page", "cluster": 0} for i in range(10)]
        rows += [{"id": f"p1-{i}", "level": "page", "cluster": 1} for i in range(10)]
        rows += [{"id": f"p2-{i}", "level": "page", "cluster": 2} for i in range(10)]
        _write_jsonl(clusters_path, rows)

        cmcv_results_path = Path(tmp) / "cmcv_results.jsonl"
        results = [{"id": f"p0-{i}", "tier": "Hard" if i < 4 else "Easy"} for i in range(5)]
        results += [{"id": f"p1-{i}", "tier": "Hard" if i < 2 else "Easy"} for i in range(5)]
        results += [{"id": f"p2-{i}", "tier": "Easy"} for i in range(5)]
        _write_jsonl(cmcv_results_path, results)

        # top 2 만 뽑으면 cluster 0, 1 만 포함되고 cluster 2(가장 낮음)는 아예 안 뽑혀야 함.
        selected = select_top_clusters_by_real_rate(
            cmcv_results_path, clusters_path, already_selected=set(), top_n=2,
        )
        assert set(selected) == {"0", "1"}
        # 부분 샘플링이 아니라 남은 풀 전체를 가져옴.
        assert set(selected["0"]) == {f"p0-{i}" for i in range(10)}
        assert set(selected["1"]) == {f"p1-{i}" for i in range(10)}


def test_select_top_clusters_by_real_rate_excludes_already_selected():
    with tempfile.TemporaryDirectory() as tmp:
        clusters_path = Path(tmp) / "clusters.jsonl"
        rows = [{"id": f"p0-{i}", "level": "page", "cluster": 0} for i in range(10)]
        _write_jsonl(clusters_path, rows)

        cmcv_results_path = Path(tmp) / "cmcv_results.jsonl"
        _write_jsonl(cmcv_results_path, [{"id": f"p0-{i}", "tier": "Hard"} for i in range(5)])

        already = {f"p0-{i}" for i in range(5)}
        selected = select_top_clusters_by_real_rate(
            cmcv_results_path, clusters_path, already_selected=already, top_n=5,
        )
        assert set(selected["0"]) == {f"p0-{i}" for i in range(5, 10)}


def test_select_top_clusters_by_real_rate_skips_clusters_with_too_few_samples():
    with tempfile.TemporaryDirectory() as tmp:
        clusters_path = Path(tmp) / "clusters.jsonl"
        # cluster 0: 표본 2개뿐(전부 hard 라 비율은 100%지만 신뢰 못함) -> 제외돼야 함.
        # cluster 1: 표본 5개, hard 비율 40%.
        rows = [{"id": f"p0-{i}", "level": "page", "cluster": 0} for i in range(2)]
        rows += [{"id": f"p1-{i}", "level": "page", "cluster": 1} for i in range(5)]
        _write_jsonl(clusters_path, rows)

        cmcv_results_path = Path(tmp) / "cmcv_results.jsonl"
        results = [{"id": f"p0-{i}", "tier": "Hard"} for i in range(2)]
        results += [{"id": f"p1-{i}", "tier": "Hard" if i < 2 else "Easy"} for i in range(5)]
        _write_jsonl(cmcv_results_path, results)

        selected = select_top_clusters_by_real_rate(
            cmcv_results_path, clusters_path, already_selected=set(), top_n=5, min_samples=5,
        )
        assert set(selected) == {"1"}


def test_write_selected_ids_flattens_to_one_id_per_line():
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "selected.txt"
        n = write_selected_ids({"0": ["a", "b"], "1": ["c"]}, out_path)
        assert n == 3
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert set(lines) == {"a", "b", "c"}
