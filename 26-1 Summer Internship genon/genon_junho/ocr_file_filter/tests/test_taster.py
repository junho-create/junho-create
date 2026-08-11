"""taster 단위 테스트: 합성 clusters.jsonl/cmcv_results.jsonl 로 sample/aggregate 검증.

    pytest tests/test_taster.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.taster.aggregate import aggregate_tier_counts, cluster_sizes  # noqa: E402
from ocr_filter.taster.sample import (  # noqa: E402
    flatten,
    load_page_clusters,
    sample_taster_ids,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _make_clusters(path: Path) -> None:
    rows = []
    # cluster 0: 페이지 10개, cluster 1: 페이지 3개 (n_per_cluster 캡 테스트용)
    for i in range(10):
        rows.append({"id": f"p0-{i}", "level": "page", "cluster": 0})
    for i in range(3):
        rows.append({"id": f"p1-{i}", "level": "page", "cluster": 1})
    rows.append({"id": "p0-0#0", "level": "element", "cluster": 5, "page_id": "p0-0"})
    _write_jsonl(path, rows)


def test_load_page_clusters_ignores_element_level():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clusters.jsonl"
        _make_clusters(path)
        mapping = load_page_clusters(path)
        assert len(mapping) == 13  # 10 + 3, element 라인 제외
        assert mapping["p0-0"] == 0
        assert mapping["p1-0"] == 1


def test_sample_taster_ids_caps_per_cluster():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clusters.jsonl"
        _make_clusters(path)
        picked = sample_taster_ids(path, n_per_cluster=4, seed=0)
        assert len(picked[0]) == 4       # 10개 중 4개만
        assert len(picked[1]) == 3       # 3개뿐이라 캡에 안 걸리고 전부
        assert set(picked[0]) <= {f"p0-{i}" for i in range(10)}


def test_flatten_unions_all_clusters():
    picked = {0: ["a", "b"], 1: ["c"]}
    assert flatten(picked) == {"a", "b", "c"}


def test_aggregate_tier_counts_groups_by_cluster():
    with tempfile.TemporaryDirectory() as tmp:
        cmcv_path = Path(tmp) / "cmcv.jsonl"
        _write_jsonl(cmcv_path, [
            {"id": "p0-0", "tier": "Easy"},
            {"id": "p0-1", "tier": "Medium"},
            {"id": "p0-2", "tier": "Hard"},
            {"id": "p1-0", "tier": "Easy"},
        ])
        id_to_cluster = {"p0-0": 0, "p0-1": 0, "p0-2": 0, "p1-0": 1}
        counts = aggregate_tier_counts(cmcv_path, id_to_cluster)
        assert counts[0] == {"easy": 1, "medium": 1, "hard": 1}
        assert counts[1] == {"easy": 1, "medium": 0, "hard": 0}


def test_cluster_sizes_counts_pages_only():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clusters.jsonl"
        _make_clusters(path)
        sizes = cluster_sizes(path)
        assert sizes == {0: 10, 1: 3}
