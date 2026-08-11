"""cluster/build.py 단위 테스트: n_clusters_element<=0 이면 요소(2단계) 클러스터링을
통째로 스킵해야 함 (taster/ddas/cmcv/report 전부 level=="page" 만 쓰고 level=="element" 는
안 쓰므로, 필요 없을 땐 크롭+임베딩 계산을 아끼려는 최적화).

    pytest tests/test_cluster_build.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ocr_filter.io.schema import Record, write_jsonl  # noqa: E402

# embed_images 는 build.py 안에서 이미 임포트해서 바인딩했으므로 거기를 패치해야 함.
import ocr_filter.cluster.build as build_mod  # noqa: E402


def _fake_embed_images(images, **kwargs):
    n = len(images)
    rng = np.random.RandomState(0)
    return rng.randn(n, 8) if n else np.zeros((0, 8))


def _make_records(tmp: Path, n: int, with_gt_elements: bool) -> list[Record]:
    records = []
    for i in range(n):
        img_path = tmp / f"page_{i}.png"
        Image.new("RGB", (100, 100), "white").save(img_path)
        gt = [{"bbox": [0, 0, 50, 50], "category": "Text"}] if with_gt_elements else None
        records.append(Record(id=f"p{i}", image_path=str(img_path), gt=gt,
                               source_type="layout", meta={}))
    return records


def test_build_clusters_skips_element_stage_when_n_clusters_element_is_zero():
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        records = _make_records(tmp, 5, with_gt_elements=True)  # GT 있는데도
        unified_path = tmp / "unified.jsonl"
        write_jsonl(records, unified_path)
        out_path = tmp / "clusters.jsonl"

        with patch.object(build_mod, "embed_images", side_effect=_fake_embed_images) as mock_embed:
            stats = build_mod.build_clusters(
                unified_path, out_path, n_clusters_page=2, n_clusters_element=0,
            )

        assert stats.n_elements == 0
        # embed_images 는 페이지 임베딩 1회만 호출돼야(요소용 두 번째 호출 없음).
        assert mock_embed.call_count == 1

        lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        assert all(d["level"] == "page" for d in lines)
        assert len(lines) == 5


def test_build_clusters_still_does_element_stage_when_requested_and_gt_present():
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        records = _make_records(tmp, 5, with_gt_elements=True)
        unified_path = tmp / "unified.jsonl"
        write_jsonl(records, unified_path)
        out_path = tmp / "clusters.jsonl"

        with patch.object(build_mod, "embed_images", side_effect=_fake_embed_images):
            stats = build_mod.build_clusters(
                unified_path, out_path, n_clusters_page=2, n_clusters_element=2,
            )

        assert stats.n_elements == 5  # 레코드마다 gt element 1개씩
        lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        assert any(d["level"] == "element" for d in lines)


def test_build_clusters_element_stage_naturally_empty_when_gt_is_none():
    # n_clusters_element>0 이어도 gt=None(신규 원본 PDF)이면 크롭할 게 없어서 결과가 빈다.
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        records = _make_records(tmp, 3, with_gt_elements=False)
        unified_path = tmp / "unified.jsonl"
        write_jsonl(records, unified_path)
        out_path = tmp / "clusters.jsonl"

        with patch.object(build_mod, "embed_images", side_effect=_fake_embed_images):
            stats = build_mod.build_clusters(
                unified_path, out_path, n_clusters_page=2, n_clusters_element=32,
            )

        assert stats.n_elements == 0
