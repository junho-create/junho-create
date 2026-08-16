"""[0] io 파서 단위 테스트: 합성 fixture 로 layout/table 소스 → 통일 스키마 검증.

    pytest tests/test_io.py
    python tests/test_io.py    # 임시 디렉터리에 fixture 만들고 build_unified 시연
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.io import build_unified, load_layout_source, load_table_source  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_fixture(root: Path) -> dict:
    """layout_src_9984 / table_src_6902 원본 구조를 흉내낸 합성 fixture."""
    images_root = root / "images"
    (images_root / "layout").mkdir(parents=True, exist_ok=True)
    (images_root / "table").mkdir(parents=True, exist_ok=True)
    (images_root / "layout" / "doc_page_001.png").write_bytes(b"fake-png")
    (images_root / "table" / "T01.jpg").write_bytes(b"fake-jpg")
    # table:test:T02 는 이미지가 없는 케이스 (image_exists=False 검증용)

    layout_jsonl = root / "layout_src" / "layout.jsonl"
    _write_jsonl(layout_jsonl, [
        {
            "id": "doc_page_001",
            "image_path": "output/full/doc_page_001.png",  # 원본 경로 (더 이상 실존 안 함)
            "layout_elements": [{"bbox": [0, 0, 100, 50], "category": "Title", "text": "hi"}],
            "status": "CONVERTED",
            "convert_path": "output/batch/doc_page_001.convert.json",
            "page_index": None,
        },
    ])

    table_dir = root / "table_src"
    _write_jsonl(table_dir / "data" / "train.jsonl", [
        {
            "image_path": "images/T01.jpg",
            "gt_html": "<table><tr><td>a</td></tr></table>",
            "original_gt_html": "<table><tr><td>a </td></tr></table>",
            "complexity": "simple",
            "prompt_style": "chandra_table_with_ocr",
            "ocr_info": [{"text": "a", "bbox": [0, 0, 10, 10]}],
        },
    ])
    _write_jsonl(table_dir / "data" / "test.jsonl", [
        {
            "image_path": "images/T02.jpg",   # 이미지 파일 없음
            "gt_html": "<table><tr><td>b</td></tr></table>",
        },
    ])

    return {"layout_src": str(layout_jsonl), "table_src": str(table_dir),
            "images": str(images_root)}


def test_load_layout_source_maps_to_images_layout_dir():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fixture(Path(tmp))
        recs = list(load_layout_source(paths["layout_src"], paths["images"]))
        assert len(recs) == 1
        r = recs[0]
        assert r.source_type == "layout"
        assert r.id == "doc_page_001"
        assert r.image_path.endswith("images/layout/doc_page_001.png")
        assert r.meta["image_exists"] is True
        assert r.gt == [{"bbox": [0, 0, 100, 50], "category": "Title", "text": "hi"}]


def test_load_table_source_reads_all_splits_and_flags_missing_image():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fixture(Path(tmp))
        recs = list(load_table_source(paths["table_src"], paths["images"]))
        by_id = {r.id: r for r in recs}
        assert set(by_id) == {"table:train:T01", "table:test:T02"}
        assert by_id["table:train:T01"].meta["image_exists"] is True
        assert by_id["table:train:T01"].gt == "<table><tr><td>a</td></tr></table>"
        assert by_id["table:test:T02"].meta["image_exists"] is False


def test_build_unified_combines_both_sources():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fixture(Path(tmp))
        out = Path(tmp) / "unified.jsonl"
        stats = build_unified(paths, out)
        assert stats.n_layout == 1
        assert stats.n_table == 2
        assert stats.n_total == 3
        assert stats.n_missing_images == 1  # table:test:T02
        assert out.exists()
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3


def _demo():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fixture(Path(tmp))
        stats = build_unified(paths, Path(tmp) / "unified.jsonl")
        print(f"layout={stats.n_layout} table={stats.n_table} "
              f"total={stats.n_total} missing_images={stats.n_missing_images}")


if __name__ == "__main__":
    _demo()
