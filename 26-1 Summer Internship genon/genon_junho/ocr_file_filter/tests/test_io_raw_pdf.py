"""io/raw_pdf.py 단위 테스트: GT 없는 신규 원본 PDF 폴더 → 페이지 이미지 + Record(gt=None).

    pytest tests/test_io_raw_pdf.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz  # noqa: E402

from ocr_filter.io.raw_pdf import build_unified_from_pdfs, pdf_to_records  # noqa: E402
from ocr_filter.io.schema import read_jsonl  # noqa: E402


def _make_pdf(path: Path, n_pages: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), "hello")
    doc.save(str(path))
    doc.close()


def test_pdf_to_records_extracts_pages_with_no_gt():
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        _make_pdf(input_dir / "company_a" / "doc1.pdf", n_pages=2)
        images_out = Path(tmp) / "images"

        records = list(pdf_to_records(input_dir, images_out))
        assert len(records) == 2
        ids = {r.id for r in records}
        assert ids == {"company_a/doc1/doc1_page_001", "company_a/doc1/doc1_page_002"}
        for r in records:
            assert r.gt is None
            assert r.source_type == "layout"
            assert Path(r.image_path).exists()
            assert r.meta["image_exists"] is True


def test_pdf_to_records_is_idempotent_on_rerun():
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        _make_pdf(input_dir / "doc1.pdf", n_pages=1)
        images_out = Path(tmp) / "images"

        first = list(pdf_to_records(input_dir, images_out))
        mtime_before = Path(first[0].image_path).stat().st_mtime
        second = list(pdf_to_records(input_dir, images_out))
        mtime_after = Path(second[0].image_path).stat().st_mtime

        assert len(first) == len(second) == 1
        assert mtime_before == mtime_after  # 이미 있으면 재변환 안 함


def test_pdf_to_records_case_insensitive_extension():
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        _make_pdf(input_dir / "doc1.PDF", n_pages=1)
        images_out = Path(tmp) / "images"

        records = list(pdf_to_records(input_dir, images_out))
        assert len(records) == 1


def test_pdf_to_records_skips_non_pdf_files():
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        input_dir.mkdir(parents=True)
        (input_dir / "notes.txt").write_text("not a pdf")
        (input_dir / "data.xlsx").write_bytes(b"\x00")
        _make_pdf(input_dir / "doc1.pdf", n_pages=1)
        images_out = Path(tmp) / "images"

        records = list(pdf_to_records(input_dir, images_out))
        assert len(records) == 1
        assert records[0].id == "doc1/doc1_page_001"


def test_pdf_to_records_skips_corrupt_pdf_without_crashing():
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        input_dir.mkdir(parents=True)
        (input_dir / "broken.pdf").write_bytes(b"not a real pdf")
        _make_pdf(input_dir / "good.pdf", n_pages=1)
        images_out = Path(tmp) / "images"

        records = list(pdf_to_records(input_dir, images_out))
        assert len(records) == 1
        assert records[0].id == "good/good_page_001"


def test_build_unified_from_pdfs_writes_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        _make_pdf(input_dir / "doc1.pdf", n_pages=1)
        images_out = Path(tmp) / "images"
        out_path = Path(tmp) / "unified.jsonl"

        n = build_unified_from_pdfs(input_dir, images_out, out_path)
        assert n == 1
        records = list(read_jsonl(out_path))
        assert len(records) == 1
        assert records[0].gt is None
