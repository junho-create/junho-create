"""[0] io: layout_src_9984 / table_src_6902 원본 3소스(데모, GT 있음) 또는
신규 원본 PDF 묶음(GT 없음) → 통일 스키마.

    Record              → {id, image_path, gt, source_type, meta}
    load_layout_source  → layout_src_9984 파서
    load_table_source   → table_src_6902 파서
    build_unified       → 위 둘을 합쳐 JSONL 로 기록
    pdf_to_records       → 신규 원본 PDF 폴더 → 페이지 이미지 + gt=None Record
    build_unified_from_pdfs → 위를 JSONL 로 기록
    image_to_records         → 신규 원본 이미지(jpg/png) 폴더 → gt=None Record (렌더링 없음)
    build_unified_from_images → 위를 JSONL 로 기록
"""

from ocr_filter.io.build import BuildStats, build_unified
from ocr_filter.io.layout import load_layout_source
from ocr_filter.io.raw_images import build_unified_from_images, image_to_records
from ocr_filter.io.raw_pdf import build_unified_from_pdfs, pdf_to_records
from ocr_filter.io.schema import Record, read_jsonl, write_jsonl
from ocr_filter.io.table import load_table_source

__all__ = [
    "Record",
    "read_jsonl",
    "write_jsonl",
    "load_layout_source",
    "load_table_source",
    "build_unified",
    "BuildStats",
    "pdf_to_records",
    "build_unified_from_pdfs",
    "image_to_records",
    "build_unified_from_images",
]
