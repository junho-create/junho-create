"""라벨/GT 없는 **신규 원본 PDF 묶음** → 통일 스키마.

`layout.py`/`table.py`는 이미 파싱+GT가 있는 데모 데이터(jhshin 쪽 사전 처리 결과) 전용이라,
사람이 그냥 폴더에 넣어둔 원본 PDF는 못 읽는다. 이 모듈이 그 갭을 메운다 — CMCV/DDAS 설계가
원래 상정한 "GT 없는 새 데이터"가 실제로 들어오는 지점(PDF 페이지를 이미지로 뽑고 gt=None인
Record 를 만든다. GT가 없어도 CMCV 쌍별 동의도 기반 티어 판정에는 전혀 지장 없음 —
`cmcv/run.py` 참고).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from ocr_filter.io.schema import Record


def _find_pdfs(input_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    pdfs: list[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".pdf" and p not in seen:
            seen.add(p)
            pdfs.append(p)
    return sorted(pdfs)


def pdf_to_records(
    input_dir: str | Path, images_out_dir: str | Path, dpi: int = 200,
) -> Iterator[Record]:
    """input_dir 아래 재귀적으로 *.pdf(확장자 대소문자 무관) 를 찾아 페이지별 PNG 로 뽑고
    gt=None Record 를 만든다.

    - id / 이미지 경로 모두 `input_dir` 기준 상대경로 구조를 그대로 보존한다
      (`{rel_dir}/{pdf_stem}_page_{NNN:03d}`) — 다른 PDF의 동일 파일명과 안 섞이게.
    - 이미 뽑힌 페이지 PNG 는 재변환하지 않는다(idempotent, 중단 후 재실행 가능).
    - 손상된 PDF 는 건너뛰고 계속 진행한다(경고만 출력, 전체 작업을 죽이지 않음).
    """
    input_dir = Path(input_dir)
    images_out_dir = Path(images_out_dir)

    for pdf_path in _find_pdfs(input_dir):
        rel_dir = pdf_path.parent.relative_to(input_dir)
        stem = pdf_path.stem
        out_dir = images_out_dir / rel_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:  # noqa: BLE001 — 손상된 PDF 는 스킵하고 계속
            print(f"[raw_pdf] PDF 열기 실패, 스킵: {pdf_path} ({type(e).__name__}: {e})")
            continue

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        try:
            for page_index in range(doc.page_count):
                page_num = page_index + 1
                image_name = f"{stem}_page_{page_num:03d}.png"
                image_path = out_dir / image_name
                if not image_path.exists():
                    pix = doc[page_index].get_pixmap(matrix=matrix)
                    pix.save(str(image_path))

                record_id = str((rel_dir / stem / f"{stem}_page_{page_num:03d}").as_posix())
                yield Record(
                    id=record_id,
                    image_path=str(image_path),
                    gt=None,
                    source_type="layout",
                    meta={
                        "raw_pdf_path": str(pdf_path),
                        "page_index": page_index,
                        "image_exists": image_path.exists(),
                    },
                )
        finally:
            doc.close()


def build_unified_from_pdfs(
    input_dir: str | Path, images_out_dir: str | Path, out_path: str | Path, dpi: int = 200,
) -> int:
    """`pdf_to_records` 결과를 그대로 JSONL 로 기록. 반환값은 총 페이지(레코드) 수."""
    from ocr_filter.io.schema import write_jsonl

    return write_jsonl(pdf_to_records(input_dir, images_out_dir, dpi=dpi), out_path)
