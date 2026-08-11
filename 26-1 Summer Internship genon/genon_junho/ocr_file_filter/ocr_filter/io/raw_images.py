"""라벨/GT 없는 **신규 이미지 묶음**(jpg/png, PDF 아님) → 통일 스키마.

`raw_pdf.py`는 PDF 를 페이지별 PNG 로 렌더링하는 단계가 있는데, 이번 5,400장 배치
(`additional_dataset/5400dataextract`)는 이미 낱장 jpg 로 추출돼 있어서 렌더링이
필요 없다 — 디렉터리를 재귀 스캔해 그대로 gt=None Record 로 감싸기만 하면 된다.

이미지 자체는 건드리지 않는다(복사/변환 없음, 원본 경로를 그대로 image_path 로 씀).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image

from ocr_filter.io.schema import Record, write_jsonl

Image.MAX_IMAGE_PIXELS = None

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _find_images(input_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS and p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def image_to_records(input_dir: str | Path, corpus_label: bool = True) -> Iterator[Record]:
    """input_dir 아래 재귀적으로 이미지 파일을 찾아 gt=None Record 를 만든다.

    - id 는 `input_dir` 기준 상대경로(확장자 제외, `/` 구분자로 통일) — 서로 다른
      코퍼스의 동일 파일명이 안 섞이게(예: `023.OCR 데이터(공공)/.../page_0001`).
    - `meta.corpus`: input_dir 바로 아래 1단계 디렉터리 이름(코퍼스 구분용, DDAS/리포트가
      코퍼스별 통계를 낼 때 씀). corpus_label=False 면 채우지 않는다.
    - 손상된 이미지(PIL 이 못 여는 파일)는 건너뛰고 계속 진행한다(경고만 출력) — 60,000장
      규모 배치에서 파일 몇 개 깨진 걸로 전체를 죽이지 않으려는 raw_pdf.py 와 동일한 정책.
    """
    input_dir = Path(input_dir)

    for img_path in _find_images(input_dir):
        rel = img_path.relative_to(input_dir)
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001 — 손상 이미지는 스킵하고 계속
            print(f"[raw_images] 이미지 열기 실패, 스킵: {img_path} ({type(e).__name__}: {e})")
            continue

        record_id = str(rel.with_suffix("").as_posix())
        meta = {"image_exists": True}
        if corpus_label:
            meta["corpus"] = rel.parts[0] if rel.parts else ""
        yield Record(
            id=record_id,
            image_path=str(img_path),
            gt=None,
            source_type="layout",
            meta=meta,
        )


def build_unified_from_images(
    input_dir: str | Path, out_path: str | Path, corpus_label: bool = True,
) -> int:
    """`image_to_records` 결과를 그대로 JSONL 로 기록. 반환값은 총 레코드 수."""
    return write_jsonl(image_to_records(input_dir, corpus_label=corpus_label), out_path)
