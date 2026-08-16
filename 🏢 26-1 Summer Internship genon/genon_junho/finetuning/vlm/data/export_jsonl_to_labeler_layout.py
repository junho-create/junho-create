#!/usr/bin/env python3
"""JSONL(레이아웃) → labeler Layout task용 디렉터리 트리 export.

기존 ``_train_data`` 는 건드리지 않고, labeler 프로젝트에서 바로 가리킬 수 있는
``input/`` · ``output/`` · ``gt/`` · ``config/`` 트리를 새로 만든다.

대상 labeler 파이프라인 (Layout)
--------------------------------
1. ``convert_only`` + ``converter.source: gt_replay``
   - VLM convert 없이 GT JSON → render(``annotated.png``) 생성
2. ``evaluate_only``
   - ``annotated.png`` 가 있으면 evaluator만 실행 → ``review.html``

권장 입력 JSONL
---------------
- ``layout_src_9984/labeler_converter_layout_source_9984.jsonl``
  (``layout_elements`` 픽셀 bbox — labeler ``convert.json`` 과 동일)
- 또는 Chandra layout split (``gt_html`` 이 div HTML / JSON 배열)

출력 구조 (``{out_dir}``)::

    labeler_ready_layout/
    ├── README.md
    ├── config/
    │   ├── layout_gt_replay.yaml      # convert_only + gt_replay (render만)
    │   └── layout_evaluate_only.yaml  # evaluate_only
    ├── input/pdfs/...                 # collect_inputs 용 PDF (symlink 또는 1페이지 PDF)
    ├── output/layout/{id}/            # labeler output_key 와 동일
    │   ├── {stem_page}.png            # → 원본 PNG symlink
    │   └── attempt_1/
    │       └── {stem_page}.convert.json
    ├── gt/layout/...                  # gt_replay 용 (convert.json 과 동일 내용)
    └── meta/
        ├── manifest.jsonl
        └── export_summary.json

사용 예 (b200)::

    cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
    source .venv/bin/activate

    python -m data.export_jsonl_to_labeler_layout \\
        --train-data-root ./_train_data \\
        --jsonl ./_train_data/layout_src_9984/labeler_converter_layout_source_9984.jsonl \\
        --image-root ./_train_data/chandra_table_layout_divhtml_16886 \\
        --out-dir ./_train_data/labeler_ready_layout \\
        --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

logger = logging.getLogger(__name__)

_PAGE_SUFFIX_RE = re.compile(r"_page_(\d+)$", re.IGNORECASE)
_DIV_RE = re.compile(
    r'<div\b[^>]*\bdata-bbox="([^"]+)"[^>]*\bdata-label="([^"]+)"[^>]*>(.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ExportStats:
    rows_read: int = 0
    exported: int = 0
    skipped_no_image: int = 0
    skipped_no_elements: int = 0
    pdf_symlink: int = 0
    pdf_single_page: int = 0
    pdf_multipage_stub: int = 0
    errors: list[str] = field(default_factory=list)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON parse error: {exc}") from exc
    return rows


def _resolve_image(image_path: str, roots: list[Path]) -> Optional[Path]:
    raw = Path(image_path)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    for root in roots:
        candidates.extend(
            [
                root / raw,
                root / raw.name,
                root / "images" / raw.name,
                root / "images" / "layout" / raw.name,
                root / "images" / "table" / raw.name,
            ]
        )
        if raw.parts and raw.parts[0] == "images":
            candidates.append(root / Path(*raw.parts))
            candidates.append(root / Path(*raw.parts[1:]))
    candidates.append(raw)
    seen: set[Path] = set()
    for c in candidates:
        c = c.expanduser()
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            return c.resolve()
    return None


def _image_size(path: Path) -> Optional[tuple[int, int]]:
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def _parse_bbox_values(raw: str | list[Any]) -> Optional[list[float]]:
    if isinstance(raw, list):
        if len(raw) < 4:
            return None
        try:
            return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
        except (TypeError, ValueError):
            return None
    text = str(raw or "").strip()
    if not text:
        return None
    parts = re.split(r"[\s,]+", text)
    if len(parts) < 4:
        return None
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
    except ValueError:
        return None


def _to_pixel_bbox(
    bbox: list[float],
    width: int,
    height: int,
    *,
    normalized_scale: Optional[int],
) -> list[int]:
    x0, y0, x1, y1 = bbox
    if normalized_scale is not None and max(abs(v) for v in bbox) <= normalized_scale + 1:
        w = max(1, width)
        h = max(1, height)
        s = float(normalized_scale)

        def px(v: float, dim: int) -> int:
            return int(max(0, min(dim, round(v / s * dim))))

        px0, py0 = px(x0, w), px(y0, h)
        px1, py1 = px(x1, w), px(y1, h)
    else:
        px0, py0, px1, py1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    if px1 < px0:
        px0, px1 = px1, px0
    if py1 < py0:
        py0, py1 = py1, py0
    return [px0, py0, px1, py1]


def _elements_from_record(
    rec: dict[str, Any],
    width: int,
    height: int,
) -> Optional[list[dict[str, Any]]]:
    elements_in = rec.get("layout_elements")
    if isinstance(elements_in, list) and elements_in:
        out: list[dict[str, Any]] = []
        for el in elements_in:
            if not isinstance(el, dict):
                continue
            bbox = _parse_bbox_values(el.get("bbox", []))
            if bbox is None:
                continue
            category = str(el.get("category") or el.get("label") or "Text").strip() or "Text"
            text = str(el.get("text") or "")
            out.append(
                {
                    "bbox": _to_pixel_bbox(bbox, width, height, normalized_scale=None),
                    "category": category,
                    "text": text,
                }
            )
        return out or None

    layout_json = rec.get("layout_json")
    if isinstance(layout_json, str) and layout_json.strip():
        try:
            payload = json.loads(layout_json)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            rec2 = dict(rec)
            rec2["layout_elements"] = payload
            return _elements_from_record(rec2, width, height)

    gt_html = str(rec.get("gt_html") or "").strip()
    if not gt_html:
        return None

    if gt_html.startswith("["):
        try:
            payload = json.loads(gt_html)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            scale = int(rec.get("bbox_scale") or 1024)
            out = []
            for el in payload:
                if not isinstance(el, dict):
                    continue
                bbox = _parse_bbox_values(el.get("bbox", []))
                if bbox is None:
                    continue
                category = str(el.get("category") or el.get("label") or "Text").strip() or "Text"
                text = str(el.get("text") or "")
                out.append(
                    {
                        "bbox": _to_pixel_bbox(bbox, width, height, normalized_scale=scale),
                        "category": category,
                        "text": text,
                    }
                )
            return out or None

    scale = int(rec.get("bbox_scale") or 1000)
    out = []
    for m in _DIV_RE.finditer(gt_html):
        bbox = _parse_bbox_values(m.group(1))
        if bbox is None:
            continue
        category = m.group(2).strip() or "Text"
        content = m.group(3).strip()
        if category.lower() == "table":
            text = content
        else:
            text = re.sub(r"<[^>]+>", "", content)
            text = re.sub(r"\s+", " ", text).strip()
        out.append(
            {
                "bbox": _to_pixel_bbox(bbox, width, height, normalized_scale=scale),
                "category": category,
                "text": text,
            }
        )
    return out or None


def _derive_ids(rec: dict[str, Any], image_path: Path) -> tuple[str, str, int]:
    """Return (output_key, stem_page, page_index)."""
    raw_id = str(rec.get("id") or "").strip()
    if raw_id:
        output_key = raw_id.replace("\\", "/").strip("/")
        stem_page = Path(output_key).name
        m = _PAGE_SUFFIX_RE.search(stem_page)
        page_index = int(m.group(1)) if m else int(rec.get("page_index") or 1)
        return output_key, stem_page, page_index

    stem = image_path.stem
    m = _PAGE_SUFFIX_RE.search(stem)
    page_index = int(m.group(1)) if m else int(rec.get("page_index") or 1)
    stem_page = stem if m else f"{stem}_page_{page_index:03d}"
    output_key = f"pages/{stem_page}"
    return output_key, stem_page, page_index


def _link_or_copy(src: Path, dst: Path, *, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def _resolve_pdf(pdf_path_raw: str, search_roots: list[Path]) -> Optional[Path]:
    raw = Path(pdf_path_raw.strip())
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    for root in search_roots:
        candidates.extend([root / raw, root / raw.name])
    candidates.append(raw)
    seen: set[Path] = set()
    for c in candidates:
        c = c.expanduser()
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            return c.resolve()
    return None


def _ensure_pdf_for_item(
    *,
    rec: dict[str, Any],
    image_path: Path,
    input_pdfs_root: Path,
    output_key: str,
    stem_page: str,
    page_index: int,
    use_symlink: bool,
    pdf_search_roots: list[Path],
    stats: ExportStats,
) -> Optional[Path]:
    pdf_path_raw = rec.get("pdf_path")
    if isinstance(pdf_path_raw, str) and pdf_path_raw.strip():
        src_pdf = _resolve_pdf(pdf_path_raw, pdf_search_roots)
        if src_pdf is not None:
            parts = Path(output_key).parts
            pdf_stem = stem_page.rsplit("_page_", 1)[0] if "_page_" in stem_page else stem_page
            rel_dir = Path(*parts[:-1]) if len(parts) > 1 else Path(".")
            dst_pdf = input_pdfs_root / rel_dir / f"{pdf_stem}.pdf"
            _link_or_copy(src_pdf, dst_pdf, use_symlink=use_symlink)
            stats.pdf_symlink += 1
            return dst_pdf

    # PNG only → labeler collect_inputs 용 PDF 생성
    pdf_stem = stem_page.rsplit("_page_", 1)[0] if "_page_" in stem_page else stem_page
    rel_dir = Path(*Path(output_key).parts[:-1]) if "/" in output_key else Path("_single_page")
    dst_pdf = input_pdfs_root / rel_dir / f"{pdf_stem}.pdf"
    dst_pdf.parent.mkdir(parents=True, exist_ok=True)

    size = _image_size(image_path)
    if size is None:
        stats.errors.append(f"image size unknown: {image_path}")
        return None
    width, height = size

    if page_index <= 1:
        if Image is None:
            stats.errors.append(f"Pillow required for PDF export: {stem_page}")
            return None
        with Image.open(image_path) as im:
            im.convert("RGB").save(dst_pdf, "PDF")
        stats.pdf_single_page += 1
        return dst_pdf

    if fitz is None:
        stats.errors.append(
            f"PyMuPDF required for page_index={page_index} stub PDF: {stem_page}"
        )
        return None

    doc = fitz.open()
    try:
        for idx in range(1, page_index + 1):
            page = doc.new_page(width=width, height=height)
            if idx == page_index:
                page.insert_image(fitz.Rect(0, 0, width, height), filename=str(image_path))
        doc.save(dst_pdf)
    finally:
        doc.close()
    stats.pdf_multipage_stub += 1
    return dst_pdf


def _write_yaml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.strip() + "\n", encoding="utf-8")


def _write_labeler_configs(out_dir: Path, *, project_root_hint: str) -> None:
    cfg_dir = out_dir / "config"
    gt_replay = f"""
# labeler Layout — GT replay render (VLM convert 없이 annotated.png 생성)
# labeler 프로젝트 루트에서:
#   labeler --config {project_root_hint}/config/layout_gt_replay.yaml
task: "layout"

input_dir: "{out_dir.as_posix()}/input/pdfs"
output_dir: "{out_dir.as_posix()}/output/layout"
results_dir: "{out_dir.as_posix()}/results/layout"
recursive: true

pdf_loader:
  dpi: 200
  skip_existing_page_dumps: true

converter:
  source: "gt_replay"

gt_replay:
  auto_start: true
  gt_dir: "{out_dir.as_posix()}/gt/layout"

converter_vlm:
  provider: "openrouter"
  base_url: "https://openrouter.ai/api/v1"
  api_key_env: "OPENROUTER_API_KEY"
  model: "unused/gt-replay"
  max_tokens: -1
  temperature: 0.0

evaluation:
  mode: "single"
  max_retries: 0

processing:
  pipeline_mode: "convert_only"
  parallel_workers: 8
  skip_already_passed: false
  retry_failed: false
  save_intermediate: true
"""
    evaluate_only = f"""
# labeler Layout — evaluate_only (annotated.png 필요)
# 1) layout_gt_replay.yaml 로 render 먼저 생성
# 2) labeler --config {project_root_hint}/config/layout_evaluate_only.yaml
task: "layout"

input_dir: "{out_dir.as_posix()}/input/pdfs"
output_dir: "{out_dir.as_posix()}/output/layout"
results_dir: "{out_dir.as_posix()}/results/layout_eval"
recursive: true

pdf_loader:
  dpi: 200
  skip_existing_page_dumps: true

converter:
  source: "gt_replay"

gt_replay:
  auto_start: true
  gt_dir: "{out_dir.as_posix()}/gt/layout"

evaluator_vlm_b:
  provider: "openrouter"
  base_url: "https://openrouter.ai/api/v1"
  api_key_env: "OPENROUTER_API_KEY"
  model: "qwen/qwen3.5-397b-a17b"
  max_tokens: 8192
  temperature: 0.2

evaluation:
  mode: "single"
  max_retries: 0

processing:
  pipeline_mode: "evaluate_only"
  parallel_workers: 4
  skip_already_passed: false
  retry_failed: false
  save_intermediate: true
"""
    _write_yaml(cfg_dir / "layout_gt_replay.yaml", gt_replay)
    _write_yaml(cfg_dir / "layout_evaluate_only.yaml", evaluate_only)


def _write_readme(out_dir: Path, *, source_jsonl: Path, exported: int) -> None:
    text = f"""# labeler_ready_layout

``_train_data`` 원본은 그대로 두고, **labeler Layout 입력용** 트리만 만든 폴더입니다.
labeler config 에서 아래 경로를 가리키면 됩니다 (b200 절대경로 기준).

- Source JSONL: ``{source_jsonl}``
- Exported pages: **{exported}**

## labeler config 에 넣을 경로

```yaml
task: layout
input_dir: "{out_dir.as_posix()}/input/pdfs"
output_dir: "{out_dir.as_posix()}/output/layout"
gt_replay:
  gt_dir: "{out_dir.as_posix()}/gt/layout"
```

## 디렉터리 구조

| 경로 | 내용 |
|------|------|
| ``input/pdfs/`` | 페이지별 collect_inputs 용 PDF |
| ``output/layout/{{id}}/`` | ``{{stem}}.png`` + ``attempt_1/{{stem}}.convert.json`` |
| ``gt/layout/`` | convert.json 과 동일 GT JSON (gt_replay 용) |
| ``meta/manifest.jsonl`` | id ↔ 이미지/convert 경로 매핑 |

## 재생성

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
OVERWRITE=1 bash scripts/export_labeler_layout.sh
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def export_jsonl_to_labeler(
    *,
    jsonl_path: Path,
    image_roots: list[Path],
    out_dir: Path,
    max_samples: Optional[int] = None,
    use_symlink: bool = True,
    overwrite: bool = False,
) -> ExportStats:
    stats = ExportStats()
    rows = _read_jsonl(jsonl_path)
    stats.rows_read = len(rows)

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    output_layout = out_dir / "output" / "layout"
    gt_layout = out_dir / "gt" / "layout"
    input_pdfs = out_dir / "input" / "pdfs"
    meta_dir = out_dir / "meta"
    manifest_path = meta_dir / "manifest.jsonl"
    meta_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    for rec in rows:
        if max_samples is not None and stats.exported >= max_samples:
            break

        image_field = rec.get("image_path")
        if not isinstance(image_field, str) or not image_field.strip():
            stats.skipped_no_image += 1
            continue

        image_path = _resolve_image(image_field, image_roots)
        if image_path is None:
            stats.skipped_no_image += 1
            continue

        size = _image_size(image_path)
        if size is None:
            stats.skipped_no_image += 1
            stats.errors.append(f"missing image size: {image_path}")
            continue
        width, height = size

        elements = _elements_from_record(rec, width, height)
        if not elements:
            stats.skipped_no_elements += 1
            continue

        output_key, stem_page, page_index = _derive_ids(rec, image_path)
        page_dir = output_layout / output_key
        attempt_dir = page_dir / "attempt_1"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        page_png = page_dir / f"{stem_page}.png"
        _link_or_copy(image_path, page_png, use_symlink=use_symlink)

        convert_path = attempt_dir / f"{stem_page}.convert.json"
        convert_path.write_text(
            json.dumps(elements, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        gt_path = gt_layout / output_key / f"{stem_page}.json"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(
            json.dumps(elements, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        pdf_path = _ensure_pdf_for_item(
            rec=rec,
            image_path=image_path,
            input_pdfs_root=input_pdfs,
            output_key=output_key,
            stem_page=stem_page,
            page_index=page_index,
            use_symlink=use_symlink,
            pdf_search_roots=image_roots,
            stats=stats,
        )

        manifest_rows.append(
            {
                "output_key": output_key,
                "stem_page": stem_page,
                "page_index": page_index,
                "source_image": str(image_path),
                "page_png": str(page_png),
                "convert_json": str(convert_path),
                "gt_json": str(gt_path),
                "input_pdf": str(pdf_path) if pdf_path else None,
                "num_elements": len(elements),
            }
        )
        stats.exported += 1

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_jsonl": str(jsonl_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "rows_read": stats.rows_read,
        "exported": stats.exported,
        "skipped_no_image": stats.skipped_no_image,
        "skipped_no_elements": stats.skipped_no_elements,
        "pdf_symlink": stats.pdf_symlink,
        "pdf_single_page": stats.pdf_single_page,
        "pdf_multipage_stub": stats.pdf_multipage_stub,
        "errors": stats.errors[:50],
        "error_count": len(stats.errors),
    }
    summary_path = meta_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_labeler_configs(out_dir, project_root_hint=out_dir.as_posix())
    _write_readme(out_dir, source_jsonl=jsonl_path, exported=stats.exported)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export layout JSONL to labeler-ready tree")
    parser.add_argument(
        "--train-data-root",
        type=Path,
        default=Path("_train_data"),
        help="train/vlm 기준 _train_data 루트",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="입력 JSONL (기본: layout_src_9984/labeler_converter_layout_source_9984.jsonl)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        action="append",
        default=None,
        help="image_path 해석용 루트 (여러 번 지정 가능)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="출력 디렉터리 (기본: {train-data-root}/labeler_ready_layout)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="symlink 대신 PNG/PDF 복사",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="out-dir 이 있으면 삭제 후 재생성",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train_root = args.train_data_root.expanduser().resolve()
    jsonl_path = (
        args.jsonl.expanduser().resolve()
        if args.jsonl
        else train_root / "layout_src_9984" / "labeler_converter_layout_source_9984.jsonl"
    )
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL not found: {jsonl_path}")

    image_roots = (
        [p.expanduser().resolve() for p in args.image_root]
        if args.image_root
        else [
            train_root / "chandra_table_layout_divhtml_16886",
            train_root / "layout_src_9984",
            train_root,
        ]
    )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else train_root / "labeler_ready_layout"
    )

    logger.info("jsonl      : %s", jsonl_path)
    logger.info("image_roots: %s", ", ".join(str(p) for p in image_roots))
    logger.info("out_dir    : %s", out_dir)

    stats = export_jsonl_to_labeler(
        jsonl_path=jsonl_path,
        image_roots=image_roots,
        out_dir=out_dir,
        max_samples=args.max_samples,
        use_symlink=not args.copy_images,
        overwrite=args.overwrite,
    )

    print(json.dumps(
        {
            "rows_read": stats.rows_read,
            "exported": stats.exported,
            "skipped_no_image": stats.skipped_no_image,
            "skipped_no_elements": stats.skipped_no_elements,
            "pdf_symlink": stats.pdf_symlink,
            "pdf_single_page": stats.pdf_single_page,
            "pdf_multipage_stub": stats.pdf_multipage_stub,
            "error_count": len(stats.errors),
            "out_dir": str(out_dir),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
