#!/usr/bin/env python3
"""통합 JSON 데이터셋(unified_ocr_*)의 GT 를 Chandra OCR_LAYOUT HTML 형식으로 변환한다.

변환 내용
--------
1. 정답(gt_html) 형식 변경: JSON element 배열 → Chandra raw HTML
   각 영역을 다음 div 로 출력한다::

       <div data-bbox="x0 y0 x1 y1" data-label="<Label>">
       ...content...
       </div>

   - data-bbox 는 0-1000 정규화 정수.
   - Table content 는 표 구조 HTML(<table>...</table>) 그대로.
   - Picture content 는 비움.
   - 그 외는 <p>...</p> (개행은 <br/>).

2. 테이블 레코드(image_path 가 images/table/...):
   - 174 dots.mocr 렌더 결과에서 얻은 Table union bbox(렌더 PNG 픽셀)를
     렌더 PNG 의 W/H 로 0-1000 정규화해 data-bbox 로 사용.
   - 이미지를 174 렌더 PNG(images/table/<key>.png)로 교체.
   - 표 HTML 은 기존 DCTN gt_html(JSON 배열의 Table element text)을 그대로 사용.
   - manifest 에 없는 레코드(예: 변환 실패로 삭제된 1건)는 원본 jpg 유지 +
     data-bbox 를 전체영역(0 0 1000 1000)으로.

3. 레이아웃 레코드: 각 element bbox 를 1024→1000 재정규화.

4. ocr_info 의 bbox 도 1024→1000 재정규화(있을 때). bbox_scale=1000 으로 기록.

입력 데이터(unified_ocr) 레코드 스키마::

    {"image_path", "gt_html"(JSON 배열 문자열), "prompt_style",
     "bbox_scale"(=1024), "output_format", "ocr_info"}

사용 예(제자리 덮어쓰기; 백업은 data_backup_<ts>/ 로)::

    python -m data.build_chandra_dataset \
        --data-dir ./_train_data/chandra_table_layout_divhtml_16886 \
        --manifest ./_train_data/table_bbox_manifest.json \
        --inplace
"""

from __future__ import annotations

import argparse
import html as _html_lib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

# 학습 GT 분포 기준 11개 카테고리(라벨).
CATS_11 = {
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title",
}

_PREFIX_RE = re.compile(r"^\d+__")
# markdown heading prefix(텍스트 맨 앞의 #, ##, ... ######)만 제거. bullet(-, *)은 보존.
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s*")


def strip_prefix(stem: str) -> str:
    return _PREFIX_RE.sub("", stem)


def _strip_md_heading(text: str) -> str:
    return _MD_HEADING_RE.sub("", str(text or "")).strip()


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def renorm_from_px(box: list, w: int, h: int, scale: int = 1000) -> list[int]:
    """픽셀 좌표(box) 를 (w,h) 기준 0-scale 정규화."""
    x0, y0, x1, y1 = box
    w = max(1, int(w)); h = max(1, int(h))
    return [
        _clamp(round(x0 * scale / w), 0, scale),
        _clamp(round(y0 * scale / h), 0, scale),
        _clamp(round(x1 * scale / w), 0, scale),
        _clamp(round(y1 * scale / h), 0, scale),
    ]


def renorm_scale(box: list, src: int = 1024, dst: int = 1000) -> list[int]:
    """이미 정규화된 좌표(0-src) 를 0-dst 로 재정규화."""
    out = []
    for v in box[:4]:
        out.append(_clamp(round(float(v) * dst / src), 0, dst))
    return out


def _esc(text: str) -> str:
    s = _html_lib.escape(str(text or ""), quote=False)
    return s.replace("\n", "<br/>")


def _content_for(category: str, text: str) -> str:
    c = str(category or "").strip().lower()
    if c == "table":
        return str(text or "").strip()
    if c == "picture":
        return ""
    return f"<p>{_esc(_strip_md_heading(text))}</p>"


def _div(bbox: list[int], label: str, content: str) -> str:
    x0, y0, x1, y1 = bbox
    if content:
        return f'<div data-bbox="{x0} {y0} {x1} {y1}" data-label="{label}">\n{content}\n</div>'
    return f'<div data-bbox="{x0} {y0} {x1} {y1}" data-label="{label}"></div>'


def _parse_elements(gt_html: str) -> list[dict]:
    try:
        d = json.loads(gt_html)
    except Exception:
        return []
    if isinstance(d, dict):
        d = [d]
    return [e for e in d if isinstance(e, dict)]


def _table_text(elements: list[dict]) -> str:
    for e in elements:
        if str(e.get("category", "")).strip().lower() == "table":
            return str(e.get("text", "") or "")
    if elements:
        return str(elements[0].get("text", "") or "")
    return ""


def convert_record(
    rec: dict,
    manifest: dict,
    *,
    scale: int = 1000,
    stats: Optional[dict] = None,
) -> dict:
    out = dict(rec)
    image_path = str(rec.get("image_path", ""))
    elements = _parse_elements(rec.get("gt_html", ""))
    is_table = "/table/" in image_path

    if is_table:
        stem = image_path.split("/")[-1].rsplit(".", 1)[0]
        key = strip_prefix(stem)
        info = manifest.get(key)
        if info is not None:
            bbox = renorm_from_px(info["bbox"], info["w"], info["h"], scale)
            # 174 렌더 이미지(다운스케일 JPEG)로 교체. key 는 prefix 제거된 stem.
            out["image_path"] = f"images/table/{key}.jpg"
            if stats is not None:
                stats["table_matched"] = stats.get("table_matched", 0) + 1
                if info.get("fallback"):
                    stats["table_fallback_bbox"] = stats.get("table_fallback_bbox", 0) + 1
        else:
            bbox = [0, 0, scale, scale]
            # image_path 유지(원본 jpg)
            if stats is not None:
                stats["table_unmatched"] = stats.get("table_unmatched", 0) + 1
        gt = _div(bbox, "Table", _content_for("Table", _table_text(elements)))
    else:
        divs = []
        for e in elements:
            label = str(e.get("category", "") or "").strip()
            box = e.get("bbox", [0, 0, 1024, 1024])
            if not (isinstance(box, list) and len(box) >= 4):
                box = [0, 0, 1024, 1024]
            b = renorm_scale(box, src=1024, dst=scale)
            divs.append(_div(b, label, _content_for(label, e.get("text", ""))))
        gt = "\n".join(divs)
        if stats is not None:
            stats["layout"] = stats.get("layout", 0) + 1

    # ocr_info 재정규화 (1024 -> scale)
    oi = rec.get("ocr_info")
    if isinstance(oi, list) and oi:
        new_oi = []
        for item in oi:
            if isinstance(item, dict) and isinstance(item.get("bbox"), list) and len(item["bbox"]) >= 4:
                ni = dict(item)
                ni["bbox"] = renorm_scale(item["bbox"], src=1024, dst=scale)
                new_oi.append(ni)
            else:
                new_oi.append(item)
        out["ocr_info"] = new_oi

    out["gt_html"] = gt
    out["bbox_scale"] = scale
    out["output_format"] = "html"
    return out


def convert_split(src: Path, dst: Path, manifest: dict, scale: int, stats: dict) -> int:
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            new = convert_record(rec, manifest, scale=scale, stats=stats)
            fout.write(json.dumps(new, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="unified_ocr 데이터셋 루트")
    ap.add_argument("--manifest", type=Path, required=True, help="table_bbox_manifest.json")
    ap.add_argument("--scale", type=int, default=1000)
    ap.add_argument("--inplace", action="store_true", help="data/*.jsonl 제자리 덮어쓰기(백업 후)")
    ap.add_argument("--out-dir", type=Path, default=None, help="inplace 가 아니면 출력 루트")
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    manifest = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))
    src_data = data_dir / "data"

    if args.inplace:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = data_dir / f"data_backup_{ts}"
        shutil.copytree(src_data, backup)
        print(f"  backup: {backup}")
        out_data = src_data
    else:
        assert args.out_dir is not None, "--out-dir 또는 --inplace 필요"
        out_data = args.out_dir.expanduser().resolve() / "data"

    stats: dict[str, Any] = {}
    summary = {"data_dir": str(data_dir), "scale": args.scale, "splits": {}}
    for split in ("train", "valid", "test"):
        src = src_data / f"{split}.jsonl"
        if not src.exists():
            print(f"  [warn] split not found: {src}")
            continue
        # inplace 면 백업본에서 읽어 원본에 쓴다
        read_src = (backup / f"{split}.jsonl") if args.inplace else src
        dst = out_data / f"{split}.jsonl"
        n = convert_split(read_src, dst, manifest, args.scale, stats)
        summary["splits"][split] = n
        print(f"  {split}: {n} records -> {dst}")

    summary["stats"] = stats
    print("  stats:", json.dumps(stats, ensure_ascii=False))
    (data_dir / "chandra_convert_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  summary: {data_dir / 'chandra_convert_summary.json'}")


if __name__ == "__main__":
    main()
