"""
AIHub 표 데이터 → 대화형 포맷 변환

AIHub '표 데이터' 구조:
    Sample/  또는  Training/ / Validation/
    ├── 01.원천데이터/
    │   ├── *.jpg          (테이블 이미지)
    │   └── *.html         (전체 HTML 문서, <table> 추출 필요)
    └── 02.라벨링데이터/
        └── *.json         (메타데이터: table_meta, table_data)

zip 파일인 경우 자동 압축 해제 후 처리.

Usage:
    # Sample 디렉토리 (이미 압축 해제됨)
    python -m data.convert_aihub \
        --input ./aihub_table_data/Sample \
        --output ./data/processed/aihub_sample.jsonl

    # Training 디렉토리 (zip 파일들 → --extract-dir 필수)
    python -m data.convert_aihub \
        --input ./aihub_table_data/Training \
        --extract-dir ./data/extracted/aihub \
        --output ./data/processed/aihub_train.jsonl

    # 여러 디렉토리 한번에
    python -m data.convert_aihub \
        --input ./aihub_table_data/Training ./aihub_table_data/Validation \
        --extract-dir ./data/extracted/aihub \
        --output ./data/processed/aihub_all.jsonl
"""

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.html_utils import (
    extract_spans_from_html,
    normalize_html,
    parse_html_table,
)
from utils.prompt_templates import build_chat_messages, build_thinking_chain
from utils.span_analyzer import analyze_span


def extract_table_html(full_html: str) -> Optional[str]:
    """
    전체 HTML 문서에서 <table>...</table> 부분만 추출한다.
    AIHub HTML은 <!doctype>, <head>, <style> 등을 포함하므로 테이블만 분리.
    """
    match = re.search(r"(<table[\s\S]*?</table>)", full_html, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def clean_aihub_html(table_html: str) -> str:
    """
    AIHub HTML의 불필요한 속성을 정리한다.
    - rowspan="1", colspan="1" 제거 (기본값)
    - 불필요한 공백 정리
    """
    # rowspan="1", colspan="1" 제거
    table_html = re.sub(r'\s*rowspan="1"', "", table_html)
    table_html = re.sub(r'\s*colspan="1"', "", table_html)
    return table_html


def load_aihub_metadata(json_path: str) -> Optional[dict]:
    """AIHub 라벨링 JSON을 로드한다."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load metadata {json_path}: {e}")
        return None


def find_pairs_from_directory(base_dir: str) -> list[dict]:
    """
    압축 해제된 디렉토리에서 image + html + json 쌍을 찾는다.

    Returns:
        [{"image_path": str, "html_path": str, "json_path": str | None, "file_id": str}, ...]
    """
    base = Path(base_dir)
    source_dir = base / "01.원천데이터"
    label_dir = base / "02.라벨링데이터"

    if not source_dir.exists():
        print(f"Warning: Source dir not found: {source_dir}")
        return []

    pairs = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # 라벨 파일 인덱스 (중첩 디렉토리 대응)
    json_index = {}
    if label_dir.exists():
        for json_file in label_dir.rglob("*.json"):
            json_index[json_file.stem] = json_file

    # 원천데이터 재귀 탐색 (zip 해제된 중첩 구조 대응)
    for img_file in sorted(source_dir.rglob("*")):
        if not img_file.is_file():
            continue
        if img_file.suffix.lower() not in image_extensions:
            continue

        file_id = img_file.stem
        html_file = img_file.with_suffix(".html")
        if not html_file.exists():
            continue

        json_file = json_index.get(file_id)

        pair = {
            "image_path": str(img_file),
            "html_path": str(html_file),
            "json_path": str(json_file) if json_file else None,
            "file_id": file_id,
        }
        pairs.append(pair)

    return pairs


def find_pairs_from_zips(
    base_dir: str, temp_dir: str, max_pairs: Optional[int] = None
) -> list[dict]:
    """
    zip 파일들을 temp_dir에 압축 해제 후 쌍을 찾는다.
    원천데이터 zip(TS_*.zip)과 라벨링데이터 zip(TL_*.zip)을 매칭.

    max_pairs 지정 시 충분한 pair가 모이면 나머지 zip은 건너뛴다.
    """
    base = Path(base_dir)
    source_zip_dir = base / "01.원천데이터"
    label_zip_dir = base / "02.라벨링데이터"

    if not source_zip_dir.exists():
        return []

    source_zips = sorted(source_zip_dir.glob("*.zip"))
    if not source_zips:
        # zip이 아닌 경우 = 이미 압축 해제된 디렉토리
        return find_pairs_from_directory(base_dir)

    # 라벨링 zip 매핑: TS_T01_C01 → TL_T01_C01
    label_zip_map = {}
    if label_zip_dir.exists():
        for lz in label_zip_dir.glob("*.zip"):
            # TL_T01_C01.zip → T01_C01
            key = lz.stem.replace("TL_", "").replace("VL_", "")
            label_zip_map[key] = lz

    all_pairs = []

    for sz in source_zips:
        # max_pairs 도달 시 나머지 zip 건너뛰기
        if max_pairs and len(all_pairs) >= max_pairs:
            print(f"  Reached {len(all_pairs)} pairs (>= max_pairs={max_pairs}), skipping remaining zips.")
            break

        # TS_T01_C01.zip → T01_C01
        key = sz.stem.replace("TS_", "").replace("VS_", "")

        # 압축 해제
        source_extract = Path(temp_dir) / "source" / key
        source_extract.mkdir(parents=True, exist_ok=True)

        print(f"  Extracting {sz.name}...")
        try:
            with zipfile.ZipFile(sz, "r") as zf:
                zf.extractall(source_extract)
        except Exception as e:
            print(f"  Warning: Failed to extract {sz}: {e}")
            continue

        label_extract = None
        lz = label_zip_map.get(key)
        if lz:
            label_extract = Path(temp_dir) / "label" / key
            label_extract.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(lz, "r") as zf:
                    zf.extractall(label_extract)
            except Exception as e:
                print(f"  Warning: Failed to extract label zip {lz}: {e}")
                label_extract = None

        # 추출된 파일에서 쌍 찾기
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        for img_file in sorted(source_extract.rglob("*")):
            if img_file.suffix.lower() not in image_extensions:
                continue

            file_id = img_file.stem
            html_file = img_file.with_suffix(".html")

            json_file = None
            if label_extract:
                # 라벨링 데이터에서 같은 파일명의 json 찾기
                json_candidates = list(label_extract.rglob(f"{file_id}.json"))
                if json_candidates:
                    json_file = json_candidates[0]

            if not html_file.exists():
                continue

            pair = {
                "image_path": str(img_file),
                "html_path": str(html_file),
                "json_path": str(json_file) if json_file else None,
                "file_id": file_id,
            }
            all_pairs.append(pair)

    return all_pairs


def convert_aihub_record(
    pair: dict,
    include_thinking: bool = True,
    prompt_idx: Optional[int] = None,
) -> Optional[dict]:
    """
    AIHub 데이터 쌍을 대화형 포맷으로 변환한다.

    Args:
        pair: {"image_path", "html_path", "json_path", "file_id"}
        include_thinking: thinking chain 포함 여부
        prompt_idx: 프롬프트 변형 인덱스 (None이면 랜덤)

    Returns:
        {"messages": [...], "metadata": {...}} or None
    """
    try:
        # HTML 로드 및 <table> 추출
        with open(pair["html_path"], "r", encoding="utf-8") as f:
            full_html = f.read()

        table_html = extract_table_html(full_html)
        if not table_html:
            print(f"  Warning: No <table> found in {pair['html_path']}")
            return None

        table_html = clean_aihub_html(table_html)

        # HTML 유효성 검증
        structure = parse_html_table(table_html)
        if structure.num_rows == 0 or structure.num_cols == 0:
            return None

        # 메타데이터 로드
        meta = None
        if pair["json_path"]:
            meta = load_aihub_metadata(pair["json_path"])

        # Thinking chain 생성
        thinking = ""
        if include_thinking:
            spans = extract_spans_from_html(table_html)

            # 메타데이터에서 header 정보 활용
            header_rows = structure.header_rows
            if meta and meta.get("table_meta", {}).get("table_meta.table_header") == "Y":
                header_rows = max(header_rows, 1)

            thinking = build_thinking_chain(
                num_rows=structure.num_rows,
                num_cols=structure.num_cols,
                spans=spans,
                header_rows=header_rows,
            )

        # 대화 메시지 구성
        normalized_html = normalize_html(table_html)
        messages = build_chat_messages(
            image_path=pair["image_path"],
            html=normalized_html,
            thinking=thinking,
            prompt_idx=prompt_idx,
        )

        # Span 통계
        span_stats = analyze_span(table_html)

        # 메타데이터 구성
        result_meta = {
            "image_path": pair["image_path"],
            "file_id": pair["file_id"],
            "num_rows": structure.num_rows,
            "num_cols": structure.num_cols,
            "num_span_cells": span_stats.span_cells,
            "complexity": span_stats.complexity,
            "complexity_score": span_stats.complexity_score,
            "source": "aihub",
        }

        # AIHub 고유 메타데이터 추가
        if meta:
            table_meta = meta.get("table_meta", {})
            result_meta["table_type"] = table_meta.get("table_meta.table_type", "")
            result_meta["table_field"] = table_meta.get("table_meta.table_field", "")
            result_meta["has_header"] = table_meta.get("table_meta.table_header") == "Y"

        return {
            "messages": messages,
            "metadata": result_meta,
        }

    except Exception as e:
        print(f"  Warning: Failed to convert {pair.get('file_id', '?')}: {e}")
        return None


def convert_dataset(
    pairs: list[dict],
    include_thinking: bool = True,
    prompt_variation: bool = True,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """전체 데이터셋을 변환한다.

    Args:
        max_samples: 지정 시 해당 건수만큼 성공적으로 변환 후 조기 종료
    """
    converted = []
    failed = 0

    for i, pair in enumerate(pairs):
        prompt_idx = None if prompt_variation else 0
        result = convert_aihub_record(pair, include_thinking, prompt_idx)

        if result is not None:
            converted.append(result)
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} (failed: {failed})")

        if max_samples and len(converted) >= max_samples:
            print(f"  Reached max_samples={max_samples}, stopping early.")
            break

    print(f"Conversion complete: {len(converted)} success, {failed} failed")
    return converted


def save_dataset(data: list[dict], output_path: str):
    """JSONL 형식으로 저장."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} records to {output_path}")

    # 통계 출력
    complexities = [item["metadata"]["complexity"] for item in data]
    from collections import Counter

    dist = Counter(complexities)
    print(f"  Complexity distribution: {dict(dist)}")

    # AIHub 고유 통계
    fields = [item["metadata"].get("table_field", "") for item in data]
    field_dist = Counter(f for f in fields if f)
    if field_dist:
        print(f"  Table field distribution: {dict(field_dist)}")

    types = [item["metadata"].get("table_type", "") for item in data]
    type_dist = Counter(t for t in types if t)
    if type_dist:
        print(f"  Table type distribution: {dict(type_dist)}")


def _load_pairs(
    input_dir: str,
    extract_dir: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> list[dict]:
    """
    단일 입력 디렉토리에서 pair 목록을 로드한다.
    zip 파일이 있으면 extract_dir에 압축 해제 후 처리.
    max_pairs 지정 시 해당 건수만큼 pair를 모으면 나머지 zip은 건너뛴다.
    """
    source_dir = Path(input_dir) / "01.원천데이터"
    has_zips = source_dir.exists() and any(source_dir.glob("*.zip"))

    if has_zips:
        if not extract_dir:
            raise ValueError(
                f"zip 파일이 감지됨: {source_dir}\n"
                "  --extract-dir 옵션으로 압축 해제 디렉토리를 지정하세요.\n"
                "  (이미지 경로가 출력 JSONL에 기록되므로 영구 경로가 필요합니다)"
            )
        os.makedirs(extract_dir, exist_ok=True)
        return find_pairs_from_zips(input_dir, extract_dir, max_pairs=max_pairs)
    else:
        return find_pairs_from_directory(input_dir)


def main():
    parser = argparse.ArgumentParser(
        description="AIHub 표 데이터를 대화형 포맷으로 변환"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="AIHub 데이터 디렉토리 경로 (Sample, Training, Validation 등). 여러 경로 지정 가능",
    )
    parser.add_argument("--output", required=True, help="출력 JSONL 경로")
    parser.add_argument(
        "--no-thinking", action="store_true", help="Thinking chain 제외"
    )
    parser.add_argument(
        "--no-prompt-variation",
        action="store_true",
        help="프롬프트 다양화 비활성화",
    )
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="zip 압축 해제 디렉토리 (zip 입력 시 필수, 이미지 경로가 여기를 가리킴)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="최대 변환 건수 (테스트용, 지정 건수 도달 시 조기 종료)",
    )
    args = parser.parse_args()

    # 1. 모든 입력에서 pair 수집
    all_pairs = []
    for input_dir in args.input:
        print(f"Loading from: {input_dir}")
        remaining = args.max_samples - len(all_pairs) if args.max_samples else None
        pairs = _load_pairs(input_dir, args.extract_dir, max_pairs=remaining)
        print(f"  Found {len(pairs)} pairs")
        all_pairs.extend(pairs)

        if args.max_samples and len(all_pairs) >= args.max_samples:
            break

    if not all_pairs:
        print("No data found. Exiting.")
        return

    # 2. 변환
    converted = convert_dataset(
        all_pairs,
        include_thinking=not args.no_thinking,
        prompt_variation=not args.no_prompt_variation,
        max_samples=args.max_samples,
    )

    # 3. 저장
    print(f"\nTotal: {len(converted)} records")
    save_dataset(converted, args.output)


if __name__ == "__main__":
    main()
