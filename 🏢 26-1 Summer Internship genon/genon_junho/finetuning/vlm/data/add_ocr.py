"""
샘플링된 JSONL에 PaddleOCR로 OCR 정보를 추가하고 프롬프트를 OCR-on 스타일로 변환한다.

train, validation, test 모든 split에 사용 가능.

Usage:
    python -m data.add_ocr \
        --input ./data/experiments/e7/train_raw.jsonl \
        --output ./data/experiments/e7/train.jsonl \
        --image_root ./data/extracted/aihub/Training/01.원천데이터 \
        --prompt_style chandra_table_with_ocr \
        --bbox_scale 1024 \
        --ocr_device gpu:0 \
        --ocr_lang korean
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ocr_extractor import PaddleOCRExtractor, normalize_ocr_items  # noqa: E402
from utils.prompt_templates import normalize_prompt_style, prompt_requires_ocr  # noqa: E402


def _extract_image_path(record: dict) -> str:
    """레코드에서 이미지 경로를 추출한다."""
    return record.get("image_path", "")


def _resolve_image_path(image_path: str, image_root: Path | None) -> str:
    """image_root가 주어지면 상대 image_path를 해당 루트 기준으로 해석한다."""
    if image_root is None:
        return image_path

    raw = Path(image_path)
    if raw.is_absolute():
        return str(raw)

    candidate = image_root / raw
    if candidate.exists():
        return str(candidate)

    # image_path가 images/... 형태인 경우 image_root 하위 직접 경로도 시도
    if raw.parts and raw.parts[0] == "images" and len(raw.parts) > 1:
        fallback = image_root / Path(*raw.parts[1:])
        if fallback.exists():
            return str(fallback)

    return str(candidate)


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None

    prompt_style = normalize_prompt_style(args.prompt_style)
    if args.prompt_style != prompt_style:
        print(f"경고: '{args.prompt_style}' -> '{prompt_style}'")

    # 입력 로드
    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"로드: {len(records)}건 from {input_path}")
    if image_root is not None:
        print(f"이미지 루트: {image_root}")
    extractor = PaddleOCRExtractor(
        lang=args.ocr_lang,
        bbox_scale=args.bbox_scale,
        device=args.ocr_device,
        ocr_version=args.ocr_version,
    )

    written = 0
    ocr_empty = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in tqdm(records, desc="OCR 추가"):
            rec = copy.deepcopy(record)
            image_path = _extract_image_path(rec)
            if not image_path:
                raise ValueError("레코드에서 image_path를 찾을 수 없습니다")

            resolved_image_path = _resolve_image_path(image_path, image_root)
            ocr_items = extractor.extract(resolved_image_path)
            if not ocr_items:
                ocr_empty += 1
            ocr_info = normalize_ocr_items(ocr_items, keep_score=False)

            rec["ocr_info"] = ocr_info
            rec["bbox_scale"] = int(args.bbox_scale)
            rec["prompt_style"] = prompt_style

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n=== OCR 추가 완료 ===")
    print(f"  출력:      {output_path}")
    print(f"  처리:      {written}건")
    print(f"  OCR 비어있음: {ocr_empty}건")
    print(f"  프롬프트:  {prompt_style}")


def main():
    parser = argparse.ArgumentParser(description="샘플링된 JSONL에 OCR 정보 추가")
    parser.add_argument("--input", required=True, help="입력 JSONL 경로")
    parser.add_argument("--output", required=True, help="출력 JSONL 경로")
    parser.add_argument(
        "--image_root",
        default="",
        help="상대 image_path 해석용 이미지 루트 디렉토리 (선택)",
    )
    parser.add_argument(
        "--prompt_style",
        default="chandra_table_with_ocr",
        help="프롬프트 스타일 (기본: chandra_table_with_ocr)",
    )
    parser.add_argument("--bbox_scale", type=int, default=1024)
    parser.add_argument(
        "--ocr_device",
        default="",
        help="OCR 추론 디바이스 (예: gpu:0, cpu). 비우면 PaddleOCR 기본값 사용",
    )
    parser.add_argument("--ocr_lang", default="korean", help="PaddleOCR 언어 코드")
    parser.add_argument(
        "--ocr_version", default="",
        help="PaddleOCR 버전 강제 지정(예: PP-OCRv5). 비우면 PaddleOCR 이 lang 기준으로 "
             "알아서 고르는데, lang=en 처럼 _PPOCRV6_LANGS 에 속한 언어는 지정 안 하면 "
             "조용히 v6 로 올라간다.",
    )
    args = parser.parse_args()

    if args.bbox_scale <= 0:
        raise ValueError("--bbox_scale는 양수여야 합니다")

    run(args)


if __name__ == "__main__":
    main()
