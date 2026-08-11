"""
이미지/텍스트 증강

이미지 증강:
- 밝기/대비 조정
- JPEG 압축 아티팩트
- 미세 회전 (±2°)
- 가우시안 노이즈

텍스트 증강:
- 프롬프트 변형 (다른 instruction 사용)

주의: 구조가 변경되는 증강(crop, flip, 큰 회전)은 사용 금지

Usage:
    python -m data.augment \
        --input data/processed/dataset.jsonl \
        --output data/processed/augmented.jsonl \
        --image_output_dir data/augmented_images/ \
        --augment_factor 2
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.prompt_templates import get_system_prompt, get_user_prompt


# =============================================================================
# Image Augmentation
# =============================================================================


def augment_brightness_contrast(img: Image.Image) -> Image.Image:
    """밝기/대비 랜덤 조정."""
    brightness_factor = random.uniform(0.85, 1.15)
    contrast_factor = random.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img


def augment_jpeg_compression(img: Image.Image) -> Image.Image:
    """JPEG 압축 아티팩트 시뮬레이션."""
    import io
    quality = random.randint(50, 85)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def augment_rotation(img: Image.Image) -> Image.Image:
    """미세 회전 (±2°). 테이블 구조를 해치지 않는 범위."""
    angle = random.uniform(-2.0, 2.0)
    return img.rotate(angle, expand=False, fillcolor=(255, 255, 255))


def augment_gaussian_noise(img: Image.Image) -> Image.Image:
    """약한 가우시안 노이즈 추가."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_color_jitter(img: Image.Image) -> Image.Image:
    """색상 미세 조정."""
    saturation_factor = random.uniform(0.9, 1.1)
    return ImageEnhance.Color(img).enhance(saturation_factor)


# 증강 함수 풀
AUGMENTATION_POOL = [
    augment_brightness_contrast,
    augment_jpeg_compression,
    augment_rotation,
    augment_gaussian_noise,
    augment_color_jitter,
]


def apply_random_augmentations(
    img: Image.Image,
    num_augments: int = 2,
) -> Image.Image:
    """
    랜덤하게 선택한 증강을 적용.

    Args:
        img: 원본 이미지
        num_augments: 적용할 증강 수 (1~3)

    Returns:
        증강된 이미지
    """
    augments = random.sample(AUGMENTATION_POOL, min(num_augments, len(AUGMENTATION_POOL)))
    for aug_fn in augments:
        img = aug_fn(img)
    return img


# =============================================================================
# Text/Prompt Augmentation
# =============================================================================


def augment_prompts(messages: list[dict]) -> list[dict]:
    """
    시스템/유저 프롬프트를 다른 변형으로 교체.
    """
    new_messages = []
    for msg in messages:
        new_msg = dict(msg)
        if msg["role"] == "system":
            new_msg["content"] = get_system_prompt()
        elif msg["role"] == "user" and isinstance(msg["content"], list):
            new_content = []
            for item in msg["content"]:
                if item.get("type") == "text":
                    new_content.append({"type": "text", "text": get_user_prompt()})
                else:
                    new_content.append(dict(item))
            new_msg["content"] = new_content
        new_messages.append(new_msg)
    return new_messages


# =============================================================================
# Batch Augmentation
# =============================================================================


def augment_dataset(
    input_path: str,
    output_path: str,
    image_output_dir: str,
    augment_factor: int = 2,
    image_augment: bool = True,
    prompt_augment: bool = True,
    span_only: bool = False,
):
    """
    데이터셋을 증강한다.

    Args:
        input_path: 입력 JSONL
        output_path: 출력 JSONL
        image_output_dir: 증강 이미지 저장 디렉토리
        augment_factor: 증강 배수 (2이면 원본 + 1개 추가)
        image_augment: 이미지 증강 여부
        prompt_augment: 프롬프트 변형 여부
        span_only: True이면 span 데이터만 증강
    """
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    output_records = []

    for i, record in enumerate(records):
        # 원본은 항상 포함
        output_records.append(record)

        # span_only 모드: span이 없는 데이터는 증강 안 함
        metadata = record.get("metadata", {})
        if span_only and metadata.get("complexity", "simple") == "simple":
            continue

        # 증강 생성
        for aug_idx in range(augment_factor - 1):
            aug_record = json.loads(json.dumps(record))  # deep copy

            # 이미지 증강
            if image_augment:
                img_path = _find_image_path(aug_record)
                if img_path and os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        aug_img = apply_random_augmentations(img)

                        # 저장
                        base = Path(img_path).stem
                        ext = Path(img_path).suffix or ".png"
                        new_name = f"{base}_aug{aug_idx}{ext}"
                        new_path = os.path.join(image_output_dir, new_name)
                        aug_img.save(new_path)

                        # 메시지 내 이미지 경로 업데이트
                        _update_image_path(aug_record, new_path)
                    except Exception as e:
                        print(f"  Warning: Image augment failed for {img_path}: {e}")

            # 프롬프트 변형
            if prompt_augment:
                aug_record["messages"] = augment_prompts(aug_record["messages"])

            # 메타데이터 업데이트
            if "metadata" in aug_record:
                aug_record["metadata"]["augmented"] = True
                aug_record["metadata"]["aug_index"] = aug_idx

            output_records.append(aug_record)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(records)}")

    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Augmentation complete: {len(records)} → {len(output_records)} records")
    print(f"  Saved to {output_path}")


def _find_image_path(record: dict) -> Optional[str]:
    """레코드에서 이미지 경로를 찾는다."""
    for msg in record.get("messages", []):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    return item.get("image", "")
    return None


def _update_image_path(record: dict, new_path: str):
    """레코드 내 이미지 경로를 업데이트한다."""
    for msg in record.get("messages", []):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    item["image"] = new_path
                    return


def main():
    parser = argparse.ArgumentParser(description="데이터 증강")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image_output_dir", default="./data/augmented_images/")
    parser.add_argument("--augment_factor", type=int, default=2)
    parser.add_argument("--no-image-augment", action="store_true")
    parser.add_argument("--no-prompt-augment", action="store_true")
    parser.add_argument("--span-only", action="store_true", help="span 데이터만 증강")
    args = parser.parse_args()

    augment_dataset(
        args.input,
        args.output,
        args.image_output_dir,
        args.augment_factor,
        image_augment=not args.no_image_augment,
        prompt_augment=not args.no_prompt_augment,
        span_only=args.span_only,
    )


if __name__ == "__main__":
    main()
