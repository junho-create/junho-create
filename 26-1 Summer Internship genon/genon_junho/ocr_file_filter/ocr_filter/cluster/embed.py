"""ViT 임베딩 추출. 페이지 전체 이미지(1단계) / 요소 크롭(2단계) 둘 다 이 함수로.

    embed_images(["a.png", "b.png"]) -> np.ndarray shape (N, hidden_size)

기본 모델은 `google/vit-base-patch16-224-in21k` (분류 헤드 없는 순수 백본, ImageNet-21k
프리트레인) — 라벨 없이 일반적인 시각 유사도만 필요해서 이걸로 충분. `--embed-model`
로 다른 HF ViT 체크포인트로 바로 교체 가능(디자인 의도상 여기 하나만 바꾸면 됨).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DEFAULT_MODEL = "google/vit-base-patch16-224-in21k"

_cache: dict[str, tuple] = {}


def _load(model_name: str):
    if model_name not in _cache:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        _cache[model_name] = (processor, model, device)
    return _cache[model_name]


@torch.no_grad()
def embed_images(
    images: list[str | Path | Image.Image],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """CLS 토큰(pooler_output 이 없으면 last_hidden_state[:,0]) 을 임베딩으로 사용.
    images 원소는 경로(str/Path)와 이미 열린 PIL.Image 를 섞어서 줄 수 있다
    (요소 크롭은 디스크에 안 쓰고 메모리 상에서 바로 넘기려고)."""
    processor, model, device = _load(model_name)
    embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        pil_images = [
            im if isinstance(im, Image.Image) else Image.open(im).convert("RGB")
            for im in batch
        ]
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        out = model(**inputs)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]
        embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 768))


def crop_element(image_path: str | Path, bbox: list[float],
                  bbox_scale: int | None = None) -> Image.Image:
    """bbox 로 이미지 크롭 (메모리 상, 저장 안 함). bbox_scale 이 주어지면(예: 1000)
    0-bbox_scale 정규화 좌표를 실제 픽셀로 환산한다."""
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = bbox
    if bbox_scale:
        x0, x1 = x0 / bbox_scale * w, x1 / bbox_scale * w
        y0, y1 = y0 / bbox_scale * h, y1 / bbox_scale * h
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, w, h  # 깨진 bbox 는 전체 이미지로 폴백
    return im.crop((x0, y0, x1, y1))
