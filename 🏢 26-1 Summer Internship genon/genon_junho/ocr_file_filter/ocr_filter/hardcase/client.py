"""[6] hardcase 용 범용 OpenAI-호환 VLM 클라이언트 (이미지 여러 장 + 텍스트).

judge/prelabel 둘 다 이 하나의 함수로 부른다 — 차이는 오직 `configs/default.yaml` 의
`hardcase.judge_vlm` / `hardcase.prelabel_vlm` 설정(엔드포인트/모델/제공자)뿐이라, 나중에
prelabel_vlm 만 OpenRouter+Gemini 로 바꿔도 이 파일은 안 건드려도 된다.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

import requests

from PIL import Image

from ocr_filter.report.render import to_data_uri


def _image_content(image: str | Path | Image.Image) -> dict:
    if isinstance(image, Image.Image):
        return {"type": "image_url", "image_url": {"url": to_data_uri(image, fmt="PNG")}}
    mime = mimetypes.guess_type(str(image))[0] or "image/png"
    data = Path(image).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def call_vlm(
    vlm_cfg: dict,
    images: list[str | Path | Image.Image],
    prompt: str,
    max_tokens: int = 16384,
    temperature: float = 0.0,
    timeout: float = 600.0,
    system: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """vlm_cfg: {"name":served_model, "endpoint":".../v1", "api_key_env"(선택)}.
    images: 경로 또는 이미 열린 PIL.Image, 여러 장 (judge 는 원본+렌더링 2장).
    system: 주면 system 롤 메시지를 앞에 붙인다 (Chandra 프롬프트처럼 system/user 분리가
        필요한 경우). enable_thinking: Qwen3.5 계열은 thinking 모델이라, False 를 주면
        vLLM 정식 키(`chat_template_kwargs.enable_thinking`)로 사고를 꺼서 (a) 사고가 길어져
        답이 max_tokens 안에서 잘리는 사고-truncation 을 막고 (b) 콜당 디코딩 토큰을 줄인다
        (export 단계가 이 모델에 이미 같은 방식으로 사고를 끈다 — cli.py 주석 참고)."""
    content = [_image_content(img) for img in images] + [{"type": "text", "text": prompt}]
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": content},
    ]
    headers = {}
    api_key_env = vlm_cfg.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env, "")
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": vlm_cfg["name"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    resp = requests.post(
        f"{vlm_cfg['endpoint']}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
