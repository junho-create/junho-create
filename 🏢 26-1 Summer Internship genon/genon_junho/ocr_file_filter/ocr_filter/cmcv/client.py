"""3모델(target/dots.ocr/paddle) OpenAI 호환 vLLM 엔드포인트에 이미지+프롬프트 던지는 클라이언트.

target(Qwen3.5-9B) 도 judge(Qwen3.5-397B) 처럼 **thinking 모델**이다(chat_template.jinja 에
enable_thinking 있음). 사고 분량이 페이지마다/호출마다 들쭉날쭉해서(관측: 같은 이미지 4연속
호출에 raw 응답 길이 3116~7874자), 예전 max_tokens=8192 로는 사고가 길어지는 날엔 실제 답
(`<div data-bbox=...>`)이 나오기 전에 잘려서 파싱 결과가 빈 리스트가 되는 경우가 있었다
(2026-07-13 실제로 재현·확인함 — Hard 로 잘못 분류된 사례의 원인). max_tokens=16384 로 인상."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import requests

from ocr_filter.cmcv.prompts import build_messages


def _image_data_uri(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def call_model(
    model_key: str,
    model_cfg: dict,
    image_path: str,
    ocr_info: list | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.0,
    timeout: float = 600.0,
) -> str:
    """model_cfg: default.yaml 의 `models.<key>` ({"name": served_name, "endpoint": ".../v1"})."""
    messages = build_messages(model_key, ocr_info)
    image_content = {"type": "image_url", "image_url": {"url": _image_data_uri(image_path)}}
    # 마지막(user) 메시지의 텍스트를 이미지(+텍스트) 멀티모달 content 로 바꾼다.
    # content 가 None(예: paddle)이면 텍스트 파트 자체를 생략 — 이미지만 보낸다.
    text = messages[-1]["content"]
    content = [image_content] if text is None else [image_content, {"type": "text", "text": text}]
    messages[-1] = {"role": messages[-1]["role"], "content": content}

    resp = requests.post(
        f"{model_cfg['endpoint']}/chat/completions",
        json={
            "model": model_cfg["name"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
