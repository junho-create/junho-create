"""
평가 메인 스크립트

학습된 모델로 테스트 데이터를 추론하고 메트릭을 계산한다.

Usage:
    python -m eval.evaluate \
        --model output/qwen3_vl_tsr_qlora_phase2/final \
        --test_data data/processed/eval.jsonl \
        --output_dir eval_results/
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from tqdm import tqdm
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (
    compute_span_metrics,
    compute_grid_metrics,
    compute_aggregate_metrics,
    normalize_complexity_label,
    to_base_complexity_bucket,
)
try:
    from eval.metrics import clear_metrics_caches
except Exception:  # pragma: no cover
    def clear_metrics_caches() -> None:
        return

try:
    # Newer metrics.py exports compute_teds_variants directly.
    from eval.metrics import compute_teds_variants  # type: ignore
except Exception:  # pragma: no cover
    compute_teds_variants = None
try:
    from eval.metrics import compute_teds_variants_nested_split  # type: ignore
except Exception:  # pragma: no cover
    compute_teds_variants_nested_split = None
from utils.prompt_templates import (
    get_user_prompt_with_style,
    normalize_prompt_style,
    PROMPT_STYLE_CHANDRA_WITH_OCR,
    PROMPT_STYLE_CHANDRA_NO_OCR,
)
try:
    from utils.html_utils import (
        extract_first_balanced_table,
        postprocess_table_html,
        parse_html_table,
        annotate_empty_cells_with_token,
    )
except Exception:
    parse_html_table = None

    def extract_first_balanced_table(html: str) -> str:
        if not html:
            return ""

        lower = html.lower()
        start = lower.find("<table")
        if start < 0:
            return ""

        depth = 0
        i = start
        n = len(lower)
        while i < n:
            next_open = lower.find("<table", i)
            next_close = lower.find("</table>", i)

            if next_open != -1 and (next_close == -1 or next_open < next_close):
                depth += 1
                i = next_open + len("<table")
                continue

            if next_close != -1:
                if depth <= 0:
                    return ""
                depth -= 1
                i = next_close + len("</table>")
                if depth == 0:
                    return html[start:i]
                continue

            break

        return ""

    def postprocess_table_html(
        html: str,
        enable_repair: bool = True,
        fill_holes: bool = True,
    ) -> tuple[str, dict]:
        del enable_repair, fill_holes
        table = extract_first_balanced_table(html) or str(html or "").strip()
        return table, {
            "applied": False,
            "canonicalized": False,
            "repairs": 0,
            "issues": [],
        }

    def annotate_empty_cells_with_token(
        html: str,
        empty_token: str = "__EMPTY__",
    ) -> str:
        del empty_token
        return str(html or "").strip()

DEFAULT_USER_PROMPT = (
    "Analyze this table image and output the complete HTML structure. "
    "Make sure to correctly identify all colspan and rowspan attributes for merged cells. "
    "If a cell is empty, output __EMPTY__ inside that cell."
)
DEFAULT_MAX_NEW_TOKENS = 10000
DEFAULT_EMPTY_CELL_TOKEN = "__EMPTY__"
_REQUESTS_THREAD_LOCAL = threading.local()
_TEDS_CALC_FALLBACK = None
_TEDS_STRUCT_CALC_FALLBACK = None
_NESTED_TEDS_MODE_CHOICES = ("legacy", "split_mean")
_HTML_POSTPROCESS_MODE_CHOICES = ("off", "canonical", "repair")
_FORCE_PLAIN_PROMPT_ENV = "EVAL_FORCE_PLAIN_PROMPT"
_API_IO_LOG_PATH_ENV = "EVAL_API_IO_LOG_PATH"
_API_IO_LOG_LOCK = threading.Lock()


def _normalize_empty_cells_html(
    html: str,
    enabled: bool,
    empty_cell_token: str,
) -> str:
    """빈 셀 표기를 empty_cell_token으로 정규화한다."""
    text = str(html or "").strip()
    if not text:
        return text
    if not enabled:
        return text
    token = str(empty_cell_token or "").strip()
    if not token:
        return text
    return annotate_empty_cells_with_token(text, empty_token=token)


def _is_truthy_env(var_name: str) -> bool:
    value = str(os.getenv(var_name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _append_api_io_log(log_path: Optional[str], payload: dict) -> None:
    if not log_path:
        return
    try:
        path = Path(log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _API_IO_LOG_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 디버그 로깅 실패는 평가 흐름을 중단시키지 않는다.
        return


def _load_eval_env() -> None:
    if load_dotenv is None:
        return
    eval_root = Path(__file__).resolve().parent.parent  # train/vlm
    repo_root = eval_root.parent.parent                 # <repo>
    candidates = [
        eval_root / ".env",        # train/vlm/.env
        repo_root / "test/.env",   # test/.env
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
    load_dotenv(override=False)


_load_eval_env()


def resolve_eval_prompt(
    prompt_style: Optional[str] = None,
    prompt_override: Optional[str] = None,
    force_plain_prompt: bool = False,
) -> str:
    """평가용 프롬프트를 결정한다.

    --prompt_style이 지정되면 학습과 동일한 스타일의 프롬프트를 사용한다.
    --prompt이 명시적으로 지정되면 그것을 우선 사용한다.

    chandra_table_with_ocr는 per-sample OCR 정보가 필요하다.
    evaluate_dataset()에서 metadata.ocr_info를 이용해 샘플별 프롬프트를 구성한다.
    본 함수에서는 공통 기본 프롬프트만 반환한다.
    """
    if force_plain_prompt:
        return prompt_override or DEFAULT_USER_PROMPT

    if prompt_override and prompt_override != DEFAULT_USER_PROMPT:
        return prompt_override
    style = normalize_prompt_style(prompt_style or PROMPT_STYLE_CHANDRA_WITH_OCR)
    if style == PROMPT_STYLE_CHANDRA_WITH_OCR:
        print(
            "  Info: chandra_table_with_ocr will use per-sample OCR metadata "
            "during evaluation."
        )
        return DEFAULT_USER_PROMPT
    return get_user_prompt_with_style(
        idx=0,
        prompt_style=style,
        ocr_info=None,
        bbox_scale=1024,
    )


def _resolve_sample_prompt(
    metadata: dict,
    default_prompt: str,
    prompt_style: Optional[str] = None,
    force_plain_prompt: bool = False,
) -> tuple[str, bool]:
    """샘플 단위 프롬프트를 반환한다.

    Returns:
        (prompt_text, used_ocr_prompt)
    """
    if force_plain_prompt or _is_truthy_env(_FORCE_PLAIN_PROMPT_ENV):
        return default_prompt, False

    style = normalize_prompt_style(prompt_style or PROMPT_STYLE_CHANDRA_WITH_OCR)
    if style == PROMPT_STYLE_CHANDRA_WITH_OCR:
        md = metadata or {}
        ocr_info = md.get("ocr_info")
        bbox_scale_raw = md.get("bbox_scale", 1024)
        try:
            bbox_scale = int(bbox_scale_raw)
        except Exception:
            bbox_scale = 1024

        # 중요 정책: OCR 정보를 프롬프트에서 임의로 축소하지 않는다.
        if isinstance(ocr_info, list) and len(ocr_info) > 0:
            return (
                get_user_prompt_with_style(
                    idx=0,
                    prompt_style=PROMPT_STYLE_CHANDRA_WITH_OCR,
                    ocr_info=ocr_info,
                    bbox_scale=bbox_scale,
                ),
                True,
            )

        # OCR required style지만 샘플에 OCR이 없으면 without_ocr로 안전 폴백
        return (
            get_user_prompt_with_style(
                idx=0,
                prompt_style=PROMPT_STYLE_CHANDRA_NO_OCR,
                ocr_info=None,
                bbox_scale=bbox_scale,
            ),
            False,
        )

    return (
        get_user_prompt_with_style(
            idx=0,
            prompt_style=style,
            ocr_info=None,
            bbox_scale=1024,
        ),
        False,
    )


def _resolve_model_class(model_name: str, trust_remote_code: bool = True):
    """모델 타입에 맞는 클래스를 자동 감지한다."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    if model_type in ("qwen2_5_vl", "qwen2_vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration

    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText


def _is_lora_adapter(model_path: str) -> bool:
    return os.path.exists(os.path.join(model_path, "adapter_config.json"))


def _load_adapter_config(model_path: str) -> dict:
    cfg_path = os.path.join(model_path, "adapter_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _torch_dtype_from_config(dtype_name: str) -> torch.dtype:
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    print(f"Warning: Unknown torch dtype '{dtype_name}', falling back to bfloat16")
    return torch.bfloat16


def _build_system_prompt(enable_thinking: bool = True) -> str:
    del enable_thinking
    # NOTE: 학습(grpo_trainer)과 동일한 system prompt를 유지한다.
    # no-thinking 제어는 prompt 텍스트 변경이 아니라 chat_template 옵션으로 처리한다.
    return (
        "You are a table structure recognition expert. "
        "Analyze the table in the image and output its HTML structure "
        "with accurate colspan and rowspan attributes."
    )


def extract_html_from_response(response: str) -> str:
    """모델 응답에서 HTML 부분만 추출한다."""
    if "</think>" in response:
        html = response.split("</think>")[-1].strip()
    else:
        html = response.strip()

    table_html = extract_first_balanced_table(html)
    if table_html:
        return table_html

    return html


def _apply_html_postprocess(
    pred_html: str,
    mode: str,
) -> tuple[str, dict]:
    """예측 HTML 후처리 (off/canonical/repair)."""
    mode = str(mode or "off").strip().lower()
    if mode not in _HTML_POSTPROCESS_MODE_CHOICES:
        mode = "off"

    info = {
        "mode": mode,
        "applied": False,
        "canonicalized": False,
        "repairs": 0,
        "issues": [],
    }
    if mode == "off":
        return pred_html, info

    processed, extra = postprocess_table_html(
        pred_html,
        enable_repair=(mode == "repair"),
        # 보수 정책: repair에서도 fill_holes는 비활성화해 과보정을 줄인다.
        fill_holes=False,
    )
    if isinstance(extra, dict):
        info.update(extra)
    info["mode"] = mode
    info["issues"] = [str(x) for x in info.get("issues", [])]
    info["repairs"] = int(info.get("repairs", 0) or 0)
    info["applied"] = bool(info.get("applied", False))
    info["canonicalized"] = bool(info.get("canonicalized", False))
    return processed, info


def _resolve_image_path(image_path: str, test_data_path: str) -> str:
    """평가 JSONL의 이미지 경로를 실제 로컬 경로로 해석한다."""
    return _resolve_any_path(image_path, test_data_path)


def _resolve_any_path(raw_path: str, test_data_path: str) -> str:
    """raw 경로를 여러 기준점으로 해석해 실제 존재 경로를 반환한다."""
    if not raw_path:
        return ""

    p = Path(raw_path).expanduser()
    candidates: list[Path] = []

    # 1) 절대 경로/현재 작업 경로 기준
    candidates.append(p)

    # 2) 테스트 데이터 파일 기준 상대 경로
    test_data_parent = Path(test_data_path).resolve().parent
    candidates.append(test_data_parent / p)

    # 2-1) 테스트 데이터 상위 디렉터리 기준 상대 경로
    # 예: test_data=.../data_ocr/train_ocr.jsonl, image_path=images/xxx.jpg
    #     실제 이미지는 .../<dataset_root>/images/xxx.jpg 인 케이스 대응
    for ancestor in test_data_parent.parents:
        candidates.append(ancestor / p)
        # 너무 위까지 탐색하지 않도록 루트 직전에서 중단
        if ancestor == ancestor.parent:
            break

    # 3) train/vlm 루트 기준 상대 경로
    eval_root = Path(__file__).resolve().parent.parent
    candidates.append(eval_root / p)

    # 4) 저장소 루트 기준 상대 경로
    repo_root = eval_root.parent.parent
    candidates.append(repo_root / p)

    seen = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return key

    return ""


def _extract_table_from_full_html(full_html: str) -> str:
    """전체 HTML 문서에서 <table>...</table> 조각을 추출한다."""
    if not full_html:
        return ""

    return extract_first_balanced_table(full_html)


def _compute_teds_with_fallback(pred_html: str, gt_html: str) -> tuple[float, float, float, float]:
    """compute_teds_variants 미지원 metrics.py와의 호환 계산."""
    if compute_teds_variants is not None:
        return compute_teds_variants(pred_html=pred_html, gt_html=gt_html)

    global _TEDS_CALC_FALLBACK, _TEDS_STRUCT_CALC_FALLBACK
    if _TEDS_CALC_FALLBACK is None or _TEDS_STRUCT_CALC_FALLBACK is None:
        from eval.metrics import TEDSCalculator

        _TEDS_CALC_FALLBACK = TEDSCalculator(structure_only=False)
        _TEDS_STRUCT_CALC_FALLBACK = TEDSCalculator(structure_only=True)

    teds = _TEDS_CALC_FALLBACK.compute(pred_html, gt_html)
    teds_struct = _TEDS_STRUCT_CALC_FALLBACK.compute(pred_html, gt_html)

    try:
        from eval.metrics import normalize_table_structure

        pred_norm = normalize_table_structure(pred_html)
        gt_norm = normalize_table_structure(gt_html)
        teds_norm = _TEDS_CALC_FALLBACK.compute(pred_norm, gt_norm)
        teds_norm_struct = _TEDS_STRUCT_CALC_FALLBACK.compute(pred_norm, gt_norm)
    except Exception:
        teds_norm = teds
        teds_norm_struct = teds_struct

    return teds, teds_struct, teds_norm, teds_norm_struct


def _compute_teds_nested_split_with_fallback(
    pred_html: str,
    gt_html: str,
) -> tuple[float, float, float, float]:
    if compute_teds_variants_nested_split is not None:
        return compute_teds_variants_nested_split(pred_html=pred_html, gt_html=gt_html)
    return _compute_teds_with_fallback(pred_html=pred_html, gt_html=gt_html)


def _html_has_nested_table(html: str) -> bool:
    text = str(html or "").strip()
    if not text:
        return False

    fragment = extract_first_balanced_table(text) or text
    if "<table" not in fragment.lower():
        return False

    if parse_html_table is not None:
        try:
            structure = parse_html_table(fragment)
            return int(getattr(structure, "nested_table_count", 0) or 0) > 0
        except Exception:
            pass

    # fallback: parser 사용이 불가능하면 문자열 기반 휴리스틱
    return fragment.lower().count("<table") > 1


def _should_apply_nested_split_teds(
    complexity: Any,
    gt_html: str,
    nested_teds_mode: str,
) -> bool:
    if nested_teds_mode != "split_mean":
        return False
    comp = normalize_complexity_label(complexity)
    if comp == "complex_nested":
        return True
    # nested split은 GT 기준으로만 적용한다.
    # pred_html에 nested hallucination이 있어도 split 평가를 강제하지 않는다.
    return _html_has_nested_table(gt_html)


def _load_aihub_gt_html(
    record: dict,
    test_data_path: str,
    normalize_empty_cells: bool = False,
    empty_cell_token: str = DEFAULT_EMPTY_CELL_TOKEN,
) -> tuple[str, str]:
    """레코드의 image_path 기반으로 AI-Hub 원본 HTML GT를 로드한다."""
    image_path_raw = record.get("image_path", "")
    image_path = _resolve_any_path(str(image_path_raw), test_data_path)
    if not image_path:
        return "", ""

    html_path = str(Path(image_path).with_suffix(".html"))
    if not os.path.exists(html_path):
        return "", ""

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            full_html = f.read()
    except Exception:
        return "", ""

    table_html = _extract_table_from_full_html(full_html)
    if not table_html:
        return "", html_path

    # 기본값 속성(span=1) 제거로 GT 비교 잡음 축소
    table_html = table_html.replace(' rowspan="1"', "")
    table_html = table_html.replace(' colspan="1"', "")
    table_html = _normalize_empty_cells_html(
        table_html,
        enabled=normalize_empty_cells,
        empty_cell_token=empty_cell_token,
    )
    return table_html, html_path


def _image_path_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_openai_message_text(response_json: dict) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        return ""
    message = (choices[0] or {}).get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "".join(texts)
    return str(content) if content is not None else ""


def _build_openai_api_messages(
    image_path: str,
    prompt: str,
    enable_thinking: bool = True,
) -> list[dict]:
    image_data_url = _image_path_to_data_url(image_path)
    return _build_openai_api_messages_from_data_url(
        image_data_url=image_data_url,
        prompt=prompt,
        enable_thinking=enable_thinking,
    )


def _build_openai_api_messages_from_data_url(
    image_data_url: str,
    prompt: str,
    enable_thinking: bool = True,
) -> list[dict]:
    return [
        {"role": "system", "content": _build_system_prompt(enable_thinking=enable_thinking)},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _get_requests_session():
    import requests

    session = getattr(_REQUESTS_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=64,
            pool_maxsize=64,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _REQUESTS_THREAD_LOCAL.session = session
    return session


def _run_openai_api_inference(
    api_url: str,
    api_model: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool = True,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    timeout_sec: int = 180,
    max_retries: int = 1,
    empty_response_retries: int = 0,
    empty_retry_backoff_sec: float = 0.5,
    empty_retry_disable_thinking: bool = False,
    force_chat_template_no_thinking: bool = False,
    truncate_prompt_tokens: Optional[int] = None,
    api_io_log_path: Optional[str] = None,
    sample_index: Optional[int] = None,
    request_id: Optional[str] = None,
) -> str:
    image_data_url = _image_path_to_data_url(image_path)
    session = _get_requests_session()

    def _call_once(thinking_enabled: bool) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": api_model,
            "messages": _build_openai_api_messages_from_data_url(
                image_data_url=image_data_url,
                prompt=prompt,
                enable_thinking=thinking_enabled,
            ),
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
        }
        if truncate_prompt_tokens is not None and int(truncate_prompt_tokens) > 0:
            payload["truncate_prompt_tokens"] = int(truncate_prompt_tokens)
        if force_chat_template_no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        last_err = None
        for attempt in range(max(0, int(max_retries)) + 1):
            _append_api_io_log(
                api_io_log_path,
                {
                    "event": "request",
                    "request_id": request_id,
                    "sample_index": sample_index,
                    "attempt": attempt,
                    "thinking_enabled": bool(thinking_enabled),
                    "api_url": api_url,
                    "api_model": api_model,
                    "image_path": image_path,
                    "prompt": prompt,
                    "has_api_key": bool(api_key),
                    "payload": payload,
                    "ts": time.time(),
                },
            )
            try:
                resp = session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout_sec,
                )
                _append_api_io_log(
                    api_io_log_path,
                    {
                        "event": "response",
                        "request_id": request_id,
                        "sample_index": sample_index,
                        "attempt": attempt,
                        "status_code": int(resp.status_code),
                        "reason": str(getattr(resp, "reason", "")),
                        "body_text": resp.text,
                        "ts": time.time(),
                    },
                )
                if resp.status_code >= 500 and attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    body = resp.text
                    if len(body) > 1000:
                        body = body[:1000] + "...(truncated)"
                    raise RuntimeError(f"HTTP {resp.status_code}: {body}")
                response_json = resp.json()
                text = _extract_openai_message_text(response_json)
                _append_api_io_log(
                    api_io_log_path,
                    {
                        "event": "parsed_result",
                        "request_id": request_id,
                        "sample_index": sample_index,
                        "attempt": attempt,
                        "response_json": response_json,
                        "response_text": text,
                        "ts": time.time(),
                    },
                )
                return text
            except Exception as e:
                last_err = e
                _append_api_io_log(
                    api_io_log_path,
                    {
                        "event": "exception",
                        "request_id": request_id,
                        "sample_index": sample_index,
                        "attempt": attempt,
                        "error": repr(e),
                        "ts": time.time(),
                    },
                )
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
        if last_err:
            raise last_err
        return ""

    max_empty_retries = max(0, int(empty_response_retries))
    for empty_attempt in range(max_empty_retries + 1):
        attempt_thinking = (
            enable_thinking
            if empty_attempt == 0 or not empty_retry_disable_thinking
            else False
        )
        _append_api_io_log(
            api_io_log_path,
            {
                "event": "empty_retry_cycle",
                "request_id": request_id,
                "sample_index": sample_index,
                "empty_attempt": empty_attempt,
                "max_empty_retries": max_empty_retries,
                "thinking_enabled": bool(attempt_thinking),
                "ts": time.time(),
            },
        )
        text = _call_once(thinking_enabled=attempt_thinking)
        if (text or "").strip():
            return text

        if empty_attempt < max_empty_retries:
            wait_sec = max(0.0, float(empty_retry_backoff_sec)) * (empty_attempt + 1)
            print(
                "Warning: empty API content "
                f"(image={os.path.basename(image_path)}, "
                f"retry={empty_attempt + 1}/{max_empty_retries}, "
                f"thinking={attempt_thinking})."
            )
            if wait_sec > 0:
                time.sleep(wait_sec)
    return ""


def run_inference_openai_api_batch(
    api_url: str,
    api_model: str,
    image_paths: list[str],
    prompt: str = DEFAULT_USER_PROMPT,
    prompts: Optional[list[str]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    enable_thinking: bool = True,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    timeout_sec: int = 180,
    max_workers: int = 8,
    empty_response_retries: int = 0,
    empty_retry_backoff_sec: float = 0.5,
    empty_retry_disable_thinking: bool = False,
    force_chat_template_no_thinking: bool = False,
    truncate_prompt_tokens: Optional[int] = None,
    api_io_log_path: Optional[str] = None,
    progress_desc: Optional[str] = None,
) -> tuple[list[str], list[float]]:
    if not image_paths:
        return [], []

    workers = max(1, int(max_workers))
    workers = min(workers, len(image_paths))
    responses = [""] * len(image_paths)
    elapsed_times = [0.0] * len(image_paths)

    prompt_list = prompts if prompts is not None else []

    def _task(i: int, path: str):
        started_at = time.time()
        prompt_i = prompt_list[i] if i < len(prompt_list) else prompt
        text = _run_openai_api_inference(
            api_url=api_url,
            api_model=api_model,
            image_path=path,
            prompt=prompt_i,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            temperature=temperature,
            api_key=api_key,
            timeout_sec=timeout_sec,
            empty_response_retries=empty_response_retries,
            empty_retry_backoff_sec=empty_retry_backoff_sec,
            empty_retry_disable_thinking=empty_retry_disable_thinking,
            force_chat_template_no_thinking=force_chat_template_no_thinking,
            truncate_prompt_tokens=truncate_prompt_tokens,
            api_io_log_path=api_io_log_path,
            sample_index=i,
            request_id=f"sample_{i}_{int(started_at * 1000)}",
        )
        return i, text, time.time() - started_at

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_task, i, path): i
            for i, path in enumerate(image_paths)
        }

        completed_iter = as_completed(future_map)
        if progress_desc:
            completed_iter = tqdm(completed_iter, total=len(future_map), desc=progress_desc)

        for future in completed_iter:
            i = future_map[future]
            try:
                idx, text, elapsed = future.result()
                responses[idx] = text
                elapsed_times[idx] = float(elapsed)
            except Exception as e:
                print(f"Warning: API inference failed at batch_index={i}: {e}")
                responses[i] = ""
                elapsed_times[i] = 0.0

    return responses, elapsed_times


def load_transformers_model_for_inference(
    model_path: str,
    base_model_path: Optional[str] = None,
    max_pixels: int = 2048 * 2048,
    min_pixels: int = 512 * 512,
):
    """
    transformers 기반 추론 모델 로드.

    LoRA 어댑터인 경우 4-bit + PeftModel로 직접 추론한다.
    """
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    is_lora = _is_lora_adapter(model_path)
    if is_lora:
        if base_model_path is None:
            adapter_cfg = _load_adapter_config(model_path)
            base_model_path = adapter_cfg.get("base_model_name_or_path")
        if base_model_path is None:
            raise ValueError("LoRA adapter requires --base_model")
        model_id = base_model_path
    else:
        model_id = model_path

    ModelClass = _resolve_model_class(model_id)
    print(f"Loading model (transformers 4-bit): {model_id} ({ModelClass.__name__})")
    model = ModelClass.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map={"": 0},
    )

    if is_lora:
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )

    model.eval()
    return model, processor


def _patch_config_for_vllm(model_id: str) -> None:
    # 일부 조합에서 tie_word_embeddings 누락으로 vLLM worker가 죽는 케이스 방지
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and not hasattr(text_cfg, "tie_word_embeddings"):
            setattr(text_cfg.__class__, "tie_word_embeddings", False)
            setattr(text_cfg, "tie_word_embeddings", False)
            print("Applied vLLM compatibility patch: text_config.tie_word_embeddings=False")
    except Exception as e:
        print(f"Warning: vLLM config patch skipped ({e})")


def _ensure_vllm_runtime_env() -> None:
    mp_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "").strip().lower()
    if mp_method in ("", "fork"):
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if "VLLM_PLUGINS" not in os.environ:
        os.environ["VLLM_PLUGINS"] = ""
    if "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" not in os.environ:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def load_vllm_model_for_inference(
    model_path: str,
    base_model_path: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    max_pixels: int = 1024 * 1024,
    min_pixels: int = 512 * 512,
    max_lora_rank: int = 128,
):
    """
    vLLM 기반 추론 모델 로드.

    - LoRA adapter 평가 시 base model + lora_request 방식 사용
    - tensor_parallel_size 기본값은 가시 GPU 전체 사용
    """
    _ensure_vllm_runtime_env()

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM backend requires CUDA visible GPUs")

    from vllm import LLM

    is_lora = _is_lora_adapter(model_path)
    lora_request = None

    if is_lora:
        adapter_cfg = _load_adapter_config(model_path)
        if base_model_path is None:
            base_model_path = adapter_cfg.get("base_model_name_or_path")
        if base_model_path is None:
            raise ValueError("LoRA adapter requires --base_model")
        model_id = base_model_path
        lora_rank = int(adapter_cfg.get("r", 64))
    else:
        model_id = model_path
        lora_rank = 0

    visible_gpus = torch.cuda.device_count()
    tp_size = int(tensor_parallel_size or visible_gpus)
    if tp_size > visible_gpus:
        raise ValueError(
            f"tensor_parallel_size={tp_size} > visible CUDA devices={visible_gpus}"
        )

    _patch_config_for_vllm(model_id)

    llm_kwargs = {
        "model": model_id,
        "trust_remote_code": True,
        "tensor_parallel_size": tp_size,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_model_len": int(max_model_len),
        "limit_mm_per_prompt": {"image": 1},
        "mm_processor_kwargs": {
            "max_pixels": int(max_pixels),
            "min_pixels": int(min_pixels),
        },
        "allowed_local_media_path": "/",
    }

    if is_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max(int(max_lora_rank), int(lora_rank))

    print("Loading model with vLLM:")
    print(f"  model: {model_id}")
    print(f"  tensor_parallel_size: {tp_size}")
    print(f"  gpu_memory_utilization: {gpu_memory_utilization}")
    print(f"  max_model_len: {max_model_len}")
    print(f"  max_pixels: {max_pixels}, min_pixels: {min_pixels}")
    print(f"  lora_enabled: {is_lora}")

    llm = LLM(**llm_kwargs)

    if is_lora:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("eval_adapter", 1, model_path)

    return llm, lora_request


def _build_transformers_inputs(
    processor,
    image_path: str,
    prompt: str,
    enable_thinking: bool = True,
):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": _build_system_prompt(enable_thinking=enable_thinking)},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
    input_len = inputs["input_ids"].shape[1]
    return inputs, input_len


def run_inference_transformers(
    model,
    processor,
    image_path: str,
    prompt: str = DEFAULT_USER_PROMPT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    enable_thinking: bool = True,
) -> str:
    inputs, input_len = _build_transformers_inputs(
        processor=processor,
        image_path=image_path,
        prompt=prompt,
        enable_thinking=enable_thinking,
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    response = processor.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response


def _build_vllm_messages(
    image_path: str,
    prompt: str,
    enable_thinking: bool = True,
) -> list[dict]:
    image_uri = Path(image_path).expanduser().resolve().as_uri()
    return [
        {"role": "system", "content": _build_system_prompt(enable_thinking=enable_thinking)},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _sampling_params_for_eval(max_new_tokens: int, temperature: float = 0.0):
    from vllm import SamplingParams

    if temperature <= 0:
        return SamplingParams(temperature=0.0, max_tokens=max_new_tokens, top_p=1.0)
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=0.9,
    )


def _extract_vllm_text(output_obj: Any) -> str:
    outputs = getattr(output_obj, "outputs", None)
    if not outputs:
        return ""
    text = getattr(outputs[0], "text", "")
    return text if isinstance(text, str) else ""


def run_inference_vllm_batch(
    llm,
    image_paths: list[str],
    prompt: str = DEFAULT_USER_PROMPT,
    prompts: Optional[list[str]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    enable_thinking: bool = True,
    temperature: float = 0.0,
    lora_request=None,
) -> list[str]:
    prompt_list = prompts if prompts is not None else []
    messages_batch = [
        _build_vllm_messages(
            image_path=img_path,
            prompt=(prompt_list[idx] if idx < len(prompt_list) else prompt),
            enable_thinking=enable_thinking,
        )
        for idx, img_path in enumerate(image_paths)
    ]
    sampling_params = _sampling_params_for_eval(
        max_new_tokens=max_new_tokens, temperature=temperature
    )

    try:
        outputs = llm.chat(
            messages=messages_batch,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        if isinstance(outputs, list):
            return [_extract_vllm_text(o) for o in outputs]
    except Exception as batch_err:
        print(f"Warning: vLLM batch chat failed, fallback to per-sample ({batch_err})")

    responses = []
    for messages in messages_batch:
        try:
            output = llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            if isinstance(output, list) and output:
                responses.append(_extract_vllm_text(output[0]))
            else:
                responses.append("")
        except Exception:
            responses.append("")
    return responses


def _build_prediction_record(
    sample: dict,
    response: str,
    elapsed: float,
    nested_teds_mode: str,
    html_postprocess_mode: str,
    normalize_empty_cells: bool = False,
    empty_cell_token: str = DEFAULT_EMPTY_CELL_TOKEN,
) -> dict:
    """추론 응답으로부터 메트릭 계산 + prediction 레코드 생성."""
    pred_html_raw = extract_html_from_response(response)
    pred_html, postprocess_info = _apply_html_postprocess(
        pred_html=pred_html_raw,
        mode=html_postprocess_mode,
    )
    gt_html = sample["gt_html"]
    pred_html_eval = _normalize_empty_cells_html(
        pred_html,
        enabled=normalize_empty_cells,
        empty_cell_token=empty_cell_token,
    )
    gt_html_eval = _normalize_empty_cells_html(
        gt_html,
        enabled=normalize_empty_cells,
        empty_cell_token=empty_cell_token,
    )
    complexity = sample.get("metadata", {}).get("complexity", "unknown")

    span_metrics = compute_span_metrics(pred_html_eval, gt_html_eval)
    grid_metrics = compute_grid_metrics(pred_html_eval, gt_html_eval)
    legacy_teds, legacy_teds_struct, legacy_teds_norm, legacy_teds_norm_struct = _compute_teds_with_fallback(
        pred_html=pred_html_eval,
        gt_html=gt_html_eval,
    )
    use_nested_split = _should_apply_nested_split_teds(
        complexity=complexity,
        gt_html=gt_html_eval,
        nested_teds_mode=nested_teds_mode,
    )

    split_teds = split_teds_struct = split_teds_norm = split_teds_norm_struct = None
    if use_nested_split:
        split_teds, split_teds_struct, split_teds_norm, split_teds_norm_struct = (
            _compute_teds_nested_split_with_fallback(
                pred_html=pred_html_eval,
                gt_html=gt_html_eval,
            )
        )

    teds_score = split_teds if split_teds is not None else legacy_teds
    teds_struct_score = split_teds_struct if split_teds_struct is not None else legacy_teds_struct
    teds_norm = split_teds_norm if split_teds_norm is not None else legacy_teds_norm
    teds_norm_struct = (
        split_teds_norm_struct if split_teds_norm_struct is not None else legacy_teds_norm_struct
    )

    record = {
        "index": sample["index"],
        "image_path": sample["image_path"],
        "image_path_raw": sample.get("image_path_raw", sample["image_path"]),
        "pred_html_raw": pred_html_raw,
        "pred_html": pred_html,
        "pred_html_eval": pred_html_eval,
        "gt_html": gt_html,
        "gt_html_eval": gt_html_eval,
        "gt_source": sample.get("gt_source", "dataset"),
        "gt_html_path": sample.get("gt_html_path", ""),
        "full_response": response,
        "complexity": complexity,
        "teds": teds_score,
        "teds_structure": teds_struct_score,
        "teds_norm": teds_norm,
        "teds_norm_structure": teds_norm_struct,
        "teds_legacy": legacy_teds,
        "teds_structure_legacy": legacy_teds_struct,
        "teds_norm_legacy": legacy_teds_norm,
        "teds_norm_structure_legacy": legacy_teds_norm_struct,
        "nested_teds_mode": nested_teds_mode,
        "nested_teds_applied": use_nested_split,
        "postprocess_mode": postprocess_info.get("mode", html_postprocess_mode),
        "postprocess_applied": bool(postprocess_info.get("applied", False)),
        "postprocess_canonicalized": bool(postprocess_info.get("canonicalized", False)),
        "postprocess_repairs": int(postprocess_info.get("repairs", 0) or 0),
        "postprocess_issues": postprocess_info.get("issues", []),
        "empty_cell_normalized": bool(normalize_empty_cells),
        "empty_cell_token": empty_cell_token if normalize_empty_cells else "",
        "span_f1": span_metrics.span_f1,
        "span_precision": span_metrics.span_precision,
        "span_recall": span_metrics.span_recall,
        "position_f1": span_metrics.position_f1,
        "attribute_accuracy": span_metrics.attribute_accuracy,
        "colspan_accuracy": span_metrics.colspan_accuracy,
        "rowspan_accuracy": span_metrics.rowspan_accuracy,
        "gca": grid_metrics.gca,
        "gsa": grid_metrics.gsa,
        "inference_time": elapsed,
    }
    if split_teds is not None:
        record["teds_split"] = split_teds
        record["teds_structure_split"] = split_teds_struct
        record["teds_norm_split"] = split_teds_norm
        record["teds_norm_structure_split"] = split_teds_norm_struct
    return record


def _resolve_batch_prompts(
    chunk: list[dict],
    default_prompt: str,
    prompt_style: str,
    force_plain_prompt: bool = False,
) -> tuple[list[str], int]:
    """배치 내 각 샘플의 프롬프트를 결정하고 OCR 사용 수를 반환한다."""
    prompts = []
    ocr_count = 0
    for sample in chunk:
        sample_prompt, used_ocr = _resolve_sample_prompt(
            metadata=sample.get("metadata", {}),
            default_prompt=default_prompt,
            prompt_style=prompt_style,
            force_plain_prompt=force_plain_prompt,
        )
        prompts.append(sample_prompt)
        if used_ocr:
            ocr_count += 1
    return prompts, ocr_count


def _load_eval_samples(
    test_data_path: str,
    max_samples: Optional[int] = None,
    gt_source: str = "aihub",
    strict_aihub_gt: bool = False,
    normalize_empty_cells: bool = False,
    empty_cell_token: str = DEFAULT_EMPTY_CELL_TOKEN,
) -> list[dict]:
    records = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    samples = []
    aihub_loaded = 0
    aihub_missing = 0
    aihub_parse_failed = 0
    for i, record in enumerate(records):
        image_path_raw = record.get("image_path", "")
        gt_html = record.get("gt_html", "")
        metadata: dict = {
            "complexity": record.get("complexity", ""),
            "prompt_style": record.get("prompt_style", ""),
        }
        if "ocr_info" in record:
            metadata["ocr_info"] = record["ocr_info"]
        if "bbox_scale" in record:
            metadata["bbox_scale"] = record["bbox_scale"]
        image_path = _resolve_image_path(image_path_raw, test_data_path)
        gt_source_used = "dataset"
        gt_html_path = ""

        if gt_source in {"aihub", "auto"}:
            aihub_gt, aihub_html_path = _load_aihub_gt_html(
                record,
                test_data_path,
                normalize_empty_cells=normalize_empty_cells,
                empty_cell_token=empty_cell_token,
            )
            if aihub_gt:
                gt_html = aihub_gt
                gt_source_used = "aihub"
                gt_html_path = aihub_html_path
                aihub_loaded += 1
            else:
                if aihub_html_path:
                    aihub_parse_failed += 1
                else:
                    aihub_missing += 1
                if gt_source == "aihub" and strict_aihub_gt:
                    raise ValueError(
                        f"AI-Hub GT load failed at sample index={i}, "
                        f"image_path={image_path_raw}"
                    )

        gt_html = _normalize_empty_cells_html(
            gt_html,
            enabled=normalize_empty_cells,
            empty_cell_token=empty_cell_token,
        )

        if not image_path or not os.path.exists(image_path):
            print(f"  Skipping {i}: image not found ({image_path_raw})")
            continue

        samples.append(
            {
                "index": i,
                "image_path": image_path,
                "image_path_raw": image_path_raw,
                "gt_html": gt_html,
                "gt_source": gt_source_used,
                "gt_html_path": gt_html_path,
                "metadata": metadata,
            }
        )

    if gt_source in {"aihub", "auto"}:
        print(
            "  GT source summary: "
            f"aihub_loaded={aihub_loaded}, "
            f"aihub_missing={aihub_missing}, "
            f"aihub_parse_failed={aihub_parse_failed}, "
            f"dataset_fallback={len(samples) - aihub_loaded}"
        )
    return samples


def _write_predictions_jsonl(path: str, predictions: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _collect_postprocess_stats(predictions: list[dict]) -> dict:
    total = len(predictions)
    applied = sum(1 for p in predictions if bool(p.get("postprocess_applied", False)))
    canonicalized = sum(
        1 for p in predictions if bool(p.get("postprocess_canonicalized", False))
    )
    repairs = sum(int(p.get("postprocess_repairs", 0) or 0) for p in predictions)
    mode = "off"
    if predictions:
        mode = str(predictions[0].get("postprocess_mode", "off"))
    return {
        "html_postprocess_mode": mode,
        "html_postprocess_applied_samples": applied,
        "html_postprocess_applied_ratio": (applied / total) if total > 0 else 0.0,
        "html_postprocess_canonicalized_samples": canonicalized,
        "html_postprocess_total_repairs": repairs,
    }


def _build_metrics_dict_for_subset(
    predictions: list[dict],
    backend: str,
    batch_size: int,
    max_new_tokens: int,
    gt_source: str,
    strict_aihub_gt: bool,
    nested_teds_mode: str,
    normalize_empty_cells: bool = False,
    empty_cell_token: str = DEFAULT_EMPTY_CELL_TOKEN,
) -> dict:
    agg = compute_aggregate_metrics(predictions)
    metrics_dict = agg.to_dict()
    total_time = sum(float(p.get("inference_time", 0.0) or 0.0) for p in predictions)
    metrics_dict["avg_inference_time"] = total_time / max(len(predictions), 1)
    metrics_dict["backend"] = backend
    metrics_dict["batch_size"] = int(max(1, batch_size))
    metrics_dict["max_new_tokens"] = int(max_new_tokens)
    metrics_dict["gt_source"] = gt_source
    metrics_dict["strict_aihub_gt"] = bool(strict_aihub_gt)
    metrics_dict["nested_teds_mode"] = nested_teds_mode
    metrics_dict["normalize_empty_cells"] = bool(normalize_empty_cells)
    metrics_dict["empty_cell_token"] = (
        str(empty_cell_token or DEFAULT_EMPTY_CELL_TOKEN)
        if normalize_empty_cells
        else ""
    )
    metrics_dict.update(_collect_postprocess_stats(predictions))
    return metrics_dict


def _write_complexity_artifacts(
    output_dir: str,
    predictions: list[dict],
    backend: str,
    batch_size: int,
    max_new_tokens: int,
    gt_source: str,
    strict_aihub_gt: bool,
    nested_teds_mode: str,
    normalize_empty_cells: bool = False,
    empty_cell_token: str = DEFAULT_EMPTY_CELL_TOKEN,
) -> None:
    # base 버킷은 항상 생성하고(simple/medium/complex),
    # 세분화 라벨(complex_col/row/mix/nested)은 별도 파일로 추가 생성한다.
    base_order = ["simple", "medium", "complex"]
    detail_order = ["complex_nested", "complex_col", "complex_row", "complex_mix"]

    by_base: dict[str, list] = {c: [] for c in base_order}
    by_raw: dict[str, list] = {}
    for pred in predictions:
        raw_comp = normalize_complexity_label(pred.get("complexity", "unknown"))
        by_raw.setdefault(raw_comp, []).append(pred)

        base_comp = to_base_complexity_bucket(raw_comp)
        if base_comp in by_base:
            by_base[base_comp].append(pred)

    ordered_raw = [c for c in detail_order if c in by_raw]
    extra_raw = sorted(
        c for c in by_raw.keys()
        if c not in set(base_order) and c not in set(detail_order)
    )
    ordered_raw.extend(extra_raw)

    artifact_sets: list[tuple[str, list]] = []
    for comp in base_order:
        artifact_sets.append((comp, by_base[comp]))
    for comp in ordered_raw:
        artifact_sets.append((comp, by_raw[comp]))

    for complexity, subset in artifact_sets:
        pred_path = os.path.join(output_dir, f"predictions_{complexity}.jsonl")
        _write_predictions_jsonl(pred_path, subset)

        metrics_path = os.path.join(output_dir, f"metrics_{complexity}.json")
        metrics_dict = _build_metrics_dict_for_subset(
            predictions=subset,
            backend=backend,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            gt_source=gt_source,
            strict_aihub_gt=strict_aihub_gt,
            nested_teds_mode=nested_teds_mode,
            normalize_empty_cells=normalize_empty_cells,
            empty_cell_token=empty_cell_token,
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)


def evaluate_dataset(
    model,
    processor,
    test_data_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    enable_thinking: bool = True,
    backend: str = "transformers",
    batch_size: int = 8,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    prompt: str = DEFAULT_USER_PROMPT,
    lora_request=None,
    api_url: Optional[str] = None,
    api_model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_timeout: int = 180,
    prompt_style: Optional[str] = None,
    gt_source: str = "aihub",
    strict_aihub_gt: bool = False,
    api_empty_response_retries: int = 0,
    api_empty_retry_backoff: float = 0.5,
    api_empty_retry_disable_thinking: bool = False,
    api_force_chat_template_no_thinking: bool = False,
    api_truncate_prompt_tokens: Optional[int] = None,
    api_io_log_path: Optional[str] = None,
    nested_teds_mode: str = "split_mean",
    html_postprocess_mode: str = "off",
    force_plain_prompt: bool = False,
    normalize_empty_cells: bool = False,
    empty_cell_token: str = DEFAULT_EMPTY_CELL_TOKEN,
):
    """테스트 데이터셋 전체를 평가한다."""
    if nested_teds_mode not in _NESTED_TEDS_MODE_CHOICES:
        raise ValueError(
            f"nested_teds_mode must be one of {_NESTED_TEDS_MODE_CHOICES}, "
            f"got: {nested_teds_mode}"
        )
    if html_postprocess_mode not in _HTML_POSTPROCESS_MODE_CHOICES:
        raise ValueError(
            f"html_postprocess_mode must be one of {_HTML_POSTPROCESS_MODE_CHOICES}, "
            f"got: {html_postprocess_mode}"
        )

    os.makedirs(output_dir, exist_ok=True)
    samples = _load_eval_samples(
        test_data_path,
        max_samples=max_samples,
        gt_source=gt_source,
        strict_aihub_gt=strict_aihub_gt,
        normalize_empty_cells=normalize_empty_cells,
        empty_cell_token=empty_cell_token,
    )

    print(f"Evaluating {len(samples)} samples...")
    print(f"  backend: {backend}")
    print(f"  batch_size: {batch_size}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  gt_source: {gt_source}")
    force_plain_prompt = bool(force_plain_prompt or _is_truthy_env(_FORCE_PLAIN_PROMPT_ENV))
    api_io_log_path = (
        str(api_io_log_path or os.getenv(_API_IO_LOG_PATH_ENV, "")).strip() or None
    )

    print(f"  nested_teds_mode: {nested_teds_mode}")
    print(f"  html_postprocess_mode: {html_postprocess_mode}")
    print(f"  force_plain_prompt: {force_plain_prompt}")
    print(f"  normalize_empty_cells: {normalize_empty_cells}")
    if normalize_empty_cells:
        print(f"  empty_cell_token: {empty_cell_token}")
    normalized_prompt_style = normalize_prompt_style(
        prompt_style or PROMPT_STYLE_CHANDRA_WITH_OCR
    )
    if prompt_style:
        print(f"  prompt_style: {normalized_prompt_style}")
    if backend == "api":
        print(f"  api_url: {api_url}")
        print(f"  api_model: {api_model}")
        print(f"  api_empty_response_retries: {api_empty_response_retries}")
        print(f"  api_empty_retry_backoff: {api_empty_retry_backoff}")
        print(f"  api_empty_retry_disable_thinking: {api_empty_retry_disable_thinking}")
        print(
            f"  api_force_chat_template_no_thinking: "
            f"{api_force_chat_template_no_thinking}"
        )
        if api_truncate_prompt_tokens is not None and int(api_truncate_prompt_tokens) > 0:
            print(f"  api_truncate_prompt_tokens: {int(api_truncate_prompt_tokens)}")
        if api_io_log_path:
            print(f"  api_io_log_path: {api_io_log_path}")

    if normalized_prompt_style == PROMPT_STYLE_CHANDRA_WITH_OCR and not force_plain_prompt:
        ocr_ready = 0
        for s in samples:
            ocr_info = (s.get("metadata") or {}).get("ocr_info")
            if isinstance(ocr_info, list) and len(ocr_info) > 0:
                ocr_ready += 1
        print(f"  OCR prompt-ready samples: {ocr_ready}/{len(samples)}")

    predictions = []
    total_time = 0.0

    if backend == "vllm":
        eff_batch_size = max(1, int(batch_size))
        ocr_prompt_used_total = 0
        for start in tqdm(range(0, len(samples), eff_batch_size), desc="Evaluating(vLLM)"):
            chunk = samples[start : start + eff_batch_size]
            image_paths = [s["image_path"] for s in chunk]
            chunk_prompts, ocr_count = _resolve_batch_prompts(
                chunk,
                prompt,
                normalized_prompt_style,
                force_plain_prompt=force_plain_prompt,
            )
            ocr_prompt_used_total += ocr_count

            chunk_start = time.time()
            responses = run_inference_vllm_batch(
                llm=model,
                image_paths=image_paths,
                prompt=prompt,
                prompts=chunk_prompts,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                temperature=temperature,
                lora_request=lora_request,
            )
            chunk_elapsed = time.time() - chunk_start
            total_time += chunk_elapsed

            per_sample_time = chunk_elapsed / max(1, len(chunk))
            for sample, response in zip(chunk, responses):
                predictions.append(
                    _build_prediction_record(
                        sample,
                        response,
                        per_sample_time,
                        nested_teds_mode=nested_teds_mode,
                        html_postprocess_mode=html_postprocess_mode,
                        normalize_empty_cells=normalize_empty_cells,
                        empty_cell_token=empty_cell_token,
                    )
                )
        if normalized_prompt_style == PROMPT_STYLE_CHANDRA_WITH_OCR and not force_plain_prompt:
            print(f"  OCR prompts used: {ocr_prompt_used_total}/{len(samples)}")
    elif backend == "api":
        if not api_url:
            raise ValueError("backend=api requires --api_url")
        if not api_model:
            raise ValueError("backend=api requires --api_model or --model")
        if api_io_log_path:
            _append_api_io_log(
                api_io_log_path,
                {
                    "event": "run_start",
                    "api_url": api_url,
                    "api_model": api_model,
                    "sample_count": len(samples),
                    "batch_size": int(eff_batch_size) if "eff_batch_size" in locals() else int(max(1, batch_size)),
                    "force_plain_prompt": bool(force_plain_prompt),
                    "prompt_style": normalized_prompt_style,
                    "ts": time.time(),
                },
            )

        eff_batch_size = max(1, int(batch_size))
        image_paths = [s["image_path"] for s in samples]
        prompts_all = []
        ocr_prompt_used_total = 0
        for sample in samples:
            sample_prompt, used_ocr = _resolve_sample_prompt(
                metadata=sample.get("metadata", {}),
                default_prompt=prompt,
                prompt_style=normalized_prompt_style,
                force_plain_prompt=force_plain_prompt,
            )
            prompts_all.append(sample_prompt)
            if used_ocr:
                ocr_prompt_used_total += 1

        inference_started = time.time()
        responses, _elapsed_times = run_inference_openai_api_batch(
            api_url=api_url,
            api_model=api_model,
            image_paths=image_paths,
            prompt=prompt,
            prompts=prompts_all,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            temperature=temperature,
            api_key=api_key,
            timeout_sec=api_timeout,
            max_workers=eff_batch_size,
            empty_response_retries=api_empty_response_retries,
            empty_retry_backoff_sec=api_empty_retry_backoff,
            empty_retry_disable_thinking=api_empty_retry_disable_thinking,
            force_chat_template_no_thinking=api_force_chat_template_no_thinking,
            truncate_prompt_tokens=api_truncate_prompt_tokens,
            api_io_log_path=api_io_log_path,
            progress_desc="Evaluating(API)",
        )
        inference_elapsed = time.time() - inference_started
        total_time += inference_elapsed
        per_sample_time = inference_elapsed / max(1, len(samples))

        for sample, response in zip(samples, responses):
            predictions.append(
                _build_prediction_record(
                    sample,
                    response,
                    per_sample_time,
                    nested_teds_mode=nested_teds_mode,
                    html_postprocess_mode=html_postprocess_mode,
                    normalize_empty_cells=normalize_empty_cells,
                    empty_cell_token=empty_cell_token,
                )
            )
        if normalized_prompt_style == PROMPT_STYLE_CHANDRA_WITH_OCR and not force_plain_prompt:
            print(f"  OCR prompts used: {ocr_prompt_used_total}/{len(samples)}")
    else:
        ocr_prompt_used_total = 0
        for sample in tqdm(samples, desc="Evaluating(transformers)"):
            start_time = time.time()
            try:
                sample_prompt, used_ocr = _resolve_sample_prompt(
                    metadata=sample.get("metadata", {}),
                    default_prompt=prompt,
                    prompt_style=normalized_prompt_style,
                    force_plain_prompt=force_plain_prompt,
                )
                if used_ocr:
                    ocr_prompt_used_total += 1
                response = run_inference_transformers(
                    model=model,
                    processor=processor,
                    image_path=sample["image_path"],
                    prompt=sample_prompt,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                )
            except Exception as e:
                print(f"  Error on {sample['index']}: {e}")
                response = ""
            elapsed = time.time() - start_time
            total_time += elapsed

            predictions.append(
                _build_prediction_record(
                    sample,
                    response,
                    elapsed,
                    nested_teds_mode=nested_teds_mode,
                    html_postprocess_mode=html_postprocess_mode,
                    normalize_empty_cells=normalize_empty_cells,
                    empty_cell_token=empty_cell_token,
                )
            )
        if normalized_prompt_style == PROMPT_STYLE_CHANDRA_WITH_OCR and not force_plain_prompt:
            print(f"  OCR prompts used: {ocr_prompt_used_total}/{len(samples)}")

    agg = compute_aggregate_metrics(predictions)

    pred_path = os.path.join(output_dir, "predictions.jsonl")
    _write_predictions_jsonl(pred_path, predictions)

    metrics_path = os.path.join(output_dir, "metrics.json")
    metrics_dict = agg.to_dict()
    metrics_dict["avg_inference_time"] = total_time / max(len(predictions), 1)
    metrics_dict["backend"] = backend
    metrics_dict["batch_size"] = int(max(1, batch_size))
    metrics_dict["max_new_tokens"] = int(max_new_tokens)
    metrics_dict["gt_source"] = gt_source
    metrics_dict["strict_aihub_gt"] = bool(strict_aihub_gt)
    metrics_dict["nested_teds_mode"] = nested_teds_mode
    metrics_dict["force_plain_prompt"] = bool(force_plain_prompt)
    metrics_dict["normalize_empty_cells"] = bool(normalize_empty_cells)
    metrics_dict["empty_cell_token"] = (
        str(empty_cell_token or DEFAULT_EMPTY_CELL_TOKEN)
        if normalize_empty_cells
        else ""
    )
    metrics_dict.update(_collect_postprocess_stats(predictions))
    if backend == "api":
        metrics_dict["api_empty_response_retries"] = int(api_empty_response_retries)
        metrics_dict["api_empty_retry_backoff"] = float(api_empty_retry_backoff)
        metrics_dict["api_empty_retry_disable_thinking"] = bool(
            api_empty_retry_disable_thinking
        )
        metrics_dict["api_force_chat_template_no_thinking"] = bool(
            api_force_chat_template_no_thinking
        )
        metrics_dict["api_truncate_prompt_tokens"] = int(
            api_truncate_prompt_tokens
            if api_truncate_prompt_tokens is not None
            else 0
        )
        metrics_dict["api_io_log_path"] = str(api_io_log_path or "")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    _write_complexity_artifacts(
        output_dir=output_dir,
        predictions=predictions,
        backend=backend,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        gt_source=gt_source,
        strict_aihub_gt=strict_aihub_gt,
        nested_teds_mode=nested_teds_mode,
        normalize_empty_cells=normalize_empty_cells,
        empty_cell_token=empty_cell_token,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples:              {agg.total_samples}")
    print(f"  Backend:              {backend}")
    print(f"  Nested TEDS Mode:     {nested_teds_mode}")
    print(f"  Avg TEDS:             {agg.avg_teds:.4f}")
    print(f"  Avg TEDS-Structure:   {agg.avg_teds_structure:.4f}")
    print(f"  Avg TEDS-Norm:        {agg.avg_teds_norm:.4f}")
    print(f"  Avg TEDS-Norm-S:      {agg.avg_teds_norm_structure:.4f}")
    print(f"  HTML Postprocess:     {html_postprocess_mode}")
    print(
        "  Postprocess Applied:  "
        f"{metrics_dict.get('html_postprocess_applied_samples', 0)}/"
        f"{len(predictions)}"
    )
    print(
        "  Postprocess Repairs:  "
        f"{metrics_dict.get('html_postprocess_total_repairs', 0)}"
    )
    print(f"  Avg Span F1:          {agg.avg_span_f1:.4f}")
    print(f"  Avg Span Precision:   {agg.avg_span_precision:.4f}")
    print(f"  Avg Span Recall:      {agg.avg_span_recall:.4f}")
    print(f"  Avg Position F1:      {agg.avg_position_f1:.4f}")
    print(f"  Avg Attr Accuracy:    {agg.avg_attribute_accuracy:.4f}")
    print(f"  Avg Colspan Accuracy: {agg.avg_colspan_accuracy:.4f}")
    print(f"  Avg Rowspan Accuracy: {agg.avg_rowspan_accuracy:.4f}")
    print(f"  ---")
    print(f"  Avg GCA:              {agg.avg_gca:.4f}")
    print(f"  Avg GSA:              {agg.avg_gsa:.4f}")
    print(f"  ---")
    print(f"  Simple TEDS:          {agg.simple_teds:.4f}")
    print(f"  Medium TEDS:          {agg.medium_teds:.4f}")
    print(f"  Complex TEDS:         {agg.complex_teds:.4f}")
    print(f"  ---")
    print(f"  Avg Inference Time:   {metrics_dict['avg_inference_time']:.2f}s")
    print("=" * 60)
    print("  Complexity artifacts:")
    print("    predictions_simple.jsonl / metrics_simple.json")
    print("    predictions_medium.jsonl / metrics_medium.json")
    print("    predictions_complex.jsonl / metrics_complex.json")
    print("    predictions_complex_{nested|col|row|mix}.jsonl (if present)")
    print("=" * 60)

    clear_metrics_caches()

    return agg


def main():
    parser = argparse.ArgumentParser(description="모델 평가")
    parser.add_argument("--model", required=True, help="모델/어댑터 경로")
    parser.add_argument("--base_model", default=None, help="베이스 모델 경로 (LoRA 시)")
    parser.add_argument("--test_data", required=True, help="테스트 JSONL")
    parser.add_argument("--output_dir", default="eval_results/")
    parser.add_argument("--max_samples", type=int, default=None, help="최대 평가 샘플 수")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--backend", choices=["transformers", "vllm", "api"], default="transformers")
    parser.add_argument("--batch_size", type=int, default=8, help="vLLM 배치 추론 크기")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--min_pixels", type=int, default=512 * 512)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_lora_rank", type=int, default=128)
    parser.add_argument("--api_url", default=os.getenv("TABLE_VLM_URL", ""))
    parser.add_argument("--api_model", default=os.getenv("TABLE_VLM_MODEL", ""))
    parser.add_argument(
        "--api_key",
        default=(
            os.getenv("TABLE_VLM_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
            or os.getenv("OPENROUTER_API_KEY", "")
        ),
    )
    parser.add_argument("--api_timeout", type=int, default=180)
    parser.add_argument(
        "--api_empty_response_retries",
        type=int,
        default=0,
        help="API 응답 content가 비어 있을 때 추가 재시도 횟수",
    )
    parser.add_argument(
        "--api_empty_retry_backoff",
        type=float,
        default=0.5,
        help="빈 응답 재시도 간 백오프 시작값(초). 재시도 번호만큼 선형 증가",
    )
    parser.add_argument(
        "--api_empty_retry_disable_thinking",
        action="store_true",
        help="빈 응답 재시도 시 thinking을 비활성화한다.",
    )
    parser.add_argument(
        "--api_force_chat_template_no_thinking",
        action="store_true",
        help=(
            "API payload에 chat_template_kwargs.enable_thinking=false를 "
            "강제로 포함한다."
        ),
    )
    parser.add_argument(
        "--api_truncate_prompt_tokens",
        type=int,
        default=0,
        help=(
            "API payload에 truncate_prompt_tokens를 전달한다. "
            "0이면 비활성화."
        ),
    )
    parser.add_argument(
        "--api_io_log_path",
        type=str,
        default="",
        help=(
            "API 요청/응답 디버그 로그 JSONL 경로. "
            "또는 환경변수 EVAL_API_IO_LOG_PATH 사용 가능."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument(
        "--force_plain_prompt",
        action="store_true",
        help=(
            "샘플별 OCR 스타일 프롬프트 생성을 비활성화하고 "
            "--prompt 값을 모든 샘플에 그대로 사용한다."
        ),
    )
    parser.add_argument(
        "--prompt_style",
        default=PROMPT_STYLE_CHANDRA_WITH_OCR,
        choices=[
            PROMPT_STYLE_CHANDRA_WITH_OCR,
            PROMPT_STYLE_CHANDRA_NO_OCR,
        ],
        help=(
            "프롬프트 스타일 "
            "(chandra_table_with_ocr/chandra_table_without_ocr). "
            "학습 config의 prompting.style과 일치시켜야 정확한 평가 가능."
        ),
    )
    parser.add_argument(
        "--gt_source",
        choices=["dataset", "aihub", "auto"],
        default="aihub",
        help=(
            "GT HTML 소스 선택. "
            "aihub: metadata.image_path 기반 원본 AI-Hub .html 우선 사용, "
            "dataset: JSONL assistant GT 사용, "
            "auto: AI-Hub 로드 성공 시 사용하고 실패 시 dataset으로 폴백."
        ),
    )
    parser.add_argument(
        "--strict_aihub_gt",
        action="store_true",
        help="--gt_source=aihub에서 원본 HTML 로드 실패 시 즉시 오류로 중단한다.",
    )
    parser.add_argument(
        "--normalize_empty_cells",
        action="store_true",
        help=(
            "평가 전 GT/Pred HTML의 빈 td/th 셀을 empty_cell_token으로 "
            "정규화한다."
        ),
    )
    parser.add_argument(
        "--empty_cell_token",
        type=str,
        default=DEFAULT_EMPTY_CELL_TOKEN,
        help="빈 셀 표기에 사용할 토큰 (기본: __EMPTY__).",
    )
    parser.add_argument(
        "--nested_teds_mode",
        choices=list(_NESTED_TEDS_MODE_CHOICES),
        default="split_mean",
        help=(
            "nested 샘플 TEDS 계산 방식. "
            "split_mean: outer/inner table을 분리해 (outer + inner_avg)/2 계산, "
            "legacy: 기존 단일 트리 기반 계산."
        ),
    )
    parser.add_argument(
        "--html_postprocess_mode",
        choices=list(_HTML_POSTPROCESS_MODE_CHOICES),
        default="off",
        help=(
            "예측 HTML 후처리 모드. "
            "off: 비활성화, canonical: 정규화만, repair: 정규화+구조 보정."
        ),
    )
    args = parser.parse_args()
    force_plain_prompt = bool(
        args.force_plain_prompt or _is_truthy_env(_FORCE_PLAIN_PROMPT_ENV)
    )

    if args.backend == "vllm":
        model, lora_request = load_vllm_model_for_inference(
            model_path=args.model,
            base_model_path=args.base_model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
            max_lora_rank=args.max_lora_rank,
        )
        processor = None
    elif args.backend == "api":
        model = args.model
        processor = None
        lora_request = None
    else:
        model, processor = load_transformers_model_for_inference(
            model_path=args.model,
            base_model_path=args.base_model,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
        )
        lora_request = None

    eval_prompt = resolve_eval_prompt(
        prompt_style=args.prompt_style,
        prompt_override=args.prompt,
        force_plain_prompt=force_plain_prompt,
    )
    if args.prompt_style:
        print(f"  Prompt style: {args.prompt_style}")
        print(f"  Resolved prompt: {eval_prompt[:80]}...")
    if force_plain_prompt:
        print(f"  Force plain prompt: true ({_FORCE_PLAIN_PROMPT_ENV}=1 compatible)")

    evaluate_dataset(
        model=model,
        processor=processor,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        enable_thinking=not args.no_thinking,
        backend=args.backend,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prompt=eval_prompt,
        prompt_style=args.prompt_style,
        lora_request=lora_request,
        api_url=args.api_url,
        api_model=(args.api_model or args.model),
        api_key=(args.api_key or None),
        api_timeout=args.api_timeout,
        api_empty_response_retries=args.api_empty_response_retries,
        api_empty_retry_backoff=args.api_empty_retry_backoff,
        api_empty_retry_disable_thinking=args.api_empty_retry_disable_thinking,
        api_force_chat_template_no_thinking=args.api_force_chat_template_no_thinking,
        api_truncate_prompt_tokens=(
            args.api_truncate_prompt_tokens
            if int(args.api_truncate_prompt_tokens) > 0
            else None
        ),
        api_io_log_path=(args.api_io_log_path or None),
        gt_source=args.gt_source,
        strict_aihub_gt=args.strict_aihub_gt,
        nested_teds_mode=args.nested_teds_mode,
        html_postprocess_mode=args.html_postprocess_mode,
        force_plain_prompt=force_plain_prompt,
        normalize_empty_cells=args.normalize_empty_cells,
        empty_cell_token=args.empty_cell_token,
    )


if __name__ == "__main__":
    main()
