"""
Teacher 모델 합성 데이터 생성기

학습된 Teacher 모델로 대량의 테이블 이미지를 추론하여
Student 학습용 합성 데이터를 생성한다.

Features:
- 이미지당 다수 생성 후 consistency check
- TEDS 기반 최적 응답 선택
- 자동 thinking chain 포함
- 진행률 추적 및 중간 저장

Usage:
    python -m distill.teacher_generate \
        --config config/distill_config.yaml \
        --image_dir data/extra_images/ \
        --output data/distill/synthetic_raw.jsonl
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import TEDSCalculator
from utils.html_utils import parse_html_table, normalize_html
from utils.span_analyzer import analyze_span
from utils.prompt_templates import (
    get_system_prompt,
    get_user_prompt_with_style,
    normalize_prompt_style,
    prompt_requires_ocr,
    PROMPT_STYLE_CHANDRA_WITH_OCR,
)


# =============================================================================
# Model Loading
# =============================================================================


def _parse_version_tuple(version: str) -> tuple[int, int, int]:
    """'4.57.1.dev0' -> (4, 57, 1) 형태로 변환."""
    nums = []
    for part in re.split(r"[^0-9]+", version):
        if not part:
            continue
        nums.append(int(part))
        if len(nums) == 3:
            break
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    """문자열 dtype을 torch dtype으로 변환."""
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    print(f"Warning: Unknown dtype '{dtype_name}', fallback to bfloat16")
    return torch.bfloat16


def _fix_moe_expert_weights(
    model,
    model_name: str,
    revision: Optional[str] = None,
) -> tuple[int, int]:
    """
    MoE expert weight shape mismatch를 자동 복구한다.

    일부 checkpoint는 expert weight가 [experts, out, in] 레이아웃으로 저장되어 있고,
    모델은 [experts, in, out]을 기대한다. 이 경우 마지막 2축 전치로 맞춘다.
    """
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except Exception as e:
        raise RuntimeError(
            "MoE auto-fix requires 'huggingface_hub' and 'safetensors'."
        ) from e

    model_dir = snapshot_download(
        repo_id=model_name,
        revision=revision,
        local_files_only=True,
    )
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise RuntimeError(f"Missing index file: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})

    # shapemismatch가 발생한 MoE expert projection만 타겟팅
    target_keys = []
    for key in weight_map.keys():
        if ".mlp.experts." not in key:
            continue
        if key.endswith(".down_proj") or key.endswith(".gate_up_proj"):
            target_keys.append(key)

    if not target_keys:
        return 0, 0

    keys_by_file: dict[str, list[str]] = {}
    for key in target_keys:
        fname = weight_map[key]
        keys_by_file.setdefault(fname, []).append(key)

    param_dict = dict(model.named_parameters())
    fixed = 0
    checked = 0

    for fname, keys in keys_by_file.items():
        file_path = os.path.join(model_dir, fname)
        with safe_open(file_path, framework="pt", device="cpu") as sf:
            for key in keys:
                checked += 1
                ckpt_tensor = sf.get_tensor(key)

                param = param_dict.get(key)
                if param is None:
                    for prefix in ("model.", "base_model.model."):
                        param = param_dict.get(prefix + key)
                        if param is not None:
                            break
                if param is None:
                    continue

                if tuple(ckpt_tensor.shape) == tuple(param.shape):
                    continue

                transposed = ckpt_tensor.transpose(-1, -2).contiguous()
                if tuple(transposed.shape) != tuple(param.shape):
                    continue

                with torch.no_grad():
                    param.data.copy_(transposed.to(device=param.device, dtype=param.dtype))
                fixed += 1

    return fixed, checked


def _build_quantization_config(quant_type: Optional[str]):
    """BitsAndBytes 양자화 설정을 생성한다. None이면 양자화 없음."""
    if not quant_type:
        return None

    quant_type = str(quant_type).strip().lower()
    if quant_type in ("4bit", "4", "nf4"):
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_type in ("8bit", "8", "int8"):
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        print(f"Warning: Unknown quantization '{quant_type}', loading without quantization")
        return None


def _model_from_pretrained(
    ModelClass,
    model_name: str,
    dtype: torch.dtype,
    device_map,
    revision: Optional[str] = None,
    force_download: bool = False,
    quantization_config=None,
):
    """dtype/torch_dtype 파라미터 호환성을 처리하는 모델 로더."""
    common_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if revision:
        common_kwargs["revision"] = revision
    if force_download:
        common_kwargs["force_download"] = True
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config

    def _load(**extra):
        try:
            return ModelClass.from_pretrained(
                model_name, dtype=dtype, **common_kwargs, **extra,
            )
        except TypeError:
            return ModelClass.from_pretrained(
                model_name, torch_dtype=dtype, **common_kwargs, **extra,
            )

    try:
        return _load()
    except RuntimeError as e:
        if "ignore_mismatched_sizes" not in str(e):
            raise
        print(
            "\n[MoE mismatch detected] Trying auto-fix: "
            "load with ignore_mismatched_sizes=True + expert transpose..."
        )
        try:
            model = _load(ignore_mismatched_sizes=True)
            fixed, checked = _fix_moe_expert_weights(
                model=model,
                model_name=model_name,
                revision=revision,
            )
            print(f"[MoE mismatch auto-fix] fixed={fixed}, checked={checked}")
            if fixed == 0:
                print(
                    "[MoE mismatch auto-fix] No weights were transposed. "
                    "If generation quality is abnormal, update transformers."
                )
            return model
        except Exception as fix_err:
            import transformers
            raise RuntimeError(
                "\n"
                "============================================================\n"
                " MoE expert weight shape mismatch!\n"
                "============================================================\n"
                f" 현재 transformers 버전: {transformers.__version__}\n"
                " 자동 복구(ignore_mismatched_sizes + transpose)를 시도했지만 실패했습니다.\n"
                "\n"
                " 해결 방법:\n"
                "   pip install -U 'git+https://github.com/huggingface/transformers'\n"
                "   또는 호환되는 버전으로 재설치 후 재실행\n"
                "\n"
                f" 원본 에러: {e}\n"
                f" 복구 에러: {fix_err}\n"
                "============================================================\n"
            ) from fix_err


def _resolve_model_class(model_name: str, trust_remote_code: bool = True):
    """모델 타입에 맞는 클래스를 자동 감지한다."""
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif model_type in ("qwen3_vl_moe", "qwen3vl_moe"):
        min_version = (4, 57, 0)
        cur_version = _parse_version_tuple(transformers.__version__)
        if cur_version < min_version:
            raise RuntimeError(
                "Qwen3-VL-235B(A22B, MoE) requires newer transformers. "
                f"Detected {transformers.__version__}, need >= {'.'.join(map(str, min_version))}. "
                "Please upgrade transformers."
            )
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration
            return Qwen3VLMoeForConditionalGeneration
        except Exception as e:
            raise RuntimeError(
                "Failed to import Qwen3VLMoeForConditionalGeneration. "
                "Please upgrade transformers to the latest release."
            ) from e
    elif model_type in ("qwen2_5_vl", "qwen2_vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText


def _load_vllm_model(teacher_cfg: dict):
    """vLLM 엔진으로 Teacher 모델을 로드한다 (FP8/대형 모델 권장)."""
    # vLLM + CUDA에서는 fork 방식이 worker 초기화 실패를 일으킬 수 있다.
    # (RuntimeError: Cannot re-initialize CUDA in forked subprocess)
    mp_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "").strip().lower()
    if mp_method == "fork":
        print(
            "  Warning: VLLM_WORKER_MULTIPROC_METHOD=fork is incompatible with CUDA workers. "
            "Overriding to 'spawn'."
        )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    elif mp_method == "":
        # 기본값을 spawn으로 명시
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    print(f"  VLLM_WORKER_MULTIPROC_METHOD: {os.environ['VLLM_WORKER_MULTIPROC_METHOD']}")

    # 외부 패키지가 vLLM plugin entrypoint를 등록한 경우(예: paddlex),
    # 무관한 plugin import 오류 로그가 반복될 수 있으므로 기본적으로 plugin auto-load를 끈다.
    if "VLLM_PLUGINS" not in os.environ:
        os.environ["VLLM_PLUGINS"] = ""
    print(f"  VLLM_PLUGINS: {os.environ.get('VLLM_PLUGINS', '<unset>')!r}")

    # paddlex 모델 호스트 연결 체크 로그를 기본 비활성화한다.
    if "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" not in os.environ:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    # Python multiprocessing start method도 spawn으로 맞춘다.
    # (일부 환경에서 fork로 고정되면 CUDA 초기화가 실패할 수 있음)
    try:
        import multiprocessing as mp

        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method != "spawn":
            if current_start_method == "fork":
                print("  Warning: multiprocessing start method is 'fork'. Overriding to 'spawn'.")
            mp.set_start_method("spawn", force=True)
            current_start_method = mp.get_start_method(allow_none=True)
        print(f"  multiprocessing start method: {current_start_method}")
    except Exception as e:
        print(f"  Warning: Failed to enforce multiprocessing start method to 'spawn': {e}")

    # NumPy major 버전은 경고만 출력한다.
    # (vLLM 0.15.x는 NumPy 2.x + OpenCV 4.13.x 조합을 사용할 수 있음)
    try:
        import numpy as np
        np_version = str(np.__version__)
        print(f"  NumPy version: {np_version}")
        np_major = int(np_version.split(".")[0])
        if np_major >= 2:
            print(
                "  Warning: NumPy 2.x detected. "
                "If import errors occur, reinstall pandas/pyarrow/scikit-learn/opencv to match NumPy ABI."
            )
    except Exception as e:
        raise RuntimeError(f"NumPy runtime check failed: {e}") from e

    # vLLM worker가 사용할 GPU 가시성 확인
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. "
            "vLLM backend requires visible CUDA devices."
        )
    visible_gpus = torch.cuda.device_count()

    try:
        from vllm import LLM
    except Exception as e:
        raise RuntimeError(
            "Failed to import vLLM runtime dependencies.\n"
            "Likely causes: NumPy ABI mismatch or incompatible binary wheels.\n"
            "Recommended fix:\n"
            "  pip install -U --force-reinstall numpy pandas pyarrow scikit-learn opencv-python-headless\n"
            "  pip install -U --force-reinstall vllm\n"
            f"Import error: {e}"
        ) from e

    model_path = teacher_cfg["name_or_path"]
    tp_size = int(teacher_cfg.get("tensor_parallel_size", visible_gpus))
    gpu_util = float(teacher_cfg.get("gpu_memory_utilization", 0.85))
    max_model_len = int(teacher_cfg.get("max_model_len", 10240))
    max_pixels = int(teacher_cfg.get("max_pixels", 1024 * 1024))
    min_pixels = int(teacher_cfg.get("min_pixels", 512 * 512))
    disable_car_cfg = teacher_cfg.get("disable_custom_all_reduce")
    if disable_car_cfg is None:
        disable_custom_all_reduce = str(
            os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "0")
        ).lower() in ("1", "true", "yes", "on")
    else:
        disable_custom_all_reduce = bool(disable_car_cfg)

    if tp_size > visible_gpus:
        raise RuntimeError(
            f"tensor_parallel_size={tp_size} > visible CUDA devices={visible_gpus}. "
            "Adjust teacher.tensor_parallel_size or CUDA_VISIBLE_DEVICES."
        )

    # vLLM(0.15.x) + 최신 transformers 조합에서
    # Qwen3VLMoeTextConfig.tie_word_embeddings 누락으로 worker 초기화가 실패할 수 있다.
    # 모델 로드 전에 text_config 클래스에 기본 속성을 주입해 호환성을 맞춘다.
    def _patch_qwen3_text_config_tie_word_embeddings() -> bool:
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is None:
                return False
            if hasattr(text_cfg, "tie_word_embeddings"):
                return False
            # class-level patch로 vLLM 내부에서 새로 생성되는 config 객체에도 적용되도록 한다.
            setattr(text_cfg.__class__, "tie_word_embeddings", False)
            setattr(text_cfg, "tie_word_embeddings", False)
            print("  Applied vLLM compatibility patch: text_config.tie_word_embeddings=False")
            return True
        except Exception as patch_err:
            print(f"  Warning: Failed to apply vLLM config compatibility patch: {patch_err}")
            return False

    _patch_qwen3_text_config_tie_word_embeddings()

    # vLLM import 경로에서 mistral_common -> cv2 import가 발생한다.
    # cv2/numpy ABI가 어긋나면 "_ARRAY_API not found"로 실패하므로 먼저 명시적으로 검사한다.
    try:
        import cv2  # noqa: F401
    except Exception as e:
        err = str(e)
        if "_ARRAY_API not found" in err or "numpy.core._multiarray_umath" in err:
            raise RuntimeError(
                "OpenCV/NumPy ABI mismatch detected before vLLM init.\n"
                "Current environment has incompatible binary wheels.\n\n"
                "Recommended fix:\n"
                "  pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless\n"
                "  pip install -U --force-reinstall \"numpy==2.2.6\" \"opencv-python-headless==4.13.0.92\"\n"
                "  pip install -U --force-reinstall pandas pyarrow scikit-learn vllm\n\n"
                "Then verify:\n"
                "  python -c \"import numpy, cv2; print(numpy.__version__, cv2.__version__)\"\n"
                "  python -c \"from vllm import LLM; print('vLLM import ok')\""
            ) from e
        raise RuntimeError(f"OpenCV import failed before vLLM init: {e}") from e

    print(f"Loading model with vLLM: {model_path}")
    print(f"  tensor_parallel_size: {tp_size}")
    print(f"  gpu_memory_utilization: {gpu_util}")
    print(f"  max_model_len: {max_model_len}")
    print(f"  max_pixels: {max_pixels}, min_pixels: {min_pixels}")
    print(f"  disable_custom_all_reduce: {disable_custom_all_reduce}")

    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_model_len,
            disable_custom_all_reduce=disable_custom_all_reduce,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
            },
            allowed_local_media_path="/",
        )
    except Exception as e:
        err = str(e)
        if "tie_word_embeddings" in err or "Qwen3VLMoeTextConfig" in err:
            raise RuntimeError(
                "vLLM/transformers compatibility error while loading Qwen3-VL MoE.\n"
                "Detected missing 'tie_word_embeddings' in Qwen3VLMoeTextConfig.\n"
                "Recommended actions:\n"
                "  1) Upgrade vLLM to a build compatible with your transformers version, or\n"
                "  2) Pin transformers to a vLLM-compatible version.\n"
                "Also keep VLLM_PLUGINS='' to disable external plugin auto-loading."
            ) from e
        raise

    print("vLLM model loaded successfully.")
    return llm


def _load_teacher_model_transformers(teacher_cfg: dict):
    """transformers backend로 Teacher 모델을 로드한다."""
    from transformers import AutoProcessor
    from peft import PeftModel

    model_path = teacher_cfg["name_or_path"]
    base_model = teacher_cfg.get("base_model")
    dtype = _torch_dtype_from_name(teacher_cfg.get("torch_dtype", "bfloat16"))
    device_map = teacher_cfg.get("device_map", "auto")
    revision = teacher_cfg.get("revision")
    force_download = bool(teacher_cfg.get("force_download", False))
    quant_type = teacher_cfg.get("quantization")
    quantization_config = _build_quantization_config(quant_type)
    if quantization_config:
        print(f"Quantization: {quant_type}")
    processor_kwargs = {
        "trust_remote_code": True,
        "max_pixels": 2048 * 2048,
        "min_pixels": 512 * 512,
    }
    if revision:
        processor_kwargs["revision"] = revision
    if force_download:
        processor_kwargs["force_download"] = True

    # LoRA 어댑터 여부 확인
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora and base_model:
        ModelClass = _resolve_model_class(base_model)
        print(f"Loading base model: {base_model} ({ModelClass.__name__})")
        model = _model_from_pretrained(
            ModelClass,
            base_model,
            dtype=dtype,
            device_map=device_map,
            revision=revision,
            force_download=force_download,
            quantization_config=quantization_config,
        )
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained(base_model, **processor_kwargs)
    else:
        ModelClass = _resolve_model_class(model_path)
        print(f"Loading model: {model_path} ({ModelClass.__name__})")
        model = _model_from_pretrained(
            ModelClass,
            model_path,
            dtype=dtype,
            device_map=device_map,
            revision=revision,
            force_download=force_download,
            quantization_config=quantization_config,
        )
        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

    model.eval()
    print(f"Teacher model loaded. Device: {next(model.parameters()).device}")
    return model, processor


def load_teacher_model(config: dict):
    """Teacher 모델을 로드한다. backend 설정에 따라 vLLM 또는 transformers 사용."""
    teacher_cfg = config["teacher"]
    backend = str(teacher_cfg.get("backend", "transformers")).strip().lower()
    # vLLM 대형 모델(235B)에서 fallback으로 transformers를 쓰면 OOM 가능성이 매우 높다.
    # 따라서 명시적으로 true를 주지 않으면 fallback을 비활성화한다.
    fallback_to_transformers = bool(teacher_cfg.get("fallback_to_transformers", False))

    if backend == "vllm":
        try:
            llm = _load_vllm_model(teacher_cfg)
            return llm, None  # processor는 vLLM 내부에서 처리
        except Exception as e:
            if not fallback_to_transformers:
                raise
            fallback_cfg = dict(teacher_cfg)
            explicit_fallback_name = teacher_cfg.get("fallback_name_or_path")
            if explicit_fallback_name:
                fallback_cfg["name_or_path"] = explicit_fallback_name
            else:
                model_name = str(teacher_cfg.get("name_or_path", ""))
                if model_name.endswith("-FP8"):
                    fallback_cfg["name_or_path"] = model_name[:-4]
            print(
                "\n[Teacher backend fallback] vLLM initialization failed. "
                "Falling back to transformers backend."
            )
            print(f"  fallback model: {fallback_cfg.get('name_or_path')}")
            print(f"  vLLM error: {e}\n")
            return _load_teacher_model_transformers(fallback_cfg)
    elif backend != "transformers":
        print(f"Warning: Unknown teacher.backend='{backend}', fallback to transformers.")

    return _load_teacher_model_transformers(teacher_cfg)


# =============================================================================
# Inference
# =============================================================================


def _build_chat_messages(
    image_path: str,
    prompt_style: str = PROMPT_STYLE_CHANDRA_WITH_OCR,
    ocr_info: Optional[list] = None,
    bbox_scale: int = 1024,
) -> list[dict]:
    """vLLM/transformers 공통 OpenAI 형식 메시지를 생성한다."""
    # 한글/공백 경로에서도 안전하도록 file URI를 표준 인코딩한다.
    try:
        image_uri = Path(image_path).resolve().as_uri()
    except Exception:
        image_uri = f"file://{os.path.abspath(image_path)}"

    return [
        {"role": "system", "content": get_system_prompt()},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_uri},
                },
                {"type": "text", "text": get_user_prompt_with_style(
                    prompt_style=prompt_style,
                    ocr_info=ocr_info,
                    bbox_scale=bbox_scale,
                )},
            ],
        },
    ]


def _generate_vllm(
    llm,
    image_path: str,
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
    prompt_style: str = PROMPT_STYLE_CHANDRA_WITH_OCR,
    ocr_info: Optional[list] = None,
    bbox_scale: int = 1024,
) -> str:
    """vLLM으로 단일 이미지 추론."""
    from vllm import SamplingParams

    messages = _build_chat_messages(
        image_path, prompt_style=prompt_style,
        ocr_info=ocr_info, bbox_scale=bbox_scale,
    )

    sampling_params = SamplingParams(
        temperature=0 if temperature <= 0 else temperature,
        max_tokens=max_new_tokens,
        top_p=0.9 if temperature > 0 else 1.0,
    )

    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    return outputs[0].outputs[0].text


def _generate_vllm_batch(
    llm,
    image_path: str,
    n_gens: int,
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
    prompt_style: str = PROMPT_STYLE_CHANDRA_WITH_OCR,
    ocr_info: Optional[list] = None,
    bbox_scale: int = 1024,
) -> list[str]:
    """vLLM으로 이미지 1장에 대해 N개 응답을 효율적으로 배치 생성한다."""
    from vllm import SamplingParams

    messages = _build_chat_messages(
        image_path, prompt_style=prompt_style,
        ocr_info=ocr_info, bbox_scale=bbox_scale,
    )

    responses = []

    # 첫 번째: greedy
    greedy_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    output = llm.chat(messages=messages, sampling_params=greedy_params)
    responses.append(output[0].outputs[0].text)

    if n_gens > 1:
        # 나머지: temperature sampling, n 파라미터로 한 번에 생성
        sample_params = SamplingParams(
            n=n_gens - 1,
            temperature=max(temperature, 0.3),
            max_tokens=max_new_tokens,
            top_p=0.9,
        )
        output = llm.chat(messages=messages, sampling_params=sample_params)
        for comp in output[0].outputs:
            responses.append(comp.text)

    return responses


def _generate_transformers(
    model,
    processor,
    image_path: str,
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
    prompt_style: str = PROMPT_STYLE_CHANDRA_WITH_OCR,
    ocr_info: Optional[list] = None,
    bbox_scale: int = 1024,
) -> str:
    """transformers로 단일 이미지 추론."""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": get_user_prompt_with_style(
                    prompt_style=prompt_style,
                    ocr_info=ocr_info,
                    bbox_scale=bbox_scale,
                )},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    target_device = next(model.parameters()).device
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    ).to(target_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9 if temperature > 0 else 1.0,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    )
    return response


def generate_single(
    model,
    processor,
    image_path: str,
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
    enable_thinking: bool = True,
    prompt_style: str = PROMPT_STYLE_CHANDRA_WITH_OCR,
    ocr_info: Optional[list] = None,
    bbox_scale: int = 1024,
) -> str:
    """단일 이미지에 대해 Teacher 추론 (vLLM/transformers 자동 디스패치)."""
    if processor is None:
        # vLLM backend
        return _generate_vllm(
            model, image_path, temperature=temperature,
            max_new_tokens=max_new_tokens, prompt_style=prompt_style,
            ocr_info=ocr_info, bbox_scale=bbox_scale,
        )
    # transformers backend
    return _generate_transformers(
        model, processor, image_path, temperature=temperature,
        max_new_tokens=max_new_tokens, prompt_style=prompt_style,
        ocr_info=ocr_info, bbox_scale=bbox_scale,
    )


def _normalize_bbox(bbox, bbox_scale: int = 1024) -> list[int]:
    """bbox를 [x0,y0,x1,y1] 형태의 int list로 정규화."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    try:
        vals = [int(round(float(v))) for v in bbox]
    except Exception:
        return []
    lo, hi = 0, int(bbox_scale)
    return [
        max(lo, min(hi, vals[0])),
        max(lo, min(hi, vals[1])),
        max(lo, min(hi, vals[2])),
        max(lo, min(hi, vals[3])),
    ]


def _extract_ocr_items_from_json(data, bbox_scale: int = 1024) -> list[dict]:
    """
    다양한 스키마에서 OCR item(text,bbox)을 추출한다.
    기대 포맷: [{"text": "...", "bbox": [x0,y0,x1,y1]}, ...]
    """
    candidates = []

    def _collect(value):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("ocr") or item.get("content")
                    bbox = item.get("bbox") or item.get("box") or item.get("quad")
                    nb = _normalize_bbox(bbox, bbox_scale=bbox_scale)
                    if text is not None and nb:
                        candidates.append({"text": str(text), "bbox": nb})
        elif isinstance(value, dict):
            for v in value.values():
                _collect(v)

    if isinstance(data, dict):
        # 우선순위 높은 키
        for key in ("ocr_info", "ocr", "ocr_items", "words", "tokens"):
            if key in data:
                _collect(data[key])
        if not candidates:
            _collect(data)
    elif isinstance(data, list):
        _collect(data)

    # 중복 제거 (text+bbox)
    uniq = []
    seen = set()
    for item in candidates:
        key = (item["text"], tuple(item["bbox"]))
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq


def _find_ocr_json_path(image_path: str) -> Optional[str]:
    """
    이미지 경로로부터 OCR json 경로를 추정한다.
    1) 동일 경로, 확장자만 .json
    2) /01.원천데이터/ -> /02.라벨링데이터/ 치환
    """
    p = Path(image_path)
    cand1 = p.with_suffix(".json")
    if cand1.exists():
        return str(cand1)

    s = str(p)
    if "01.원천데이터" in s:
        cand2 = Path(s.replace("01.원천데이터", "02.라벨링데이터")).with_suffix(".json")
        if cand2.exists():
            return str(cand2)

    return None


def load_ocr_info_for_image(image_path: str, bbox_scale: int = 1024) -> list[dict]:
    """이미지의 sidecar json에서 OCR 정보를 로드한다. 없으면 빈 리스트."""
    json_path = _find_ocr_json_path(image_path)
    if not json_path:
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _extract_ocr_items_from_json(data, bbox_scale=bbox_scale)
    except Exception:
        return []


def extract_html_from_response(response: str) -> str:
    """응답에서 HTML 추출."""
    if "</think>" in response:
        html_part = response.split("</think>")[-1].strip()
    else:
        html_part = response.strip()

    match = re.search(r"<table[\s\S]*?</table>", html_part, re.IGNORECASE)
    return match.group(0) if match else html_part


# =============================================================================
# Consistency Selection
# =============================================================================


def select_best_response(
    responses: list[str],
    min_consistency: float = 0.8,
) -> Optional[dict]:
    """
    여러 응답 중 가장 일관되고 품질 높은 응답을 선택한다.

    Strategy:
    1. 각 응답에서 HTML 추출
    2. 쌍별 TEDS-Structure 계산
    3. 평균 TEDS가 가장 높은 응답 선택 (다수결 효과)
    4. 최소 consistency 미달 시 None 반환

    Returns:
        {"response": str, "html": str, "avg_consistency": float} or None
    """
    teds_calc = TEDSCalculator(structure_only=True)

    htmls = []
    valid_indices = []
    for i, resp in enumerate(responses):
        html = extract_html_from_response(resp)
        try:
            structure = parse_html_table(html)
            if structure.num_rows >= 2 and structure.num_cols >= 2:
                htmls.append(html)
                valid_indices.append(i)
        except Exception:
            continue

    if len(valid_indices) < 2:
        # 유효 응답이 1개 이하면 있는 것이라도 반환
        if valid_indices:
            idx = valid_indices[0]
            return {
                "response": responses[idx],
                "html": htmls[0],
                "avg_consistency": 1.0,
            }
        return None

    # 쌍별 TEDS 계산
    best_idx = 0
    best_avg = 0.0

    for i in range(len(htmls)):
        scores = []
        for j in range(len(htmls)):
            if i != j:
                score = teds_calc.compute(htmls[i], htmls[j])
                scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.0
        if avg > best_avg:
            best_avg = avg
            best_idx = i

    if best_avg < min_consistency:
        return None

    orig_idx = valid_indices[best_idx]
    return {
        "response": responses[orig_idx],
        "html": htmls[best_idx],
        "avg_consistency": best_avg,
    }


# =============================================================================
# Batch Generation
# =============================================================================


def collect_image_paths(sources: list[str]) -> list[str]:
    """이미지 소스 디렉토리/파일 목록에서 이미지 경로를 수집."""
    extensions = {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"}
    paths = set()

    for source in sources:
        if os.path.isfile(source):
            paths.add(source)
        elif os.path.isdir(source):
            for ext in extensions:
                paths.update(glob.glob(os.path.join(source, ext)))
                paths.update(glob.glob(os.path.join(source, "**", ext), recursive=True))
        else:
            # glob 패턴
            paths.update(glob.glob(source))

    return sorted(paths)


def generate_synthetic_dataset(
    model,
    processor,
    config: dict,
    image_paths: Optional[list[str]] = None,
):
    """
    대량의 합성 데이터를 생성한다.

    Features:
    - 이미지당 N회 생성 → consistency check → 최적 선택
    - 중간 저장 (100개마다)
    - 실패/스킵 추적
    """
    gen_cfg = config["synthetic_generation"]

    if image_paths is None:
        image_paths = collect_image_paths(gen_cfg["image_sources"])

    output_path = gen_cfg["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_gens = gen_cfg.get("num_generations_per_image", 3)
    temperature = gen_cfg.get("temperature", 0.3)
    max_tokens = gen_cfg.get("max_new_tokens", 4096)
    enable_thinking = gen_cfg.get("enable_thinking", True)
    target_samples = gen_cfg.get("target_samples")
    max_oom_errors = int(gen_cfg.get("max_oom_errors", 5))
    max_errors = int(gen_cfg.get("max_errors", 200))
    min_consistency = config.get("quality_filter", {}).get("min_consistency_teds", 0.8)
    prompting_cfg = config.get("prompting", {})
    raw_prompt_style = prompting_cfg.get("style", PROMPT_STYLE_CHANDRA_WITH_OCR)
    prompt_style = normalize_prompt_style(raw_prompt_style)
    bbox_scale = int(prompting_cfg.get("bbox_scale", 1024))
    use_ocr = prompt_requires_ocr(prompt_style)
    if raw_prompt_style != prompt_style:
        print(
            f"  Warning: unsupported prompting.style='{raw_prompt_style}', "
            f"fallback to '{prompt_style}'."
        )

    print(f"Generating synthetic data:")
    print(f"  Images: {len(image_paths)}")
    print(f"  Generations per image: {n_gens}")
    print(f"  Temperature: {temperature}")
    print(f"  Prompt style: {prompt_style}")
    print(f"  OCR mode: {'on' if use_ocr else 'off'}")
    if target_samples:
        print(f"  Target samples: {target_samples}")
    print(f"  Output: {output_path}")

    if not image_paths:
        print("  No images found. Check synthetic_generation.image_sources in config.")
        return []

    results = []
    stats = {
        "success": 0,
        "low_consistency": 0,
        "error": 0,
        "oom": 0,
        "skipped": 0,
        "target_stop": 0,
    }
    total_time = 0

    # 기존 결과 이어서 생성
    processed_images = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_images.add(item.get("image_path", ""))
                    results.append(item)
        print(f"  Resuming: {len(results)} already generated")

    if target_samples and len(results) >= target_samples:
        print(f"  Target already satisfied by existing output ({len(results)} >= {target_samples})")
        return results

    output_file = open(output_path, "a", encoding="utf-8")

    try:
        for i, img_path in enumerate(tqdm(image_paths, desc="Generating")):
            if target_samples and len(results) >= target_samples:
                stats["target_stop"] += 1
                break

            if img_path in processed_images:
                stats["skipped"] += 1
                continue

            start = time.time()

            try:
                ocr_info = load_ocr_info_for_image(img_path, bbox_scale=bbox_scale) if use_ocr else []

                # N회 생성 (vLLM: 배치 생성, transformers: 순차 생성)
                if processor is None:
                    # vLLM 배치 생성 (greedy 1 + sampled n-1)
                    responses = _generate_vllm_batch(
                        model, img_path, n_gens=n_gens,
                        temperature=temperature,
                        max_new_tokens=max_tokens,
                        prompt_style=prompt_style,
                        ocr_info=ocr_info,
                        bbox_scale=bbox_scale,
                    )
                else:
                    responses = []
                    for gen_idx in range(n_gens):
                        resp = generate_single(
                            model, processor, img_path,
                            temperature=temperature if gen_idx > 0 else 0.0,
                            max_new_tokens=max_tokens,
                            enable_thinking=enable_thinking,
                            prompt_style=prompt_style,
                            ocr_info=ocr_info,
                            bbox_scale=bbox_scale,
                        )
                        responses.append(resp)

                # 최적 응답 선택
                best = select_best_response(responses, min_consistency)

                if best is None:
                    stats["low_consistency"] += 1
                    continue

                # Span 분석
                try:
                    span_stats = analyze_span(best["html"])
                    complexity = span_stats.complexity
                    complexity_score = span_stats.complexity_score
                except Exception:
                    complexity = "unknown"
                    complexity_score = 0.0

                record = {
                    "image_path": img_path,
                    "response": best["response"],
                    "html": best["html"],
                    "avg_consistency": best["avg_consistency"],
                    "complexity": complexity,
                    "complexity_score": complexity_score,
                    "source": "teacher",
                    "n_generations": n_gens,
                }
                if use_ocr:
                    record["ocr_info"] = ocr_info
                    record["bbox_scale"] = bbox_scale

                # 스트리밍 저장
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                results.append(record)
                stats["success"] += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    stats["oom"] += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if stats["oom"] <= 10:
                        print(f"\n  OOM on {img_path}: {e}")
                    if stats["oom"] >= max_oom_errors:
                        raise RuntimeError(
                            "Repeated CUDA OOM during generation. "
                            f"oom_count={stats['oom']} (max_oom_errors={max_oom_errors}).\n"
                            "Suggestions:\n"
                            "  1) Fix vLLM environment and use backend=vllm\n"
                            "  2) Reduce synthetic_generation.max_new_tokens\n"
                            "  3) Use smaller teacher or quantization (4bit/8bit)\n"
                            "  4) Disable OCR prompt for smoke test (prompting.style=chandra_table_without_ocr)"
                        ) from e
                else:
                    stats["error"] += 1
                    if stats["error"] <= 10:
                        print(f"\n  Error on {img_path}: {e}")
                    if stats["error"] >= max_errors:
                        raise RuntimeError(
                            "Too many generation errors in a row. "
                            f"error_count={stats['error']} (max_errors={max_errors}).\n"
                            "Likely causes:\n"
                            "  1) vLLM multimodal input/path format issue (file:// URI)\n"
                            "  2) Broken runtime dependency (vLLM/cv2/numpy)\n"
                            "  3) Invalid image files in image_sources\n"
                            "Check the first printed 'Error on ...' message for root cause."
                        ) from e
            except Exception as e:
                stats["error"] += 1
                if stats["error"] <= 10:
                    print(f"\n  Error on {img_path}: {e}")
                if stats["error"] >= max_errors:
                    raise RuntimeError(
                        "Too many generation errors in a row. "
                        f"error_count={stats['error']} (max_errors={max_errors}).\n"
                        "Likely causes:\n"
                        "  1) vLLM multimodal input/path format issue (file:// URI)\n"
                        "  2) Broken runtime dependency (vLLM/cv2/numpy)\n"
                        "  3) Invalid image files in image_sources\n"
                        "Check the first printed 'Error on ...' message for root cause."
                    ) from e

            elapsed = time.time() - start
            total_time += elapsed

            # 진행 상황 출력
            if (i + 1) % 100 == 0:
                avg_time = total_time / max(i + 1 - stats["skipped"], 1)
                print(
                    f"\n  [{i+1}/{len(image_paths)}] "
                    f"success={len(results)}, "
                    f"low_consistency={stats['low_consistency']}, "
                    f"errors={stats['error']}, "
                    f"avg_time={avg_time:.1f}s/img"
                )

    finally:
        output_file.close()

    # 최종 통계
    print(f"\nGeneration complete:")
    print(f"  Total images:     {len(image_paths)}")
    print(f"  Success:          {len(results)}")
    print(f"  Low consistency:  {stats['low_consistency']}")
    print(f"  OOM:              {stats['oom']}")
    print(f"  Errors:           {stats['error']}")
    print(f"  Skipped (resume): {stats['skipped']}")
    if target_samples and stats["target_stop"] > 0:
        print(f"  Stopped at target_samples={target_samples}")
    print(f"  Output: {output_path}")

    return results


def main():
    import yaml

    parser = argparse.ArgumentParser(description="Teacher 합성 데이터 생성")
    parser.add_argument("--config", default="config/distill_config.yaml")
    parser.add_argument("--image_dir", default=None, help="이미지 디렉토리 (config 오버라이드)")
    parser.add_argument("--output", default=None, help="출력 경로 (config 오버라이드)")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.output:
        config["synthetic_generation"]["output_path"] = args.output

    # 모델 로드
    model, processor = load_teacher_model(config)

    # 이미지 수집
    sources = config["synthetic_generation"]["image_sources"]
    if args.image_dir:
        sources = [args.image_dir]
    image_paths = collect_image_paths(sources)

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    # 생성
    generate_synthetic_dataset(model, processor, config, image_paths)


if __name__ == "__main__":
    main()
