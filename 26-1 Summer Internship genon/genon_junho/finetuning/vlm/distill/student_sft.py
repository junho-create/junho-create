"""
Student SFT (Supervised Fine-Tuning)

Teacher가 생성한 합성 데이터로 Student(소형) 모델을 학습한다.
기존 train_qlora.py의 로직을 재활용하되, distill_config.yaml의
Student 설정을 사용한다.

Usage:
    python -m distill.student_sft --config config/distill_config.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.monitoring import (
    audit_long_table_capacity,
    write_json_report,
    write_training_progress_html,
)

MIN_ENFORCED_MAX_SEQ_LENGTH = 8192

try:
    from transformers import TrainerCallback as _HFTrainerCallback
except Exception:
    class _HFTrainerCallback:  # pragma: no cover
        pass


def _resolve_model_class(model_name: str, trust_remote_code: bool = True):
    """모델 타입에 맞는 클래스를 자동 감지한다."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif model_type in ("qwen2_5_vl", "qwen2_vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText


def _torch_dtype_from_config(dtype_name: str) -> torch.dtype:
    """문자열 dtype을 torch dtype으로 안전하게 변환한다."""
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    print(f"Warning: Unknown torch dtype '{dtype_name}', falling back to bfloat16")
    return torch.bfloat16


def _configure_sdpa_runtime(attn_implementation: str) -> None:
    """SDPA runtime backend를 안전하게 설정한다.

    목적:
    - `attn_implementation=sdpa`는 유지
    - cuDNN SDPA plan build 오류(`No valid execution plans built`)가 날 경우를
      피하기 위해 cuDNN backend를 기본 비활성화
    """
    if str(attn_implementation or "").strip().lower() != "sdpa":
        return
    if not torch.cuda.is_available():
        return

    disable_cudnn = str(os.environ.get("TSR_DISABLE_CUDNN_SDPA", "1")).strip().lower()
    disable_cudnn = disable_cudnn not in {"0", "false", "no"}

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if disable_cudnn and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    status = {}
    for key, fn_name in [
        ("flash", "flash_sdp_enabled"),
        ("mem_efficient", "mem_efficient_sdp_enabled"),
        ("math", "math_sdp_enabled"),
        ("cudnn", "cudnn_sdp_enabled"),
    ]:
        fn = getattr(torch.backends.cuda, fn_name, None)
        if callable(fn):
            status[key] = bool(fn())
    print(
        "[SDPA runtime]"
        f" disable_cudnn={disable_cudnn}, backends={status}"
    )


class SpanWeightedTrainer:
    """
    Trainer를 동적으로 서브클래싱하여 WeightedRandomSampler를 적용한다.

    HuggingFace Trainer는 _get_train_sampler()로 sampler를 결정하므로,
    이를 오버라이드하여 span 복잡도 기반 가중 샘플링을 지원한다.
    """

    @staticmethod
    def create(trainer_cls, sample_weights):
        """Trainer 클래스를 서브클래싱하여 가중 샘플러가 적용된 클래스를 반환한다."""
        weights = sample_weights

        class _WeightedTrainer(trainer_cls):
            def _get_train_sampler(self, train_dataset=None):
                if weights is not None:
                    return WeightedRandomSampler(
                        weights,
                        num_samples=len(weights),
                        replacement=True,
                    )
                return super()._get_train_sampler(train_dataset)

        return _WeightedTrainer


class EpochProgressCallback(_HFTrainerCallback):
    """Epoch 단위 진행상황을 간단히 출력한다."""

    def __init__(self, total_epochs: int):
        self.total_epochs = max(1, int(total_epochs))
        self._train_start = None
        self._epoch_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.time()
        print(f"[Progress] Epoch 0/{self.total_epochs} (0.0%)")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.time()
        epoch_idx = int(state.epoch or 0) + 1
        if epoch_idx > self.total_epochs:
            epoch_idx = self.total_epochs
        print(f"[Epoch {epoch_idx}/{self.total_epochs}] start")

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = 0.0
        if self._epoch_start is not None:
            elapsed = time.time() - self._epoch_start
        epoch_idx = int(round(state.epoch or 0))
        if epoch_idx < 1:
            epoch_idx = 1
        if epoch_idx > self.total_epochs:
            epoch_idx = self.total_epochs
        pct = 100.0 * epoch_idx / float(self.total_epochs)
        print(
            f"[Progress] Epoch {epoch_idx}/{self.total_epochs} "
            f"({pct:.1f}%) - {elapsed:.1f}s"
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            epoch_idx = int(round(state.epoch or 0))
            if epoch_idx < 1:
                epoch_idx = 1
            if epoch_idx > self.total_epochs:
                epoch_idx = self.total_epochs
            print(f"[Eval] epoch={epoch_idx}/{self.total_epochs} eval_loss={eval_loss:.6f}")

    def on_train_end(self, args, state, control, **kwargs):
        total_elapsed = 0.0
        if self._train_start is not None:
            total_elapsed = time.time() - self._train_start
        print(f"[Progress] Training finished in {total_elapsed/60.0:.1f} min")


def train_student_sft(
    config: dict,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
):
    """Student SFT 학습."""
    from transformers import (
        AutoProcessor,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    from torch.utils.data import Subset

    from train.collator import MultimodalCollator
    from train.train_qlora import TSRDataset

    student_cfg = config["student"]
    sft_cfg = config["student_sft"]
    teacher_cfg = config.get("teacher", {})
    use_gradient_checkpointing = bool(sft_cfg.get("gradient_checkpointing", True))

    # --- 모델 로드 ---
    model_name = student_cfg["name_or_path"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"Loading student model: {model_name} (local_rank={local_rank}, world_size={world_size})")
    attn_implementation = student_cfg.get("attn_implementation", "flash_attention_2")
    _configure_sdpa_runtime(attn_implementation)

    # 모델 클래스 자동 감지
    ModelClass = _resolve_model_class(model_name)

    use_lora = bool(sft_cfg.get("use_lora", True))

    # QLoRA quantization (옵션)
    # - 기본값은 기존 동작과 동일하게 LoRA 사용 시 4bit on
    # - B200 등 특정 환경에서는 4bit 커널 호환 이슈가 있어 config에서 끌 수 있다.
    quant_cfg = sft_cfg.get("quantization", {})
    if not isinstance(quant_cfg, dict):
        quant_cfg = {}
    load_in_4bit = bool(
        quant_cfg.get(
            "load_in_4bit",
            sft_cfg.get("load_in_4bit", use_lora),
        )
    )

    bnb_config = None
    if use_lora and load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch,
                quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
            ),
            bnb_4bit_use_double_quant=bool(
                quant_cfg.get("bnb_4bit_use_double_quant", True)
            ),
        )
    elif use_lora:
        print("  4bit quantization disabled (student_sft.quantization.load_in_4bit=false)")

    model = ModelClass.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=_torch_dtype_from_config(student_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        device_map={"": local_rank},
    )

    if bnb_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    # LoRA 적용
    # - resume_from_checkpoint: HF Trainer가 optimizer/scheduler/rng/global_step까지 복원
    #   (체크포인트에 optimizer.pt/scheduler.pt/trainer_state.json이 있어야 함). 여기서는
    #   일반 LoRA(use_lora)로 구조만 초기화해두고, 실제 가중치/스텝은 trainer.train()의
    #   resume_from_checkpoint가 로드 -> 체크포인트 번호가 이어짐(예: 4150 -> 4200).
    # - resume_from_adapter: LoRA 가중치만 이어받고 스텝은 0부터 새로 시작(옵티마이저 상태 없음).
    resume_adapter = sft_cfg.get("resume_from_adapter")
    resume_ckpt = sft_cfg.get("resume_from_checkpoint")
    if resume_adapter and not resume_ckpt:
        adapter_cfg_path = os.path.join(resume_adapter, "adapter_config.json")
        if not os.path.exists(adapter_cfg_path):
            raise FileNotFoundError(
                f"resume_from_adapter={resume_adapter} specified but "
                f"{adapter_cfg_path} not found. Check the path."
            )
        print(f"  Resuming LoRA adapter from: {resume_adapter}")
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
        model.print_trainable_parameters()
    elif use_lora:
        default_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            # "gate_proj", "up_proj", "down_proj",
        ]
        lora_cfg = sft_cfg.get("lora", {}) if isinstance(sft_cfg.get("lora"), dict) else {}

        lora_r = int(lora_cfg.get("lora_r", lora_cfg.get("r", sft_cfg.get("lora_r", 128))))
        lora_alpha = int(
            lora_cfg.get(
                "lora_alpha",
                lora_cfg.get("alpha", sft_cfg.get("lora_alpha", 256)),
            )
        )
        lora_dropout = float(
            lora_cfg.get(
                "lora_dropout",
                lora_cfg.get("dropout", sft_cfg.get("lora_dropout", 0.05)),
            )
        )
        lora_bias = str(lora_cfg.get("bias", sft_cfg.get("lora_bias", "none")))
        lora_task_type = str(
            lora_cfg.get("task_type", sft_cfg.get("lora_task_type", "CAUSAL_LM"))
        )

        raw_target_modules = lora_cfg.get(
            "target_modules",
            sft_cfg.get("target_modules", default_target_modules),
        )
        if isinstance(raw_target_modules, str):
            target_modules = [
                module.strip() for module in raw_target_modules.split(",") if module.strip()
            ]
        elif isinstance(raw_target_modules, (list, tuple)):
            target_modules = [str(module).strip() for module in raw_target_modules if str(module).strip()]
        else:
            raise TypeError(
                "student_sft.target_modules (or student_sft.lora.target_modules) must be "
                "a list/tuple or comma-separated string"
            )
        if not target_modules:
            raise ValueError("LoRA target_modules is empty; configure at least one module")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=lora_task_type,
            target_modules=target_modules,
        )
        print(
            "  LoRA settings:"
            f" r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, "
            f"bias={lora_bias}, task_type={lora_task_type}, target_modules={target_modules}"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if use_gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    # 프로세서
    max_pixels = int(
        sft_cfg.get(
            "max_pixels",
            teacher_cfg.get("max_pixels", 1024 * 1024),
        )
    )
    min_pixels = int(
        sft_cfg.get(
            "min_pixels",
            teacher_cfg.get("min_pixels", 512 * 512),
        )
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )
    if not getattr(processor, "chat_template", None) and hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "chat_template", None):
        processor.chat_template = processor.tokenizer.chat_template

    # --- 데이터셋 ---
    train_path = sft_cfg["train_file"]
    eval_path = sft_cfg["eval_file"]

    # distill 데이터가 없으면 기본 학습 데이터 사용
    if not os.path.exists(train_path):
        print(f"  Warning: {train_path} not found, using default data path")
        train_path = config.get("data", {}).get("train_file", train_path)
    if not os.path.exists(eval_path):
        eval_path = config.get("data", {}).get("eval_file", eval_path)

    if not train_path or not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train dataset not found: {sft_cfg.get('train_file')}. "
            "Run Phase 2 (quality filter) first or update student_sft.train_file."
        )

    strip_ocr_score = bool(sft_cfg.get("strip_ocr_score", True))
    train_ocr_mix_ratio = sft_cfg.get("train_ocr_mix_ratio")
    train_ocr_mix_seed = int(sft_cfg.get("train_ocr_mix_seed", 42))
    annotate_empty_cells = bool(sft_cfg.get("annotate_empty_cells", False))
    empty_cell_token = str(sft_cfg.get("empty_cell_token", "__EMPTY__") or "__EMPTY__")
    # 통합 JSON 타겟의 표 HTML 속성 따옴표를 작은따옴표로 정규화해 \" escape 를 없앤다.
    single_quote_html_attrs = bool(sft_cfg.get("single_quote_html_attrs", False))
    # Prompt-side empty-cell instruction also follows existing annotate_empty_cells option.
    include_empty_cell_prompt_instruction = annotate_empty_cells
    if train_ocr_mix_ratio is not None:
        train_ocr_mix_ratio = float(train_ocr_mix_ratio)
    image_dir = sft_cfg.get("image_dir") or config.get("data", {}).get("image_dir")
    if image_dir:
        print(f"  Using image_dir override: {image_dir}")
    train_dataset = TSRDataset(
        train_path,
        image_dir=image_dir,
        strip_ocr_score=strip_ocr_score,
        ocr_mix_ratio=train_ocr_mix_ratio,
        ocr_mix_seed=train_ocr_mix_seed,
        annotate_empty_cells=annotate_empty_cells,
        empty_cell_token=empty_cell_token,
        single_quote_html_attrs=single_quote_html_attrs,
    )
    eval_dataset = (
        TSRDataset(
            eval_path,
            image_dir=image_dir,
            strip_ocr_score=strip_ocr_score,
            annotate_empty_cells=annotate_empty_cells,
            empty_cell_token=empty_cell_token,
            single_quote_html_attrs=single_quote_html_attrs,
        )
        if eval_path and os.path.exists(eval_path)
        else None
    )

    max_train_samples = (
        max_train_samples
        if max_train_samples is not None
        else sft_cfg.get("max_train_samples")
    )
    max_eval_samples = (
        max_eval_samples
        if max_eval_samples is not None
        else sft_cfg.get("max_eval_samples")
    )

    if max_train_samples is not None:
        max_train_samples = min(len(train_dataset), int(max_train_samples))
        train_dataset = Subset(train_dataset, list(range(max_train_samples)))
        print(f"  Using train subset: {max_train_samples} samples")

    if eval_dataset is not None and max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), int(max_eval_samples))
        eval_dataset = Subset(eval_dataset, list(range(max_eval_samples)))
        print(f"  Using eval subset: {max_eval_samples} samples")

    requested_max_seq_length = int(
        sft_cfg.get("max_seq_length", MIN_ENFORCED_MAX_SEQ_LENGTH)
    )
    include_thinking = bool(sft_cfg.get("include_thinking", True))
    token_weight_cfg = (
        sft_cfg.get("token_weighting", {})
        if isinstance(sft_cfg.get("token_weighting", {}), dict)
        else {}
    )
    token_weighting_enabled = bool(token_weight_cfg.get("enabled", False))
    span_attribute_weight = float(token_weight_cfg.get("span_attribute_weight", 1.0))
    bbox_weight = float(token_weight_cfg.get("bbox_weight", 1.0))
    if requested_max_seq_length < MIN_ENFORCED_MAX_SEQ_LENGTH:
        print(
            "[Config][warning] max_seq_length is below enforced minimum:"
            f" requested={requested_max_seq_length},"
            f" enforced={MIN_ENFORCED_MAX_SEQ_LENGTH}"
        )
    configured_max_seq_length = max(
        requested_max_seq_length,
        MIN_ENFORCED_MAX_SEQ_LENGTH,
    )
    auto_expand_max_seq_length = bool(sft_cfg.get("auto_expand_max_seq_length", True))
    requested_max_seq_length_cap = int(sft_cfg.get("max_seq_length_cap", 12288))
    max_seq_length_cap = max(requested_max_seq_length_cap, configured_max_seq_length)
    if max_seq_length_cap != requested_max_seq_length_cap:
        print(
            "[Config][warning] max_seq_length_cap was raised to satisfy enforced minimum:"
            f" requested_cap={requested_max_seq_length_cap},"
            f" effective_cap={max_seq_length_cap}"
        )
    long_table_context_margin_tokens = int(
        sft_cfg.get("long_table_context_margin_tokens", 512)
    )

    # Collator
    collator = MultimodalCollator(
        processor=processor,
        max_seq_length=configured_max_seq_length,
        truncate_to_assistant=bool(sft_cfg.get("truncate_to_assistant", True)),
        include_thinking=include_thinking,
        include_empty_cell_prompt_instruction=(
            include_empty_cell_prompt_instruction
        ),
        empty_cell_token=empty_cell_token,
        token_weighting_enabled=token_weighting_enabled,
        span_attribute_weight=span_attribute_weight,
        bbox_weight=bbox_weight,
    )

    # --- Training Arguments ---
    output_dir = sft_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    num_train_epochs = int(sft_cfg.get("num_train_epochs", 3))
    progress_by_epoch = bool(sft_cfg.get("progress_by_epoch", False))
    es_cfg = sft_cfg.get("early_stopping", {})
    es_enabled = bool(eval_dataset and es_cfg.get("enabled", False))
    es_metric = es_cfg.get("metric", "eval_loss")
    es_greater_is_better = bool(es_cfg.get("greater_is_better", False))
    es_patience = int(es_cfg.get("patience", 2))
    es_threshold = float(es_cfg.get("threshold", 0.0))
    default_logging_strategy = "epoch" if progress_by_epoch else "steps"
    logging_strategy = sft_cfg.get("logging_strategy", default_logging_strategy)
    save_strategy = sft_cfg.get("save_strategy", default_logging_strategy)
    if eval_dataset:
        default_eval_strategy = "epoch" if progress_by_epoch else "steps"
        eval_strategy = sft_cfg.get("eval_strategy", default_eval_strategy)
    else:
        eval_strategy = "no"
    disable_tqdm = bool(sft_cfg.get("disable_tqdm", progress_by_epoch))

    def _records_for_audit(ds):
        if ds is None:
            return []
        if isinstance(ds, Subset):
            base = ds.dataset
            if hasattr(base, "records"):
                return [base.records[i] for i in ds.indices]
            return []
        return getattr(ds, "records", [])

    def _ocr_audit_for(ds):
        if ds is None:
            return None
        if hasattr(ds, "ocr_score_audit"):
            return ds.ocr_score_audit
        if isinstance(ds, Subset) and hasattr(ds.dataset, "ocr_score_audit"):
            return ds.dataset.ocr_score_audit
        return None

    def _ocr_mix_for(ds):
        if ds is None:
            return None
        if hasattr(ds, "ocr_mix_stats"):
            return ds.ocr_mix_stats
        if isinstance(ds, Subset) and hasattr(ds.dataset, "ocr_mix_stats"):
            return ds.dataset.ocr_mix_stats
        return None

    generation_max_new_tokens = (
        config.get("synthetic_generation", {}).get("max_new_tokens")
        if isinstance(config.get("synthetic_generation", {}), dict)
        else None
    )
    html_audit_sample_limit = sft_cfg.get("html_audit_sample_limit", 2000)
    train_html_audit = audit_long_table_capacity(
        _records_for_audit(train_dataset),
        tokenizer=processor.tokenizer,
        dataset_name="train",
        max_seq_length=configured_max_seq_length,
        max_new_tokens=generation_max_new_tokens,
        sample_limit=html_audit_sample_limit,
        include_thinking=include_thinking,
    )
    eval_html_audit = None
    if eval_dataset is not None:
        eval_html_audit = audit_long_table_capacity(
            _records_for_audit(eval_dataset),
            tokenizer=processor.tokenizer,
            dataset_name="eval",
            max_seq_length=configured_max_seq_length,
            max_new_tokens=generation_max_new_tokens,
            sample_limit=html_audit_sample_limit,
            include_thinking=include_thinking,
        )

    required_assistant_tokens = int(train_html_audit.get("recommended_max_new_tokens", 0))
    if eval_html_audit is not None:
        required_assistant_tokens = max(
            required_assistant_tokens,
            int(eval_html_audit.get("recommended_max_new_tokens", 0)),
        )
    required_with_margin = required_assistant_tokens + long_table_context_margin_tokens
    effective_max_seq_length = configured_max_seq_length
    auto_adjust_applied = False
    auto_adjust_clamped = False
    if auto_expand_max_seq_length and required_with_margin > configured_max_seq_length:
        proposed = min(max_seq_length_cap, required_with_margin)
        if proposed > configured_max_seq_length:
            effective_max_seq_length = proposed
            collator.max_seq_length = effective_max_seq_length
            auto_adjust_applied = True
        if required_with_margin > max_seq_length_cap:
            auto_adjust_clamped = True

    preflight_report = {
        "token_weighting": {
            "enabled": token_weighting_enabled,
            "span_attribute_weight": span_attribute_weight,
            "bbox_weight": bbox_weight,
        },
        "ocr_score_audit": {
            "train": _ocr_audit_for(train_dataset),
            "eval": _ocr_audit_for(eval_dataset),
            "strip_ocr_score": strip_ocr_score,
        },
        "ocr_mix": {
            "train": _ocr_mix_for(train_dataset),
            "eval": _ocr_mix_for(eval_dataset),
            "train_ocr_mix_ratio": train_ocr_mix_ratio,
            "train_ocr_mix_seed": train_ocr_mix_seed if train_ocr_mix_ratio is not None else None,
        },
        "empty_cell_token": {
            "enabled": annotate_empty_cells,
            "token": empty_cell_token if annotate_empty_cells else "",
        },
        "long_table_capacity_audit": {
            "train": train_html_audit,
            "eval": eval_html_audit,
            "configured_max_seq_length": configured_max_seq_length,
            "effective_max_seq_length": effective_max_seq_length,
            "auto_expand_max_seq_length": auto_expand_max_seq_length,
            "auto_adjust_applied": auto_adjust_applied,
            "auto_adjust_clamped_by_cap": auto_adjust_clamped,
            "max_seq_length_cap": max_seq_length_cap,
            "long_table_context_margin_tokens": long_table_context_margin_tokens,
            "required_assistant_tokens": required_assistant_tokens,
            "required_with_margin": required_with_margin,
            "configured_generation_max_new_tokens": generation_max_new_tokens,
            "include_thinking": include_thinking,
        },
    }
    preflight_path = Path(output_dir) / "preflight_checks.json"
    write_json_report(preflight_path, preflight_report)
    print(f"[Preflight] report saved: {preflight_path}")
    if auto_adjust_applied:
        print(
            "[Preflight] auto-adjusted max_seq_length:"
            f" {configured_max_seq_length} -> {effective_max_seq_length}"
        )
    if auto_adjust_clamped:
        print(
            "[Preflight][warning] required_with_margin exceeds max_seq_length_cap:"
            f" required={required_with_margin}, cap={max_seq_length_cap}"
        )
    for warning in train_html_audit.get("warnings", []):
        print(f"[Preflight][train] {warning}")
    if eval_html_audit is not None:
        for warning in eval_html_audit.get("warnings", []):
            print(f"[Preflight][eval] {warning}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=int(sft_cfg.get("max_steps", -1)),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=sft_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=sft_cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.05),
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=sft_cfg.get("bf16", True),
        tf32=sft_cfg.get("tf32", True),
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=sft_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=sft_cfg.get("dataloader_pin_memory", True),
        auto_find_batch_size=sft_cfg.get("auto_find_batch_size", False),
        optim=sft_cfg.get("optim", "adamw_torch_fused"),
        remove_unused_columns=False,
        disable_tqdm=disable_tqdm,
        # Logging
        logging_strategy=logging_strategy,
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_strategy=save_strategy,
        save_steps=sft_cfg.get("save_steps", 200),
        save_total_limit=sft_cfg.get("save_total_limit", None),
        eval_strategy=eval_strategy,
        eval_steps=sft_cfg.get("eval_steps", 200) if eval_dataset else None,
        load_best_model_at_end=es_enabled,
        metric_for_best_model=es_metric if es_enabled else None,
        greater_is_better=es_greater_is_better if es_enabled else None,
        report_to=sft_cfg.get("report_to", "none"),
        run_name=sft_cfg.get("run_name", "student_sft"),
        # DDP
        ddp_find_unused_parameters=False,
    )

    callbacks = []
    if progress_by_epoch:
        callbacks.append(EpochProgressCallback(total_epochs=num_train_epochs))
    if es_enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=es_patience,
                early_stopping_threshold=es_threshold,
            )
        )
        print(
            "  Early stopping enabled:"
            f" metric={es_metric}, patience={es_patience}, threshold={es_threshold}"
        )

    # --- 가중 샘플링 (task 1:1 우선, 없으면 span 복잡도) ---
    def _base_and_indices(ds):
        """Subset 이면 (base_dataset, indices), 아니면 (ds, None)."""
        if isinstance(ds, Subset):
            return ds.dataset, list(ds.indices)
        return ds, None

    task_mix_sampling = sft_cfg.get("task_mix_sampling", False)
    span_weighted = sft_cfg.get("span_weighted_sampling", False)
    sample_weights = None

    if task_mix_sampling:
        # 테이블/레이아웃을 1:1(또는 지정 비율)로 균형 샘플링한다.
        base_ds, subset_indices = _base_and_indices(train_dataset)
        if hasattr(base_ds, "get_task_sampling_weights"):
            task_ratios = sft_cfg.get("task_sampling_ratios")  # 예: {table:0.5, layout:0.5}
            base_weights = base_ds.get_task_sampling_weights(task_ratios)
            task_counts = base_ds.get_task_counts()
            if base_weights is None:
                print(
                    f"  Task mix sampling skipped: only one task present {task_counts}"
                )
            else:
                if subset_indices is not None:
                    sample_weights = [base_weights[i] for i in subset_indices]
                else:
                    sample_weights = base_weights
                print(
                    f"  Task mix sampling enabled: counts={task_counts},"
                    f" ratios={task_ratios or 'uniform(1:1)'}"
                )
        else:
            print("  Warning: dataset does not support task mix sampling")

    if sample_weights is None and span_weighted and not isinstance(train_dataset, Subset):
        target_ratios = sft_cfg.get(
            "span_sampling_ratios",
            {"simple": 0.10, "medium": 0.20, "complex": 0.70},
        )
        print(f"  Span weighted sampling enabled: {target_ratios}")
        sample_weights = train_dataset.get_sampling_weights(target_ratios)
    elif sample_weights is None and span_weighted:
        print(
            "  Warning: span_weighted_sampling is disabled because "
            "max_train_samples created a Subset. Remove max_train_samples "
            "to enable weighted sampling."
        )

    # --- NaNSafeTrainer: eval loss NaN → 0 대체로 EarlyStopping 오동작 방지 ---
    class NaNSafeTrainer(Trainer):
        """eval prediction_step에서 NaN/Inf loss를 0으로 대체한다.

        DDP eval all_gather 또는 all-labels=-100 배치에서 발생하는
        NaN eval_loss가 EarlyStopping을 비활성화하는 문제를 방지한다.
        """

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            token_weights = inputs.pop("token_weights", None)
            outputs = model(**inputs)

            # token weight 비활성/미제공 시 기본 loss 사용
            if token_weights is None:
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            logits = outputs.logits
            labels = inputs.get("labels")
            if labels is None:
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            # Causal LM shift 정렬: t 시점 logits이 t+1 label을 예측
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = token_weights[..., 1:].contiguous().to(shift_logits.dtype)

            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction="none",
            )
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view_as(shift_labels)

            valid_mask = (shift_labels != -100).to(shift_logits.dtype)
            effective_weights = shift_weights * valid_mask
            denom = effective_weights.sum().clamp(min=1.0)
            loss = (per_token_loss * effective_weights).sum() / denom

            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            loss, logits, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            if loss is not None and not torch.isfinite(loss):
                print(
                    f"[Warning] eval prediction_step got non-finite loss:"
                    f" {loss.item()}, replacing with 0"
                )
                loss = torch.zeros_like(loss)
            return loss, logits, labels

    # --- Trainer ---
    TrainerClass = NaNSafeTrainer
    if sample_weights is not None:
        TrainerClass = SpanWeightedTrainer.create(NaNSafeTrainer, sample_weights)
        print(f"  Using SpanWeightedTrainer (NaNSafe) with {len(sample_weights)} weights")

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # --- 학습 ---
    print(f"\n{'='*60}")
    print(f"Student SFT Training")
    print(f"  Model:  {model_name}")
    if resume_adapter:
        print(f"  Resume: {resume_adapter}")
    print(f"  Train:  {len(train_dataset)} samples")
    print(f"  Eval:   {len(eval_dataset) if eval_dataset else 0} samples")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  LR:     {sft_cfg.get('learning_rate', 5e-5)}")
    print(f"  MaxPx:  {max_pixels} (min={min_pixels})")
    print(f"  MaxSeq: {configured_max_seq_length} -> {collator.max_seq_length}")
    print(f"  TruncateToAssistant: {collator.truncate_to_assistant}")
    print(f"  IncludeThinking: {include_thinking}")
    print(
        "  EmptyCellToken:"
        f" enabled={annotate_empty_cells},"
        f" token={empty_cell_token if annotate_empty_cells else '(off)'}"
    )
    print(
        "  TokenWeighting:"
        f" enabled={token_weighting_enabled},"
        f" span_attribute_weight={span_attribute_weight:.2f},"
        f" bbox_weight={bbox_weight:.2f}"
    )
    print(f"  GradCkpt: {use_gradient_checkpointing}")
    train_mix_stats = _ocr_mix_for(train_dataset)
    if train_mix_stats:
        print(
            "  OCRMix(train):"
            f" on={train_mix_stats['applied_ocr_on']},"
            f" off={train_mix_stats['applied_ocr_off']},"
            f" ratio={train_mix_stats['effective_ratio']:.4f},"
            f" seed={train_mix_stats['seed']}"
        )
    print(f"  SpanWeighted: {span_weighted}")
    print(f"  ProgressByEpoch: {progress_by_epoch}")
    print(f"  Strategies: logging={logging_strategy}, save={save_strategy}, eval={eval_strategy}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    trainer.train()

    progress_path = Path(output_dir) / "training_progress.html"
    progress_summary = write_training_progress_html(
        trainer.state.log_history,
        progress_path,
        title=sft_cfg.get("run_name", "student_sft"),
    )
    write_json_report(Path(output_dir) / "training_progress.json", progress_summary)
    print(f"[Progress] chart saved: {progress_path}")

    # 저장
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    print(f"\nStudent SFT complete. Model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Student SFT")
    parser.add_argument("--config", default="config/distill_config.yaml")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_student_sft(
        config,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
