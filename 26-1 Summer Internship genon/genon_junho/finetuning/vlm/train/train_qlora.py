"""
Qwen3-VL QLoRA Fine-tuning 학습 스크립트

Features:
- QLoRA (4-bit quantization + LoRA)
- DeepSpeed ZeRO-3 지원
- 2-Phase 학습 (기초 → Span 특화)
- Span 가중 샘플링
- Wandb 로깅
- Gradient checkpointing

Usage:
    # 단일 GPU
    python -m train.train_qlora --config config/training_config.yaml

    # 멀티 GPU (DeepSpeed)
    deepspeed --num_gpus=4 -m train.train_qlora \
        --config config/training_config.yaml \
        --deepspeed config/deepspeed_zero3.json

    # 또는 run_train.sh 사용
    bash train/run_train.sh
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.monitoring import (
    audit_and_sanitize_ocr_scores,
    audit_long_table_capacity,
    write_json_report,
    write_training_progress_html,
)
from utils.fill_empty_cells import fill_empty_cells
from utils.prompt_templates import (
    PROMPT_STYLE_CHANDRA_NO_OCR,
    PROMPT_STYLE_CHANDRA_WITH_OCR,
)

MIN_ENFORCED_MAX_SEQ_LENGTH = 8192


# =============================================================================
# 통합(JSON) 타겟 따옴표 정규화
# =============================================================================
# 통합 정답은 JSON 배열이고, Table element 의 text 에 표 HTML 이 들어간다. HTML 속성을
# 큰따옴표(colspan="2", align="center")로 쓰면 JSON 으로 직렬화될 때 전부 \" 로 escape 되어
# 모델이 따옴표 상태를 놓치기 쉽다(예: "category": "Table, 처럼 wrapper 따옴표 누락 → 파싱 실패).
# 속성을 작은따옴표로 바꾸면 JSON 문자열 안에 " 가 사라져 escape 가 불필요해지고, 작은따옴표
# 속성은 valid HTML 이며 bs4/lxml 파싱 결과(TEDS/span)도 동일하다.
_HTML_ATTR_DQUOTE_RE = re.compile(r'([A-Za-z_:][\w:.\-]*)\s*=\s*"([^"\']*)"')


def single_quote_html_attributes(html: str) -> str:
    """HTML 태그 속성의 큰따옴표를 작은따옴표로 바꾼다(값에 따옴표가 없는 경우만)."""
    if not html or '"' not in html:
        return html
    return _HTML_ATTR_DQUOTE_RE.sub(r"\1='\2'", html)


def single_quote_unified_target(gt_html: str) -> str:
    """통합 정답(JSON 문자열) 안의 element text(표 HTML 포함) 속성 따옴표를 작은따옴표로 정규화."""
    s = str(gt_html or "")
    if not s:
        return s
    try:
        elements = json.loads(s)
    except Exception:
        # 통합 JSON 이 아니면(레거시 순수 HTML) HTML 변환만 적용
        return single_quote_html_attributes(s)
    if not isinstance(elements, list):
        return s
    changed = False
    for el in elements:
        if isinstance(el, dict) and isinstance(el.get("text"), str) and "<" in el["text"]:
            new_text = single_quote_html_attributes(el["text"])
            if new_text != el["text"]:
                el["text"] = new_text
                changed = True
    return json.dumps(elements, ensure_ascii=False) if changed else s


# =============================================================================
# Dataset
# =============================================================================


class TSRDataset(Dataset):
    """테이블 구조 인식 학습 데이터셋."""

    def __init__(
        self,
        data_path: str,
        image_dir: Optional[str] = None,
        strip_ocr_score: bool = True,
        ocr_mix_ratio: Optional[float] = None,
        ocr_mix_seed: int = 42,
        annotate_empty_cells: bool = False,
        empty_cell_token: str = "__EMPTY__",
        single_quote_html_attrs: bool = False,
    ):
        self.image_dir = str(Path(image_dir).expanduser()) if image_dir else None
        data_parent = Path(data_path).expanduser().resolve().parent
        self.image_path_resolve_stats = {"resolved": 0, "unresolved": 0}
        self.annotate_empty_cells = bool(annotate_empty_cells)
        self.empty_cell_token = str(empty_cell_token or "").strip() or "__EMPTY__"
        self.single_quote_html_attrs = bool(single_quote_html_attrs)
        self.single_quote_converted = 0
        self.records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    image_path = record.get("image_path")
                    if isinstance(image_path, str) and image_path.strip():
                        resolved = self._resolve_image_path(image_path, data_parent)
                        record["image_path"] = resolved
                        if resolved != image_path:
                            self.image_path_resolve_stats["resolved"] += 1
                        elif not Path(image_path).expanduser().exists():
                            self.image_path_resolve_stats["unresolved"] += 1
                    # 빈 셀 채우기는 표 셀(<td>)에만 영향을 주므로 task_type 분기 없이
                    # 활성화 시 모든 레코드에 동일 적용한다(표가 없으면 무영향).
                    if self.annotate_empty_cells:
                        record["gt_html"] = fill_empty_cells(
                            str(record.get("gt_html", "") or ""),
                            marker=self.empty_cell_token,
                        )
                    # 통합 JSON 타겟의 HTML 속성 따옴표를 작은따옴표로 정규화(\" escape 제거)
                    if self.single_quote_html_attrs:
                        before = str(record.get("gt_html", "") or "")
                        after = single_quote_unified_target(before)
                        if after != before:
                            self.single_quote_converted += 1
                        record["gt_html"] = after
                    self.records.append(record)

        print(f"Loaded {len(self.records)} records from {data_path}")
        if self.annotate_empty_cells:
            print(
                "[Empty cell token]"
                f" enabled=True, token={self.empty_cell_token}"
            )
        if self.single_quote_html_attrs:
            print(
                "[HTML attr quotes] single-quote normalization enabled,"
                f" converted={self.single_quote_converted}/{len(self.records)} records"
            )
        if self.image_dir:
            print(
                "[Image path resolve]"
                f" image_dir={self.image_dir},"
                f" resolved={self.image_path_resolve_stats['resolved']},"
                f" unresolved={self.image_path_resolve_stats['unresolved']}"
            )

        self.ocr_mix_stats = None
        if ocr_mix_ratio is not None:
            self.ocr_mix_stats = self._apply_ocr_mix(
                ratio=float(ocr_mix_ratio),
                seed=int(ocr_mix_seed),
            )
            print(
                "[OCR mix]"
                f" requested_ratio={self.ocr_mix_stats['requested_ratio']:.4f},"
                f" effective_ratio={self.ocr_mix_stats['effective_ratio']:.4f},"
                f" on={self.ocr_mix_stats['applied_ocr_on']},"
                f" off={self.ocr_mix_stats['applied_ocr_off']},"
                f" ocr_available={self.ocr_mix_stats['records_with_ocr_info_before_mix']},"
                f" seed={self.ocr_mix_stats['seed']}"
            )
            if self.ocr_mix_stats["capped_by_ocr_availability"]:
                print(
                    "[OCR mix][warning]"
                    " requested OCR-on count exceeds OCR-available samples."
                    f" requested={self.ocr_mix_stats['requested_ocr_on']},"
                    f" available={self.ocr_mix_stats['records_with_ocr_info_before_mix']}"
                )

        self.ocr_score_audit = audit_and_sanitize_ocr_scores(
            self.records,
            remove_score=strip_ocr_score,
        )
        print(
            "[OCR score check]"
            f" records_with_ocr={self.ocr_score_audit['records_with_ocr_info']},"
            f" records_with_score={self.ocr_score_audit['records_with_ocr_score']},"
            f" score_items={self.ocr_score_audit['ocr_items_with_score']},"
            f" removed={self.ocr_score_audit['ocr_items_score_removed']}"
        )

    @staticmethod
    def _record_has_ocr_info(record: dict) -> bool:
        ocr_info = record.get("ocr_info")
        return isinstance(ocr_info, list) and len(ocr_info) > 0

    def _apply_ocr_mix(self, ratio: float, seed: int) -> dict:
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                f"ocr_mix_ratio must be within [0.0, 1.0], got {ratio}"
            )

        total = len(self.records)
        if total == 0:
            return {
                "enabled": True,
                "requested_ratio": ratio,
                "effective_ratio": 0.0,
                "seed": seed,
                "total_records": 0,
                "mixable_records": 0,
                "layout_records_excluded": 0,
                "records_with_ocr_info_before_mix": 0,
                "requested_ocr_on": 0,
                "target_ocr_on": 0,
                "applied_ocr_on": 0,
                "applied_ocr_off": 0,
                "capped_by_ocr_availability": False,
            }

        # 통합 단일 파이프라인: 테이블/레이아웃 구분(task_type) 없이 모든 레코드를
        # OCR-mix 대상으로 한다. OCR 정보가 있는 레코드만 OCR-on 후보가 된다.
        mixable_indices = list(range(total))
        layout_count = 0

        available_indices = [
            idx for idx in mixable_indices
            if self._record_has_ocr_info(self.records[idx])
        ]
        mixable_total = len(mixable_indices)
        requested_on = int(round(mixable_total * ratio))
        target_on = min(requested_on, len(available_indices))

        rng = random.Random(seed)
        rng.shuffle(available_indices)
        selected_on = set(available_indices[:target_on])

        applied_on = 0
        for idx in mixable_indices:
            record = self.records[idx]
            if idx in selected_on:
                record["prompt_style"] = PROMPT_STYLE_CHANDRA_WITH_OCR
                if "bbox_scale" not in record:
                    record["bbox_scale"] = 1024
                applied_on += 1
            else:
                record["prompt_style"] = PROMPT_STYLE_CHANDRA_NO_OCR
                record.pop("ocr_info", None)

        applied_off = mixable_total - applied_on
        effective_ratio = (applied_on / mixable_total) if mixable_total > 0 else 0.0
        return {
            "enabled": True,
            "requested_ratio": ratio,
            "effective_ratio": effective_ratio,
            "seed": seed,
            "total_records": total,
            "mixable_records": mixable_total,
            "layout_records_excluded": layout_count,
            "records_with_ocr_info_before_mix": len(available_indices),
            "requested_ocr_on": requested_on,
            "target_ocr_on": target_on,
            "applied_ocr_on": applied_on,
            "applied_ocr_off": applied_off,
            "capped_by_ocr_availability": target_on < requested_on,
        }

    def _resolve_image_path(self, image_path: str, data_parent: Path) -> str:
        raw_path = Path(image_path).expanduser()
        if raw_path.is_absolute():
            return str(raw_path)

        candidates = []
        if self.image_dir:
            image_root = Path(self.image_dir)
            candidates.extend(
                [
                    image_root / raw_path,
                    image_root / raw_path.name,
                ]
            )
        candidates.extend(
            [
                data_parent / raw_path,
                data_parent / raw_path.name,
                data_parent.parent / raw_path,
                data_parent.parent / raw_path.name,
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

        return image_path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def get_sampling_weights(
        self, target_ratios: Optional[dict] = None
    ) -> list[float]:
        """Span 복잡도 기반 샘플링 가중치."""
        from utils.span_analyzer import compute_sampling_weights

        html_list = [record.get("gt_html", "") for record in self.records]
        return compute_sampling_weights(html_list, target_ratios)

    def get_task_counts(self) -> dict:
        """task_type 별 레코드 수를 반환한다."""
        counts: dict = {}
        for record in self.records:
            task = str(record.get("task_type", "table"))
            counts[task] = counts.get(task, 0) + 1
        return counts

    def get_task_sampling_weights(
        self, target_ratios: Optional[dict] = None
    ) -> Optional[list[float]]:
        """task_type 기반 가중치를 반환한다 (예: table:layout = 1:1).

        target_ratios 미지정 시 존재하는 모든 task 를 균등(1:1) 비율로 맞춘다.
        각 샘플 가중치 = (해당 task 목표비율) / (해당 task 레코드 수).
        단일 task 만 존재하면 None 을 반환한다(가중 샘플링 불필요).
        """
        counts = self.get_task_counts()
        tasks = [t for t, c in counts.items() if c > 0]
        if len(tasks) < 2:
            return None

        if not target_ratios:
            target_ratios = {t: 1.0 / len(tasks) for t in tasks}

        total_ratio = sum(float(target_ratios.get(t, 0.0)) for t in tasks)
        if total_ratio <= 0:
            return None

        per_sample = {}
        for t in tasks:
            ratio = float(target_ratios.get(t, 0.0)) / total_ratio
            per_sample[t] = (ratio / counts[t]) if counts[t] > 0 else 0.0

        return [
            per_sample.get(str(record.get("task_type", "table")), 0.0)
            for record in self.records
        ]


# =============================================================================
# Model Setup
# =============================================================================


def _resolve_model_class(model_name: str, trust_remote_code: bool = True):
    """모델 타입에 맞는 클래스를 자동 감지한다."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "")
    print(f"Detected model_type: {model_type}")

    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif model_type in ("qwen2_5_vl", "qwen2_vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import AutoModelForImageTextToText
        print(f"Warning: Unknown model_type '{model_type}', falling back to AutoModelForImageTextToText")
        return AutoModelForImageTextToText


def _configure_sdpa_runtime(attn_implementation: str) -> None:
    """SDPA runtime backend를 안전하게 설정한다.

    `attn_implementation=sdpa`는 유지하되, cuDNN SDPA plan build 오류를
    피하기 위해 cuDNN backend를 기본 비활성화한다.
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


def load_model_and_processor(config: dict, lora_config: dict):
    """모델과 프로세서를 로드한다."""
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_name = config["model"]["name_or_path"]
    trust_remote_code = config["model"].get("trust_remote_code", True)

    # 모델 클래스 자동 감지
    ModelClass = _resolve_model_class(model_name, trust_remote_code)

    # 프로세서 로드
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        max_pixels=config["vision"].get("max_pixels", 2048 * 2048),
        min_pixels=config["vision"].get("min_pixels", 512 * 512),
    )

    # Quantization 설정
    quant_cfg = lora_config.get("quantization", {})
    bnb_config = None
    if quant_cfg.get("load_in_4bit", True):
        compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )

    # 모델 로드
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    attn_implementation = config["model"].get("attn_implementation", "flash_attention_2")
    _configure_sdpa_runtime(attn_implementation)
    print(f"Loading model: {model_name} (class={ModelClass.__name__}, local_rank={local_rank})")
    model = ModelClass.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, config["model"].get("torch_dtype", "bfloat16")),
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        device_map={"": local_rank},
    )

    # kbit training 준비
    if bnb_config:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # LoRA 설정
    lora_cfg = lora_config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


# =============================================================================
# Training
# =============================================================================


def train(
    config: dict,
    lora_config: dict,
    phase: str = "phase1",
    resume_from: Optional[str] = None,
):
    """
    QLoRA 학습을 실행한다.

    Args:
        config: training_config.yaml 내용
        lora_config: lora_config.yaml 내용
        phase: "phase1" 또는 "phase2"
        resume_from: 체크포인트 경로 (이어서 학습)
    """
    from transformers import TrainingArguments, Trainer

    # Phase 설정 오버라이드
    phase_cfg = config.get("phases", {}).get(phase, {})
    train_cfg = config["training"]

    num_epochs = phase_cfg.get("epochs", train_cfg.get("num_train_epochs", 3))
    learning_rate = phase_cfg.get("learning_rate", train_cfg.get("learning_rate", 2e-5))

    # 모델/프로세서 로드
    model, processor = load_model_and_processor(config, lora_config)

    # 데이터셋 로드
    data_cfg = config["data"]
    strip_ocr_score = bool(data_cfg.get("strip_ocr_score", True))
    include_thinking = bool(
        train_cfg.get(
            "include_thinking",
            data_cfg.get("include_thinking", True),
        )
    )
    token_weight_cfg = (
        train_cfg.get("token_weighting", {})
        if isinstance(train_cfg.get("token_weighting", {}), dict)
        else {}
    )
    token_weighting_enabled = bool(token_weight_cfg.get("enabled", False))
    span_attribute_weight = float(token_weight_cfg.get("span_attribute_weight", 1.0))
    train_ocr_mix_ratio = data_cfg.get("train_ocr_mix_ratio")
    train_ocr_mix_seed = int(data_cfg.get("train_ocr_mix_seed", 42))
    annotate_empty_cells = bool(data_cfg.get("annotate_empty_cells", False))
    empty_cell_token = str(data_cfg.get("empty_cell_token", "__EMPTY__") or "__EMPTY__")
    # Prompt-side empty-cell instruction also follows existing annotate_empty_cells option.
    include_empty_cell_prompt_instruction = annotate_empty_cells
    if train_ocr_mix_ratio is not None:
        train_ocr_mix_ratio = float(train_ocr_mix_ratio)
    train_dataset = TSRDataset(
        data_cfg["train_file"],
        data_cfg.get("image_dir"),
        strip_ocr_score=strip_ocr_score,
        ocr_mix_ratio=train_ocr_mix_ratio,
        ocr_mix_seed=train_ocr_mix_seed,
        annotate_empty_cells=annotate_empty_cells,
        empty_cell_token=empty_cell_token,
    )
    eval_dataset = TSRDataset(
        data_cfg["eval_file"],
        data_cfg.get("image_dir"),
        strip_ocr_score=strip_ocr_score,
        annotate_empty_cells=annotate_empty_cells,
        empty_cell_token=empty_cell_token,
    )

    requested_max_seq_length = int(
        train_cfg.get("max_seq_length", MIN_ENFORCED_MAX_SEQ_LENGTH)
    )
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
    auto_expand_max_seq_length = bool(train_cfg.get("auto_expand_max_seq_length", True))
    requested_max_seq_length_cap = int(train_cfg.get("max_seq_length_cap", 12288))
    max_seq_length_cap = max(requested_max_seq_length_cap, configured_max_seq_length)
    if max_seq_length_cap != requested_max_seq_length_cap:
        print(
            "[Config][warning] max_seq_length_cap was raised to satisfy enforced minimum:"
            f" requested_cap={requested_max_seq_length_cap},"
            f" effective_cap={max_seq_length_cap}"
        )
    long_table_context_margin_tokens = int(
        train_cfg.get("long_table_context_margin_tokens", 512)
    )

    # Collator
    from train.collator import MultimodalCollator
    collator = MultimodalCollator(
        processor=processor,
        max_seq_length=configured_max_seq_length,
        include_thinking=include_thinking,
        include_empty_cell_prompt_instruction=(
            include_empty_cell_prompt_instruction
        ),
        empty_cell_token=empty_cell_token,
        token_weighting_enabled=token_weighting_enabled,
        span_attribute_weight=span_attribute_weight,
    )

    # 출력 디렉토리
    output_dir = f"{train_cfg['output_dir']}_{phase}"
    os.makedirs(output_dir, exist_ok=True)

    # 데이터셋 사전 점검 (긴 HTML 출력 용량/테이블 닫힘)
    html_audit_sample_limit = data_cfg.get("html_audit_sample_limit", 2000)
    generation_max_new_tokens = (
        config.get("synthetic_generation", {}).get("max_new_tokens")
        if isinstance(config.get("synthetic_generation", {}), dict)
        else None
    )
    train_html_audit = audit_long_table_capacity(
        train_dataset.records,
        tokenizer=processor.tokenizer,
        dataset_name="train",
        max_seq_length=configured_max_seq_length,
        max_new_tokens=generation_max_new_tokens,
        sample_limit=html_audit_sample_limit,
        include_thinking=include_thinking,
    )
    eval_html_audit = audit_long_table_capacity(
        eval_dataset.records,
        tokenizer=processor.tokenizer,
        dataset_name="eval",
        max_seq_length=configured_max_seq_length,
        max_new_tokens=generation_max_new_tokens,
        sample_limit=html_audit_sample_limit,
        include_thinking=include_thinking,
    )
    required_assistant_tokens = max(
        int(train_html_audit.get("recommended_max_new_tokens", 0)),
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
        "phase": phase,
        "token_weighting": {
            "enabled": token_weighting_enabled,
            "span_attribute_weight": span_attribute_weight,
        },
        "ocr_score_audit": {
            "train": train_dataset.ocr_score_audit,
            "eval": eval_dataset.ocr_score_audit,
            "strip_ocr_score": strip_ocr_score,
        },
        "ocr_mix": {
            "train": train_dataset.ocr_mix_stats,
            "eval": eval_dataset.ocr_mix_stats,
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
    for warning in eval_html_audit.get("warnings", []):
        print(f"[Preflight][eval] {warning}")

    # TrainingArguments
    log_cfg = config.get("logging", {})
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=learning_rate,
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        tf32=train_cfg.get("tf32", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
        remove_unused_columns=False,
        # Logging & Saving
        logging_strategy=log_cfg.get("logging_strategy", "steps"),
        logging_steps=log_cfg.get("logging_steps", 10),
        save_strategy=log_cfg.get("save_strategy", "steps"),
        save_steps=log_cfg.get("save_steps", 200),
        save_total_limit=log_cfg.get("save_total_limit", 5),
        eval_strategy=log_cfg.get("eval_strategy", "steps"),
        eval_steps=log_cfg.get("eval_steps", 200),
        report_to=log_cfg.get("report_to", "wandb"),
        run_name=f"{log_cfg.get('run_name', 'tsr')}_{phase}",
        # DDP
        ddp_find_unused_parameters=False,
        # Resume
        resume_from_checkpoint=resume_from,
    )

    # Span-aware loss를 위한 안전한 Trainer (token_weights 미사용 시 기본 loss)
    class SpanAwareTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            token_weights = inputs.pop("token_weights", None)
            outputs = model(**inputs)

            if token_weights is None:
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            logits = outputs.logits
            labels = inputs.get("labels")
            if labels is None:
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

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

    # Trainer
    trainer = SpanAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # 학습 실행
    print(f"\n{'='*60}")
    print(f"Starting {phase}: {phase_cfg.get('description', '')}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print(f"  IncludeThinking: {include_thinking}")
    print(
        "  EmptyCellToken:"
        f" enabled={annotate_empty_cells},"
        f" token={empty_cell_token if annotate_empty_cells else '(off)'}"
    )
    print(
        "  TokenWeighting:"
        f" enabled={token_weighting_enabled},"
        f" span_attribute_weight={span_attribute_weight:.2f}"
    )
    if train_dataset.ocr_mix_stats:
        mix_stats = train_dataset.ocr_mix_stats
        print(
            "  OCRMix(train):"
            f" on={mix_stats['applied_ocr_on']},"
            f" off={mix_stats['applied_ocr_off']},"
            f" ratio={mix_stats['effective_ratio']:.4f},"
            f" seed={mix_stats['seed']}"
        )
    print(f"  MaxSeq: {configured_max_seq_length} -> {collator.max_seq_length}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=resume_from)

    progress_title = f"{log_cfg.get('run_name', 'tsr')}_{phase}"
    progress_path = Path(output_dir) / "training_progress.html"
    progress_summary = write_training_progress_html(
        trainer.state.log_history,
        progress_path,
        title=progress_title,
    )
    write_json_report(Path(output_dir) / "training_progress.json", progress_summary)
    print(f"[Progress] chart saved: {progress_path}")

    # 모델 저장
    trainer.save_model(os.path.join(output_dir, "final"))
    processor.save_pretrained(os.path.join(output_dir, "final"))

    print(f"\n{phase} training complete. Model saved to {output_dir}/final")
    return output_dir


# =============================================================================
# LoRA Merge
# =============================================================================


def merge_lora(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    config: dict,
):
    """LoRA 어댑터를 베이스 모델에 병합."""
    from transformers import AutoProcessor
    from peft import PeftModel

    print(f"Merging LoRA adapter: {lora_adapter_path}")

    # 모델 클래스 자동 감지
    ModelClass = _resolve_model_class(base_model_path, trust_remote_code=True)

    # 베이스 모델 (quantization 없이)
    model = ModelClass.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    # 병합
    model = model.merge_and_unload()

    # 저장
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL QLoRA Training")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--lora_config", default="config/lora_config.yaml")
    parser.add_argument(
        "--phase", choices=["phase1", "phase2", "both"], default="both",
        help="학습 단계 (phase1/phase2/both)"
    )
    parser.add_argument("--resume_from", default=None, help="체크포인트 경로")
    parser.add_argument("--merge", action="store_true", help="학습 후 LoRA 병합")

    # torchrun이 추가하는 인자 무시
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # 설정 로드
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.lora_config, "r") as f:
        lora_config = yaml.safe_load(f)

    # 학습 실행
    if args.phase in ("phase1", "both"):
        output_dir = train(
            config, lora_config,
            phase="phase1", resume_from=args.resume_from,
        )

        if args.phase == "both":
            # Phase 2는 Phase 1의 결과를 이어서 학습
            train(
                config, lora_config,
                phase="phase2",
                resume_from=os.path.join(output_dir, "final"),
            )

    elif args.phase == "phase2":
        train(
            config, lora_config,
            phase="phase2", resume_from=args.resume_from,
        )

    # LoRA 병합
    if args.merge:
        lora_cfg = lora_config.get("merge", {})
        merged_dir = lora_cfg.get("output_merged_dir", "./output/merged_model")
        merge_lora(
            config["model"]["name_or_path"],
            os.path.join(config["training"]["output_dir"] + "_phase2", "final"),
            merged_dir,
            config,
        )


if __name__ == "__main__":
    main()
