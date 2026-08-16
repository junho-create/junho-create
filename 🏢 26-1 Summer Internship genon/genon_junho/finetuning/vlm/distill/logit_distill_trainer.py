"""
Logit Distillation Trainer

Teacher의 확률 분포(soft targets)를 Student가 학습하는 트레이너.
단순 SFT보다 더 풍부한 정보를 전달하여 Student 성능을 높인다.

지원 모드:
1. Offline: 사전 저장된 Teacher logit 사용 (save_teacher_logits.py로 생성)
2. Online: Teacher를 메모리에 유지하며 실시간 logit 계산

Loss = alpha * KL_div(student_soft, teacher_soft) + (1-alpha) * CE(student, hard_label)

Usage:
    # Offline mode (권장)
    python -m distill.logit_distill_trainer \
        --config config/distill_config.yaml \
        --mode offline

    # Online mode (Teacher + Student 동시 메모리)
    python -m distill.logit_distill_trainer \
        --config config/distill_config.yaml \
        --mode online
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIN_ENFORCED_MAX_SEQ_LENGTH = 8192


def get_distributed_context() -> tuple[int, int, int, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    return local_rank, rank, world_size, distributed


def init_distributed(local_rank: int, distributed: bool) -> None:
    if not distributed:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed logit distillation requires CUDA.")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _torch_dtype_from_config(dtype_name: str) -> torch.dtype:
    """문자열 dtype을 torch dtype으로 변환한다."""
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    print(f"Warning: Unknown torch dtype '{dtype_name}', falling back to bfloat16")
    return torch.bfloat16


def _enforce_min_max_seq_length(value: int, source_label: str) -> int:
    resolved = int(value)
    if resolved < MIN_ENFORCED_MAX_SEQ_LENGTH:
        print(
            "[Config][warning] max_seq_length is below enforced minimum:"
            f" {source_label}={resolved} -> {MIN_ENFORCED_MAX_SEQ_LENGTH}"
        )
        return MIN_ENFORCED_MAX_SEQ_LENGTH
    return resolved


def load_student_for_distillation(config: dict):
    """
    Distillation 시작 Student를 로드한다.

    우선순위:
      1) logit_distillation.student_init_path
      2) student_sft.output_dir/final (존재 시)
      3) student.name_or_path
    """
    from transformers import AutoProcessor
    from peft import PeftModel, LoraConfig, get_peft_model
    from distill.teacher_generate import _resolve_model_class

    student_cfg = config["student"]
    sft_cfg = config.get("student_sft", {})
    ld_cfg = config.get("logit_distillation", {})

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dtype = _torch_dtype_from_config(student_cfg.get("torch_dtype", "bfloat16"))

    init_path = ld_cfg.get("student_init_path")
    if not init_path:
        sft_final = os.path.join(sft_cfg.get("output_dir", "./output/student_sft"), "final")
        init_path = sft_final if os.path.exists(sft_final) else student_cfg["name_or_path"]

    is_adapter = os.path.isdir(init_path) and os.path.exists(os.path.join(init_path, "adapter_config.json"))

    if is_adapter:
        with open(os.path.join(init_path, "adapter_config.json"), "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg.get("base_model_name_or_path", student_cfg["name_or_path"])
        ModelClass = _resolve_model_class(base_model)
        print(f"Loading student base: {base_model} ({ModelClass.__name__})")
        model = ModelClass.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=student_cfg.get("attn_implementation", "flash_attention_2"),
            device_map={"": local_rank},
        )
        print(f"Loading student adapter: {init_path}")
        model = PeftModel.from_pretrained(model, init_path, is_trainable=True)
        processor_source = init_path if os.path.exists(os.path.join(init_path, "preprocessor_config.json")) else base_model
    else:
        ModelClass = _resolve_model_class(init_path)
        print(f"Loading student model: {init_path} ({ModelClass.__name__})")
        model = ModelClass.from_pretrained(
            init_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=student_cfg.get("attn_implementation", "flash_attention_2"),
            device_map={"": local_rank},
        )

        # Base 모델에서 바로 시작할 때만 신규 LoRA를 붙인다.
        if init_path == student_cfg["name_or_path"] and sft_cfg.get("use_lora", True):
            lora_config = LoraConfig(
                r=sft_cfg.get("lora_r", 128),
                lora_alpha=sft_cfg.get("lora_alpha", 256),
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        processor_source = init_path

    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        max_pixels=2048 * 2048,
        min_pixels=512 * 512,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, processor, init_path


# =============================================================================
# Distillation Dataset
# =============================================================================


class DistillationDataset(Dataset):
    """
    Logit Distillation용 데이터셋.

    각 샘플은:
    - Student 입력 (messages → tokenized)
    - Teacher soft targets (top-k logit)
    - Hard labels (원래 정답 토큰)
    """

    def __init__(
        self,
        data_path: str,
        teacher_logits_dir: Optional[str] = None,
        processor=None,
        max_seq_length: int = 8192,
        require_teacher_logits: bool = False,
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.require_teacher_logits = require_teacher_logits

        # 학습 데이터 로드
        self.records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        # Teacher logits 로드 (offline mode)
        self.teacher_logits = {}
        if teacher_logits_dir and os.path.isdir(teacher_logits_dir):
            self._load_teacher_logits(teacher_logits_dir)
            if self.require_teacher_logits and len(self.teacher_logits) == 0:
                raise ValueError(
                    "No teacher logits were loaded from "
                    f"{teacher_logits_dir}. Check save_teacher_logits output."
                )

    def _load_teacher_logits(self, logits_dir: str):
        """사전 계산된 Teacher logits를 메모리에 로드."""
        from distill.save_teacher_logits import load_logits_batch

        files = sorted(glob.glob(os.path.join(logits_dir, "logits_*.npz")))
        for f in files:
            items = load_logits_batch(f)
            for item in items:
                self.teacher_logits[item["index"]] = item

        print(f"  Loaded teacher logits: {len(self.teacher_logits)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        teacher_logit = self.teacher_logits.get(idx)

        return {
            "record": record,
            "teacher_logit": teacher_logit,
            "index": idx,
        }


# =============================================================================
# Distillation Loss
# =============================================================================


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_top_k_indices: torch.Tensor,
    teacher_top_k_values: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """
    Distillation loss를 계산한다.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_top_k_indices: [batch, seq_len, top_k] - Teacher의 top-k 토큰 인덱스
        teacher_top_k_values: [batch, seq_len, top_k] - Teacher의 top-k logit 값
        hard_labels: [batch, seq_len] - 정답 토큰 ID
        temperature: soft target temperature
        alpha: distill_loss 가중치

    Returns:
        {"loss": total_loss, "distill_loss": ..., "hard_loss": ...}
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    top_k = teacher_top_k_indices.shape[-1]
    zero_loss = student_logits.sum() * 0.0
    valid_mask_2d = (hard_labels != ignore_index)

    # === Hard Loss (Standard CE) ===
    if valid_mask_2d.any():
        hard_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            hard_labels.view(-1),
            ignore_index=ignore_index,
        )
    else:
        hard_loss = zero_loss

    # === Soft Loss (KL Divergence on top-k) ===
    # Teacher soft targets (top-k만 사용, 나머지는 0으로 근사)
    # 1) Teacher top-k에 temperature 적용
    teacher_soft_topk = F.softmax(teacher_top_k_values.float() / temperature, dim=-1)

    # 2) Student에서 같은 top-k 위치의 logit 추출
    # student_logits: [batch, seq, vocab] → gather top-k indices
    student_topk_logits = torch.gather(
        student_logits, dim=-1, index=teacher_top_k_indices.long()
    ).float()
    student_soft_topk = F.log_softmax(student_topk_logits / temperature, dim=-1)

    # 3) KL divergence (top-k 영역에서만)
    # 유효한 위치만 (hard_labels != ignore_index)
    mask = valid_mask_2d.unsqueeze(-1).expand_as(student_soft_topk)
    if mask.any():
        mask = mask.float()
        kl_div = F.kl_div(
            student_soft_topk, teacher_soft_topk,
            reduction="none", log_target=False,
        )
        kl_div = (kl_div * mask).sum() / mask.sum().clamp(min=1.0)
        distill_loss = kl_div * (temperature ** 2)
    else:
        distill_loss = zero_loss

    if not torch.isfinite(hard_loss):
        hard_loss = zero_loss
    if not torch.isfinite(distill_loss):
        distill_loss = zero_loss

    # === Combined Loss ===
    total_loss = alpha * distill_loss + (1 - alpha) * hard_loss
    if not torch.isfinite(total_loss):
        finite_parts = []
        if torch.isfinite(distill_loss):
            finite_parts.append(alpha * distill_loss)
        if torch.isfinite(hard_loss):
            finite_parts.append((1 - alpha) * hard_loss)
        if finite_parts:
            total_loss = sum(finite_parts)
        else:
            total_loss = zero_loss

    return {
        "loss": total_loss,
        "distill_loss": distill_loss.detach(),
        "hard_loss": hard_loss.detach(),
    }


# =============================================================================
# Distillation Trainer
# =============================================================================


class DistillationTrainer:
    """
    Logit Distillation 학습 트레이너.

    Offline mode: 사전 저장된 logit 사용
    Online mode: Teacher 모델 실시간 추론
    """

    def __init__(
        self,
        student_model,
        processor,
        config: dict,
        teacher_model=None,  # online mode
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.processor = processor
        self.config = config

        ld_cfg = config.get("logit_distillation", {})
        self.temperature = ld_cfg.get("temperature", 2.0)
        self.alpha = ld_cfg.get("alpha", 0.5)
        self.online_mode = ld_cfg.get("online_mode", False) and teacher_model is not None
        self.top_k = ld_cfg.get("save_top_k_logits", 50)
        self.max_seq_length = _enforce_min_max_seq_length(
            ld_cfg.get(
                "max_seq_length",
                config.get("student_sft", {}).get(
                    "max_seq_length", MIN_ENFORCED_MAX_SEQ_LENGTH
                ),
            ),
            "logit_distillation.max_seq_length",
        )
        self.error_count = 0
        self.local_rank, self.rank, self.world_size, _ = get_distributed_context()
        self.distributed = dist.is_available() and dist.is_initialized() and self.world_size > 1
        self.is_main_process = self.rank == 0
        self.device = next(self.student.parameters()).device

        self.output_dir = ld_cfg.get("output_dir", "./output/student_logit_distill")
        if self.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
        if self.distributed:
            dist.barrier()

    def _zero_losses(self) -> dict[str, torch.Tensor]:
        first_param = next((p for p in self.student.parameters() if p.requires_grad), None)
        if first_param is None:
            zero = torch.zeros((), device=self.device, requires_grad=True)
        else:
            zero = first_param.sum() * 0.0
        return {
            "loss": zero,
            "distill_loss": zero.detach(),
            "hard_loss": zero.detach(),
        }

    def train(
        self,
        train_dataset: DistillationDataset,
        eval_dataset: Optional[DistillationDataset] = None,
    ):
        """학습 실행."""
        from transformers import get_cosine_schedule_with_warmup

        ld_cfg = self.config.get("logit_distillation", {})
        use_fused_adamw = bool(ld_cfg.get("use_fused_adamw", True))
        use_tf32 = bool(ld_cfg.get("tf32", True))

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
            torch.backends.cudnn.allow_tf32 = use_tf32

        # Optimizer
        optimizer_kwargs = {
            "lr": ld_cfg.get("learning_rate", 1e-5),
            "weight_decay": 0.01,
        }
        if torch.cuda.is_available() and use_fused_adamw:
            try:
                optimizer = torch.optim.AdamW(
                    self.student.parameters(),
                    fused=True,
                    **optimizer_kwargs,
                )
            except TypeError:
                optimizer = torch.optim.AdamW(
                    self.student.parameters(),
                    **optimizer_kwargs,
                )
        else:
            optimizer = torch.optim.AdamW(
                self.student.parameters(),
                **optimizer_kwargs,
            )

        num_epochs = ld_cfg.get("num_train_epochs", 3)
        batch_size = ld_cfg.get("per_device_train_batch_size", 1)
        grad_accum = ld_cfg.get("gradient_accumulation_steps", 8)
        es_cfg = ld_cfg.get("early_stopping", {})
        es_enabled = bool(eval_dataset and es_cfg.get("enabled", False))
        es_patience = int(es_cfg.get("patience", 2))
        es_min_delta = float(es_cfg.get("min_delta", 0.0))
        bad_epochs = 0

        if batch_size != 1:
            raise ValueError(
                "Current DistillationTrainer supports batch_size=1 only. "
                "Set logit_distillation.per_device_train_batch_size=1."
            )

        if not self.online_mode:
            teacher_count = len(train_dataset.teacher_logits)
            if teacher_count == 0:
                raise ValueError(
                    "Offline logit distillation requires precomputed teacher logits, "
                    "but loaded count is 0."
                )
            coverage = teacher_count / max(len(train_dataset), 1)
            if coverage < 0.5:
                print(
                    f"Warning: teacher logits coverage is low ({coverage:.1%}). "
                    "Training quality may degrade."
                )

        if es_enabled:
            print(
                "Early stopping enabled for logit distillation:"
                f" patience={es_patience}, min_delta={es_min_delta}"
            )

        # DataLoader
        train_sampler = None
        if self.distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=self._collate_fn,
        )
        if len(train_loader) == 0:
            raise ValueError("Train dataloader is empty. Check student_sft.train_file.")

        eval_sampler = None
        if eval_dataset and self.distributed:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )

        total_steps = max(1, len(train_loader) * num_epochs // max(grad_accum, 1))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.05),
            num_training_steps=total_steps,
        )

        # Training loop
        self.student.train()
        if self.teacher:
            self.teacher.eval()

        global_step = 0
        best_eval_loss = float("inf")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_distill_loss = 0.0
            epoch_hard_loss = 0.0
            num_batches = 0

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            iterator = train_loader
            pbar = None
            if self.is_main_process:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
                iterator = pbar

            for step, batch in enumerate(iterator):
                losses = self._training_step(batch)

                if losses is None:
                    continue

                loss = losses["loss"] / grad_accum
                if not torch.isfinite(loss):
                    self.error_count += 1
                    if self.is_main_process and self.error_count <= 10:
                        print("\n[distill step warning] non-finite aggregated loss skipped.")
                    if self.distributed:
                        loss = self._zero_losses()["loss"]
                    else:
                        continue
                if not loss.requires_grad:
                    self.error_count += 1
                    if self.is_main_process and self.error_count <= 10:
                        print("\n[distill step warning] loss has no grad_fn, skipping batch.")
                    if self.distributed:
                        loss = self._zero_losses()["loss"]
                    else:
                        continue
                try:
                    loss.backward()
                except RuntimeError as e:
                    if self._is_cuda_oom_error(e):
                        if self.distributed:
                            raise
                        self.error_count += 1
                        if self.is_main_process and self.error_count <= 10:
                            print(f"\n[distill step warning] backward OOM skipped: {e}")
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

                epoch_loss += losses["loss"].item()
                epoch_distill_loss += losses["distill_loss"].item()
                epoch_hard_loss += losses["hard_loss"].item()
                num_batches += 1

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    try:
                        optimizer.step()
                        scheduler.step()
                    except RuntimeError as e:
                        if self._is_cuda_oom_error(e):
                            if self.distributed:
                                raise
                            self.error_count += 1
                            if self.is_main_process and self.error_count <= 10:
                                print(f"\n[distill step warning] optimizer OOM skipped: {e}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            raise
                    optimizer.zero_grad()
                    global_step += 1

                    if pbar is not None and global_step % 10 == 0 and num_batches > 0:
                        avg_loss = epoch_loss / num_batches
                        avg_dl = epoch_distill_loss / num_batches
                        avg_hl = epoch_hard_loss / num_batches
                        pbar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "distill": f"{avg_dl:.4f}",
                            "hard": f"{avg_hl:.4f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        })

                    # 체크포인트 저장
                    if self.is_main_process and global_step % 200 == 0:
                        self._save_checkpoint(global_step)

            # grad_accum으로 나눠떨어지지 않는 마지막 step 처리
            if num_batches > 0 and (num_batches % grad_accum) != 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Epoch 종료
            avg_loss = epoch_loss / max(num_batches, 1)
            if self.is_main_process:
                print(f"\nEpoch {epoch+1} complete: avg_loss={avg_loss:.4f}")

            # Evaluation
            if eval_dataset:
                eval_loss = self._evaluate(eval_dataset, eval_sampler=eval_sampler)
                if self.is_main_process:
                    print(f"  Eval loss: {eval_loss:.4f}")
                if eval_loss < (best_eval_loss - es_min_delta):
                    best_eval_loss = eval_loss
                    bad_epochs = 0
                    if self.is_main_process:
                        self._save_checkpoint("best")
                else:
                    bad_epochs += 1
                    if es_enabled and bad_epochs >= es_patience:
                        if self.is_main_process:
                            print(
                                f"  Early stopping triggered at epoch {epoch+1}: "
                                f"no improvement for {bad_epochs} eval checks."
                            )
                        break

        # 최종 저장
        if self.is_main_process:
            self._save_checkpoint("final")
            print(f"\nTraining complete. Model saved to {self.output_dir}")
        if self.distributed:
            dist.barrier()

    def _training_step(self, batch: dict) -> Optional[dict]:
        """단일 학습 스텝."""
        try:
            record = batch["record"]
            messages = record["messages"]

            # Student forward
            student_inputs = self._prepare_inputs(messages)
            if student_inputs is None:
                return self._zero_losses() if self.distributed else None

            student_outputs = self.student(**student_inputs)
            student_logits = student_outputs.logits

            # Teacher logits
            if self.online_mode:
                # 실시간 Teacher 추론
                with torch.no_grad():
                    teacher_outputs = self.teacher(**student_inputs)
                    teacher_logits_full = teacher_outputs.logits

                # top-k 추출
                top_k_values, top_k_indices = torch.topk(
                    teacher_logits_full, k=self.top_k, dim=-1
                )
            else:
                # 사전 저장된 logits 사용
                teacher_logit = batch.get("teacher_logit")
                if teacher_logit is None:
                    # Offline logit이 없으면 hard loss만 사용
                    labels = self._create_labels(messages, student_inputs["input_ids"])
                    valid = (labels != -100)
                    if valid.any():
                        hard_loss = F.cross_entropy(
                            student_logits.view(-1, student_logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                        )
                    else:
                        hard_loss = student_logits.sum() * 0.0
                    return {
                        "loss": hard_loss,
                        "distill_loss": hard_loss.detach() * 0.0,
                        "hard_loss": hard_loss.detach(),
                    }

                top_k_indices = torch.tensor(
                    teacher_logit["top_k_indices"], dtype=torch.long
                ).unsqueeze(0).to(student_logits.device)
                top_k_values = torch.tensor(
                    teacher_logit["top_k_values"], dtype=torch.float32
                ).unsqueeze(0).to(student_logits.device)

                # 시퀀스 길이 맞춤
                assistant_start = teacher_logit["assistant_start"]
                seq_len = min(
                    top_k_indices.shape[1],
                    student_logits.shape[1] - assistant_start,
                )
                if seq_len <= 0:
                    return self._zero_losses() if self.distributed else None
                student_logits = student_logits[:, assistant_start:assistant_start + seq_len]
                top_k_indices = top_k_indices[:, :seq_len]
                top_k_values = top_k_values[:, :seq_len]

            # Hard labels
            hard_labels = self._create_labels(messages, student_inputs["input_ids"])
            if self.online_mode:
                pass  # 전체 시퀀스
            else:
                hard_labels = hard_labels[:, assistant_start:assistant_start + seq_len]

            # Distillation loss 계산
            losses = compute_distillation_loss(
                student_logits, top_k_indices, top_k_values,
                hard_labels,
                temperature=self.temperature,
                alpha=self.alpha,
            )
            if not torch.isfinite(losses["loss"]):
                self.error_count += 1
                if self.is_main_process and self.error_count <= 5:
                    print(
                        "\n[distill step warning] non-finite loss detected. "
                        "Skipping this batch."
                    )
                return self._zero_losses() if self.distributed else None
            return losses

        except Exception as e:
            self.error_count += 1
            if self.is_main_process and self.error_count <= 5:
                print(f"\n[distill step warning] {e}")
            return self._zero_losses() if self.distributed else None

    def _prepare_inputs(self, messages) -> Optional[dict]:
        """메시지를 모델 입력으로 변환."""
        from PIL import Image

        images = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        try:
                            images.append(
                                Image.open(item["image"]).convert("RGB")
                            )
                        except Exception:
                            return None

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
        ).to(self.device)

        input_ids = inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor) and input_ids.shape[1] > self.max_seq_length:
            # pixel_values/image_grid_thw는 시퀀스 축이 아니므로 제외한다.
            skip_keys = {"pixel_values", "image_grid_thw"}
            for key, value in list(inputs.items()):
                if key in skip_keys:
                    continue
                if isinstance(value, torch.Tensor) and value.dim() >= 2:
                    inputs[key] = value[:, :self.max_seq_length]

        return inputs

    @staticmethod
    def _is_cuda_oom_error(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cublas_status_alloc_failed" in msg
            or "cuda error" in msg and "alloc" in msg
        )

    def _create_labels(self, messages, input_ids) -> torch.Tensor:
        """Assistant 부분만 라벨로 설정 (<|im_start|>assistant 토큰 경계 기반)."""
        tokenizer = self.processor.tokenizer
        labels = torch.full_like(input_ids, -100)

        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_header_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

        ids_list = input_ids[0].tolist()
        i = 0
        while i < len(ids_list):
            if ids_list[i] != im_start_id:
                i += 1
                continue

            header_start = i + 1
            header_end = header_start + len(assistant_header_ids)
            if (
                header_end <= len(ids_list)
                and ids_list[header_start:header_end] == assistant_header_ids
            ):
                content_start = header_end
                content_end = len(ids_list)
                for e in range(content_start, len(ids_list)):
                    if ids_list[e] == im_end_id:
                        content_end = e + 1
                        break
                labels[0, content_start:content_end] = input_ids[0, content_start:content_end]
                i = content_end
            else:
                i += 1

        return labels

    def _evaluate(
        self,
        eval_dataset: DistillationDataset,
        eval_sampler: Optional[DistributedSampler] = None,
    ) -> float:
        """평가 loss 계산."""
        self.student.eval()
        total_loss = 0.0
        count = 0

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=(eval_sampler is None),
            sampler=eval_sampler,
            collate_fn=self._collate_fn,
        )

        iterator = eval_loader
        if self.is_main_process:
            iterator = tqdm(eval_loader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for batch in iterator:
                losses = self._training_step(batch)
                if losses:
                    total_loss += losses["loss"].item()
                    count += 1

        if self.distributed:
            stats = torch.tensor(
                [total_loss, float(count)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss = stats[0].item()
            count = int(stats[1].item())

        self.student.train()
        return total_loss / max(count, 1)

    def _collate_fn(self, batch):
        """현재는 batch_size=1만 지원한다."""
        if len(batch) != 1:
            raise ValueError(f"Unsupported batch size: {len(batch)}")
        return batch[0]

    def _save_checkpoint(self, step):
        """체크포인트 저장."""
        if not self.is_main_process:
            return
        save_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = self.student.module if hasattr(self.student, "module") else self.student
        model_to_save.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Logit Distillation Training")
    parser.add_argument("--config", default="config/distill_config.yaml")
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    local_rank, rank, world_size, distributed = get_distributed_context()
    init_distributed(local_rank, distributed)
    is_main_process = rank == 0

    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)

        ld_cfg = config.get("logit_distillation", {})
        sft_cfg = config.get("student_sft", {})
        resolved_max_seq_length = _enforce_min_max_seq_length(
            ld_cfg.get(
                "max_seq_length",
                sft_cfg.get("max_seq_length", MIN_ENFORCED_MAX_SEQ_LENGTH),
            ),
            "logit_distillation.max_seq_length",
        )

        # Student 로드
        student, processor, init_path = load_student_for_distillation(config)
        if is_main_process:
            print(f"Student init path: {init_path}")

        if distributed:
            student = DDP(
                student,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        # Teacher (online mode만)
        teacher = None
        if args.mode == "online":
            from distill.teacher_generate import load_teacher_model
            teacher, _ = load_teacher_model(config)

        train_path = sft_cfg.get("train_file")
        if not train_path or not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Train data not found: {train_path}. "
                "Run quality_filter first to create student_sft.train_file."
            )

        # 데이터셋
        train_dataset = DistillationDataset(
            data_path=train_path,
            teacher_logits_dir=(
                ld_cfg.get("teacher_logits_dir") if args.mode == "offline" else None
            ),
            processor=processor,
            max_seq_length=resolved_max_seq_length,
            require_teacher_logits=(args.mode == "offline"),
        )

        eval_dataset = None
        eval_path = sft_cfg.get("eval_file")
        if eval_path and os.path.exists(eval_path):
            eval_dataset = DistillationDataset(
                data_path=eval_path,
                teacher_logits_dir=(
                    ld_cfg.get("teacher_logits_dir") if args.mode == "offline" else None
                ),
                processor=processor,
                max_seq_length=resolved_max_seq_length,
                require_teacher_logits=False,
            )
        elif is_main_process:
            print(f"Eval dataset not found, skipping eval: {eval_path}")

        # 학습
        trainer = DistillationTrainer(
            student_model=student,
            processor=processor,
            config=config,
            teacher_model=teacher,
        )
        trainer.train(train_dataset, eval_dataset)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
