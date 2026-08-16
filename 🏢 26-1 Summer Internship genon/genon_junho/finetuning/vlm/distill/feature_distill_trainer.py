"""
Feature-level Distillation

Teacher와 Student의 중간 레이어 hidden state를 매칭하여
Student가 Teacher의 내부 표현을 학습하도록 한다.

Teacher (235B, hidden=8192, 94 layers) → Student (8B, hidden=4096, 32 layers)

주의: 같은 아키텍처 계열(Qwen3-VL)에서만 효과적.

Usage:
    python -m distill.feature_distill_trainer \
        --config config/distill_config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Projection Layers
# =============================================================================


class LinearProjection(nn.Module):
    """Teacher hidden dim → Student hidden dim 선형 변환."""

    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class MLPProjection(nn.Module):
    """2-layer MLP projection (더 풍부한 변환)."""

    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        mid_dim = (teacher_dim + student_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(teacher_dim, mid_dim, bias=False),
            nn.GELU(),
            nn.Linear(mid_dim, student_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class AttentionProjection(nn.Module):
    """Attention-based projection for selective feature matching."""

    def __init__(self, teacher_dim: int, student_dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Linear(student_dim, student_dim, bias=False)
        self.key = nn.Linear(teacher_dim, student_dim, bias=False)
        self.value = nn.Linear(teacher_dim, student_dim, bias=False)
        self.num_heads = num_heads
        self.scale = (student_dim // num_heads) ** -0.5

    def forward(self, teacher_hidden, student_hidden):
        """student가 teacher에서 필요한 정보를 attention으로 추출."""
        q = self.query(student_hidden)
        k = self.key(teacher_hidden)
        v = self.value(teacher_hidden)

        # 간단한 scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


# =============================================================================
# Feature Distillation Loss
# =============================================================================


def compute_feature_loss(
    teacher_hiddens: list[torch.Tensor],
    student_hiddens: list[torch.Tensor],
    projections: nn.ModuleList,
    layer_mapping: dict[int, int],
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Teacher와 Student의 hidden state 간 feature loss를 계산한다.

    Args:
        teacher_hiddens: Teacher의 hidden states [layer][batch, seq, hidden]
        student_hiddens: Student의 hidden states
        projections: Teacher→Student 차원 변환 모듈
        layer_mapping: {teacher_layer: student_layer}
        loss_type: "mse", "cosine", "smooth_l1"

    Returns:
        feature loss (scalar)
    """
    total_loss = 0.0
    n_pairs = 0

    for proj_idx, (t_layer, s_layer) in enumerate(sorted(layer_mapping.items())):
        if t_layer >= len(teacher_hiddens) or s_layer >= len(student_hiddens):
            continue

        t_hidden = teacher_hiddens[t_layer]  # [batch, seq, teacher_dim]
        s_hidden = student_hiddens[s_layer]  # [batch, seq, student_dim]

        # 시퀀스 길이 맞춤
        min_seq = min(t_hidden.shape[1], s_hidden.shape[1])
        t_hidden = t_hidden[:, :min_seq]
        s_hidden = s_hidden[:, :min_seq]

        # Teacher → Student 차원으로 projection
        t_projected = projections[proj_idx](t_hidden)

        # Loss 계산
        if loss_type == "mse":
            loss = F.mse_loss(t_projected, s_hidden)
        elif loss_type == "cosine":
            loss = 1.0 - F.cosine_similarity(
                t_projected.reshape(-1, t_projected.shape[-1]),
                s_hidden.reshape(-1, s_hidden.shape[-1]),
                dim=-1,
            ).mean()
        elif loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(t_projected, s_hidden)
        else:
            loss = F.mse_loss(t_projected, s_hidden)

        total_loss += loss
        n_pairs += 1

    return total_loss / max(n_pairs, 1)


# =============================================================================
# Combined Distillation Trainer (Feature + Logit + Hard)
# =============================================================================


class CombinedDistillationTrainer:
    """
    Feature + Logit + Hard label을 조합한 종합 증류 트레이너.

    Total Loss = w_feature * feature_loss
               + w_logit * logit_loss
               + w_hard * hard_label_loss
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        processor,
        config: dict,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.processor = processor
        self.config = config

        fd_cfg = config.get("feature_distillation", {})
        ld_cfg = config.get("logit_distillation", {})

        # Layer mapping
        self.layer_mapping = {
            int(k): int(v)
            for k, v in fd_cfg.get("layer_mapping", {}).items()
        }

        # Projection layers
        teacher_dim = fd_cfg.get("teacher_hidden_dim", 8192)
        student_dim = fd_cfg.get("student_hidden_dim", 4096)
        proj_type = fd_cfg.get("projection_type", "linear")

        ProjectionClass = {
            "linear": LinearProjection,
            "mlp": MLPProjection,
        }.get(proj_type, LinearProjection)

        self.projections = nn.ModuleList([
            ProjectionClass(teacher_dim, student_dim)
            for _ in self.layer_mapping
        ])

        # Device 설정
        if hasattr(student_model, "device"):
            self.projections = self.projections.to(student_model.device)

        # Loss weights
        self.w_feature = fd_cfg.get("feature_loss_weight", 0.3)
        self.w_logit = fd_cfg.get("logit_loss_weight", 0.5)
        self.w_hard = fd_cfg.get("hard_loss_weight", 0.2)

        # Distillation params
        self.temperature = ld_cfg.get("temperature", 2.0)
        self.top_k = ld_cfg.get("save_top_k_logits", 50)

        self.output_dir = fd_cfg.get("output_dir", "./output/student_feature_distill")
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, train_data_path: str, eval_data_path: str = None):
        """학습 실행."""
        fd_cfg = self.config.get("feature_distillation", {})
        ld_cfg = self.config.get("logit_distillation", {})

        # 데이터 로드
        records = []
        with open(train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Optimizer (Student + Projections)
        all_params = list(self.student.parameters()) + list(self.projections.parameters())
        optimizer = torch.optim.AdamW(
            all_params,
            lr=ld_cfg.get("learning_rate", 1e-5),
            weight_decay=0.01,
        )

        num_epochs = ld_cfg.get("num_train_epochs", 3)
        grad_accum = ld_cfg.get("gradient_accumulation_steps", 8)

        self.teacher.eval()
        self.student.train()

        global_step = 0
        log_interval = 10

        for epoch in range(num_epochs):
            epoch_losses = {"total": 0, "feature": 0, "logit": 0, "hard": 0}
            num_steps = 0

            pbar = tqdm(records, desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, record in enumerate(pbar):
                try:
                    losses = self._combined_step(record)
                except Exception as e:
                    continue

                if losses is None:
                    continue

                loss = losses["total"] / grad_accum
                loss.backward()

                for key in epoch_losses:
                    epoch_losses[key] += losses[key].item()
                num_steps += 1

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % log_interval == 0:
                        n = max(num_steps, 1)
                        pbar.set_postfix({
                            "loss": f"{epoch_losses['total']/n:.4f}",
                            "feat": f"{epoch_losses['feature']/n:.4f}",
                            "logit": f"{epoch_losses['logit']/n:.4f}",
                            "hard": f"{epoch_losses['hard']/n:.4f}",
                        })

                    if global_step % 200 == 0:
                        self._save_checkpoint(global_step)

            n = max(num_steps, 1)
            print(f"\nEpoch {epoch+1}: "
                  f"total={epoch_losses['total']/n:.4f}, "
                  f"feature={epoch_losses['feature']/n:.4f}, "
                  f"logit={epoch_losses['logit']/n:.4f}, "
                  f"hard={epoch_losses['hard']/n:.4f}")

        self._save_checkpoint("final")
        print(f"\nTraining complete. Saved to {self.output_dir}")

    def _combined_step(self, record: dict) -> Optional[dict]:
        """Feature + Logit + Hard loss를 결합한 단일 스텝."""
        from PIL import Image

        messages = record.get("messages", [])

        # 입력 준비
        images = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        try:
                            images.append(Image.open(item["image"]).convert("RGB"))
                        except Exception:
                            return None

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text], images=images if images else None,
            return_tensors="pt",
        ).to(self.student.device)

        # Teacher forward (hidden states 포함)
        with torch.no_grad():
            teacher_out = self.teacher(
                **inputs, output_hidden_states=True
            )
            teacher_logits = teacher_out.logits
            teacher_hiddens = teacher_out.hidden_states

        # Student forward (hidden states 포함)
        student_out = self.student(
            **inputs, output_hidden_states=True
        )
        student_logits = student_out.logits
        student_hiddens = student_out.hidden_states

        # Labels (assistant만)
        labels = self._create_labels(messages, inputs["input_ids"])

        # === Feature Loss ===
        feature_loss = compute_feature_loss(
            list(teacher_hiddens), list(student_hiddens),
            self.projections, self.layer_mapping,
        )

        # === Logit Loss ===
        top_k_values, top_k_indices = torch.topk(
            teacher_logits, k=self.top_k, dim=-1
        )
        from distill.logit_distill_trainer import compute_distillation_loss
        logit_losses = compute_distillation_loss(
            student_logits, top_k_indices, top_k_values, labels,
            temperature=self.temperature, alpha=1.0,  # pure distill
        )
        logit_loss = logit_losses["distill_loss"]

        # === Hard Loss ===
        vocab_size = student_logits.shape[-1]
        hard_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )

        # === Combined ===
        total = (
            self.w_feature * feature_loss
            + self.w_logit * logit_loss
            + self.w_hard * hard_loss
        )

        return {
            "total": total,
            "feature": feature_loss.detach(),
            "logit": logit_loss.detach() if isinstance(logit_loss, torch.Tensor) else torch.tensor(logit_loss),
            "hard": hard_loss.detach(),
        }

    def _create_labels(self, messages, input_ids):
        """Assistant 부분만 라벨 생성 (<|im_start|>assistant 토큰 경계 기반)."""
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

    def _save_checkpoint(self, step):
        save_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        self.student.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        # Projection 레이어도 저장
        torch.save(
            self.projections.state_dict(),
            os.path.join(save_dir, "projections.pt"),
        )


def main():
    parser = argparse.ArgumentParser(description="Feature Distillation Training")
    parser.add_argument("--config", default="config/distill_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from distill.teacher_generate import load_teacher_model
    from distill.logit_distill_trainer import load_student_for_distillation

    # Teacher
    teacher, _ = load_teacher_model(config)

    # Student
    student, processor, init_path = load_student_for_distillation(config)
    print(f"Student init path: {init_path}")
    sft_cfg = config.get("student_sft", {})

    # Trainer
    trainer = CombinedDistillationTrainer(teacher, student, processor, config)
    trainer.train(
        train_data_path=sft_cfg["train_file"],
        eval_data_path=sft_cfg.get("eval_file"),
    )


if __name__ == "__main__":
    main()
