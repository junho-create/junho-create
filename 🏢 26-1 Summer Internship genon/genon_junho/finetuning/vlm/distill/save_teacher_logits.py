"""
Teacher Logit 사전 계산 및 저장

Offline Logit Distillation을 위해 Teacher 모델의 출력 확률 분포를
사전 계산하여 디스크에 저장한다.

전체 vocabulary에 대한 logit을 저장하면 디스크 사용량이 매우 크므로,
top-k logit만 저장하는 옵션을 제공한다.

Storage format per sample:
    {
        "index": int,
        "image_path": str,
        "token_ids": [int, ...],         # Teacher가 생성한 토큰 시퀀스
        "top_k_logits": [                 # 각 토큰 위치의 top-k logit
            {"indices": [int, ...], "values": [float, ...]},
            ...
        ],
    }

Usage:
    python -m distill.save_teacher_logits \
        --config config/distill_config.yaml \
        --input data/distill/train.jsonl \
        --output_dir data/distill/teacher_logits/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distill.teacher_generate import load_teacher_model
from utils.prompt_templates import get_system_prompt, get_user_prompt

MIN_ENFORCED_MAX_SEQ_LENGTH = 8192


def _enforce_min_max_seq_length(value: int, source_label: str) -> int:
    resolved = int(value)
    if resolved < MIN_ENFORCED_MAX_SEQ_LENGTH:
        print(
            "[Config][warning] max_seq_length is below enforced minimum:"
            f" {source_label}={resolved} -> {MIN_ENFORCED_MAX_SEQ_LENGTH}"
        )
        return MIN_ENFORCED_MAX_SEQ_LENGTH
    return resolved


def _assistant_content(messages: list[dict]) -> str:
    """첫 번째 assistant 메시지 content를 반환."""
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _extract_image_path(messages: list[dict]) -> str:
    """messages에서 첫 번째 image path를 추출한다."""
    for msg in messages:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for item in msg["content"]:
            if isinstance(item, dict) and item.get("type") == "image":
                return str(item.get("image", ""))
    return ""


def _to_vllm_messages(messages: list[dict]) -> list[dict]:
    """학습 메시지 포맷(type=image)을 vLLM chat 포맷(image_url)으로 변환."""
    converted = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")

        if not isinstance(content, list):
            converted.append({"role": role, "content": str(content)})
            continue

        out_items = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                out_items.append({"type": "text", "text": str(item.get("text", ""))})
            elif item_type == "image":
                image_path = str(item.get("image", "")).strip()
                if not image_path:
                    continue
                try:
                    image_uri = Path(image_path).expanduser().resolve().as_uri()
                except Exception:
                    image_uri = f"file://{os.path.abspath(image_path)}"
                out_items.append({"type": "image_url", "image_url": {"url": image_uri}})
            elif item_type == "image_url":
                out_items.append(item)

        converted.append({"role": role, "content": out_items})

    return converted


def _extract_topk_from_vllm_logprobs(
    logprob_entry: Any,
    generated_token_id: int,
    top_k: int,
) -> tuple[list[int], list[float]]:
    """
    vLLM output logprobs entry를 (indices, values) 형태로 정규화한다.
    values는 logprob를 사용한다.
    """
    candidates: list[tuple[int, float]] = []

    if isinstance(logprob_entry, dict):
        iterator = logprob_entry.items()
    elif isinstance(logprob_entry, (list, tuple)):
        iterator = enumerate(logprob_entry)
    else:
        iterator = []

    for key, value in iterator:
        token_id = getattr(value, "token_id", None)
        if token_id is None:
            try:
                token_id = int(key)
            except Exception:
                continue

        logprob = getattr(value, "logprob", None)
        if logprob is None:
            try:
                logprob = float(value)
            except Exception:
                continue

        try:
            candidates.append((int(token_id), float(logprob)))
        except Exception:
            continue

    if not any(tok == generated_token_id for tok, _ in candidates):
        # chosen token이 top-k 후보에 없으면 매우 작은 점수로라도 포함
        candidates.append((int(generated_token_id), -1e9))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:top_k]

    indices = [tok for tok, _ in candidates]
    values = [val for _, val in candidates]

    if len(indices) < top_k:
        pad_n = top_k - len(indices)
        indices.extend([0] * pad_n)
        values.extend([-1e9] * pad_n)

    return indices, values


def _extract_completion_token_ids(completion: Any, tokenizer) -> list[int]:
    """vLLM completion에서 token ids를 추출한다. 없으면 text를 재토크나이즈한다."""
    token_ids = getattr(completion, "token_ids", None) or []
    if token_ids:
        return [int(x) for x in token_ids]

    text = getattr(completion, "text", "")
    if isinstance(text, str) and text:
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            return [int(x) for x in token_ids]
        except Exception:
            return []
    return []


def _build_onehot_topk_from_tokens(
    token_ids: list[int],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    logprobs를 받을 수 없는 환경에서 사용하는 fallback top-k.
    각 위치에서 생성 토큰 1개만 확률 1(로그 0)에 가깝게 둔다.
    """
    seq_len = len(token_ids)
    indices = np.zeros((seq_len, top_k), dtype=np.int32)
    values = np.full((seq_len, top_k), -1e9, dtype=np.float32)
    for i, tok in enumerate(token_ids):
        indices[i, 0] = int(tok)
        values[i, 0] = 0.0
    return indices, values


def compute_teacher_logits_vllm(
    llm,
    tokenizer,
    record: dict,
    top_k: int = 50,
    max_seq_length: int = 8192,
) -> Optional[dict]:
    """
    vLLM backend에서 teacher top-k logprob를 추출한다.

    Note:
      - offline logit distillation에 맞추기 위해 assistant 시작 위치를 함께 저장한다.
      - values는 raw logits가 아닌 logprob이며, soft target으로 사용 시 동일하게 동작한다.
    """
    try:
        from vllm import SamplingParams
    except Exception:
        return None

    messages = record.get("messages", [])
    if not messages:
        return None

    image_path = _extract_image_path(messages)
    if not image_path or not os.path.exists(image_path):
        return None

    assistant_text = _assistant_content(messages)
    if not assistant_text:
        return None

    assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
    if not assistant_tokens:
        return None

    # assistant 시작 오프셋 추정 (학습 입력 토큰 기준)
    try:
        full_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(full_token_ids, torch.Tensor):
            full_token_ids = full_token_ids.tolist()
        if isinstance(full_token_ids, list) and full_token_ids and isinstance(full_token_ids[0], list):
            full_token_ids = full_token_ids[0]
    except Exception:
        full_token_ids = []

    assistant_start = _find_subsequence(full_token_ids, assistant_tokens)
    if assistant_start < 0:
        assistant_start = max(len(full_token_ids) - len(assistant_tokens), 0)

    prompt_messages = [m for m in messages if m.get("role") != "assistant"]
    vllm_messages = _to_vllm_messages(prompt_messages)

    target_max_tokens = min(max(len(assistant_tokens) + 8, 32), max_seq_length)
    sampling_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(target_max_tokens),
    }

    try:
        outputs = llm.chat(
            messages=vllm_messages,
            sampling_params=SamplingParams(logprobs=int(top_k), **sampling_kwargs),
        )
    except Exception as e_with_logprobs:
        # 일부 vLLM/mm 조합은 chat+logprobs를 지원하지 않을 수 있다.
        # 이 경우 logprobs 없이 생성한 뒤 one-hot soft target으로 대체한다.
        try:
            outputs = llm.chat(
                messages=vllm_messages,
                sampling_params=SamplingParams(**sampling_kwargs),
            )
        except Exception as e_no_logprobs:
            raise RuntimeError(
                "vLLM chat failed both with/without logprobs. "
                f"with_logprobs={e_with_logprobs}; without_logprobs={e_no_logprobs}"
            ) from e_no_logprobs

    if not isinstance(outputs, list) or not outputs:
        return None

    req_output = outputs[0]
    completions = getattr(req_output, "outputs", None)
    if not completions:
        return None

    completion = completions[0]
    generated_token_ids = _extract_completion_token_ids(completion, tokenizer)
    if not generated_token_ids:
        return None

    token_logprobs = getattr(completion, "logprobs", None)
    has_logprobs = isinstance(token_logprobs, (list, tuple)) and len(token_logprobs) > 0

    if has_logprobs:
        seq_len = min(len(generated_token_ids), len(token_logprobs), max_seq_length)
    else:
        seq_len = min(len(generated_token_ids), max_seq_length)

    if seq_len <= 0:
        return None

    if has_logprobs:
        top_k_indices = np.zeros((seq_len, top_k), dtype=np.int32)
        top_k_values = np.full((seq_len, top_k), -1e9, dtype=np.float32)

        for pos in range(seq_len):
            inds, vals = _extract_topk_from_vllm_logprobs(
                token_logprobs[pos],
                int(generated_token_ids[pos]),
                top_k,
            )
            top_k_indices[pos] = np.asarray(inds, dtype=np.int32)
            top_k_values[pos] = np.asarray(vals, dtype=np.float32)
    else:
        top_k_indices, top_k_values = _build_onehot_topk_from_tokens(
            generated_token_ids[:seq_len], top_k
        )

    return {
        "token_ids": full_token_ids if isinstance(full_token_ids, list) else [],
        "assistant_start": int(assistant_start),
        "assistant_token_ids": [int(x) for x in generated_token_ids[:seq_len]],
        "top_k_indices": top_k_indices,
        "top_k_values": top_k_values.astype(np.float16),
    }


def compute_teacher_logits(
    model,
    processor,
    record: dict,
    top_k: int = 50,
    max_seq_length: int = 8192,
) -> Optional[dict]:
    """
    단일 레코드에 대해 Teacher 모델의 logit을 계산한다.

    Args:
        model: Teacher 모델
        processor: 프로세서
        record: {"messages": [...]}
        top_k: 저장할 상위 k개 logit
        max_seq_length: 최대 시퀀스 길이

    Returns:
        {
            "token_ids": list[int],        # 전체 토큰 시퀀스
            "assistant_start": int,        # assistant 응답 시작 위치
            "top_k_indices": np.array,     # [seq_len, top_k] int32
            "top_k_values": np.array,      # [seq_len, top_k] float16
        }
    """
    messages = record.get("messages", [])

    # 이미지 추출
    images = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    img_path = item.get("image", "")
                    try:
                        images.append(Image.open(img_path).convert("RGB"))
                    except Exception:
                        return None

    # 전체 대화를 토큰화
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    input_ids = inputs["input_ids"]
    if input_ids.shape[1] > max_seq_length:
        # 너무 길면 잘라냄 (pixel_values, image_grid_thw는 이미지 패치 차원이므로 제외)
        skip_keys = {"pixel_values", "image_grid_thw"}
        for key in inputs:
            if key in skip_keys:
                continue
            if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() >= 2:
                inputs[key] = inputs[key][:, :max_seq_length]
        input_ids = inputs["input_ids"]

    # Forward pass (logit 계산)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    logits = logits[0]  # [seq_len, vocab_size]

    # Assistant 시작 위치 찾기 (대략적으로)
    assistant_content = ""
    for msg in messages:
        if msg["role"] == "assistant":
            assistant_content = msg.get("content", "")
            break

    assistant_tokens = processor.tokenizer.encode(
        assistant_content, add_special_tokens=False
    )
    assistant_start = _find_subsequence(
        input_ids[0].tolist(), assistant_tokens
    )
    if assistant_start < 0:
        assistant_start = input_ids.shape[1] // 2  # fallback

    # Top-k logit 추출 (assistant 부분만)
    assistant_logits = logits[assistant_start:]
    top_k_values, top_k_indices = torch.topk(assistant_logits, k=top_k, dim=-1)

    return {
        "token_ids": input_ids[0].cpu().tolist(),
        "assistant_start": assistant_start,
        "assistant_token_ids": input_ids[0, assistant_start:].cpu().tolist(),
        "top_k_indices": top_k_indices.cpu().numpy().astype(np.int32),
        "top_k_values": top_k_values.cpu().to(torch.float16).numpy(),
    }


def _find_subsequence(sequence: list, subsequence: list) -> int:
    """시퀀스에서 하위 시퀀스의 시작 위치를 찾는다."""
    if not subsequence:
        return -1
    # 처음 5개 토큰으로 매칭 (긴 시퀀스에서 효율)
    prefix = subsequence[:min(5, len(subsequence))]
    n, m = len(sequence), len(prefix)
    for i in range(n - m + 1):
        if sequence[i:i + m] == prefix:
            return i
    return -1


def save_logits_batch(
    model,
    processor,
    tokenizer,
    data_path: str,
    output_dir: str,
    top_k: int = 50,
    max_seq_length: int = 8192,
    max_samples: int = None,
):
    """
    데이터셋 전체의 Teacher logit을 사전 계산하여 저장한다.

    파일 구조:
        output_dir/
            metadata.json          # 설정 정보
            logits_0000.npz        # 배치별 numpy 압축 파일
            logits_0001.npz
            ...
    """
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 로드
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    print(f"Computing teacher logits for {len(records)} samples...")
    print(f"  Top-k: {top_k}, Max seq len: {max_seq_length}")
    print(f"  Output: {output_dir}")

    # 메타데이터 저장
    metadata = {
        "num_samples": len(records),
        "top_k": top_k,
        "max_seq_length": max_seq_length,
        "samples_per_file": 100,
    }

    batch_size = metadata["samples_per_file"]
    batch_data = []
    batch_idx = 0
    processed = 0
    errors = 0

    for i, record in enumerate(tqdm(records, desc="Computing logits")):
        try:
            if processor is None:
                result = compute_teacher_logits_vllm(
                    model,
                    tokenizer,
                    record,
                    top_k=top_k,
                    max_seq_length=max_seq_length,
                )
            else:
                result = compute_teacher_logits(
                    model, processor, record, top_k, max_seq_length
                )
            if result is None:
                errors += 1
                continue

            batch_data.append({
                "index": i,
                "image_path": record.get("metadata", {}).get("image_path", ""),
                "assistant_start": result["assistant_start"],
                "assistant_token_ids": result["assistant_token_ids"],
                "top_k_indices": result["top_k_indices"],
                "top_k_values": result["top_k_values"],
            })
            processed += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  Error on record {i}: {e}")

        # 배치 저장
        if len(batch_data) >= batch_size:
            _save_batch(output_dir, batch_idx, batch_data)
            batch_data = []
            batch_idx += 1

    # 남은 배치 저장
    if batch_data:
        _save_batch(output_dir, batch_idx, batch_data)
        batch_idx += 1

    metadata["num_batches"] = batch_idx
    metadata["num_processed"] = processed
    metadata["num_errors"] = errors

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nLogit computation complete:")
    print(f"  Processed: {processed}/{len(records)}")
    print(f"  Errors: {errors}")
    print(f"  Batches: {batch_idx}")

    # 디스크 사용량
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
    )
    print(f"  Disk usage: {total_size / 1024 / 1024:.1f} MB")


def _save_batch(output_dir: str, batch_idx: int, batch_data: list):
    """배치를 npz 파일로 저장."""
    filepath = os.path.join(output_dir, f"logits_{batch_idx:04d}.npz")

    save_dict = {}
    for i, item in enumerate(batch_data):
        prefix = f"s{i}_"
        save_dict[f"{prefix}index"] = np.array(item["index"])
        save_dict[f"{prefix}assistant_start"] = np.array(item["assistant_start"])
        save_dict[f"{prefix}assistant_token_ids"] = np.array(
            item["assistant_token_ids"], dtype=np.int32
        )
        save_dict[f"{prefix}top_k_indices"] = item["top_k_indices"]
        save_dict[f"{prefix}top_k_values"] = item["top_k_values"]

    save_dict["batch_size"] = np.array(len(batch_data))
    np.savez_compressed(filepath, **save_dict)


def load_logits_batch(filepath: str) -> list[dict]:
    """npz 배치 파일에서 logit 데이터를 로드."""
    data = np.load(filepath)
    batch_size = int(data["batch_size"])

    items = []
    for i in range(batch_size):
        prefix = f"s{i}_"
        items.append({
            "index": int(data[f"{prefix}index"]),
            "assistant_start": int(data[f"{prefix}assistant_start"]),
            "assistant_token_ids": data[f"{prefix}assistant_token_ids"],
            "top_k_indices": data[f"{prefix}top_k_indices"],
            "top_k_values": data[f"{prefix}top_k_values"],
        })

    return items


def main():
    import yaml
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description="Teacher logit 사전 계산")
    parser.add_argument("--config", default="config/distill_config.yaml")
    parser.add_argument("--input", default=None, help="학습 데이터 JSONL")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ld_cfg = config.get("logit_distillation", {})

    model, processor = load_teacher_model(config)

    tokenizer = None
    if processor is None:
        teacher_cfg = config.get("teacher", {})
        tokenizer_source = (
            teacher_cfg.get("fallback_name_or_path")
            or teacher_cfg.get("name_or_path")
        )
        print(
            "Teacher backend is vLLM. Loading tokenizer/processor for chat template parsing:",
            tokenizer_source,
        )
        tokenizer_processor = AutoProcessor.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
        )
        tokenizer = tokenizer_processor.tokenizer
    else:
        tokenizer = processor.tokenizer

    resolved_max_seq_length = (
        args.max_seq_length
        or ld_cfg.get("max_seq_length")
        or config.get("student_sft", {}).get("max_seq_length")
        or MIN_ENFORCED_MAX_SEQ_LENGTH
    )
    resolved_max_seq_length = _enforce_min_max_seq_length(
        resolved_max_seq_length,
        "save_teacher_logits.max_seq_length",
    )

    save_logits_batch(
        model, processor,
        tokenizer,
        data_path=args.input or config["student_sft"]["train_file"],
        output_dir=args.output_dir or ld_cfg.get("teacher_logits_dir", "data/distill/teacher_logits"),
        top_k=args.top_k or ld_cfg.get("save_top_k_logits", 50),
        max_seq_length=int(resolved_max_seq_length),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
