"""
멀티모달 데이터 Collator

Qwen3-VL의 이미지+텍스트 입력을 처리하는 커스텀 collator.
- processor를 사용하여 이미지 토큰을 올바르게 확장
- 대화형 메시지 → 토큰 변환
- Assistant turn만 loss 마스킹
"""

import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.prompt_templates import (  # noqa: E402
    build_chat_messages,
    PROMPT_STYLE_CHANDRA_WITH_OCR,
)


@dataclass
class MultimodalCollator:
    """
    Qwen3-VL 멀티모달 학습을 위한 Data Collator.

    processor를 사용하여 이미지+텍스트를 토큰화하고,
    assistant 응답 부분에만 loss를 계산하도록 마스킹한다.
    """

    processor: Any
    max_seq_length: int = 8192
    IGNORE_INDEX: int = -100
    truncate_to_assistant: bool = True
    include_thinking: bool = True
    include_empty_cell_prompt_instruction: bool = False
    empty_cell_token: str = "__EMPTY__"
    token_weighting_enabled: bool = False
    span_attribute_weight: float = 1.0
    bbox_weight: float = 1.0

    def __call__(self, examples: list[dict]) -> dict:
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_grid_thw = []
        batch_mm_token_type_ids = []

        for example in examples:
            result = self._process_single(example)
            batch_input_ids.append(result["input_ids"])
            batch_labels.append(result["labels"])
            batch_mm_token_type_ids.append(result.get("mm_token_type_ids"))
            if "pixel_values" in result:
                batch_pixel_values.append(result["pixel_values"])
                batch_image_grid_thw.append(result["image_grid_thw"])

        # Padding
        max_len = min(
            max(len(ids) for ids in batch_input_ids),
            self.max_seq_length,
        )
        pad_token_id = self.processor.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        padded_mm_token_type_ids = []
        has_mm_token_type_ids = any(v is not None for v in batch_mm_token_type_ids)

        for ids, labs, mm_token_type_ids in zip(
            batch_input_ids,
            batch_labels,
            batch_mm_token_type_ids,
        ):
            if self.truncate_to_assistant:
                start, end = self._compute_truncate_window(ids, labs, max_len)
                ids = ids[start:end]
                labs = labs[start:end]
                if mm_token_type_ids is not None:
                    mm_token_type_ids = mm_token_type_ids[start:end]
            else:
                ids = ids[:max_len]
                labs = labs[:max_len]
                if mm_token_type_ids is not None:
                    mm_token_type_ids = mm_token_type_ids[:max_len]
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_labels.append(labs + [self.IGNORE_INDEX] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
            if has_mm_token_type_ids:
                mm_ids = (
                    mm_token_type_ids
                    if mm_token_type_ids is not None
                    else [0] * len(ids)
                )
                padded_mm_token_type_ids.append(mm_ids + [0] * pad_len)

        output = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

        if self.token_weighting_enabled and (
            self.span_attribute_weight > 1.0 or self.bbox_weight > 1.0
        ):
            batch_weights = [
                self._build_span_token_weights(ids, labs)
                for ids, labs in zip(padded_input_ids, padded_labels)
            ]
            output["token_weights"] = torch.tensor(batch_weights, dtype=torch.float)

        if batch_pixel_values:
            output["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            output["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)
        if has_mm_token_type_ids:
            output["mm_token_type_ids"] = torch.tensor(
                padded_mm_token_type_ids,
                dtype=torch.long,
            )

        return output

    # colspan/rowspan 값(따옴표 유무 포함).
    _SPAN_ATTR_RE = re.compile(
        r"\b(?:colspan|rowspan)\s*=\s*(?:\"(\d+)\"|'(\d+)'|(\d+))",
        flags=re.IGNORECASE,
    )
    # data-bbox="x0 y0 x1 y1" (공백 구분 정수 4개; 픽셀/정규화 무관).
    _BBOX_ATTR_RE = re.compile(
        r"data-bbox\s*=\s*[\"'](-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)[\"']",
        flags=re.IGNORECASE,
    )

    def _token_char_offsets(
        self,
        tokenizer: Any,
        sub_ids: list[int],
        full_text: str,
    ):
        """assistant 토큰들의 (char_start, char_end) 오프셋 리스트를 반환한다.

        Qwen 등 byte-level BPE 에서는 토큰을 하나씩 decode 하면 멀티바이트 문자
        (한글/특수문자)가 토큰 경계에서 쪼개져 문자열 복원이 깨진다. 따라서
        fast tokenizer 의 offset_mapping 을 1순위로 쓰고, 실패 시 누적(prefix)
        decode 로 UTF-8 안전하게 폴백한다.
        """
        # 1순위: fast tokenizer offset mapping (re-encode 가 원 토큰열과 일치할 때).
        try:
            if getattr(tokenizer, "is_fast", False):
                enc = tokenizer(
                    full_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                if list(enc["input_ids"]) == list(sub_ids):
                    return list(enc["offset_mapping"])
        except Exception:
            pass

        # 폴백: 누적 prefix decode (멀티바이트 안전, O(n) decode 호출).
        try:
            offsets = []
            prev_len = 0
            for k in range(len(sub_ids)):
                cur_len = len(
                    tokenizer.decode(sub_ids[: k + 1], skip_special_tokens=True)
                )
                offsets.append((prev_len, cur_len))
                prev_len = cur_len
            return offsets
        except Exception:
            return None

    def _build_span_token_weights(
        self,
        input_ids: list[int],
        labels: list[int],
    ) -> list[float]:
        """assistant HTML의 특정 숫자 토큰에 loss 가중치를 부여한다.

        - colspan/rowspan 값 토큰 → span_attribute_weight
        - data-bbox 좌표(x0 y0 x1 y1) 숫자 토큰 → bbox_weight
        나머지는 모두 1.0. (tsr_test 의 per-token weighting 방식을 bbox 로 확장)

        char→token 매핑은 offset_mapping 기반이라 한글 등 멀티바이트 본문에서도
        안전하다.
        """
        weights = [1.0] * len(input_ids)
        span_w = float(self.span_attribute_weight)
        bbox_w = float(self.bbox_weight)
        if not self.token_weighting_enabled or (span_w <= 1.0 and bbox_w <= 1.0):
            return weights

        assistant_positions = [
            i for i, lab in enumerate(labels) if lab != self.IGNORE_INDEX
        ]
        if not assistant_positions:
            return weights

        start = assistant_positions[0]
        end = assistant_positions[-1] + 1
        sub_ids = input_ids[start:end]

        tokenizer = self.processor.tokenizer
        full_text = tokenizer.decode(sub_ids, skip_special_tokens=True)
        if not full_text:
            return weights

        # (char_start, char_end, weight) 범위 수집. 서로 겹치지 않는 속성들이므로
        # start 기준 정렬 후 단일 커서 스캔으로 토큰에 매핑한다.
        value_ranges: list[tuple[int, int, float]] = []

        if span_w > 1.0:
            for m in self._SPAN_ATTR_RE.finditer(full_text):
                value = next((g for g in m.groups() if g is not None), None)
                if not value:
                    continue
                vs = m.start() + m.group(0).find(value)
                value_ranges.append((vs, vs + len(value), span_w))

        if bbox_w > 1.0:
            for m in self._BBOX_ATTR_RE.finditer(full_text):
                # 4개 좌표 숫자 각각의 char 범위에 가중치 부여(공백/따옴표 제외).
                for gi in range(1, 5):
                    value_ranges.append((m.start(gi), m.end(gi), bbox_w))

        if not value_ranges:
            return weights

        offsets = self._token_char_offsets(tokenizer, sub_ids, full_text)
        if offsets is None:
            return weights

        value_ranges.sort(key=lambda r: r[0])
        n_ranges = len(value_ranges)
        range_idx = 0
        for offset, (token_start, token_end) in enumerate(offsets):
            if token_end <= token_start:
                continue  # 특수/빈 토큰

            while range_idx < n_ranges and value_ranges[range_idx][1] <= token_start:
                range_idx += 1
            if range_idx >= n_ranges:
                break

            r_start, r_end, r_weight = value_ranges[range_idx]
            if (token_start < r_end) and (token_end > r_start):
                weights[start + offset] = r_weight

        return weights

    def _process_single(self, record: dict) -> dict:
        """
        단일 레코드를 처리한다.

        slim 포맷(image_path, gt_html, thinking, prompt_style, ocr_info, bbox_scale)
        에서 messages를 동적 생성한 뒤, processor로 이미지+텍스트를 토큰화하고
        assistant 응답 부분에만 loss를 계산하도록 라벨을 생성한다.
        """
        # slim 포맷에서 messages 동적 생성
        messages = build_chat_messages(
            image_path=record["image_path"],
            html=record["gt_html"],
            thinking=record.get("thinking", ""),
            include_thinking=self.include_thinking,
            prompt_idx=None,
            prompt_style=record.get(
                "prompt_style",
                PROMPT_STYLE_CHANDRA_WITH_OCR,
            ),
            ocr_info=record.get("ocr_info"),
            bbox_scale=record.get("bbox_scale", 1024),
            include_empty_cell_prompt_instruction=(
                self.include_empty_cell_prompt_instruction
            ),
            empty_cell_token=self.empty_cell_token,
            bbox_space=record.get("bbox_space", "norm_1000"),
            image_width=record.get("image_width"),
            image_height=record.get("image_height"),
        )

        # 이미지 추출
        images = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item["type"] == "image":
                        img_path = item.get("image", "")
                        try:
                            images.append(Image.open(img_path).convert("RGB"))
                        except Exception:
                            pass

        # apply_chat_template으로 이미지 플레이스홀더가 포함된 텍스트 생성
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        # processor()로 이미지 토큰을 올바르게 확장하여 토큰화
        if images:
            inputs = self.processor(
                text=[text], images=images, return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text], return_tensors="pt",
            )

        input_ids = inputs["input_ids"][0].tolist()

        # Assistant 부분만 loss 계산하도록 라벨 생성
        labels = self._create_labels(input_ids)

        # 모든 라벨이 IGNORE_INDEX이면 이미지 로딩 실패 또는 assistant 구간 탐색 실패
        if all(l == self.IGNORE_INDEX for l in labels):
            img_path = record.get("image_path", "")
            print(
                f"[Warning] All labels are IGNORE_INDEX for sample"
                f" (image={img_path!r}). "
                "Image load failure or assistant turn not found."
            )

        result = {"input_ids": input_ids, "labels": labels}
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
            result["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            result["mm_token_type_ids"] = inputs["mm_token_type_ids"][0].tolist()

        return result

    def _compute_truncate_window(
        self,
        input_ids: list[int],
        labels: list[int],
        target_len: int,
    ) -> tuple[int, int]:
        """assistant 라벨 구간을 최대한 보존하는 자르기 구간을 계산한다."""
        if target_len <= 0 or len(input_ids) <= target_len:
            return 0, len(input_ids)

        label_positions = [
            i for i, token in enumerate(labels)
            if token != self.IGNORE_INDEX
        ]
        n = len(input_ids)

        # assistant 라벨이 없으면 일반 tail-truncation
        if not label_positions:
            return max(0, n - target_len), n

        first_label = label_positions[0]
        last_label = label_positions[-1]
        label_span_len = last_label - first_label + 1

        if label_span_len >= target_len:
            # assistant 자체가 더 길면, 우선 assistant 끝쪽(테이블 마감부) 보존
            start = max(0, last_label - target_len + 1)
            end = min(n, start + target_len)
            return start, end

        # assistant 전체를 포함하면서 가능한 한 많은 앞 컨텍스트 보존
        end = min(n, max(last_label + 1, target_len))
        start = max(0, end - target_len)
        if start > first_label:
            start = first_label
            end = min(n, start + target_len)
        return start, end

    def _truncate_keep_assistant(
        self,
        input_ids: list[int],
        labels: list[int],
        target_len: int,
    ) -> tuple[list[int], list[int]]:
        """
        길이 초과 시 assistant 라벨 구간을 우선 보존한다.
        기존 앞부분 절단(ids[:target_len])은 긴 HTML 정답을 잘라 학습 신호를 약하게 만들 수 있다.
        """
        start, end = self._compute_truncate_window(input_ids, labels, target_len)
        return input_ids[start:end], labels[start:end]

    def _create_labels(self, input_ids: list[int]) -> list[int]:
        """
        Assistant 응답 부분만 학습하도록 라벨을 생성한다.

        <|im_start|>assistant\n ... <|im_end|> 구간의 content만 라벨로 설정.
        """
        tokenizer = self.processor.tokenizer

        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # "assistant\n" 토큰 시퀀스
        assistant_header_ids = tokenizer.encode(
            "assistant\n", add_special_tokens=False,
        )

        labels = [self.IGNORE_INDEX] * len(input_ids)

        i = 0
        while i < len(input_ids):
            if input_ids[i] != im_start_id:
                i += 1
                continue

            # <|im_start|> 다음이 "assistant\n"인지 확인
            header_start = i + 1
            header_end = header_start + len(assistant_header_ids)
            if (
                header_end <= len(input_ids)
                and input_ids[header_start:header_end] == assistant_header_ids
            ):
                # assistant content 시작
                content_start = header_end
                # <|im_end|> 찾기
                content_end = len(input_ids)
                for e in range(content_start, len(input_ids)):
                    if input_ids[e] == im_end_id:
                        content_end = e + 1  # <|im_end|> 포함
                        break

                for idx in range(content_start, content_end):
                    labels[idx] = input_ids[idx]

                i = content_end
            else:
                i += 1

        return labels
