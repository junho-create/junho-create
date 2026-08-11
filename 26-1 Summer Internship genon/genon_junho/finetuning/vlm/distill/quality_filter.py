"""
합성 데이터 품질 필터링 및 밸런싱

Teacher가 생성한 raw 데이터를 필터링하고,
span 복잡도 분포를 밸런싱하여 Student 학습에 최적화된 데이터셋을 만든다.

Filters:
1. HTML 유효성 (파싱 가능, 올바른 구조)
2. 테이블 크기 범위
3. Thinking chain 존재 여부
4. Consistency score 임계값
5. HTML 길이 제한
6. 중복 제거

Balancing:
- 복잡도별 목표 비율에 맞게 언더/오버샘플링

Usage:
    python -m distill.quality_filter \
        --config config/distill_config.yaml \
        --input data/distill/synthetic_raw.jsonl \
        --output data/distill/synthetic_filtered.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.html_utils import parse_html_table, normalize_html
from utils.span_analyzer import analyze_span
from utils.prompt_templates import (
    build_chat_messages,
    normalize_prompt_style,
    prompt_requires_ocr,
    PROMPT_STYLE_CHANDRA_WITH_OCR,
)
from utils.html_utils import extract_spans_from_html


# =============================================================================
# Filters
# =============================================================================


class QualityFilter:
    """합성 데이터 품질 필터."""

    def __init__(self, config: dict):
        fc = config.get("quality_filter", {})
        self.min_consistency = fc.get("min_consistency_teds", 0.8)
        self.min_rows = fc.get("min_table_rows", 2)
        self.min_cols = fc.get("min_table_cols", 2)
        self.max_rows = fc.get("max_table_rows", 50)
        self.max_cols = fc.get("max_table_cols", 20)
        self.require_thinking = fc.get("require_thinking", True)
        self.require_valid_html = fc.get("require_valid_html", True)
        self.require_closed_table = fc.get("require_closed_table", True)
        self.max_html_length = fc.get("max_html_length", 16000)

        self.stats = Counter()

    def check(self, record: dict) -> tuple[bool, str]:
        """
        레코드 검증. (통과 여부, 실패 사유) 반환.
        """
        response = record.get("response", "")
        html = record.get("html", "")

        # 1. Consistency
        consistency = record.get("avg_consistency", 0.0)
        if consistency < self.min_consistency:
            self.stats["low_consistency"] += 1
            return False, f"consistency={consistency:.3f} < {self.min_consistency}"

        # 2. Thinking chain
        if self.require_thinking:
            if "<think>" not in response or "</think>" not in response:
                self.stats["no_thinking"] += 1
                return False, "missing thinking chain"

        # 3. HTML 유효성
        if self.require_valid_html:
            if not html or "<table" not in html:
                self.stats["invalid_html"] += 1
                return False, "invalid HTML"
            if self.require_closed_table and "</table>" not in html.lower():
                self.stats["missing_table_close"] += 1
                return False, "missing </table> (likely truncated HTML)"

            try:
                structure = parse_html_table(html)
            except Exception as e:
                self.stats["parse_error"] += 1
                return False, f"HTML parse error: {e}"

            # 테이블 크기
            if structure.num_rows < self.min_rows:
                self.stats["too_few_rows"] += 1
                return False, f"rows={structure.num_rows} < {self.min_rows}"
            if structure.num_cols < self.min_cols:
                self.stats["too_few_cols"] += 1
                return False, f"cols={structure.num_cols} < {self.min_cols}"
            if structure.num_rows > self.max_rows:
                self.stats["too_many_rows"] += 1
                return False, f"rows={structure.num_rows} > {self.max_rows}"
            if structure.num_cols > self.max_cols:
                self.stats["too_many_cols"] += 1
                return False, f"cols={structure.num_cols} > {self.max_cols}"

            # 빈 테이블 체크
            if len(structure.cells) == 0:
                self.stats["empty_table"] += 1
                return False, "empty table (no cells)"

            # span 속성 유효성 (colspan/rowspan이 테이블 범위를 초과하는지)
            for cell in structure.cells:
                if cell.row + cell.rowspan > structure.num_rows:
                    self.stats["invalid_span"] += 1
                    return False, f"rowspan exceeds table at ({cell.row},{cell.col})"
                if cell.col + cell.colspan > structure.num_cols:
                    self.stats["invalid_span"] += 1
                    return False, f"colspan exceeds table at ({cell.row},{cell.col})"

        # 4. HTML 길이
        if len(html) > self.max_html_length:
            self.stats["too_long"] += 1
            return False, f"HTML length={len(html)} > {self.max_html_length}"

        # 5. 이미지 존재
        img_path = record.get("image_path", "")
        if img_path and not os.path.exists(img_path):
            self.stats["missing_image"] += 1
            return False, f"image not found: {img_path}"

        self.stats["passed"] += 1
        return True, "ok"

    def report(self) -> str:
        total = sum(self.stats.values())
        passed = self.stats.get("passed", 0)
        lines = [
            f"Quality Filter Report:",
            f"  Total checked: {total}",
            f"  Passed:        {passed} ({passed/max(total,1):.1%})",
            f"  Filtered out:  {total - passed}",
            f"  Breakdown:",
        ]
        for reason, count in self.stats.most_common():
            if reason != "passed":
                lines.append(f"    {reason}: {count}")
        return "\n".join(lines)


# =============================================================================
# Deduplication
# =============================================================================


def deduplicate(records: list[dict]) -> list[dict]:
    """
    HTML 구조 기반 중복 제거.
    정확히 같은 HTML 구조를 가진 레코드는 하나만 유지.
    """
    seen = set()
    unique = []

    for record in records:
        html = record.get("html", "")
        # 내용 제거 후 구조만으로 해시
        try:
            normalized = normalize_html(html)
            # 셀 내용 제거하여 구조만 비교
            import re
            structure_only = re.sub(r">([^<]+)<", "><", normalized)
            h = hashlib.md5(structure_only.encode()).hexdigest()
        except Exception:
            h = hashlib.md5(html.encode()).hexdigest()

        if h not in seen:
            seen.add(h)
            unique.append(record)

    removed = len(records) - len(unique)
    if removed > 0:
        print(f"  Deduplication: {len(records)} → {len(unique)} ({removed} duplicates removed)")

    return unique


# =============================================================================
# Balancing
# =============================================================================


def balance_by_complexity(
    records: list[dict],
    target_distribution: dict[str, float],
    max_total: int = None,
    seed: int = 42,
) -> list[dict]:
    """
    복잡도별 목표 비율에 맞게 밸런싱한다.

    카테고리 누락/소량 데이터에서도 0개가 되지 않도록
    목표 분포를 가중치로 사용해 샘플링한다.
    """
    random.seed(seed)

    # 카테고리별 분류
    by_complexity = defaultdict(list)
    for record in records:
        comp = record.get("complexity", "unknown")
        # unknown은 simple로 처리
        if comp not in target_distribution:
            comp = "simple"
        by_complexity[comp].append(record)

    total = len(records)
    if max_total:
        total = min(total, max_total)

    available_complexities = [
        comp for comp, samples in by_complexity.items() if samples
    ]
    if not available_complexities:
        print("  Warning: No available records for balancing")
        return []

    weights = []
    for comp in available_complexities:
        if comp not in target_distribution:
            print(f"  Warning: Complexity '{comp}' not in target_distribution, using fallback weight")
        weights.append(target_distribution.get(comp, 0.0))
    if sum(weights) <= 0:
        weights = [1.0 for _ in available_complexities]

    missing = [comp for comp in target_distribution if comp not in by_complexity]
    for comp in missing:
        print(f"  Warning: No samples for complexity '{comp}'")

    balanced = []
    for _ in range(total):
        comp = random.choices(available_complexities, weights=weights, k=1)[0]
        balanced.append(random.choice(by_complexity[comp]))

    random.shuffle(balanced)

    print(f"  Balancing: {len(records)} → {len(balanced)}")
    dist = Counter(r.get("complexity", "unknown") for r in balanced)
    for comp, count in sorted(dist.items()):
        print(f"    {comp}: {count} ({count/len(balanced):.1%})")

    return balanced


# =============================================================================
# Format Conversion (raw → training format)
# =============================================================================


def convert_to_training_format(
    records: list[dict],
    prompt_style: str = PROMPT_STYLE_CHANDRA_WITH_OCR,
    bbox_scale: int = 1024,
) -> list[dict]:
    """
    필터링된 레코드를 학습용 대화 포맷으로 변환한다.
    Teacher 응답을 그대로 assistant 메시지로 사용.
    """
    training_records = []

    use_ocr = prompt_requires_ocr(prompt_style)
    missing_ocr_count = 0

    for record in records:
        image_path = record["image_path"]
        response = record["response"]
        html = record.get("html", "")
        ocr_info = record.get("ocr_info", [])
        if not isinstance(ocr_info, list):
            ocr_info = []
        try:
            record_bbox_scale = int(record.get("bbox_scale", bbox_scale))
        except Exception:
            record_bbox_scale = int(bbox_scale)

        if use_ocr and not ocr_info:
            missing_ocr_count += 1

        # thinking과 html 분리
        thinking = ""
        if "<think>" in response and "</think>" in response:
            think_match = response.split("<think>")[1].split("</think>")[0]
            thinking = think_match.strip()

        # 대화 메시지 구성
        messages = build_chat_messages(
            image_path=image_path,
            html=html,
            thinking=thinking,
            prompt_idx=None,  # 랜덤 프롬프트
            prompt_style=prompt_style,
            ocr_info=ocr_info,
            bbox_scale=record_bbox_scale,
        )

        training_record = {
            "messages": messages,
            "metadata": {
                "image_path": image_path,
                "complexity": record.get("complexity", "unknown"),
                "complexity_score": record.get("complexity_score", 0.0),
                "source": "teacher_distill",
                "avg_consistency": record.get("avg_consistency", 0.0),
                "prompt_style": prompt_style,
                "bbox_scale": record_bbox_scale,
            },
        }
        if ocr_info:
            training_record["metadata"]["ocr_info"] = ocr_info
        training_records.append(training_record)

    if use_ocr and missing_ocr_count > 0:
        print(
            f"  Warning: {missing_ocr_count} records have empty OCR info "
            f"while prompt_style='{prompt_style}'."
        )

    return training_records


# =============================================================================
# Main Pipeline
# =============================================================================


def run_quality_pipeline(config: dict, input_path: str = None, output_path: str = None):
    """
    전체 품질 필터링 파이프라인 실행.

    1. 로드
    2. 필터링
    3. 중복 제거
    4. 밸런싱
    5. 학습 포맷 변환
    6. Train/Eval 분리
    7. 저장
    """
    fc = config.get("quality_filter", {})
    prompting_cfg = config.get("prompting", {})
    raw_prompt_style = prompting_cfg.get("style", PROMPT_STYLE_CHANDRA_WITH_OCR)
    prompt_style = normalize_prompt_style(raw_prompt_style)
    bbox_scale = int(prompting_cfg.get("bbox_scale", 1024))
    if raw_prompt_style != prompt_style:
        print(
            f"  Warning: unsupported prompting.style='{raw_prompt_style}', "
            f"fallback to '{prompt_style}'."
        )

    input_path = input_path or fc.get("input_path", "data/distill/synthetic_raw.jsonl")
    output_path = output_path or fc.get("output_path", "data/distill/synthetic_filtered.jsonl")

    # 1. 로드
    print("=" * 60)
    print("Quality Filtering Pipeline")
    print("=" * 60)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"\n[1/6] Loaded {len(records)} records from {input_path}")

    # 2. 필터링
    print(f"\n[2/6] Filtering...")
    qf = QualityFilter(config)
    filtered = []
    for record in records:
        passed, reason = qf.check(record)
        if passed:
            filtered.append(record)
    print(qf.report())

    # 3. 중복 제거
    print(f"\n[3/6] Deduplication...")
    filtered = deduplicate(filtered)

    # 4. 밸런싱
    print(f"\n[4/6] Balancing...")
    target_dist = fc.get("target_distribution", {
        "simple": 0.20, "medium": 0.30, "complex": 0.50,
    })
    balanced = balance_by_complexity(filtered, target_dist)

    # 5. 학습 포맷 변환
    print(f"\n[5/6] Converting to training format...")
    print(f"  Prompt style: {prompt_style}")
    print(f"  BBox scale:   {bbox_scale}")
    training_data = convert_to_training_format(
        balanced,
        prompt_style=prompt_style,
        bbox_scale=bbox_scale,
    )

    # 6. Train/Eval 분리
    print(f"\n[6/6] Splitting train/eval...")
    random.seed(42)
    random.shuffle(training_data)
    if len(training_data) == 0:
        raise ValueError(
            "No samples remained after filtering. "
            "Relax filter thresholds or increase synthetic_generation.target_samples."
        )

    # 최소 1개 train 샘플은 보장
    eval_size = max(1, int(len(training_data) * 0.1)) if len(training_data) >= 2 else 0
    eval_size = min(eval_size, max(len(training_data) - 1, 0))
    eval_data = training_data[:eval_size]
    train_data = training_data[eval_size:]

    # 저장
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # 필터링 결과 (raw)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in balanced:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Filtered data: {output_path} ({len(balanced)} records)")

    # 학습 데이터
    train_path = os.path.join(output_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Train data: {train_path} ({len(train_data)} records)")

    eval_path = os.path.join(output_dir, "eval.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Eval data: {eval_path} ({len(eval_data)} records)")

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!")
    print(f"  Input:     {len(records)} raw records")
    print(f"  Filtered:  {len(balanced)} balanced records")
    print(f"  Train:     {len(train_data)}")
    print(f"  Eval:      {len(eval_data)}")
    print(f"{'=' * 60}")

    return {
        "filtered": balanced,
        "train": train_data,
        "eval": eval_data,
    }


def main():
    import yaml

    parser = argparse.ArgumentParser(description="합성 데이터 품질 필터링")
    parser.add_argument("--config", default="config/distill_config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_quality_pipeline(config, args.input, args.output)


if __name__ == "__main__":
    main()
