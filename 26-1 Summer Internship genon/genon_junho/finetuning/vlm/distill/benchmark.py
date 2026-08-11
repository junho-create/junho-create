"""
Teacher vs Student 성능 비교 벤치마크

두 모델의 성능을 동일한 테스트 셋에서 비교하고
상세 분석 리포트를 생성한다.

비교 항목:
- TEDS / TEDS-Structure
- Span F1 / Attribute Accuracy
- 복잡도별 성능 분석
- 추론 속도
- 모델 크기 / 메모리 사용량

Usage:
    python -m distill.benchmark \
        --teacher_model output/teacher/final \
        --student_model output/student_sft/final \
        --test_data data/processed/eval.jsonl \
        --output_dir eval_results/benchmark/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate import (
    load_model_for_inference,
    run_inference,
    extract_html_from_response,
)
from eval.metrics import (
    TEDSCalculator,
    compute_span_metrics,
    compute_aggregate_metrics,
)


def get_model_info(model) -> dict:
    """모델 크기 및 메모리 정보."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # GPU 메모리
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
    else:
        mem_allocated = 0
        mem_reserved = 0

    return {
        "total_params": total_params,
        "total_params_b": total_params / 1e9,
        "trainable_params": trainable_params,
        "gpu_memory_allocated_gb": mem_allocated,
        "gpu_memory_reserved_gb": mem_reserved,
    }


def evaluate_model(
    model,
    processor,
    test_records: list[dict],
    model_name: str,
) -> list[dict]:
    """단일 모델을 평가하고 결과를 반환."""
    predictions = []

    for i, record in enumerate(tqdm(test_records, desc=f"Eval {model_name}")):
        messages = record.get("messages", [])
        metadata = record.get("metadata", {})

        # 이미지 경로
        image_path = None
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        image_path = item.get("image", "")

        # GT HTML
        gt_html = ""
        for msg in messages:
            if msg["role"] == "assistant":
                gt_html = extract_html_from_response(msg.get("content", ""))

        if not image_path or not os.path.exists(image_path):
            continue

        # 추론
        start = time.time()
        try:
            response = run_inference(model, processor, image_path)
            pred_html = extract_html_from_response(response)
        except Exception:
            pred_html = ""
        elapsed = time.time() - start

        # 메트릭
        span_m = compute_span_metrics(pred_html, gt_html)

        predictions.append({
            "index": i,
            "image_path": image_path,
            "pred_html": pred_html,
            "gt_html": gt_html,
            "complexity": metadata.get("complexity", "unknown"),
            "span_f1": span_m.span_f1,
            "attribute_accuracy": span_m.attribute_accuracy,
            "inference_time": elapsed,
        })

    return predictions


def compare_models(
    teacher_preds: list[dict],
    student_preds: list[dict],
) -> dict:
    """Teacher와 Student의 결과를 상세 비교."""
    teacher_agg = compute_aggregate_metrics(teacher_preds)
    student_agg = compute_aggregate_metrics(student_preds)

    # 샘플별 비교
    teacher_by_idx = {p["index"]: p for p in teacher_preds}
    student_by_idx = {p["index"]: p for p in student_preds}

    common_indices = set(teacher_by_idx) & set(student_by_idx)

    student_better = 0
    teacher_better = 0
    equal = 0
    degradation_samples = []

    for idx in common_indices:
        t = teacher_by_idx[idx]
        s = student_by_idx[idx]

        if s["span_f1"] > t["span_f1"] + 0.01:
            student_better += 1
        elif t["span_f1"] > s["span_f1"] + 0.01:
            teacher_better += 1
            # Student가 크게 뒤지는 케이스 기록
            if t["span_f1"] - s["span_f1"] > 0.2:
                degradation_samples.append({
                    "index": idx,
                    "image_path": t["image_path"],
                    "complexity": t["complexity"],
                    "teacher_span_f1": t["span_f1"],
                    "student_span_f1": s["span_f1"],
                    "gap": t["span_f1"] - s["span_f1"],
                })
        else:
            equal += 1

    # 추론 속도 비교
    teacher_avg_time = sum(p["inference_time"] for p in teacher_preds) / max(len(teacher_preds), 1)
    student_avg_time = sum(p["inference_time"] for p in student_preds) / max(len(student_preds), 1)
    speedup = teacher_avg_time / max(student_avg_time, 0.001)

    comparison = {
        "teacher": teacher_agg.to_dict(),
        "student": student_agg.to_dict(),
        "comparison": {
            "teds_gap": teacher_agg.avg_teds - student_agg.avg_teds,
            "teds_struct_gap": teacher_agg.avg_teds_structure - student_agg.avg_teds_structure,
            "span_f1_gap": teacher_agg.avg_span_f1 - student_agg.avg_span_f1,
            "student_better_count": student_better,
            "teacher_better_count": teacher_better,
            "equal_count": equal,
            "teacher_avg_time": teacher_avg_time,
            "student_avg_time": student_avg_time,
            "speedup": speedup,
            "retention_rate": {
                "teds": student_agg.avg_teds / max(teacher_agg.avg_teds, 0.001),
                "teds_structure": student_agg.avg_teds_structure / max(teacher_agg.avg_teds_structure, 0.001),
                "span_f1": student_agg.avg_span_f1 / max(teacher_agg.avg_span_f1, 0.001),
            },
            "degradation_samples": sorted(
                degradation_samples, key=lambda x: -x["gap"]
            )[:20],
        },
    }

    return comparison


def print_benchmark_report(comparison: dict, teacher_info: dict, student_info: dict):
    """벤치마크 결과 리포트 출력."""
    t = comparison["teacher"]
    s = comparison["student"]
    c = comparison["comparison"]

    print("\n" + "=" * 70)
    print("  BENCHMARK REPORT: Teacher vs Student")
    print("=" * 70)

    # 모델 정보
    print(f"\n{'Model Info':-^70}")
    print(f"  {'':30} {'Teacher':>15} {'Student':>15}")
    print(f"  {'Parameters':30} {teacher_info['total_params_b']:>12.1f}B {student_info['total_params_b']:>12.1f}B")
    print(f"  {'GPU Memory':30} {teacher_info['gpu_memory_allocated_gb']:>11.1f}GB {student_info['gpu_memory_allocated_gb']:>11.1f}GB")
    print(f"  {'Avg Inference Time':30} {c['teacher_avg_time']:>12.2f}s {c['student_avg_time']:>12.2f}s")
    print(f"  {'Speedup':30} {'':>15} {c['speedup']:>12.1f}x")

    # 성능 비교
    print(f"\n{'Performance':-^70}")
    print(f"  {'Metric':30} {'Teacher':>12} {'Student':>12} {'Gap':>8} {'Retention':>10}")
    print(f"  {'-'*72}")

    metrics = [
        ("TEDS", t["avg_teds"], s["avg_teds"], "teds"),
        ("TEDS-Structure", t["avg_teds_structure"], s["avg_teds_structure"], "teds_structure"),
        ("Span F1", t["avg_span_f1"], s["avg_span_f1"], "span_f1"),
        ("Span Precision", t["avg_span_precision"], s["avg_span_precision"], None),
        ("Span Recall", t["avg_span_recall"], s["avg_span_recall"], None),
        ("Position F1", t["avg_position_f1"], s["avg_position_f1"], None),
        ("Attribute Accuracy", t["avg_attribute_accuracy"], s["avg_attribute_accuracy"], None),
    ]

    for name, tv, sv, ret_key in metrics:
        gap = tv - sv
        gap_str = f"{gap:+.4f}"
        retention = c["retention_rate"].get(ret_key, sv / max(tv, 0.001)) if ret_key else sv / max(tv, 0.001)
        ret_str = f"{retention:.1%}"

        # 색 표시 (터미널에서)
        indicator = "✓" if retention >= 0.95 else ("⚠" if retention >= 0.85 else "✗")
        print(f"  {name:30} {tv:>12.4f} {sv:>12.4f} {gap_str:>8} {ret_str:>8} {indicator}")

    # 복잡도별
    print(f"\n{'By Complexity (TEDS-Structure)':-^70}")
    for comp in ("simple", "medium", "complex"):
        tv = t.get(f"{comp}_teds", 0)
        sv = s.get(f"{comp}_teds", 0)
        gap = tv - sv
        print(f"  {comp:30} {tv:>12.4f} {sv:>12.4f} {gap:>+8.4f}")

    # 샘플별 비교
    print(f"\n{'Sample Comparison':-^70}")
    print(f"  Student better:  {c['student_better_count']}")
    print(f"  Teacher better:  {c['teacher_better_count']}")
    print(f"  Equal (±0.01):   {c['equal_count']}")

    # 성능 하락 심한 케이스
    degradation = c.get("degradation_samples", [])
    if degradation:
        print(f"\n{'Top Degradation Cases (Student ≪ Teacher)':-^70}")
        for d in degradation[:5]:
            print(
                f"  [{d['complexity']:>8}] {os.path.basename(d['image_path']):30} "
                f"T={d['teacher_span_f1']:.3f} S={d['student_span_f1']:.3f} "
                f"gap={d['gap']:.3f}"
            )

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Teacher vs Student 벤치마크")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--teacher_base_model", default=None)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--student_base_model", default=None)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_dir", default="eval_results/benchmark/")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 테스트 데이터 로드
    records = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"Test samples: {len(records)}")

    # --- Teacher 평가 ---
    print("\n=== Loading Teacher ===")
    teacher_model, teacher_proc = load_model_for_inference(
        args.teacher_model, args.teacher_base_model
    )
    teacher_info = get_model_info(teacher_model)
    teacher_preds = evaluate_model(teacher_model, teacher_proc, records, "Teacher")

    # 메모리 해제
    del teacher_model
    torch.cuda.empty_cache()

    # --- Student 평가 ---
    print("\n=== Loading Student ===")
    student_model, student_proc = load_model_for_inference(
        args.student_model, args.student_base_model
    )
    student_info = get_model_info(student_model)
    student_preds = evaluate_model(student_model, student_proc, records, "Student")

    # --- 비교 ---
    comparison = compare_models(teacher_preds, student_preds)

    # 리포트 출력
    print_benchmark_report(comparison, teacher_info, student_info)

    # 저장
    with open(os.path.join(args.output_dir, "benchmark.json"), "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)

    with open(os.path.join(args.output_dir, "teacher_predictions.jsonl"), "w") as f:
        for p in teacher_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(os.path.join(args.output_dir, "student_predictions.jsonl"), "w") as f:
        for p in student_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
