"""통합(Table + Layout) 모델 평가 메인 스크립트.

기존 `eval.evaluate` 는 모델 출력이 "표 HTML 그대로" 라고 가정하지만,
통합 모델은 layout element 의 JSON 배열을 출력한다. 이 스크립트는

  1) 통합 system prompt + 샘플별 unified prompt 로 추론하고
  2) eval.unified_metrics 로 테이블/레이아웃 지표를 분리 계산한다.

지표
----
- 공통: json_parse_success_rate (출력이 JSON 배열로 파싱되는 비율)
- table 샘플: TEDS / TEDS-structure / span F1 (eval.metrics 재사용)
- layout 샘플: element F1(IoU+category) / category accuracy / mean IoU / text 유사도

백엔드
------
- api    : OpenAI 호환 /v1/chat/completions (서버 vLLM 서빙). 기본.
- vllm   : 로컬 vLLM
- transformers: 로컬 transformers(4-bit)

사용 예 (서버 API)::

    python -m eval.evaluate_unified \
        --backend api \
        --api_url http://192.168.75.173:8010/v1/chat/completions \
        --api_model <served-model> \
        --test_data _train_data/unified_smoke/data/test.jsonl \
        --output_dir eval_results/unified_smoke_test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.unified_metrics import (
    aggregate_unified_metrics,
    evaluate_unified_sample,
)
from utils.prompt_templates import (
    SYSTEM_PROMPT,
    get_user_prompt_with_style,
    normalize_prompt_style,
    prompt_requires_ocr,
    PROMPT_STYLE_CHANDRA_NO_OCR,
    PROMPT_STYLE_CHANDRA_WITH_OCR,
)

# evaluate.py 의 검증된 헬퍼 재사용 (이미지 경로 해석/모델 로딩/저수준 추론)
from eval.evaluate import (
    _resolve_any_path,
    _image_path_to_data_url,
    _get_requests_session,
    load_vllm_model_for_inference,
    load_transformers_model_for_inference,
)

DEFAULT_MAX_NEW_TOKENS = 10000


# =============================================================================
# 샘플 로딩 + 프롬프트 구성
# =============================================================================


def _resolve_sample_prompt_style(record: dict) -> str:
    """레코드의 prompt_style/ocr_info 로 2종 prompt style 중 하나를 결정한다."""
    raw_style = record.get("prompt_style")
    if raw_style:
        return normalize_prompt_style(raw_style)

    ocr_info = record.get("ocr_info")
    if isinstance(ocr_info, list) and len(ocr_info) > 0:
        return PROMPT_STYLE_CHANDRA_WITH_OCR
    return PROMPT_STYLE_CHANDRA_NO_OCR


def _build_user_prompt(record: dict, prompt_style: str) -> str:
    bbox_scale_raw = record.get("bbox_scale", 1024)
    try:
        bbox_scale = int(bbox_scale_raw)
    except Exception:
        bbox_scale = 1024
    ocr_info = record.get("ocr_info") if prompt_requires_ocr(prompt_style) else None
    # 픽셀 좌표계(image_px) 데이터면 학습과 동일하게 이미지 크기를 프롬프트에 명시한다.
    bbox_space = str(record.get("bbox_space", "norm_1000"))
    image_width = record.get("image_width")
    image_height = record.get("image_height")
    return get_user_prompt_with_style(
        idx=0,
        prompt_style=prompt_style,
        ocr_info=ocr_info,
        bbox_scale=bbox_scale,
        bbox_space=bbox_space,
        image_width=image_width,
        image_height=image_height,
    )


def _load_samples(
    test_data_path: str,
    max_samples: Optional[int] = None,
) -> list[dict]:
    records = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if max_samples:
        records = records[:max_samples]

    samples = []
    skipped = 0
    for i, rec in enumerate(records):
        image_path = _resolve_any_path(str(rec.get("image_path", "")), test_data_path)
        if not image_path or not os.path.exists(image_path):
            skipped += 1
            continue
        prompt_style = _resolve_sample_prompt_style(rec)
        # 통합 단일 파이프라인: table/layout 구분을 두지 않는다.
        # manifest 에 task_type 이 없으면 빈 값("")으로 두어 카테고리를 강제하지
        # 않는다. table-crop 이미지든 여러 카테고리가 담긴 페이지든, 모델은 동일
        # 프롬프트로 layout element JSON 배열([{bbox,category,text}])을 생성하고,
        # 평가도 동일한 layout(JSON) 경로로 처리한다.
        task_type = str(rec.get("task_type", "")).strip().lower()
        samples.append(
            {
                "index": i,
                "image_path": image_path,
                "task_type": task_type,
                "gt_html": rec.get("gt_html", ""),
                "prompt_style": prompt_style,
                "prompt": _build_user_prompt(rec, prompt_style),
                "output_format": str(rec.get("output_format", "json")).strip().lower() or "json",
                "system_prompt": SYSTEM_PROMPT,
            }
        )
    if skipped:
        print(f"  Skipped {skipped} samples (image not found)")
    return samples


# =============================================================================
# 추론 백엔드
# =============================================================================


def _infer_api(
    sample: dict,
    api_url: str,
    api_model: str,
    api_key: Optional[str],
    max_new_tokens: int,
    temperature: float,
    timeout_sec: int,
    enable_thinking: bool = True,
) -> str:
    session = _get_requests_session()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    image_data_url = _image_path_to_data_url(sample["image_path"])
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": sample.get("system_prompt", SYSTEM_PROMPT)},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_new_tokens),
    }
    # 학습은 JSON-only(thinking 미포함)이므로, 평가에서도 thinking을 끈다.
    # Qwen3 계열은 thinking이 기본 ON이라 chat_template_kwargs로 hard-switch off 한다.
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    resp = session.post(api_url, headers=headers, json=payload, timeout=timeout_sec)
    if resp.status_code >= 400:
        body = resp.text[:500]
        raise RuntimeError(f"HTTP {resp.status_code}: {body}")
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    content = (choices[0] or {}).get("message", {}).get("content", "")
    if isinstance(content, list):
        return "".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return str(content or "")


def _build_unified_messages_for_vllm(sample: dict) -> list[dict]:
    image_uri = Path(sample["image_path"]).expanduser().resolve().as_uri()
    return [
        {"role": "system", "content": sample.get("system_prompt", SYSTEM_PROMPT)},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
    ]


def _vllm_chat_kwargs(enable_thinking: bool) -> dict:
    """vLLM llm.chat 에 넘길 추가 kwargs.

    학습이 JSON-only(thinking 미포함)이므로 평가에서도 thinking 을 끈다.
    Qwen3 계열은 thinking 이 기본 ON 이라 chat_template_kwargs 로 hard-switch off 한다.
    (vLLM>=0.9.0 에서 LLM.chat 의 chat_template_kwargs 지원)
    """
    if enable_thinking:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _infer_vllm_batch(
    llm,
    samples: list[dict],
    max_new_tokens: int,
    temperature: float,
    lora_request=None,
    enable_thinking: bool = True,
) -> list[str]:
    from vllm import SamplingParams

    if temperature <= 0:
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, top_p=1.0)
    else:
        sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, top_p=0.9)

    messages_batch = [_build_unified_messages_for_vllm(s) for s in samples]
    chat_kwargs = _vllm_chat_kwargs(enable_thinking)
    try:
        outputs = llm.chat(
            messages=messages_batch,
            sampling_params=sp,
            lora_request=lora_request,
            **chat_kwargs,
        )
        results = []
        for o in outputs:
            outs = getattr(o, "outputs", None)
            results.append(getattr(outs[0], "text", "") if outs else "")
        return results
    except Exception as e:
        print(f"Warning: vLLM batch failed, per-sample fallback ({e})")
        results = []
        for messages in messages_batch:
            try:
                o = llm.chat(
                    messages=messages,
                    sampling_params=sp,
                    lora_request=lora_request,
                    **chat_kwargs,
                )
                outs = getattr(o[0], "outputs", None) if o else None
                results.append(getattr(outs[0], "text", "") if outs else "")
            except Exception:
                results.append("")
        return results


def _infer_transformers(
    model, processor, sample: dict, max_new_tokens: int, enable_thinking: bool = True
) -> str:
    import torch
    from PIL import Image

    image = Image.open(sample["image_path"]).convert("RGB")
    messages = [
        {"role": "system", "content": sample.get("system_prompt", SYSTEM_PROMPT)},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
    ]
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # 일부 processor 는 enable_thinking 인자를 받지 않는다.
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


# =============================================================================
# 평가 루프
# =============================================================================


def evaluate_unified(
    test_data_path: str,
    output_dir: str,
    backend: str = "api",
    max_samples: Optional[int] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    iou_threshold: float = 0.5,
    batch_size: int = 8,
    enable_thinking: bool = True,
    # api
    api_url: Optional[str] = None,
    api_model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_timeout: int = 180,
    # 로컬 백엔드 모델
    model: Any = None,
    processor: Any = None,
    lora_request: Any = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    samples = _load_samples(test_data_path, max_samples=max_samples)

    n_table = sum(1 for s in samples if s["task_type"] == "table")
    # task_type 이 비어 있으면(통합 추론) layout 으로 집계.
    n_layout = sum(1 for s in samples if s["task_type"] in ("layout", ""))
    print(f"Evaluating {len(samples)} samples (table={n_table}, layout={n_layout})")
    print(f"  backend: {backend}, iou_threshold: {iou_threshold}")
    print(f"  enable_thinking: {enable_thinking}")

    predictions = []
    total_time = 0.0

    if backend == "vllm":
        eff_bs = max(1, int(batch_size))
        for start in tqdm(range(0, len(samples), eff_bs), desc="Evaluating(vLLM)"):
            chunk = samples[start : start + eff_bs]
            t0 = time.time()
            responses = _infer_vllm_batch(
                model, chunk, max_new_tokens, temperature, lora_request,
                enable_thinking=enable_thinking,
            )
            elapsed = time.time() - t0
            total_time += elapsed
            per = elapsed / max(1, len(chunk))
            for s, r in zip(chunk, responses):
                predictions.append(_make_pred(s, r, per, iou_threshold))
    elif backend == "transformers":
        for s in tqdm(samples, desc="Evaluating(transformers)"):
            t0 = time.time()
            try:
                r = _infer_transformers(
                    model, processor, s, max_new_tokens,
                    enable_thinking=enable_thinking,
                )
            except Exception as e:
                print(f"  Error on {s['index']}: {e}")
                r = ""
            elapsed = time.time() - t0
            total_time += elapsed
            predictions.append(_make_pred(s, r, elapsed, iou_threshold))
    else:  # api
        if not api_url:
            raise ValueError("backend=api requires --api_url")
        if not api_model:
            raise ValueError("backend=api requires --api_model")
        for s in tqdm(samples, desc="Evaluating(API)"):
            t0 = time.time()
            try:
                r = _infer_api(
                    s, api_url, api_model, api_key,
                    max_new_tokens, temperature, api_timeout,
                    enable_thinking=enable_thinking,
                )
            except Exception as e:
                print(f"  Error on {s['index']}: {e}")
                r = ""
            elapsed = time.time() - t0
            total_time += elapsed
            predictions.append(_make_pred(s, r, elapsed, iou_threshold))

    agg = aggregate_unified_metrics(predictions)
    agg["avg_inference_time"] = total_time / max(len(predictions), 1)
    agg["backend"] = backend
    agg["iou_threshold"] = iou_threshold
    agg["enable_thinking"] = bool(enable_thinking)

    pred_path = os.path.join(output_dir, "predictions_unified.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(output_dir, "metrics_unified.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    _print_summary(agg)
    return agg


def _make_pred(sample: dict, response: str, elapsed: float, iou_threshold: float) -> dict:
    rec = evaluate_unified_sample(
        pred_text=response,
        gt_html=sample["gt_html"],
        task_type=sample["task_type"],
        iou_threshold=iou_threshold,
        output_format=sample.get("output_format", "json"),
    )
    rec["index"] = sample["index"]
    rec["image_path"] = sample["image_path"]
    rec["prompt_style"] = sample["prompt_style"]
    rec["full_response"] = response
    rec["inference_time"] = elapsed
    return rec


def _print_summary(agg: dict) -> None:
    is_html = agg.get("output_format") == "html"
    print("\n" + "=" * 60)
    print("UNIFIED EVALUATION RESULTS" + ("  [HTML]" if is_html else "  [JSON]"))
    print("=" * 60)
    print(f"  Total samples:          {agg.get('total_samples', 0)}")
    print(f"  Table / Layout:         {agg.get('table_samples', 0)} / {agg.get('layout_samples', 0)}")
    if is_html:
        print(f"  HTML parse success:     {agg.get('html_parse_success_rate', 0.0):.4f}")
    else:
        print(f"  JSON parse success:     {agg.get('json_parse_success_rate', 0.0):.4f}")
    if "table" in agg:
        t = agg["table"]
        print("  --- Table ---")
        print(f"    Avg TEDS:             {t['avg_teds']:.4f}")
        print(f"    Avg TEDS-Structure:   {t['avg_teds_structure']:.4f}")
        print(f"    Avg Span F1:          {t['avg_span_f1']:.4f}")
        if is_html:
            print(f"    HTML parse success:   {t.get('html_parse_success_rate', 0.0):.4f}")
        else:
            print(f"    JSON parse success:   {t.get('json_parse_success_rate', 0.0):.4f}")
    if "layout" in agg:
        l = agg["layout"]
        print("  --- Layout ---")
        if is_html:
            print(f"    Avg TEDS:             {l['avg_layout_teds']:.4f}")
            print(f"    Avg TEDS-Structure:   {l['avg_layout_teds_structure']:.4f}")
            print(f"    TEDS nonzero rate:    {l['layout_teds_nonzero_rate']:.4f}")
            print(f"    HTML parse success:   {l.get('html_parse_success_rate', 0.0):.4f}")
        else:
            print(f"    Avg Element F1:       {l['avg_layout_f1']:.4f}")
            print(f"    Avg Precision:        {l['avg_layout_precision']:.4f}")
            print(f"    Avg Recall:           {l['avg_layout_recall']:.4f}")
            print(f"    Avg Category Acc:     {l['avg_layout_category_accuracy']:.4f}")
            print(f"    Avg Mean IoU:         {l['avg_layout_mean_iou']:.4f}")
            print(f"    Avg Text Sim:         {l['avg_layout_text_sim']:.4f}")
            print(f"    JSON parse success:   {l.get('json_parse_success_rate', 0.0):.4f}")
    print(f"  Avg inference time:     {agg.get('avg_inference_time', 0.0):.2f}s")
    print(f"  enable_thinking:        {agg.get('enable_thinking', True)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="통합 Table+Layout 모델 평가")
    parser.add_argument("--test_data", required=True, help="통합 테스트 JSONL")
    parser.add_argument("--output_dir", default="eval_results/unified")
    parser.add_argument("--backend", choices=["api", "vllm", "transformers"], default="api")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    # 학습이 JSON-only(thinking 미포함)이므로 평가도 thinking 을 끄는 것을 권장한다.
    # 기본은 thinking ON(과거 동작 호환). --no-thinking 으로 끈다.
    parser.add_argument(
        "--no-thinking",
        dest="no_thinking",
        action="store_true",
        help="추론 시 thinking 을 비활성화한다(학습=JSON-only 와 정합).",
    )
    # 로컬 모델 로딩 옵션
    parser.add_argument("--model", default=None, help="모델/어댑터 경로(vllm/transformers)")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--min_pixels", type=int, default=512 * 512)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--max_lora_rank", type=int, default=128)
    # api 옵션
    parser.add_argument("--api_url", default=os.getenv("TABLE_VLM_URL", ""))
    parser.add_argument("--api_model", default=os.getenv("TABLE_VLM_MODEL", ""))
    parser.add_argument(
        "--api_key",
        default=(os.getenv("TABLE_VLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")),
    )
    parser.add_argument("--api_timeout", type=int, default=180)
    args = parser.parse_args()

    model = processor = lora_request = None
    if args.backend == "vllm":
        if not args.model:
            raise ValueError("backend=vllm requires --model")
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
    elif args.backend == "transformers":
        if not args.model:
            raise ValueError("backend=transformers requires --model")
        model, processor = load_transformers_model_for_inference(
            model_path=args.model,
            base_model_path=args.base_model,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
        )

    evaluate_unified(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        backend=args.backend,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        iou_threshold=args.iou_threshold,
        batch_size=args.batch_size,
        enable_thinking=(not args.no_thinking),
        api_url=args.api_url,
        api_model=(args.api_model or args.model),
        api_key=(args.api_key or None),
        api_timeout=args.api_timeout,
        model=model,
        processor=processor,
        lora_request=lora_request,
    )


if __name__ == "__main__":
    main()
