"""
Claude Opus 4.6 (OpenRouter)를 활용한 GT HTML 생성

학습 데이터의 gt_html을 Claude가 독립적으로 생성하여 교차 검증 자료를 만든다.
기존 eval.evaluate의 API 호출/프롬프트 구성 함수를 재사용한다.

Usage:
    python -m data.generate_gt_with_vlm \
        --input data/experiments/e7_gtfilter/train.jsonl \
        --output data/gt_validation/train_vlm.jsonl \
        --data_root /path/to/images \
        --api_model anthropic/claude-opus-4.6 \
        --max_samples 100

model:
    - anthropic/claude-opus-4.6
    - qwen/qwen3.5-397b-a17b
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate import (
    _build_system_prompt,
    _image_path_to_data_url,
    _run_openai_api_inference,
    extract_html_from_response,
)
from utils.prompt_templates import (
    PROMPT_STYLE_CHANDRA_NO_OCR,
    PROMPT_STYLE_CHANDRA_WITH_OCR,
    get_user_prompt_with_style,
)


def _resolve_image_path(image_path_raw: str, data_root: str) -> str:
    """이미지 경로를 해석한다. 절대경로 → 그대로, 상대경로 → data_root 기준."""
    p = Path(image_path_raw)
    if p.is_absolute() and p.exists():
        return str(p)

    # data_root 기준
    candidate = Path(data_root) / image_path_raw
    if candidate.exists():
        return str(candidate.resolve())

    # 원본 경로 그대로 시도
    if p.exists():
        return str(p.resolve())

    return ""


def _build_prompt_for_sample(record: dict) -> str:
    """샘플의 OCR 유무에 따라 적절한 프롬프트를 생성한다."""
    ocr_info = record.get("ocr_info")
    bbox_scale = int(record.get("bbox_scale", 1024))

    if isinstance(ocr_info, list) and len(ocr_info) > 0:
        return get_user_prompt_with_style(
            idx=0,
            prompt_style=PROMPT_STYLE_CHANDRA_WITH_OCR,
            ocr_info=ocr_info,
            bbox_scale=bbox_scale,
        )

    return get_user_prompt_with_style(
        idx=0,
        prompt_style=PROMPT_STYLE_CHANDRA_NO_OCR,
        ocr_info=None,
        bbox_scale=bbox_scale,
    )


def _load_existing_results(output_path: str) -> set[str]:
    """이미 처리 완료된 image_path 집합을 로드한다 (중단 재개용)."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ip = rec.get("image_path", "")
                if ip:
                    done.add(ip)
            except json.JSONDecodeError:
                continue
    return done


def _process_single(
    record: dict,
    image_path: str,
    api_url: str,
    api_model: str,
    api_key: str,
    max_new_tokens: int,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> dict:
    """단일 샘플에 대해 Claude API 호출 후 결과 레코드를 반환한다."""
    prompt = _build_prompt_for_sample(record)
    result = {
        "image_path": record.get("image_path", ""),
        "gt_html": "",  # Claude 생성 HTML로 대체
        "thinking": record.get("thinking", ""),
        "complexity": record.get("complexity", ""),
        "prompt_style": record.get("prompt_style", ""),
    }

    # OCR 필드 보존
    if "ocr_info" in record:
        result["ocr_info"] = record["ocr_info"]
    if "bbox_scale" in record:
        result["bbox_scale"] = record["bbox_scale"]

    try:
        raw_response = _run_openai_api_inference(
            api_url=api_url,
            api_model=api_model,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
            temperature=temperature,
            api_key=api_key,
            timeout_sec=timeout,
            max_retries=max_retries,
        )
        claude_html = extract_html_from_response(raw_response)
        result["gt_html"] = claude_html
    except Exception as e:
        result["gt_html"] = ""
        result["error"] = str(e)

    return result


def generate_gt_with_claude(
    input_path: str,
    output_path: str,
    data_root: str,
    api_url: str,
    api_model: str,
    api_key: str,
    max_new_tokens: int = 10000,
    temperature: float = 0.0,
    timeout: int = 300,
    max_retries: int = 3,
    max_workers: int = 4,
    max_samples: int | None = None,
    batch_flush: int = 10,
) -> dict:
    """Claude API로 GT HTML을 생성한다.

    Returns:
        처리 통계 dict
    """
    # 입력 데이터 로드
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    # 중단 재개: 이미 처리된 항목 건너뛰기
    done_set = _load_existing_results(output_path)
    pending = []
    for rec in records:
        ip = rec.get("image_path", "")
        if ip in done_set:
            continue
        pending.append(rec)

    print(f"전체: {len(records)}건, 완료: {len(done_set)}건, 대기: {len(pending)}건")

    if not pending:
        print("모든 샘플이 이미 처리되었습니다.")
        return {"total": len(records), "already_done": len(done_set), "processed": 0, "errors": 0}

    # 이미지 경로 해석 및 유효성 검사
    valid_tasks: list[tuple[dict, str]] = []
    skipped = 0
    for rec in pending:
        ip_raw = rec.get("image_path", "")
        image_path = _resolve_image_path(ip_raw, data_root)
        if not image_path:
            print(f"  Skip: 이미지 없음 — {ip_raw}")
            skipped += 1
            continue
        valid_tasks.append((rec, image_path))

    print(f"유효 샘플: {len(valid_tasks)}건 (건너뜀: {skipped}건)")
    print(f"API: {api_url} / 모델: {api_model}")
    print(f"workers: {max_workers}, timeout: {timeout}s, max_retries: {max_retries}")

    # 출력 파일 append 모드
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "a", encoding="utf-8")

    errors = 0
    processed = 0
    buffer: list[str] = []

    def _flush():
        nonlocal buffer
        if buffer:
            out_file.write("".join(buffer))
            out_file.flush()
            buffer = []

    workers = min(max(1, max_workers), len(valid_tasks))
    pbar = tqdm(total=len(valid_tasks), desc="Claude GT 생성")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for rec, img_path in valid_tasks:
            fut = executor.submit(
                _process_single,
                record=rec,
                image_path=img_path,
                api_url=api_url,
                api_model=api_model,
                api_key=api_key,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
            )
            future_map[fut] = rec.get("image_path", "")

        for future in as_completed(future_map):
            try:
                result = future.result()
                buffer.append(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1
                if "error" in result:
                    errors += 1
            except Exception as e:
                ip = future_map[future]
                # 예외 발생 시에도 레코드를 남겨 중단 재개 가능
                error_rec = {"image_path": ip, "gt_html": "", "error": str(e)}
                buffer.append(json.dumps(error_rec, ensure_ascii=False) + "\n")
                processed += 1
                errors += 1

            pbar.update(1)

            # 주기적 flush
            if len(buffer) >= batch_flush:
                _flush()

    _flush()
    out_file.close()
    pbar.close()

    stats = {
        "total": len(records),
        "already_done": len(done_set),
        "processed": processed,
        "errors": errors,
        "skipped": skipped,
    }

    print("\n" + "=" * 50)
    print("Claude GT 생성 완료")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  출력: {output_path}")
    print("=" * 50)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Claude Opus로 GT HTML 생성")
    parser.add_argument("--input", required=True, help="입력 slim JSONL")
    parser.add_argument("--output", required=True, help="출력 JSONL")
    parser.add_argument("--data_root", required=True, help="이미지 루트 디렉토리")
    parser.add_argument(
        "--api_url",
        default="https://openrouter.ai/api/v1/chat/completions",
        help="API URL",
    )
    parser.add_argument(
        "--api_model",
        default="anthropic/claude-opus-4.6",
        help="API 모델명",
    )
    parser.add_argument(
        "--api_key",
        default=os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
        help="API 키 (기본: env OPENROUTER_API_KEY 또는 OPENAI_API_KEY)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300, help="API 타임아웃(초)")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_workers", type=int, default=4, help="병렬 worker 수")
    parser.add_argument("--max_samples", type=int, default=None, help="최대 처리 샘플 수")
    parser.add_argument("--batch_flush", type=int, default=10, help="N건마다 디스크 flush")
    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: API 키가 필요합니다. --api_key 또는 OPENROUTER_API_KEY 환경변수를 설정하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not Path(args.input).exists():
        print(f"Error: 입력 파일을 찾을 수 없습니다: {args.input}", file=sys.stderr)
        sys.exit(1)

    generate_gt_with_claude(
        input_path=args.input,
        output_path=args.output,
        data_root=args.data_root,
        api_url=args.api_url,
        api_model=args.api_model,
        api_key=args.api_key,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        max_samples=args.max_samples,
        batch_flush=args.batch_flush,
    )


if __name__ == "__main__":
    main()
