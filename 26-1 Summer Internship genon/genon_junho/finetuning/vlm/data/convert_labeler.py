"""
labeler 출력 디렉토리 → Slim JSONL 변환 스크립트

output/ 폴더 구조:
  output/{image_name}/
    final_result.json  → status(PASS/FAIL), attempts(총 시도 횟수)
    attempt_{n}/
      generated.html   → 생성된 테이블 HTML

사용법:
  # 714_set (기본)
  python data/convert_labeler.py --base-dir _train_data/714_set

  # image-prefix 명시
  python data/convert_labeler.py \
    --base-dir _train_data/714_set \
    --image-prefix "train/vlm/_train_data/714_set/input"

  # 출력 위치/이름 변경
  python data/convert_labeler.py \
    --base-dir _train_data/my_set \
    --image-dir input \
    --output-dir results/ \
    --output-name my_set
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

# utils를 import하기 위해 train/vlm 루트를 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
vlm_root = os.path.dirname(script_dir)
sys.path.insert(0, vlm_root)

from utils.span_analyzer import analyze_span


def find_generated_html(folder_path: str, attempts: int) -> str | None:
    """
    attempt_{attempts}/generated.html을 찾는다.
    없으면 낮은 번호부터 역순 탐색.
    """
    for n in range(attempts, 0, -1):
        candidate = os.path.join(folder_path, f"attempt_{n}", "generated.html")
        if os.path.exists(candidate):
            return candidate
    return None


def find_input_image(input_dir: str, image_prefix: str, image_name_no_ext: str) -> str | None:
    """
    input/ 폴더에서 이미지 파일을 탐색한다.
    .png → .jpg 순으로 시도.
    """
    for ext in (".png", ".jpg"):
        path = os.path.join(input_dir, image_name_no_ext + ext)
        if os.path.exists(path):
            filename = image_name_no_ext + ext
            return f"{image_prefix}/{filename}"
    return None


def extract_table_html(html_content: str) -> str:
    """
    HTML 파일에서 <table>...</table> 부분만 추출한다.
    전체가 테이블이면 그대로 반환.
    """
    match = re.search(r"<table[\s\S]*?</table>", html_content, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return html_content.strip()


def convert(base_dir: str, image_prefix: str, image_dir: str, output_dir: str, output_name: str):
    output_dir_abs = os.path.join(base_dir, "output")
    input_dir = os.path.join(base_dir, image_dir)

    pass_records = []
    fail_records = []
    error_count = 0
    skip_count = 0

    folders = sorted(os.listdir(output_dir_abs))
    total = len(folders)

    for i, folder_name in enumerate(folders, 1):
        folder_path = os.path.join(output_dir_abs, folder_name)
        if not os.path.isdir(folder_path):
            continue

        result_path = os.path.join(folder_path, "final_result.json")
        if not os.path.exists(result_path):
            print(f"[SKIP] final_result.json 없음: {folder_name}")
            skip_count += 1
            continue

        # final_result.json 읽기
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)

        status = result.get("status", "")
        attempts = result.get("attempts", 1)

        # generated.html 탐색
        html_path = find_generated_html(folder_path, attempts)
        if html_path is None:
            print(f"[SKIP] generated.html 없음: {folder_name}")
            skip_count += 1
            continue

        # HTML 읽기
        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()

        gt_html = extract_table_html(html_content)

        # input 이미지 탐색 (폴더명 = 이미지명 확장자 제외)
        image_path = find_input_image(input_dir, image_prefix, folder_name)
        if image_path is None:
            print(f"[SKIP] input 이미지 없음: {folder_name}")
            skip_count += 1
            continue

        # complexity 계산
        try:
            span_stats = analyze_span(gt_html)
            complexity = span_stats.complexity
        except Exception as e:
            print(f"[WARN] complexity 계산 실패 ({folder_name}): {e}")
            complexity = "simple"
            error_count += 1

        record = {
            "image_path": image_path,
            "gt_html": gt_html,
            "thinking": "",
            "complexity": complexity,
            "prompt_style": "chandra_table_without_ocr",
        }

        if status == "PASS":
            pass_records.append(record)
        else:
            fail_records.append(record)

        if i % 100 == 0:
            print(f"  처리 중: {i}/{total} (pass={len(pass_records)}, fail={len(fail_records)})")

    # 출력 파일 저장
    os.makedirs(output_dir, exist_ok=True)
    pass_path = os.path.join(output_dir, f"{output_name}_pass.jsonl")
    fail_path = os.path.join(output_dir, f"{output_name}_fail.jsonl")

    with open(pass_path, "w", encoding="utf-8") as f:
        for record in pass_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(fail_path, "w", encoding="utf-8") as f:
        for record in fail_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    print("=== 변환 완료 ===")
    print(f"PASS: {len(pass_records)}건 → {pass_path}")
    print(f"FAIL: {len(fail_records)}건 → {fail_path}")
    print(f"SKIP: {skip_count}건 (파일 없음)")
    print(f"WARN: {error_count}건 (complexity 계산 실패, simple로 대체)")
    print(f"합계: {len(pass_records) + len(fail_records) + skip_count}/{total}")

    # complexity 분포 출력
    for label, records in [("PASS", pass_records), ("FAIL", fail_records)]:
        if not records:
            continue
        dist = Counter(r["complexity"] for r in records)
        print(f"\n{label} complexity: " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))


def main():
    parser = argparse.ArgumentParser(
        description="labeler 출력 디렉토리를 Slim JSONL 포맷으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python data/convert_labeler.py --base-dir _train_data/714_set
  python data/convert_labeler.py --base-dir _train_data/my_set --output-name my_set
        """,
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="labeler 출력 루트 (input/, output/ 포함)",
    )
    parser.add_argument(
        "--image-dir",
        default="input",
        help="labeler 출력 디렉토리 내 이미지 폴더 (기본: input)",
    )
    parser.add_argument(
        "--image-prefix",
        default=None,
        help="JSONL의 image_path 앞에 붙을 prefix (기본: output-dir 기준 상대경로)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="JSONL 저장 위치 (기본: --base-dir과 동일)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="출력 파일명 prefix ({name}_pass.jsonl) (기본: base-dir 폴더명)",
    )

    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)

    # output_dir: 미지정 시 base_dir
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else base_dir

    # image_prefix: 미지정 시 output_dir 기준 상대경로 사용
    image_prefix = args.image_prefix if args.image_prefix else os.path.relpath(os.path.join(base_dir, args.image_dir), output_dir)
    # 선행 ../ 제거 (output_dir이 base_dir 하위 폴더일 때 ../input → input)
    while image_prefix.startswith("../"):
        image_prefix = image_prefix[3:]

    # output_name: 미지정 시 base_dir 폴더명
    output_name = args.output_name if args.output_name else os.path.basename(base_dir)

    print(f"base-dir    : {base_dir}")
    print(f"image-prefix: {image_prefix}")
    print(f"output-dir  : {output_dir}")
    print(f"output-name : {output_name}")
    print()

    convert(base_dir, image_prefix, args.image_dir, output_dir, output_name)


if __name__ == "__main__":
    main()
