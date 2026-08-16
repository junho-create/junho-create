#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 전용 전체 필드 추출 CLI.

사용법:
  python cli_llm.py samples/sample.pdf          # 단일 파일 처리
  python cli_llm.py --batch                     # samples/ 폴더의 모든 PDF 처리
  python cli_llm.py --all                       # --batch와 동일
  python cli_llm.py samples/                    # samples/ 폴더의 모든 PDF 처리

기존 cli.py와 거의 동일하지만:
- app.pipeline.run 대신 app.llm.run.run_llm_on_file 를 호출하고
- 결과를 outputs/llm/*.json 에 저장한다.
"""
from pathlib import Path
import json
import argparse

from app.llm.run import run_llm_on_file

OUT_DIR = Path(__file__).parent / "outputs" / "llm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = Path(__file__).parent / "samples"


def process_single_file(pdf_path: Path) -> bool:
    """단일 PDF 파일을 처리하고 성공 여부를 반환."""
    try:
        result = run_llm_on_file(str(pdf_path))
        out_path = OUT_DIR / (pdf_path.stem + ".json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[OK][LLM] {pdf_path.name} → {out_path}")
        return True
    except Exception as e:
        print(f"[ERR][LLM] {pdf_path.name}: {e}")
        return False


def process_batch(samples_dir: Path = None) -> None:
    """samples/ 폴더의 모든 PDF 파일을 처리."""
    if samples_dir is None:
        samples_dir = SAMPLES_DIR

    if not samples_dir.exists():
        print(f"[ERR] Directory not found: {samples_dir}")
        raise SystemExit(1)

    pdf_files = sorted(samples_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] No PDF files found in {samples_dir}")
        return

    print(f"[INFO][LLM] Processing {len(pdf_files)} PDF file(s) from {samples_dir}")
    print("-" * 60)

    success_count = 0
    for pdf_path in pdf_files:
        if process_single_file(pdf_path):
            success_count += 1

    print("-" * 60)
    print(f"[INFO][LLM] Completed: {success_count}/{len(pdf_files)} files processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM으로 PDF 계약서에서 전체 필드를 추출합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  %(prog)s samples/sample.pdf          # 단일 파일 처리
  %(prog)s --batch                     # samples/ 폴더의 모든 PDF 처리
  %(prog)s --all                       # --batch와 동일
  %(prog)s samples/                    # 지정한 폴더의 모든 PDF 처리
        """
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="처리할 PDF 파일 경로 또는 폴더 경로 (생략 시 --batch 옵션 필요)",
    )
    parser.add_argument(
        "--batch",
        "--all",
        dest="batch",
        action="store_true",
        help="samples/ 폴더의 모든 PDF 파일을 배치 처리",
    )

    args = parser.parse_args()

    # 배치 모드
    if args.batch or (args.input and Path(args.input).is_dir()):
        if args.input and Path(args.input).is_dir():
            process_batch(Path(args.input))
        else:
            process_batch()
    # 단일 파일 모드
    elif args.input:
        pdf_path = Path(args.input)
        if not pdf_path.exists():
            print(f"[ERR] File not found: {pdf_path}")
            raise SystemExit(1)
        if pdf_path.suffix.lower() != ".pdf":
            print(f"[ERR] Not a PDF file: {pdf_path}")
            raise SystemExit(1)
        process_single_file(pdf_path)
    else:
        parser.print_help()
        raise SystemExit(1)
