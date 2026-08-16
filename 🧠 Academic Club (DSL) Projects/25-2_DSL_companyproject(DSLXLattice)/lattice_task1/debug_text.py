#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF에서 추출된 텍스트 블록을 디버깅 출력하는 스크립트

사용법:
    python debug_text.py samples/sample.pdf
    python debug_text.py samples/sample.pdf --page 1
    python debug_text.py samples/sample.pdf --page 1 --lines 1-10
    python debug_text.py samples/sample.pdf --search "계약"
    python debug_text.py samples/sample.pdf --no-ocr-fallback  # OCR 폴백 비활성화
    python debug_text.py samples/sample.pdf --ocr-threshold 5  # OCR 임계값 변경
"""
import sys
from pathlib import Path
from app.ingest.loader_pdf import load_pdf_as_blocks, load_pdf_with_fallback_ocr

def print_blocks(blocks, page_filter=None, line_range=None, search_term=None):
    """텍스트 블록들을 출력"""
    print("=" * 80)
    print("추출된 텍스트 블록")
    print("=" * 80)
    print(f"총 {len(blocks)}개 블록\n")
    
    filtered_blocks = blocks
    
    # 페이지 필터링
    if page_filter is not None:
        filtered_blocks = [b for b in filtered_blocks if b.page == page_filter]
        print(f"[필터] 페이지 {page_filter}만 표시")
    
    # 라인 범위 필터링
    if line_range:
        start, end = line_range
        filtered_blocks = [b for b in filtered_blocks 
                          if start <= b.line_no <= end]
        print(f"[필터] 라인 {start}-{end}만 표시")
    
    # 검색어 필터링
    if search_term:
        filtered_blocks = [b for b in filtered_blocks 
                          if search_term.lower() in b.text.lower()]
        print(f"[필터] '{search_term}' 포함된 블록만 표시")
    
    if page_filter or line_range or search_term:
        print(f"필터링 후: {len(filtered_blocks)}개 블록\n")
    
    # 블록 출력
    current_page = None
    for i, block in enumerate(filtered_blocks, 1):
        # 페이지 변경 시 구분선 출력
        if current_page != block.page:
            if current_page is not None:
                print()
            print(f"\n{'='*80}")
            print(f"페이지 {block.page}")
            print('='*80)
            current_page = block.page
        
        # 블록 정보 출력
        print(f"[{i:4d}] 라인 {block.line_no:3d} | {block.text}")
    
    print("\n" + "=" * 80)
    print(f"총 {len(filtered_blocks)}개 블록 출력 완료")
    print("=" * 80)

def print_statistics(blocks):
    """통계 정보 출력"""
    print("\n" + "=" * 80)
    print("통계 정보")
    print("=" * 80)
    
    # 페이지별 통계
    pages = {}
    for block in blocks:
        if block.page not in pages:
            pages[block.page] = 0
        pages[block.page] += 1
    
    print(f"\n총 페이지 수: {len(pages)}")
    print(f"총 블록 수: {len(blocks)}")
    print(f"\n페이지별 블록 수:")
    for page in sorted(pages.keys()):
        print(f"  페이지 {page}: {pages[page]}개 블록")
    
    # 텍스트 길이 통계
    lengths = [len(block.text) for block in blocks]
    if lengths:
        print(f"\n텍스트 길이 통계:")
        print(f"  평균: {sum(lengths) / len(lengths):.1f}자")
        print(f"  최소: {min(lengths)}자")
        print(f"  최대: {max(lengths)}자")
    
    # 빈 줄 제외 통계
    non_empty = [b for b in blocks if b.text.strip()]
    print(f"\n빈 줄 제외: {len(non_empty)}개 블록")

def main():
    if len(sys.argv) < 2:
        print("사용법: python debug_text.py <pdf_file> [옵션]")
        print("\n옵션:")
        print("  --page N              특정 페이지만 표시")
        print("  --lines N-M           특정 라인 범위만 표시")
        print("  --search TERM         검색어가 포함된 블록만 표시")
        print("  --stats               통계 정보만 표시")
        print("  --no-ocr-fallback     OCR 폴백 비활성화 (기본값: 활성화)")
        print("  --ocr-threshold N     OCR 폴백 임계값 설정 (기본값: 10)")
        print("\n예시:")
        print("  python debug_text.py samples/sample.pdf")
        print("  python debug_text.py samples/sample.pdf --page 1")
        print("  python debug_text.py samples/sample.pdf --page 1 --lines 1-10")
        print("  python debug_text.py samples/sample.pdf --search '계약'")
        print("  python debug_text.py samples/sample.pdf --no-ocr-fallback")
        print("  python debug_text.py samples/sample.pdf --ocr-threshold 5")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)
    
    # 옵션 파싱
    page_filter = None
    line_range = None
    search_term = None
    stats_only = False
    use_ocr_fallback = True  # 기본값: OCR 폴백 활성화
    ocr_threshold = 10  # 기본값: 10개
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--page" and i + 1 < len(sys.argv):
            page_filter = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--lines" and i + 1 < len(sys.argv):
            start, end = map(int, sys.argv[i + 1].split("-"))
            line_range = (start, end)
            i += 2
        elif sys.argv[i] == "--search" and i + 1 < len(sys.argv):
            search_term = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--stats":
            stats_only = True
            i += 1
        elif sys.argv[i] == "--no-ocr-fallback":
            use_ocr_fallback = False
            i += 1
        elif sys.argv[i] == "--ocr-threshold" and i + 1 < len(sys.argv):
            ocr_threshold = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # PDF 로딩
    print(f"PDF 파일 로딩 중: {pdf_path}")
    try:
        if use_ocr_fallback:
            print(f"[INFO] OCR 폴백 활성화 (임계값: {ocr_threshold}개 블록)")
            blocks = load_pdf_with_fallback_ocr(
                str(pdf_path), 
                min_blocks_threshold=ocr_threshold
            )
        else:
            print("[INFO] OCR 폴백 비활성화 (기존 방식만 사용)")
            blocks = load_pdf_as_blocks(str(pdf_path))
        print(f"로딩 완료: {len(blocks)}개 블록 추출됨\n")
    except Exception as e:
        print(f"[오류] PDF 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 통계 정보 출력
    print_statistics(blocks)
    
    # 통계만 출력하는 경우 종료
    if stats_only:
        return
    
    # 블록 출력
    print_blocks(blocks, page_filter, line_range, search_term)

if __name__ == "__main__":
    main()

