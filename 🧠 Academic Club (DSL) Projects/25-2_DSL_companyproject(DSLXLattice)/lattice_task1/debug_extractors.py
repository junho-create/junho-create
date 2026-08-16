#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
추출기들이 어떻게 작동하는지 디버깅 출력하는 스크립트

사용법:
    python debug_extractors.py samples/sample.pdf
    python debug_extractors.py samples/sample.pdf --extractor ConDateExtractor
"""
import sys
import json
from pathlib import Path
from app.ingest.loader_pdf import load_pdf_as_blocks
#from app.extractors.cnt_identifier import IdentifierExtractor
from app.extractors.cnt_name import NameExtractor
from app.extractors.cnt_con_date import ConDateExtractor
from app.extractors.cnt_period import PeriodExtractor
from app.extractors.cnt_amount import AmountExtractor
from app.extractors.cnt_auto_renewal import AutoRenewalExtractor
from app.extractors.cnt_renewal_flag import RenewalFlagExtractor

EXTRACTORS = {
   # "IdentifierExtractor": IdentifierExtractor(),
    "NameExtractor": NameExtractor(),
    "ConDateExtractor": ConDateExtractor(),
    "PeriodExtractor": PeriodExtractor(),
    "AmountExtractor": AmountExtractor(),
    "AutoRenewalExtractor": AutoRenewalExtractor(),
    "RenewalFlagExtractor": RenewalFlagExtractor(),
}

def print_candidates(extractor_name, candidates):
    """추출된 후보들을 출력"""
    print(f"\n{'='*80}")
    print(f"{extractor_name} - 추출된 후보: {len(candidates)}개")
    print('='*80)
    
    if not candidates:
        print("  (후보 없음)")
        return
    
    for i, cand in enumerate(candidates, 1):
        print(f"\n[후보 {i}]")
        print(f"  필드: {cand.field}")
        print(f"  원시 값: {cand.raw_value}")
        print(f"  출처: {cand.source}")
        print(f"  점수: {cand.score:.3f}")
        print(f"  특징: {cand.features}")
        print(f"  근거:")
        print(f"    - 페이지: {cand.evidence.page}")
        print(f"    - 라인: {cand.evidence.lines}")
        print(f"    - 스니펫: {cand.evidence.snippet[:100]}...")

def print_matching_blocks(blocks, pattern_name, pattern):
    """패턴에 매칭되는 블록들을 출력"""
    import re
    if isinstance(pattern, str):
        pattern = re.compile(pattern, re.IGNORECASE)
    
    print(f"\n{'='*80}")
    print(f"패턴 매칭: {pattern_name}")
    print('='*80)
    
    matches = []
    for block in blocks:
        if pattern.search(block.text):
            matches.append(block)
    
    print(f"매칭된 블록: {len(matches)}개\n")
    
    for i, block in enumerate(matches, 1):
        match = pattern.search(block.text)
        matched_text = match.group(0) if match else ""
        print(f"[{i}] 페이지 {block.page}, 라인 {block.line_no}")
        print(f"    매칭: '{matched_text}'")
        print(f"    전체: {block.text[:80]}...")
        print()

def main():
    if len(sys.argv) < 2:
        print("사용법: python debug_extractors.py <pdf_file> [옵션]")
        print("\n옵션:")
        print("  --extractor NAME    특정 추출기만 실행")
        print("  --all               모든 추출기 실행 (기본값)")
        print("  --json              JSON 형식으로 출력")
        print("\n사용 가능한 추출기:")
        for name in EXTRACTORS.keys():
            print(f"  - {name}")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)
    
    # 옵션 파싱
    extractor_filter = None
    json_output = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--extractor" and i + 1 < len(sys.argv):
            extractor_filter = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--json":
            json_output = True
            i += 1
        else:
            i += 1
    
    # PDF 로딩
    print(f"PDF 파일 로딩 중: {pdf_path}")
    try:
        blocks = load_pdf_as_blocks(str(pdf_path))
        print(f"로딩 완료: {len(blocks)}개 블록 추출됨\n")
    except Exception as e:
        print(f"[오류] PDF 로딩 실패: {e}")
        sys.exit(1)
    
    # 추출기 실행
    results = {}
    
    extractors_to_run = [extractor_filter] if extractor_filter else EXTRACTORS.keys()
    
    for extractor_name in extractors_to_run:
        if extractor_name not in EXTRACTORS:
            print(f"[경고] 알 수 없는 추출기: {extractor_name}")
            continue
        
        extractor = EXTRACTORS[extractor_name]
        try:
            candidates = extractor.extract(blocks)
            results[extractor_name] = candidates
            
            if json_output:
                # JSON 형식으로 변환
                json_data = []
                for cand in candidates:
                    json_data.append({
                        "field": cand.field,
                        "raw_value": cand.raw_value,
                        "source": cand.source,
                        "score": cand.score,
                        "features": cand.features,
                        "evidence": {
                            "page": cand.evidence.page,
                            "lines": cand.evidence.lines,
                            "snippet": cand.evidence.snippet[:200]
                        }
                    })
                print(json.dumps({extractor_name: json_data}, ensure_ascii=False, indent=2))
            else:
                print_candidates(extractor_name, candidates)
        except Exception as e:
            print(f"[오류] {extractor_name} 실행 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 요약 출력
    if not json_output:
        print(f"\n{'='*80}")
        print("요약")
        print('='*80)
        for extractor_name, candidates in results.items():
            print(f"{extractor_name}: {len(candidates)}개 후보 추출")
        print('='*80)

if __name__ == "__main__":
    main()

