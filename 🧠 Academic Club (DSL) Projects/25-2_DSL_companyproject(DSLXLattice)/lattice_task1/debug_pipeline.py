#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 파이프라인을 단계별로 디버깅 출력하는 스크립트

사용법:
    python debug_pipeline.py samples/sample.pdf
    python debug_pipeline.py samples/sample.pdf --step extract
"""
import sys
from pathlib import Path
from app.ingest.loader_pdf import load_pdf_as_blocks
#from app.extractors.cnt_identifier import IdentifierExtractor 
# 주석 처리된 이유: IdentifierExtractor을 우리가 함수로 구현하지 않아도 될거 같음!
from app.extractors.cnt_name import NameExtractor
from app.extractors.cnt_con_date import ConDateExtractor
from app.extractors.cnt_period import PeriodExtractor
from app.extractors.cnt_amount import AmountExtractor
from app.extractors.cnt_auto_renewal import AutoRenewalExtractor
from app.extractors.cnt_renewal_flag import RenewalFlagExtractor
from app.postprocess import scoring, normalize, validate

REGISTERED_EXTRACTORS = [
    #IdentifierExtractor(),
    NameExtractor(),
    ConDateExtractor(),
    PeriodExtractor(),
    AmountExtractor(),
    AutoRenewalExtractor(),
    RenewalFlagExtractor(),
]

def print_step(step_name, data, max_items=10):
    """단계별 출력"""
    print(f"\n{'='*80}")
    print(f"[{step_name}]")
    print('='*80)
    
    if isinstance(data, list):
        print(f"총 {len(data)}개 항목")
        if len(data) > max_items:
            print(f"(처음 {max_items}개만 표시)\n")
            data = data[:max_items]
        
        for i, item in enumerate(data, 1):
            print(f"\n[{i}]")
            if hasattr(item, '__dict__'):
                for key, value in item.__dict__.items():
                    if key == 'evidence' and hasattr(value, '__dict__'):
                        print(f"  {key}:")
                        for ek, ev in value.__dict__.items():
                            if ek == 'snippet' and len(str(ev)) > 100:
                                print(f"    {ek}: {str(ev)[:100]}...")
                            else:
                                print(f"    {ek}: {ev}")
                    elif isinstance(value, dict):
                        print(f"  {key}: {value}")
                    elif isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {item}")
    elif isinstance(data, dict):
        print(f"총 {len(data)}개 필드\n")
        for key, value in data.items():
            print(f"  {key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, str) and len(v) > 100:
                        print(f"    {k}: {v[:100]}...")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"    {value}")
    else:
        print(data)
    
    print('='*80)

def main():
    if len(sys.argv) < 2:
        print("사용법: python debug_pipeline.py <pdf_file> [옵션]")
        print("\n옵션:")
        print("  --step STEP        특정 단계만 출력 (load|extract|score|normalize|validate|final)")
        print("  --max-items N      최대 출력 항목 수 (기본: 10)")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)
    
    # 옵션 파싱
    step_filter = None
    max_items = 10
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--step" and i + 1 < len(sys.argv):
            step_filter = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--max-items" and i + 1 < len(sys.argv):
            max_items = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # 1단계: PDF 로딩
    if not step_filter or step_filter == "load":
        print(f"PDF 파일 로딩 중: {pdf_path}")
        try:
            blocks = load_pdf_as_blocks(str(pdf_path))
            print(f"로딩 완료: {len(blocks)}개 블록 추출됨")
            print_step("1. PDF 로딩", blocks, max_items)
        except Exception as e:
            print(f"[오류] PDF 로딩 실패: {e}")
            sys.exit(1)
    else:
        blocks = load_pdf_as_blocks(str(pdf_path))
    
    # 2단계: 추출
    if not step_filter or step_filter == "extract":
        all_candidates = []
        for extractor in REGISTERED_EXTRACTORS:
            try:
                candidates = extractor.extract(blocks)
                all_candidates.extend(candidates)
            except Exception as e:
                print(f"[WARN] extractor {extractor.__class__.__name__}: {e}")
        
        print_step("2. 필드 추출", all_candidates, max_items)
        
        # 필드별 그룹화
        from collections import defaultdict
        by_field = defaultdict(list)
        for cand in all_candidates:
            by_field[cand.field].append(cand)
        
        print(f"\n필드별 후보 수:")
        for field, items in sorted(by_field.items()):
            print(f"  {field}: {len(items)}개")
    else:
        all_candidates = []
        for extractor in REGISTERED_EXTRACTORS:
            try:
                all_candidates.extend(extractor.extract(blocks))
            except Exception:
                pass
    
    # 3단계: 점수화
    if not step_filter or step_filter == "score":
        scored = scoring.apply(all_candidates)
        print_step("3. 점수화", scored, max_items)
        
        # 점수 분포
        scores = [c.score for c in scored]
        if scores:
            print(f"\n점수 분포:")
            print(f"  평균: {sum(scores) / len(scores):.3f}")
            print(f"  최소: {min(scores):.3f}")
            print(f"  최대: {max(scores):.3f}")
    else:
        scored = scoring.apply(all_candidates)
    
    # 4단계: 정규화
    if not step_filter or step_filter == "normalize":
        normed = normalize.apply(scored)
        print_step("4. 정규화", normed, max_items)
    else:
        normed = normalize.apply(scored)
    
    # 5단계: 검증 및 최종 선택
    if not step_filter or step_filter == "validate":
        final_result = validate.select_best(normed)
        print_step("5. 검증 및 최종 선택", final_result)
    else:
        final_result = validate.select_best(normed)
    
    # 최종 결과
    if not step_filter or step_filter == "final":
        from app.models.result import make_empty_result
        result = make_empty_result()
        result.update(final_result)
        if result["CNT_CONCLUDED"]["value"] is None:
            result["CNT_CONCLUDED"]["note"] = "사용자 입력 필요(O/X)"
        
        print_step("6. 최종 결과", result)
        
        # 추출된 필드 요약
        extracted_fields = {k: v for k, v in result.items() if v.get("value") is not None}
        print(f"\n추출된 필드: {len(extracted_fields)}개")
        for field, data in extracted_fields.items():
            print(f"  {field}: {data.get('value')} (신뢰도: {data.get('confidence', 'N/A')})")

if __name__ == "__main__":
    main()

