# -*- coding: utf-8 -*-
"""로더→추출→점수→정규화→검증 라우팅."""
from app.ingest.loader_pdf import load_pdf_as_blocks, load_pdf_with_fallback_ocr
from app.extractors.cnt_name import NameExtractor
from app.extractors.cnt_con_date import ConDateExtractor
from app.extractors.cnt_period import PeriodExtractor
from app.extractors.cnt_amount import AmountExtractor
from app.extractors.cnt_auto_renewal import AutoRenewalExtractor
from app.extractors.cnt_renewal_flag import RenewalFlagExtractor
from app.postprocess import scoring, normalize, validate
from app.models.result import make_empty_result

# 등록된 추출기 목록
# 참고: IdentifierExtractor는 현재 사용하지 않음 (필요시 주석 해제)
REGISTERED_EXTRACTORS = [
    NameExtractor(),
    ConDateExtractor(),
    PeriodExtractor(),
    AmountExtractor(),
    AutoRenewalExtractor(),
    RenewalFlagExtractor(),
]


def process(
    pdf_path: str,
    use_ocr_fallback: bool = True,
    min_blocks_threshold: int = 10
) -> dict:
    """
    PDF 경로→최종 결과(dict).
    
    Args:
        pdf_path: PDF 파일 경로
        use_ocr_fallback: True이면 텍스트 추출 후 블록이 적을 때 OCR 폴백 사용 (기본값: True)
        min_blocks_threshold: 이 값 미만이면 OCR 호출 (기본값: 10)
    
    Returns:
        dict: 추출된 필드 정보가 담긴 딕셔너리
    """
    if use_ocr_fallback:
        blocks = load_pdf_with_fallback_ocr(pdf_path, min_blocks_threshold=min_blocks_threshold)
    else:
        blocks = load_pdf_as_blocks(pdf_path)

    # 모든 필드 모듈에서 후보 수집
    all_candidates = []
    for extractor in REGISTERED_EXTRACTORS:
        try:
            all_candidates.extend(extractor.extract(blocks))
        except Exception as e:
            # 개별 모듈 오류가 전체 파이프라인을 막지 않도록 방어
            print(f"[WARN] extractor {extractor.__class__.__name__}: {e}")

    # 점수화 → 정규화 → 검증/최종 선택
    scored = scoring.apply(all_candidates)
    normed = normalize.apply(scored)
    final_result = validate.select_best(normed)

    # 누락 필드 보정(체결 여부는 사용자 입력)
    result = make_empty_result()
    result.update(final_result)
    if result["CNT_CONCLUDED"]["value"] is None:
        result["CNT_CONCLUDED"]["note"] = "사용자 입력 필요(O/X)"
    return result