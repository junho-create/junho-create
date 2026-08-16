# -*- coding: utf-8 -*-
import re

# 날짜: 2024.07.01. / 2024.07.01 / 2024-07-01 / 2024년 7월 1일 / 1 July 2024
RE_DATE_GENERIC = re.compile(
    r"(\d{4}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}[일]?\.)"  # YYYY.MM.DD. 등
    r"|((\d{4})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})[일]?)"  # YYYY.MM.DD / YYYY년 M월 D일 (공백/점/슬래시)
    r"|(\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일?)"          
    r"|(\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",
    re.IGNORECASE,
)

# 기간 라인 패턴(~, from-to, 부터-까지) – 체결일 후보에서 제외 판단에 사용
RE_RANGE1 = re.compile(
    r"(\d{4}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}[일]?)\s*[~\-–]\s*(\d{4}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}[일]?)"
)
RE_RANGE2 = re.compile(
    r"(\d{4}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}[일]?)\s*(부터|from)\s*(\d{4}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}[일]?)(까지|to)",
    re.IGNORECASE,
)

# 금액 + 통화(근접)
RE_AMOUNT_LINE = re.compile(
    r"(총액|계약\s*금액|Total\s*Amount|Contract\s*Value)?[^\n]{0,20}?"
    r"(₩|\$|€|¥|￦)?\s*([\d,]+(?:\.\d+)?)\s*(KRW|원|USD|달러|EUR|€|JPY|엔)?",
    re.IGNORECASE,
)

# 계약명 라벨 패턴
RE_NAME_LABEL = re.compile(
    r"계약명\s*[:：]\s*(.+)|"
    r"계약명\s+(.+)|"
    r"계약건명\s*[:：]\s*(.+)|"
    r"계약건명\s+(.+)|"
    r"Contract\s*Name\s*[:：]?\s*(.+)|"
    r"Agreement\s*Name\s*[:：]?\s*(.+)",
    re.IGNORECASE,
)

# 체결일 라벨 및 서명/날인 컨텍스트
RE_CON_LABEL = re.compile(
    r"계약\s*체결일|체결일|서명일|날인일|Executed\s+on|Signed\s+on",
    re.IGNORECASE,
)
RE_SIGN_CONTEXT = re.compile(
    r"서명|날인|갑|을|대리인|대표이사|법인|회사",
    re.IGNORECASE,
)

# 재계약/갱신 문서 키워드 + 선행계약 참조 힌트(추후 재계약 여부 추정에 활용)
RE_RENEWAL_WORD = re.compile(
    r"재계약|갱신계약|연장계약|Renewal|Extension|Addendum|Amendment",
    re.IGNORECASE,
)
RE_PRIOR_REF = re.compile(
    r"기체결|원계약|계약번호|Contract\s*No\.?|Agreement\s*ID|\d{4}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}"
)

# 자동 갱신 관련 정규식 패턴
RE_AUTO_TRIGGER = re.compile(
    r"("
    r"자동\s*갱신|"            # '자동갱신', '자동 갱신'
    r"자동\s*연장|"            # '자동연장', '자동 연장'
    r"자동으로\s*갱신|"        # '자동으로 갱신'
    r"자동으로\s*연장|"        # '자동으로 연장'
    r"자동적(?:으로)?|"        # '자동적', '자동적으로'
    r"Auto\s*Renewal|"        # 영어
    r"Auto\s*Extension"
    r")",
    re.IGNORECASE,
)
RE_AUTO_PERIOD = re.compile(
    r"(\d+)\s*("
    r"년(?:간|동안|씩)?|"        # 1년, 1년간, 1년동안, 1년씩
    r"개월(?:간|동안|씩)?|"      # 6개월, 6개월간, 6개월씩
    r"월(?:간|동안|씩)?|"        # 월 단위 표현 (필요시)
    r"일(?:간|동안|씩)?|"        # 30일, 30일간, 30일씩
    r"years?|months?|days?|year|month|day"
    r")",
    re.IGNORECASE,
)