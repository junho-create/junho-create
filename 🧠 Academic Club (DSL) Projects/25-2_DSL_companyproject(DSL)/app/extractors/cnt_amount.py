# -*- coding: utf-8 -*-
from typing import List
import re 
from app.extractors.base import FieldExtractor, Candidate, Evidence
from app.ingest.loader_pdf import TextBlock
from app.utils.regexes import RE_AMOUNT_LINE


INCLUDE_VERBS = ("으로 한다","로 한다","정한다","지급한다","산정한다","합계는","총액은","로 산정")
EXCLUDE_VERBS = ("부담한다","분담한다","공제한다","납부한다","환급한다","예치한다")

GOOD_TOKENS = (
    "계약금액", "계약 금액", "총액", "총 금액", "총 계약금액", "대가", "대금",
    "Contract Value", "Total Amount", "Contract Amount", "Agreement Amount",
)

BAD_TOKENS = (
    "인지세", "인지대", "인지세액", "지체상금", "지연배상금", "위약금", "벌금", "가산금",
    "공제", "차감", "수수료", "이자", "이율", "예치금",
    "보증금", "선급금",
)

def _strip_leading_index(line: str) -> str:
    # 예: "5. 계약금액..." / "5) 계약금액..." / "5 ) 계약금액..."
    return re.sub(r"^\s*\d+\s*[.)]\s*", "", line)


def _select_amount_match(line: str):
    """
    RE_AMOUNT_LINE 으로 찾은 여러 금액 후보 중에서
    - 통화 기호(₩, ￦, $, € 등)나 통화 코드(KRW, 원, 달러 등)가 붙은 것 우선
    - 그게 없으면 자릿수가 가장 큰 숫자(큰 금액) 우선
    """
    cleaned = _strip_leading_index(line)
    matches = list(RE_AMOUNT_LINE.finditer(cleaned))
    if not matches:
        return None

    # 1순위: symbol 또는 ccy 그룹이 채워진 매치
    for m in matches:
        sym = m.group(2)
        num = m.group(3)
        ccy = m.group(4)
        if sym or ccy:
            return m

    # 2순위: 숫자 길이가 가장 긴 매치 (콤마 제거 후)
    def _num_len(m):
        n = m.group(3) or ""
        return len(n.replace(",", ""))

    return max(matches, key=_num_len)


class AmountExtractor(FieldExtractor):
    def extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        cands: List[Candidate] = []
        if not blocks:
            return cands

        good = GOOD_TOKENS
        bad  = BAD_TOKENS

        for b in blocks:
            line = b.text
            m = _select_amount_match(line)
            if not m:
                continue

            symbol, number, ccy = m.group(2), m.group(3), m.group(4)
            if not number:
                continue

            # 의미 필터(문맥)
            has_good = any(tok in line for tok in good)
            has_bad = any(tok in line for tok in bad)
            incl_v = any(v in line for v in INCLUDE_VERBS)
            excl_v = any(v in line for v in EXCLUDE_VERBS)
            has_ccy = bool(ccy or symbol) or any(tok in line for tok in ("KRW", "USD", "EUR", "JPY", "원", "₩", "$", "€", "¥", "엔"))

            # 나쁜 토큰 + 제외 동사 → 스킵
            if has_bad and (excl_v or not has_good):
                continue
            
            # 좋은 신호 하나도 없으면 스킵(무의미 숫자 방지)
            if not (has_good or incl_v or has_ccy):
                continue

            # 숫자는 콤마 제거해서 정규화
            normalized_number = number.replace(",", "")

            ev = Evidence(page=b.page, lines=[b.line_no], snippet=line)
            features = {
                "kw": 1 if has_good else 0,
                "verb": 1 if incl_v else 0,
                "ccy": 1 if has_ccy else 0,
            }
            cands.append(Candidate("CNT_AMT", normalized_number, "header", ev, features))

            if ccy or symbol:
                cands.append(
                    Candidate(
                        "CNT_AMT_CRY",
                        (ccy or symbol),
                        "header",
                        ev,
                        {"currency": 1},
                    )
                )
        return cands