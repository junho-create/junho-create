# -*- coding: utf-8 -*-

def make_empty_result() -> dict:
    fields = [
        "TEMP_KEY","CNT_NAME","CNT_CON_DATE","CNT_ST_DATE","CNT_END_DATE",
        "CNT_CONCLUDED","CNT_RENEWAL","CNT_AMT","CNT_AMT_CRY",
        "CNT_AUTO_RNW_TERM_AMT","CNT_AUTO_RNW_TERM_UNIT",
    ]
    return {f: {"value": None, "confidence": None, "evidence": None} for f in fields}