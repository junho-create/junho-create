# -*- coding: utf-8 -*-
"""스코어 가중치/임계치 등 공통 설정."""

WEIGHTS = {
    "source": {  # 후보 출처별 가중치
        "header": 0.45,
        "sentence": 0.35,
        "table": 0.35,
        "title": 0.20,
    },
    "format_ok": 0.20,   # 날짜/금액/단위 포맷 파서 성공 가점
    "logic_ok": 0.20,    # 시작≤종료 등 교차 논리 가점(필드에 따라 적용)
    "currency_pair": 0.20,  # 금액+통화 동시 검출 가점
}

THRESHOLDS = {
    "accept": 0.75,  # 이 점수 미만이면 검수 필요로 플래그
}