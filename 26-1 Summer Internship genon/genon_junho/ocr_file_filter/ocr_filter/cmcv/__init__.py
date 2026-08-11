"""[2]/[4] CMCV: target/dots.ocr/paddle 교차검증.

    normalize   → 3모델 각자 출력 형식을 [{"category","text"}] 공통형으로
    client      → vLLM OpenAI 호환 엔드포인트 호출 (모델별 프롬프트 자동 선택)
    run_cmcv    → unified.jsonl 순회하며 3모델 추론 + 채점 + 티어 배정
"""

from ocr_filter.cmcv.normalize import normalize, plain_text, table_htmls
from ocr_filter.cmcv.run import run_cmcv

__all__ = ["normalize", "plain_text", "table_htmls", "run_cmcv"]
