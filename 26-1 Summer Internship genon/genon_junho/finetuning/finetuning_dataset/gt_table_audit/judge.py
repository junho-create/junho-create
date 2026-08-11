#!/usr/bin/env python3
"""GenOS serving 776 (Qwen3.5-397B-A17B-FP8-Instruct) judge 클라이언트.

labeler 의 `VLMClient` 를 상속하되 `call` 만 갈아끼운다 — 원본은 3회 재시도가
하드코딩(`vlm_client.py:69`)이라 5,030건에서 너무 느려진다. 여기서는 **호출 1회**만
하고, 실패하면 `JudgeError` 를 올려 그 페이지를 사람 검수 큐로 보낸다.

`_to_base64`(PNG 재인코딩 + data URI), `call_with_images` 의 메시지 조립,
빈 `choices` 감지 로직은 상속해서 그대로 쓴다.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv

from genos.doc_parser.labeler.core.vlm_client import VLMClient

# genos_key 가 들어 있는 .env (사용자 지정)
GENOS_ENV_PATH = Path("/home/jhyeo/finetuning/eval_318/tsr_test_eval/.env")

# "low" | "medium" | "high" | None(모델 기본) | "off"(thinking 끔 — 판별력 붕괴, 쓰지 말 것)
#
# None(모델 기본)을 쓴다. effort=low 는 단발 프로브에서 출력 1,794 토큰으로 싸 보였지만
# 실제 감사 호출(이미지 2장) 26회 평균은 5,228 토큰으로 오히려 full(4,278)보다 많았다.
# 절감이 없는데 검출률만 80%→71% 로 떨어지므로 굳이 쓸 이유가 없다.
DEFAULT_REASONING_EFFORT = None

JUDGE_CONFIG = {
    # serving 776 은 OpenAI 호환 게이트웨이이고, model 이름이 리터럴 "model" 이다.
    "base_url": "https://genos.genon.ai/api/gateway/rep/serving/776/v1",
    "model": "model",
    "api_key_env": "GENOS_KEY",
    # 776 은 thinking 모델이다. **thinking 을 켠 채로 둔다.**
    #
    # 끄면 토큰은 크게 아끼지만(완성 토큰 209→2) 판별력이 무너진다 — 판별력 테스트에서
    # 훼손 검출률이 80%(thinking on) → 38%(off) 로 떨어졌고, 특히 "행 하나 통째 삭제"를
    # 10번 중 6번 놓쳤다. `ocr_filter/hardcase/prompts.py:68-77` 이 경고한 그 함정이다.
    #
    # 참고로 이 게이트웨이에서 thinking 을 끄는 방법은 `extra_body.reasoning.enabled`
    # 뿐이다. chat_template_kwargs.enable_thinking 과 "/no_think" 는 무시된다(실측).
    #
    # thinking 이 켜져 있으면 reasoning_content 가 max_tokens 를 함께 쓰므로 예산을
    # 넉넉히 잡아야 한다. 모자라면 content 가 빈 채로 돌아온다.
    #
    # thinking 길이는 `reasoning.effort` 로 조절된다(실측, 표 1개 페이지 기준):
    #   effort=low     출력 1,794 토큰 / 24.0초 / thinking 3,477자
    #   default(full)  출력 2,037~4,278 토큰 / 57초 / thinking 4,121자
    #   effort=medium  출력 2,920 토큰 / 89.7초 / thinking 6,883자  ← full 보다 길다
    #   enabled=false  출력 ~700 토큰 / 빠름  ← 하지만 검출률 38% 로 붕괴
    "max_tokens": 24000,
    "temperature": 0.2,
    "reasoning_effort": DEFAULT_REASONING_EFFORT,
    "extra_params": {},
}


class JudgeError(RuntimeError):
    """judge 호출/파싱 실패. 재시도하지 않고 사람 검수 큐로 보낸다."""


def load_genos_key() -> str:
    """tsr_test_eval/.env 의 genos_key 를 GENOS_KEY 환경변수로 올린다."""
    if os.environ.get("GENOS_KEY"):
        return os.environ["GENOS_KEY"]
    load_dotenv(GENOS_ENV_PATH)
    key = os.environ.get("genos_key") or os.environ.get("GENOS_KEY")
    if not key:
        raise RuntimeError(f"{GENOS_ENV_PATH} 에서 genos_key 를 못 찾았다")
    os.environ["GENOS_KEY"] = key
    return key


class SingleShotVLMClient(VLMClient):
    """재시도 없는 VLMClient. timeout 도 config 로 뺀다(원본은 120초 하드코딩).

    토큰 사용량을 누적해 둔다 — thinking 이 켜져 있어서 완성 토큰이 얼마나 나가는지가
    max_tokens 예산과 전량 실행 시간 추정의 근거가 된다.
    """

    def __init__(self, cfg: dict, timeout: int = 180):
        super().__init__(cfg)
        self.timeout = timeout
        # 부모의 `reasoning_enabled` 는 {"enabled": bool} 만 보내서 길이 조절이 안 된다.
        # 이 게이트웨이는 {"effort": "low"} 를 받으므로 직접 넣는다.
        effort = cfg.get("reasoning_effort")
        if effort:
            extra_body = dict(self.extra_params.get("extra_body") or {})
            extra_body["reasoning"] = ({"enabled": False} if effort == "off"
                                       else {"effort": effort})
            self.extra_params = {**self.extra_params, "extra_body": extra_body}
        self._usage_lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "max_completion": 0, "truncated": 0}

    def _record(self, response) -> None:
        u = getattr(response, "usage", None)
        if u is None:
            return
        with self._usage_lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += u.prompt_tokens or 0
            self.usage["completion_tokens"] += u.completion_tokens or 0
            self.usage["max_completion"] = max(self.usage["max_completion"],
                                               u.completion_tokens or 0)
            if getattr(response.choices[0], "finish_reason", "") == "length":
                self.usage["truncated"] += 1

    def call(self, messages: list, max_tokens: int = None, temperature: float = None) -> str:
        request_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "timeout": self.timeout,
            **self.extra_params,
        }
        eff = max_tokens if max_tokens is not None else self.max_tokens
        if eff is not None and eff != -1:
            request_kwargs["max_tokens"] = eff

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as e:  # 타임아웃·API 오류·네트워크 — 전부 1회로 끝낸다
            raise JudgeError(f"api_error: {type(e).__name__}: {e}") from e

        choices = getattr(response, "choices", None)
        if not choices or getattr(choices[0], "message", None) is None:
            err = getattr(response, "error", None)
            raise JudgeError(f"empty_response: {err or 'no choices'}")
        self._record(response)
        content = choices[0].message.content or ""
        if not content.strip():
            # thinking 이 max_tokens 를 다 먹고 content 가 안 나온 경우가 대부분이다.
            fr = getattr(choices[0], "finish_reason", "?")
            raise JudgeError(f"empty_response: blank content (finish_reason={fr})")
        return content


def make_judge(timeout: int = 180, max_tokens: int | None = None,
               reasoning_effort: str | None = None) -> SingleShotVLMClient:
    load_genos_key()
    cfg = dict(JUDGE_CONFIG)
    if max_tokens is not None:
        cfg["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        cfg["reasoning_effort"] = None if reasoning_effort == "default" else reasoning_effort
    return SingleShotVLMClient(cfg, timeout=timeout)


# ── 응답 파싱 ────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.S)

METRIC_KEYS = (
    "table_coverage",
    "basic_structure",
    "complex_structure",
    "cell_correspondence",
    "text_accuracy",
    "auxiliary_visual_fidelity",
)


def extract_json(raw: str) -> dict:
    """raw → ```json 펜스 → 최외곽 {...} 3단 파싱.

    `EvaluatorBase._extract_json` 과 같은 전략인데, 그쪽은 인스턴스 메서드라
    평가자 객체 없이 못 쓰므로 여기 옮겨 놓았다.
    """
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise JudgeError("parse_error: JSON 을 못 찾음")


def validate_verdict(verdict: dict, n_tables: int) -> dict:
    """스키마를 검증하고 정규화한다. 어긋나면 JudgeError — 재시도는 하지 않는다."""
    if not isinstance(verdict, dict):
        raise JudgeError("parse_error: 최상위가 object 가 아님")

    tables = verdict.get("tables")
    if not isinstance(tables, list):
        raise JudgeError("parse_error: tables 가 배열이 아님")
    if len(tables) != n_tables:
        raise JudgeError(f"table_count_mismatch: {len(tables)} != {n_tables}")

    norm_tables = []
    for pos, t in enumerate(tables, start=1):
        if not isinstance(t, dict):
            raise JudgeError(f"parse_error: tables[{pos-1}] 가 object 가 아님")
        metrics = t.get("metrics")
        if not isinstance(metrics, dict):
            raise JudgeError(f"parse_error: tables[{pos-1}].metrics 없음")
        norm_metrics = {}
        for k in METRIC_KEYS:
            m = metrics.get(k)
            if not isinstance(m, dict) or "score" not in m:
                raise JudgeError(f"parse_error: metrics.{k} 누락 (표 {pos})")
            try:
                score = int(m["score"])
            except (TypeError, ValueError):
                raise JudgeError(f"parse_error: metrics.{k}.score 가 정수가 아님 (표 {pos})")
            if not 1 <= score <= 5:
                raise JudgeError(f"parse_error: metrics.{k}.score={score} 범위 밖 (표 {pos})")
            et = m.get("error_types") or []
            norm_metrics[k] = {
                "score": score,
                "severity": m.get("severity") or "NONE",
                "error_types": [str(x) for x in et] if isinstance(et, list) else [],
                "reason": str(m.get("reason") or ""),
            }
        # index 는 모델이 틀리게 붙이는 경우가 있어 배열 위치를 신뢰한다.
        norm_tables.append({
            "index": pos,
            "metrics": norm_metrics,
            "min_score": min(v["score"] for v in norm_metrics.values()),
            "summary": str(t.get("summary") or ""),
        })

    unlabeled = verdict.get("unlabeled_tables") or []
    if not isinstance(unlabeled, list):
        unlabeled = []
    overall = verdict.get("overall") or {}
    try:
        confidence = float(overall.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "schema_version": "gt_table_audit_v1",
        "tables": norm_tables,
        "unlabeled_tables": [u for u in unlabeled if isinstance(u, dict)],
        "overall": {
            "min_table_score": min(t["min_score"] for t in norm_tables),
            "confidence": max(0.0, min(1.0, confidence)),
            "summary": str(overall.get("summary") or ""),
        },
    }
