"""cdm.py(수식 CDM 스코어러) 단위 테스트: 가짜 점수 방지 정책(렌더/설치 실패시 None) 위주.
TeX Live 가 이 프로젝트 안에 설치돼 있으면(scripts/setup_cdm_texlive.sh) 실제 렌더링까지
도는 통합 테스트도 같이 돈다 -- 미설치 상태면 그 테스트들은 스킵된다.

    pytest tests/test_metrics_cdm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

import ocr_filter.metrics.cdm as cdm_mod  # noqa: E402
from ocr_filter.metrics.cdm import cdm_score  # noqa: E402

_TEXLIVE_INSTALLED = cdm_mod._PDFLATEX.is_file()


def test_cdm_score_none_for_empty_input():
    assert cdm_score("", "x^2") is None
    assert cdm_score("x^2", "") is None
    assert cdm_score("", "") is None


@pytest.mark.skipif(_TEXLIVE_INSTALLED, reason="TeX Live 설치돼 있으면 미설치 시뮬레이션 케이스는 무의미")
def test_cdm_score_none_when_texlive_not_installed(monkeypatch):
    monkeypatch.setattr(cdm_mod, "_env_ready", False)
    monkeypatch.setattr(cdm_mod, "_PDFLATEX", Path("/nonexistent/pdflatex"))
    assert cdm_score("x^2", "x^2") is None


@pytest.mark.skipif(not _TEXLIVE_INSTALLED, reason="scripts/setup_cdm_texlive.sh 먼저 실행해야 함")
def test_cdm_score_identical_formula_is_high():
    score = cdm_score("x^2 + y^2 = z^2", "x^2 + y^2 = z^2")
    assert score is not None
    assert score > 0.9


@pytest.mark.skipif(not _TEXLIVE_INSTALLED, reason="scripts/setup_cdm_texlive.sh 먼저 실행해야 함")
def test_cdm_score_different_formula_is_low():
    score = cdm_score("x^2 + y^2 = z^2", "alpha plus beta plus gamma delta")
    assert score is not None
    assert score < 0.5
