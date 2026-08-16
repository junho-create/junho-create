"""CDM(수식 유사도) — GT/pred LaTeX을 각각 렌더링해 문자 단위로 매칭한 F1 스코어.

구현은 `ocr_filter/metrics/_vendor_cdm/`에 vendoring 되어 있다(원본: hckim/upstream_OmniDocBench
의 `src/metrics/cdm/`, 읽기 전용으로 복사만 함 — 원본은 건드리지 않음). 렌더링에 필요한 TeX
Live/ImageMagick은 이 프로젝트 자체 `_texlive/`에 설치해서 쓴다(`scripts/setup_cdm_texlive.sh`).
전부 이 파일(`Path(__file__)`) 기준 상대경로로 찾으므로, 프로젝트를 통째로 다른 서버로 옮겨도
동일하게(스크립트만 재실행하면) 동작한다 — 다른 사용자 디렉토리나 서버 고정 절대경로에는
의존하지 않는다.

TeX Live가 아직 설치 안 됐거나 렌더링이 실패하면(폰트 누락 등) 가짜 점수를 내지 않기 위해
None을 반환한다 — 원래 더미 구현의 원칙을 그대로 유지.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ocr_filter/metrics/cdm.py -> ocr_file_filter/
_TEXLIVE_DIR = _PROJECT_ROOT / "_texlive"
_TEXLIVE_BIN = _TEXLIVE_DIR / "2025" / "bin" / "x86_64-linux"
_MAGICK_SHIM_BIN = _TEXLIVE_DIR / "bin"  # magick -> convert 위임 shim (scripts/setup_cdm_texlive.sh 3단계)
_IM_POLICY_DIR = _TEXLIVE_DIR / "im_policy"
_PDFLATEX = _TEXLIVE_BIN / "pdflatex"
_TMP_BASE = _PROJECT_ROOT / "_work" / "cdm_tmp"

_env_ready = False


def _system_texlive() -> dict | None:
    """시스템에 이미 설치된 TeX Live/ImageMagick으로 CDM을 돌릴 수 있으면 그 경로들을 반환.

    `_texlive/`(setup_cdm_texlive.sh) 는 폐쇄망/이식성을 위한 프로젝트 자체 설치본인데,
    이미 시스템에 pdflatex+kpsewhich+magick 과 필수 패키지(amsmath/standalone)가 있는
    환경에서는 수 GB를 다시 받을 이유가 없다. 2026-07-30 이 서버에서 확인:
    /usr/bin/pdflatex, /usr/bin/kpsewhich, /usr/local/bin/magick, amsmath.sty,
    standalone.cls 전부 존재. 이 폴백이 없으면 cdm_score 가 항상 None 을 반환해서
    **Formula 슬롯이 통째로 죽는다** (신규 배치의 1/6이 ArxivFormula 라 영향이 크다).
    """
    from shutil import which
    pdflatex, kpsewhich = which("pdflatex"), which("kpsewhich")
    magick = which("magick") or which("convert")
    if not (pdflatex and kpsewhich and magick):
        return None
    return {"pdflatex": pdflatex, "kpsewhich": kpsewhich,
            "bin": str(Path(pdflatex).parent), "magick_bin": str(Path(magick).parent)}


def _ensure_env() -> bool:
    """TeX Live/magick 을 PATH 등에 등록. 프로젝트 자체 설치본(_texlive/)을 우선 쓰고,
    없으면 시스템 설치본으로 폴백한다. 둘 다 없으면 False(→ cdm_score 가 None)."""
    global _env_ready
    if _env_ready:
        return True
    if not _PDFLATEX.is_file():
        sys_tl = _system_texlive()
        if sys_tl is None:
            return False
        os.environ["CDM_PDFLATEX"] = sys_tl["pdflatex"]
        os.environ["CDM_KPSEWHICH"] = sys_tl["kpsewhich"]
        os.environ["CDM_TEXLIVE_BIN"] = sys_tl["bin"]
        os.environ.setdefault("CDM_CJK_FONT", "mj")
        # 시스템 설치본은 IM policy 를 프로젝트가 덮어쓰지 않는다(정책 파일이 없음) —
        # PDF 변환이 policy.xml 로 막힌 배포판이면 여기서 렌더가 실패하고, cdm_score 는
        # 예외를 삼켜 None 을 내므로 가짜 점수가 나올 위험은 없다.
        _env_ready = True
        logger.info("CDM: 시스템 TeX Live 사용 (%s)", sys_tl["pdflatex"])
        return True

    os.environ["CDM_TEXLIVE_BIN"] = str(_TEXLIVE_BIN)
    os.environ["CDM_TEXLIVE_ROOT"] = str(_TEXLIVE_DIR / "2025")
    os.environ["CDM_PDFLATEX"] = str(_PDFLATEX)
    os.environ["CDM_KPSEWHICH"] = str(_TEXLIVE_BIN / "kpsewhich")
    os.environ.setdefault("CDM_CJK_FONT", "mj")
    os.environ["MAGICK_CONFIGURE_PATH"] = str(_IM_POLICY_DIR)

    path_items = [str(_TEXLIVE_BIN), str(_MAGICK_SHIM_BIN)]
    for item in os.environ.get("PATH", "").split(os.pathsep):
        if item and item not in path_items:
            path_items.append(item)
    os.environ["PATH"] = os.pathsep.join(path_items)

    _env_ready = True
    return True


# 렌더가 구조적으로 안 되는 환경(TeX 는 있는데 CDM 토큰 추출이 실패하는 경우 등)에서
# 매 호출마다 pdflatex+magick 을 돌리면 콜당 수 초를 버린다 — 수식 페이지가 많은 배치
# (ArxivFormula 1만 장)에서는 이것만으로 실행 시간이 몇 시간씩 늘어난다. 연속 실패가
# 임계치를 넘으면 이 프로세스에서는 CDM 을 포기하고 즉시 None 을 돌려준다.
_CDM_FAIL_LIMIT = 8
_consecutive_failures = 0
_cdm_disabled = False


def _note_failure() -> None:
    global _consecutive_failures, _cdm_disabled
    _consecutive_failures += 1
    if _consecutive_failures >= _CDM_FAIL_LIMIT and not _cdm_disabled:
        _cdm_disabled = True
        logger.warning(
            "CDM: 연속 %d회 실패 — 이 프로세스에서는 CDM 을 비활성화한다(호출당 수 초 낭비 방지). "
            "수식 채점이 필요하면 렌더 환경을 먼저 고칠 것.", _CDM_FAIL_LIMIT)


def _note_success() -> None:
    global _consecutive_failures
    _consecutive_failures = 0


def cdm_score(gt: str, pred: str) -> float | None:
    if not gt or not pred:
        return None
    if _cdm_disabled:
        return None
    if not _ensure_env():
        logger.warning(
            "CDM: TeX Live 미설치(%s 없음) -- scripts/setup_cdm_texlive.sh 먼저 실행 필요, 이번 호출은 스킵",
            _PDFLATEX,
        )
        return None

    from ocr_filter.metrics._vendor_cdm.cdm import cdm_metrics as _cdm_metrics_impl

    _TMP_BASE.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="cdm_", dir=str(_TMP_BASE)) as tmp_dir:
            metrics = _cdm_metrics_impl(gt, pred, tmp_dir=tmp_dir)
    except Exception:
        logger.exception("CDM 계산 실패 (gt=%r, pred=%r) -- 스킵", gt[:80], pred[:80])
        _note_failure()
        return None

    # cdm_metrics()는 렌더링이 통째로 실패해도(pdflatex/magick 깨짐 등) 예외를 던지지 않고
    # gt_tokens=0 또는 pred_tokens=0 인 채로 F1_score=0.0 을 조용히 돌려준다 — 이건 "내용이
    # 진짜 다 틀림"과 구분이 안 되는 가짜 0점이라, 렌더 자체가 실패한 경우(토큰 하나도 못
    # 뽑음)는 따로 걸러서 None 으로 스킵한다. 실제로 렌더는 됐는데 매칭이 하나도 안 된
    # 경우(F1=0.0, gt/pred_tokens>0)는 진짜 0점이므로 그대로 반환.
    if metrics.get("gt_tokens", 0) == 0 or metrics.get("pred_tokens", 0) == 0:
        logger.warning(
            "CDM: 렌더링 실패로 추정(gt_tokens=%s, pred_tokens=%s) -- 가짜 0점 방지, 스킵. "
            "TeX Live/magick 설치 상태 확인 필요(scripts/setup_cdm_texlive.sh).",
            metrics.get("gt_tokens"), metrics.get("pred_tokens"),
        )
        _note_failure()
        return None
    _note_success()
    return float(metrics.get("F1_score", 0.0))
