"""vLLM 서버 3개 기동/상태/종료.

각 모델을 백그라운드 프로세스로 띄우고 log_dir 에 <key>.log / <key>.pid 를 남긴다.
status 는 /v1/models 헬스체크, stop 은 pid 로 종료.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import requests

from ocr_filter.models.manifest import Manifest, Model


def _pid_file(manifest: Manifest, model: Model) -> Path:
    return manifest.log_dir / f"{model.key}.pid"


def _log_file(manifest: Manifest, model: Model) -> Path:
    return manifest.log_dir / f"{model.key}.log"


def _compile_cache_env() -> dict[str, str]:
    """/tmp 가 noexec 로 마운트돼 있어 triton/torchinductor 컴파일 캐시가
    mmap PROT_EXEC 에 실패한다 ("failed to map segment from shared object").
    exec 가능한 홈 디렉터리 밑으로 캐시 경로를 돌린다."""
    cache_root = Path.home() / ".cache"
    return {
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
    }


def command_for(model: Model) -> str:
    """dry-run 표시용: 실제로 실행될 명령 문자열."""
    if model.external:
        return "(외부/원격 모델 — 이 서버에서 기동 불가. 터널/VPN 연결 후 status 로 확인)"
    if model.serve_script:
        return f"bash {model.serve_script} {model.gpus} {model.port}"
    env = f"CUDA_VISIBLE_DEVICES={model.gpus} "
    return env + " ".join(model.vllm_command())


def serve_all(manifest: Manifest, only: list[str] | None = None,
              dry_run: bool = False) -> None:
    manifest.log_dir.mkdir(parents=True, exist_ok=True)
    targets = [m for m in manifest if (only is None or m.key in only)]

    for m in targets:
        if m.external:
            print(f"[{m.key}] 외부/원격 모델 — 스킵 (터널/VPN 연결 후 status 로 확인)")
            continue

        if dry_run:
            print(f"[{m.key}] {command_for(m)}")
            continue

        # 이미 떠 있으면 스킵
        pidf = _pid_file(manifest, m)
        if pidf.exists() and _pid_alive(int(pidf.read_text().strip() or 0)):
            print(f"[{m.key}] 이미 실행 중 (pid {pidf.read_text().strip()}) — 스킵")
            continue

        logf = _log_file(manifest, m)
        with open(logf, "w") as log:
            if m.serve_script:
                env = dict(os.environ, **_compile_cache_env())
                proc = subprocess.Popen(
                    ["bash", m.serve_script, m.gpus, str(m.port)],
                    stdout=log, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True,
                )
            else:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=m.gpus, **_compile_cache_env())
                proc = subprocess.Popen(
                    m.vllm_command(),
                    stdout=log, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True,  # 부모 죽어도 유지
                )
        pidf.write_text(str(proc.pid))
        print(f"[{m.key}] 기동 pid={proc.pid} port={m.port} log={logf}")

    if not dry_run:
        print("\n헬스체크: `python -m ocr_filter.cli models status` "
              "(모델 로딩에 수분 소요될 수 있음)")


def status_all(manifest: Manifest) -> None:
    for m in manifest:
        up, detail = _probe(m)
        mark = "UP  " if up else "DOWN"
        print(f"[{mark}] {m.key:<11} {m.served_name:<32} {m.endpoint}  {detail}")


def stop_all(manifest: Manifest, only: list[str] | None = None) -> None:
    targets = [m for m in manifest if (only is None or m.key in only)]
    for m in targets:
        if m.external:
            print(f"[{m.key}] 외부/원격 모델 — 스킵")
            continue
        pidf = _pid_file(manifest, m)
        if not pidf.exists():
            print(f"[{m.key}] pid 파일 없음 — 스킵")
            continue
        pid = int(pidf.read_text().strip() or 0)
        if pid and _pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                print(f"[{m.key}] 종료 pid={pid}")
            except ProcessLookupError:
                print(f"[{m.key}] 이미 종료됨 pid={pid}")
        pidf.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False           # 그런 프로세스 없음
    except PermissionError:
        return True            # 존재하지만 소유자가 달라 시그널 못 보냄 → 살아있음
    except OSError:
        return False


def _probe(model: Model, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        r = requests.get(f"{model.endpoint}/models", timeout=timeout)
        if r.status_code == 200:
            return True, "ok"
        return False, f"http {r.status_code}"
    except requests.RequestException as e:
        return False, type(e).__name__
