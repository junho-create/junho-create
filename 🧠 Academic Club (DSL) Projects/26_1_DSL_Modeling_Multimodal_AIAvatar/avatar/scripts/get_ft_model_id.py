#!/usr/bin/env python3
"""
파인튜닝 job ID(ftjob-...)로 완료 후 발급된 모델 ID를 조회합니다.
이 모델 ID를 .env.local에 OPENAI_LLM_MODEL=... 로 설정하면 에이전트가 파인튜닝 모델을 사용합니다.

실행 (avatar 폴더에서):
  uv run python scripts/get_ft_model_id.py ftjob-hEcrLzi4EvegNoHOP3Tv1214

OPENAI_API_KEY는 .env.local에 있으면 dotenv으로 로드됩니다.
"""
import os
import sys
from pathlib import Path

# avatar 루트에서 .env.local 로드
AVATAR_DIR = Path(__file__).resolve().parent.parent
env_local = AVATAR_DIR / ".env.local"
if env_local.exists():
    for line in env_local.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if " #" in value:
            value = value.split(" #")[0].strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/get_ft_model_id.py <job_id>", file=sys.stderr)
        print("Example: python scripts/get_ft_model_id.py ftjob-hEcrLzi4EvegNoHOP3Tv1214", file=sys.stderr)
        sys.exit(1)
    job_id = sys.argv[1].strip()
    if not job_id.startswith("ftjob-"):
        print("Job ID는 보통 ftjob-... 형식입니다.", file=sys.stderr)

    try:
        from openai import OpenAI
    except ImportError:
        print("openai 패키지 필요: uv add openai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY가 없습니다. .env.local 또는 환경 변수를 설정하세요.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    job = client.fine_tuning.jobs.retrieve(job_id)
    model_id = getattr(job, "fine_tuned_model", None) or (job.model if hasattr(job, "model") else None)
    status = getattr(job, "status", "unknown")

    if model_id:
        print(model_id)
        print("", file=sys.stderr)
        print(".env.local에 다음 한 줄 추가 후 에이전트 실행:", file=sys.stderr)
        print(f"  OPENAI_LLM_MODEL={model_id}", file=sys.stderr)
    else:
        print(f"Job 상태: {status}. 완료(succeeded) 후에 fine_tuned_model이 발급됩니다.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
