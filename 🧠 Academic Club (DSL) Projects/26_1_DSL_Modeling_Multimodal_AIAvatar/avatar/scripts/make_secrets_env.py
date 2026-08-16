#!/usr/bin/env python3
"""
.env.local에서 주석·빈 줄을 제거한 secrets.env를 생성합니다.
배포 시: lk agent deploy --secrets-file secrets.env

실행: avatar 폴더에서 uv run python scripts/make_secrets_env.py
"""
from pathlib import Path

AVATAR_DIR = Path(__file__).resolve().parent.parent
ENV_LOCAL = AVATAR_DIR / ".env.local"
SECRETS_ENV = AVATAR_DIR / "secrets.env"


def main() -> None:
    if not ENV_LOCAL.exists():
        print(f".env.local not found: {ENV_LOCAL}")
        return
    lines_out = []
    for raw in ENV_LOCAL.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # 값 끝의 '# 주석' 제거 (값 안에 # 이 있으면 그대로 둠)
        value = value.strip()
        if " #" in value:
            value = value.split(" #")[0].strip()
        if not key:
            continue
        lines_out.append(f"{key}={value}")
    SECRETS_ENV.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines_out)} entries to {SECRETS_ENV}")
    print("Keys:", ", ".join(l.split("=")[0] for l in lines_out))


if __name__ == "__main__":
    main()
