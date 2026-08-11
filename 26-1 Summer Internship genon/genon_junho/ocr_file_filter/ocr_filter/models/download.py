"""HF 모델 다운로드 + 폐쇄망 이동용 번들링.

huggingface_hub 가 있으면 snapshot_download 를, 없으면 `hf`/`huggingface-cli` 를
서브프로세스로 호출한다. 타겟(로컬 체크포인트)은 건너뛴다.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

from ocr_filter.models.manifest import Manifest, Model


def _download_one(model: Model) -> Path:
    dest = model.download_dir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] {model.key}: {model.repo_id}@{model.revision} → {dest}")

    try:
        from huggingface_hub import snapshot_download  # type: ignore

        snapshot_download(
            repo_id=model.repo_id,
            revision=model.revision,
            local_dir=str(dest),
        )
        return dest
    except ImportError:
        pass  # CLI 폴백

    cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if not cli:
        raise RuntimeError(
            "huggingface_hub 미설치 & hf/huggingface-cli 없음. "
            "`pip install huggingface_hub` 후 다시 실행하세요."
        )
    if cli.endswith("hf"):
        cmd = [cli, "download", model.repo_id, "--revision", model.revision,
               "--local-dir", str(dest)]
    else:
        cmd = [cli, "download", model.repo_id, "--revision", model.revision,
               "--local-dir", str(dest)]
    subprocess.run(cmd, check=True)
    return dest


def download_all(manifest: Manifest, only: list[str] | None = None) -> None:
    """다운로드가 필요한 모델을 모두 받는다. only 로 특정 키만 받을 수 있다."""
    targets = [m for m in manifest if (only is None or m.key in only)]
    for m in targets:
        if not m.needs_download:
            print(f"[skip] {m.key}: 로컬 체크포인트 ({m.local_path}) — 다운로드 불필요")
            continue
        _download_one(m)
    print(f"[done] 다운로드 완료. 캐시 루트: {manifest.cache_dir}")


def bundle(manifest: Manifest, out_path: str | Path) -> Path:
    """cache_dir 전체를 tar.gz 로 묶는다 (폐쇄망 서버로 scp 이동용)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = manifest.cache_dir
    if not src.exists():
        raise FileNotFoundError(f"캐시 폴더가 없습니다: {src} (먼저 download 하세요)")
    print(f"[bundle] {src} → {out} (오래 걸릴 수 있음)")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(src, arcname=src.name)
    print(f"[done] 번들 생성: {out}  ({out.stat().st_size / 1e9:.1f} GB)")
    return out
