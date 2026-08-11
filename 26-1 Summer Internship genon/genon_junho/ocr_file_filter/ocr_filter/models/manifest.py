"""모델 매니페스트 로딩 + 경로/명령 해석.

configs/models.yaml 을 읽어 각 모델의 로컬 경로, 서빙 엔드포인트, vLLM 기동 명령을
계산한다. 다운로드/서빙 코드는 모두 이 Model 객체를 통해 동작한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Model:
    key: str                       # target / external_a / external_b
    served_name: str               # OpenAI API 에 노출되는 이름
    port: int
    gpus: str                      # "0" 또는 "0,1" (CUDA_VISIBLE_DEVICES)
    host: str
    cache_dir: Path
    repo_id: str | None = None     # HF repo (다운로드 대상)
    revision: str = "main"
    local_path: str | None = None  # 로컬 체크포인트 (다운로드 안 함)
    vllm_args: list[str] = field(default_factory=list)
    default_vllm_args: list[str] = field(default_factory=list)
    lora_path: str | None = None   # LoRA 어댑터 경로 (base 모델에 붙여서 서빙)
    lora_rank: int = 0
    serve_script: str | None = None  # 있으면 vllm serve 대신 이 스크립트로 기동 (외부 venv 등)
    external: bool = False         # true면 이 서버가 직접 못 띄움 (원격/터널 필요) — status만 확인
    vllm_bin: str = "vllm"          # 다른 venv 의 vllm 바이너리를 쓸 때 절대경로 지정
    parallel_mode: str = "tensor"  # "tensor"(기본) | "pipeline" — gpus 개수만큼 TP 대신 PP 로

    @property
    def needs_download(self) -> bool:
        return self.local_path is None and self.repo_id is not None

    @property
    def download_dir(self) -> Path:
        """HF 모델을 받을 로컬 폴더 (repo_id 를 안전한 폴더명으로)."""
        safe = re.sub(r"[^A-Za-z0-9._-]", "__", self.repo_id or self.key)
        return self.cache_dir / safe

    @property
    def model_path(self) -> str:
        """vLLM 에 넘길 경로: 로컬 체크포인트면 그 경로, 아니면 다운로드 폴더."""
        if self.local_path:
            return self.local_path
        return str(self.download_dir)

    @property
    def endpoint(self) -> str:
        # 헬스체크/클라이언트용. 0.0.0.0 은 접속용으론 localhost 로 바꾼다.
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/v1"

    @property
    def tensor_parallel_size(self) -> int:
        return len([g for g in self.gpus.split(",") if g.strip() != ""]) or 1

    def vllm_command(self) -> list[str]:
        """`vllm serve ...` 인자 리스트 (CUDA_VISIBLE_DEVICES 는 env 로 별도 지정)."""
        # LoRA 서빙 시 base 모델은 별도 이름으로 숨기고, 클라이언트가 부르는
        # served_name 은 --lora-modules 쪽에 붙인다 (이름 충돌 방지).
        base_served_name = f"{self.served_name}-base" if self.lora_path else self.served_name
        cmd = [
            self.vllm_bin, "serve", self.model_path,
            "--served-model-name", base_served_name,
            "--host", self.host,
            "--port", str(self.port),
        ]
        n = self.tensor_parallel_size
        if n > 1:
            # PP(pipeline-parallel)는 head 수 제약 없이 레이어를 GPU 개수만큼 나눈다 —
            # attention head 수가 GPU 개수로 안 나눠떨어지는 모델(TP 불가)에 쓴다.
            # 예: Qwen3.5-122B-A10B(num_attention_heads=32, num_key_value_heads=2)는
            # 3으로 안 나눠져서 물리 GPU 3장으로는 TP=3 이 불가능 → PP=3 으로 우회
            # (48 레이어가 3으로 정확히 나눠짐, 2026-07-30 확인).
            flag = "--pipeline-parallel-size" if self.parallel_mode == "pipeline" else "--tensor-parallel-size"
            cmd += [flag, str(n)]
        cmd += list(self.default_vllm_args) + list(self.vllm_args)
        if self.lora_path:
            cmd += [
                "--enable-lora",
                "--max-lora-rank", str(self.lora_rank or 128),
                "--max-loras", "1",
                "--lora-modules", f"{self.served_name}={self.lora_path}",
            ]
        return cmd


@dataclass
class Manifest:
    cache_dir: Path
    log_dir: Path
    host: str
    models: dict[str, Model]

    def __iter__(self):
        return iter(self.models.values())

    def get(self, key: str) -> Model:
        if key not in self.models:
            raise KeyError(f"모델 키 '{key}' 없음. 사용 가능: {list(self.models)}")
        return self.models[key]


def load_manifest(path: str | Path) -> Manifest:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg.get("cache_dir", "./_models"))
    log_dir = Path(cfg.get("log_dir", "./_work/serve"))
    host = cfg.get("host", "0.0.0.0")
    default_args = cfg.get("default_vllm_args", [])

    models: dict[str, Model] = {}
    for key, m in cfg["models"].items():
        models[key] = Model(
            key=key,
            served_name=m["served_name"],
            port=int(m["port"]),
            gpus=str(m.get("gpus", "0")),
            host=host,
            cache_dir=cache_dir,
            repo_id=m.get("repo_id"),
            revision=m.get("revision", "main"),
            local_path=m.get("local_path"),
            vllm_args=list(m.get("vllm_args", [])),
            default_vllm_args=list(default_args),
            lora_path=m.get("lora_path"),
            lora_rank=int(m.get("lora_rank", 0)),
            serve_script=m.get("serve_script"),
            external=bool(m.get("external", False)),
            vllm_bin=m.get("vllm_bin", "vllm"),
            parallel_mode=m.get("parallel_mode", "tensor"),
        )
    return Manifest(cache_dir=cache_dir, log_dir=log_dir, host=host, models=models)
