# 모델 재설치 가이드

이 프로젝트는 `_models/` 디렉토리에 대용량 모델 파일을 사용합니다.  
아래 모델들은 이전 시 포함되지 않으므로, 새 서버에서 직접 다운로드해야 합니다.

---

## 1. Qwen3.5-9B (19GB)

**출처**: Hugging Face — `Qwen/Qwen3.5-9B`  
**라이선스**: Apache 2.0

```bash
# huggingface-cli 사용 (권장)
pip install huggingface_hub

huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir ./_models/Qwen3.5-9B \
  --local-dir-use-symlinks False
```

또는 Python으로:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3.5-9B",
    local_dir="./_models/Qwen3.5-9B"
)
```

---

## 2. rednote-hilab/dots.ocr (5.5GB)

**출처**: Hugging Face — `rednote-hilab/dots.ocr`  
**라이선스**: MIT

```bash
huggingface-cli download rednote-hilab/dots.ocr \
  --local-dir ./_models/rednote-hilab__dots.ocr \
  --local-dir-use-symlinks False
```

또는 Python으로:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="rednote-hilab/dots.ocr",
    local_dir="./_models/rednote-hilab__dots.ocr"
)
```

---

## 3. chandra_all_e18align_20260623_lora (153MB)

> **⚠️ 이 모델은 별도로 이전(복사)됩니다.**  
> `_models/chandra_all_e18align_20260623_lora/` 디렉토리가 tar 아카이브에 포함되어 있습니다.

---

## 다운로드 전 참고사항

- HuggingFace 접근이 제한된 환경이라면 `HF_ENDPOINT` 환경변수로 미러 서버를 지정할 수 있습니다:
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- 네트워크 속도에 따라 Qwen3.5-9B는 수십 분 이상 소요될 수 있습니다.
- `venv`는 `requirements.txt`로 재설치합니다:
  ```bash
  python -m venv venv
  source venv/bin/activate   # Windows: venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
- TexLive는 패키지 매니저로 재설치합니다:
  ```bash
  sudo apt install texlive-full   # Ubuntu/Debian 계열
  ```
