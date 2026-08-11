"""PaddleOCR-VL 1.6 **실제 파이프라인**(레이아웃검출 PP-DocLayoutV3 + vLLM 인식) 배치 호출.

`ocr_filter.cmcv.client.call_model` 로 external_b(paddle) 을 부르면 텍스트 프롬프트 없이
이미지만 줘도 인식 텍스트만 나오고 레이아웃 박스가 없다 (그 0.9B 모델은 원래 크롭당 인식만
하는 컴포넌트라서 — `ocr_filter/cmcv/normalize.py` 주석 참고). report 갤러리에서 paddle 패널에도
bbox+category 를 보여주려면, 레이아웃검출까지 포함한 진짜 `PaddleOCRVL` 파이썬 파이프라인을
통째로 호출해야 한다 (`jhyeo/paddleOCRVL1.6/scripts/run_ocr.py` 와 같은 방식).

이 파이프라인은 별도 venv(venv_paddle) 에서 동작하고 초기화가 무거워서
(레이아웃검출 모델 로딩 등), 이미지 여러 장을 **한 서브프로세스에서 배치로** 처리한다.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 2026-07-30 경로 정정. 기존 값(NHN B200 서버 기준)은 이 머신에 전부 존재하지 않아서
# 서브프로세스가 즉시 실패했다 — 그 결과 external_b(paddle) 요소가 전량 빈 리스트가 되고
# **모든 페이지가 Hard 로 떨어진다**(빈 리스트 vs 실제 출력이라 어떤 쌍도 agree_min 을
# 못 넘김). CMCV 를 돌리기 전 반드시 확인해야 하는 지점.
#
# - venv_paddle/bin/python 은 /home/shkim/.local/share/uv/... 를 가리키는 죽은 심링크라
#   못 쓴다. 시스템 python3 에 paddleocr 3.7.0 + paddle 3.3.0 이 설치돼 있어 그걸 쓴다.
# - PaddleOCR-VL 인식 서버 포트: `models.yaml` 의 external_b 를 따른다(2026-07-30 재기동
#   이후 표준 :8080/GPU3). 임시로 :8118 에 떠 있던 시절도 있었는데, 필요하면
#   OCR_FILTER_PADDLE_VLLM_URL 로 덮어쓸 수 있게 열어둔다.
import os

PADDLE_ROOT = Path("/home/jhyeo/paddleOCRVL1.6")
VENV_PADDLE_PYTHON = Path(os.environ.get("OCR_FILTER_PADDLE_PYTHON", "/usr/bin/python3"))
VLLM_SERVER_URL = os.environ.get("OCR_FILTER_PADDLE_VLLM_URL", "http://127.0.0.1:8080/v1")

# PP-DocLayoutV3 라벨 → 우리 캐노니컬 카테고리 (다른 두 모델과 같은 색으로 렌더링하려고).
# 매핑에 없는 라벨은 원래 이름 그대로 씀 (DEFAULT_COLOR 로 렌더링).
_LABEL_MAP = {
    "doc_title": "Title",
    "paragraph_title": "Section-header",
    "text": "Text",
    "table": "Table",
    "table_title": "Caption",
    "figure_title": "Caption",
    "header": "Page-header",
    "footer": "Page-footer",
    "header_image": "Picture",
    "footer_image": "Picture",
    "image": "Picture",
    "figure": "Picture",
    "formula": "Formula",
    "formula_number": "Formula",
    "number": "List-item",
    "reference": "Footnote",
}

_WORKER_SCRIPT = r"""
import json, sys

in_path, out_path, server_url = sys.argv[1], sys.argv[2], sys.argv[3]
image_paths = json.load(open(in_path, encoding="utf-8"))

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="vllm-server",
                        vl_rec_server_url=server_url, device="gpu")

results = {}
for img in image_paths:
    elements = []
    try:
        for res in pipeline.predict(img):
            for block in res["parsing_res_list"]:
                # PaddleOCRVLBlock 객체 — dict 아님, attribute 로 접근 (label/content/bbox)
                elements.append({
                    "category": getattr(block, "label", "") or "",
                    "text": getattr(block, "content", "") or "",
                    "bbox": getattr(block, "bbox", None),
                })
    except Exception:
        elements = []
    results[img] = elements

json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
"""


def _run_paddle_chunk(image_paths: list[str], timeout: float) -> dict[str, list[dict]]:
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.json"
        out_path = Path(tmp) / "out.json"
        script_path = Path(tmp) / "worker.py"
        in_path.write_text(json.dumps(image_paths, ensure_ascii=False), encoding="utf-8")
        script_path.write_text(_WORKER_SCRIPT, encoding="utf-8")

        proc = subprocess.run(
            [str(VENV_PADDLE_PYTHON), str(script_path), str(in_path), str(out_path), VLLM_SERVER_URL],
            cwd=str(PADDLE_ROOT), timeout=timeout, capture_output=True, text=True,
        )
        if not out_path.exists():
            raise RuntimeError(f"paddle 파이프라인 청크 호출 실패:\n{proc.stdout}\n{proc.stderr}")
        return json.loads(out_path.read_text(encoding="utf-8"))


def get_paddle_elements_batch(
    image_paths: list[str], timeout: float = 900.0, chunk_size: int = 200,
) -> dict[str, list[dict]]:
    """{image_path: [{"category","text","bbox"(pixel)}]}. bbox 는 gt/dots.ocr 과 같은
    원본 픽셀 좌표라 draw_boxes(..., coord_system="pixel") 그대로 쓰면 된다.

    청크(chunk_size 장씩) 단위로 서브프로세스를 나눠 호출한다 — 극단적으로 손상되거나
    거대한 이미지 때문에 서브프로세스가 세그폴트로 죽어도, 그 청크만 결과 없음으로
    처리하고 나머지 청크는 계속 진행한다. (예전엔 전체를 서브프로세스 1번에 몰아서 호출해서,
    무인 장시간 실행 중 마지막 청크 하나가 죽으면 그 전까지의 처리 결과가 통째로 날아가는
    구조였다.)"""
    chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

    # 청크를 몇 개까지 동시에 돌릴지. 기본 1 = 종전과 완전히 같은 순차 동작.
    # 클라이언트(crop/인코딩/블록 조립)가 CPU 단일 스레드라 한 프로세스로는 vLLM 을 25%
    # 밖에 못 채운다 — `paddlevl16_infer/run_all.sh` 가 같은 이유로 샤드 6개를 쓴다.
    # 배치가 클 때(코퍼스 9,100장 = 46청크) 순차로 돌리면 청크마다 모델 로드(~80s)까지
    # 겹쳐 3.8시간이 걸린다. 실측 기준값이 필요한 곳에서만 env 로 올려 쓸 것.
    par = max(1, int(os.environ.get("OCR_FILTER_PADDLE_PARALLEL", "1")))

    raw: dict[str, list[dict]] = {}

    def _one(idx_chunk):
        i, chunk = idx_chunk
        try:
            return _run_paddle_chunk(chunk, timeout)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"[paddle_layout] 청크 {i}~{i + len(chunk)} 실패, "
                  f"해당 페이지는 paddle 결과 없음으로 처리: {e}")
            return {img: [] for img in chunk}

    indexed = [(i * chunk_size, c) for i, c in enumerate(chunks)]
    if par == 1:
        for item in indexed:
            raw.update(_one(item))
    else:
        with ThreadPoolExecutor(max_workers=par) as pool:
            for res in pool.map(_one, indexed):
                raw.update(res)

    return {
        img: [{"category": _LABEL_MAP.get(e["category"], e["category"]),
               "text": e["text"], "bbox": e["bbox"]} for e in elements]
        for img, elements in raw.items()
    }
