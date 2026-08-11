"""[6] Hard case 정제 파이프라인의 결과를 labeler UI 로 연동하는 어댑터.

_work/hardcase_prelabel.jsonl 파일(2단계 prelabel 결과)을 읽어서
labeler 의 LayoutTask(evaluate 등 포함) 파이프라인을 그대로 실행합니다.
이 스크립트는 `convert` 단계만 가로채서(미리 생성된 bbox 사용) 불필요한 추론을 피하고,
나머지 렌더링, 평가, UI 리포팅은 labeler 기존 로직을 100% 재활용합니다.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# labeler 디렉토리가 sys.path 에 없으면 추가
_LABELER_ROOT = Path(__file__).resolve().parent.parent.parent / "labeler"
if str(_LABELER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LABELER_ROOT))

from genos.doc_parser.labeler.core import pipeline
from genos.doc_parser.labeler.core.types import ConvertResult, InputItem
from genos.doc_parser.labeler.layout.task import LayoutTask

logger = logging.getLogger(__name__)


class HardcaseLabelerTask(LayoutTask):
    """Hardcase 결과를 labeler Layout 파이프라인에 태우기 위한 어댑터 태스크."""
    name = "hardcase_layout"

    def __init__(self, jsonl_path: Path):
        super().__init__()
        self.jsonl_path = jsonl_path

    def collect_inputs(self, input_dir: Path, config: dict) -> list[InputItem]:
        items = []
        if not self.jsonl_path.exists():
            logger.warning("File not found: %s", self.jsonl_path)
            return items

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                image_path = Path(record["image_path"])
                record_id = record["id"]
                display_name = image_path.name
                
                # prelabel_elements가 없으면 빈 리스트
                prelabel_elements = record.get("prelabel_elements", [])
                
                items.append(
                    InputItem(
                        source_path=image_path,
                        output_key=record_id,
                        display_name=display_name,
                        relative_path=Path(record_id),
                        metadata={
                            "prelabel_elements": prelabel_elements,
                            "stem_page": image_path.stem,
                        }
                    )
                )
        return items

    def convert(
        self, item: InputItem, attempt_dir: Path, prev_feedback: str | None = None
    ) -> ConvertResult:
        """vLLM 호출 대신 _work/hardcase_prelabel.jsonl 에 이미 있는 결과를 리턴."""
        elements = item.metadata.get("prelabel_elements", [])
        payload = json.dumps({"elements": elements}, ensure_ascii=False)
        artifact_name = f"{item.metadata['stem_page']}.convert.json"
        artifact = attempt_dir / artifact_name
        artifact.write_text(payload, encoding="utf-8")
        return ConvertResult(content=payload, artifact_path=artifact)

    def _ensure_page_image(self, item: InputItem) -> None:
        """pdf_loader 우회 — 원본이 이미 이미지임."""
        if not Path(item.source_path).exists():
            raise FileNotFoundError(f"Image not found: {item.source_path}")


def run_export(
    jsonl_path: str | Path,
    output_dir: str | Path,
    results_dir: str | Path,
    config: dict,
) -> None:
    jsonl_path = Path(jsonl_path)
    # labeler 프레임워크가 evaluate 까지 하도록 pipeline_mode 를 full 로 강제
    config.setdefault("processing", {})
    config["processing"]["pipeline_mode"] = "full"
    
    # input_dir 은 dummy (jsonl 에서 직접 읽으므로)
    config["input_dir"] = str(jsonl_path.parent)
    config["output_dir"] = str(output_dir)
    config["results_dir"] = str(results_dir)
    
    # hardcase 태스크 생성 및 파이프라인 실행
    task = HardcaseLabelerTask(jsonl_path)
    fresh_results = pipeline.run(task, config)

    # 렌더링된 결과로 UI HTML 굽기
    from genos.doc_parser.labeler.layout.report import generate_review_html, generate_summary_report
    
    intermediate_json = Path(results_dir) / "intermediate.json"
    if intermediate_json.exists():
        try:
            from genos.doc_parser.labeler.core.types import ProcessResult
            raw = json.loads(intermediate_json.read_text("utf-8"))
            
            # _load_cached_result를 위한 우회 대신 ProcessResult.from_dict가 실패하면
            # json dict에 있는 값을 기반으로 직접 로드하도록 할 수 있지만
            # labeler가 intermediate.json 을 생성했으니 그냥 가져온다.
            # 하지만 ProcessResult 에는 item, status 등이 있다.
            
            # python 3.10+ dataclasses
            # 안전하게 fresh_results 와 캐시 결과를 섞어서 저장하는 함수를 사용하는 편이 좋으나,
            # generate_review_html 에는 ProcessResult 객체 리스트가 필요.
            
            # fresh_results 만 넘겨도 어차피 1번 실행하면 다 처리됨.
            if fresh_results:
                generate_summary_report(fresh_results, config)
                generate_review_html(fresh_results, config, Path(results_dir) / "review.html")
                logger.info("Successfully generated review UI at %s/review.html", results_dir)
        except Exception as e:
            logger.error("Failed to generate review UI: %s", e)
