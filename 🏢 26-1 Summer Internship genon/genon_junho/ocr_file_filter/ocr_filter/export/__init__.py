"""[7] export: 큐레이션된 라벨(`final_dataset.jsonl`)을 실제 SFT 학습 포맷(gt_html)으로 변환.

    python -m ocr_filter.cli export gt-html

`_vendor_chandra_convert.py`는 jhshin/tsr_test/train/vlm/data/build_chandra_dataset.py 를
읽기 전용으로 복사한 것(원본은 안 건드림) — 이 프로젝트를 통째로 다른 서버로 옮겨도 jhshin
저장소에 대한 런타임 의존 없이 그대로 재현되도록 vendoring 했다(ocr_filter/metrics/_vendor_cdm/
와 동일한 패턴).
"""

from ocr_filter.export.gt_html import (
    build_gt_html_dataset,
    elements_to_gt_html,
    normalize_bbox_to_1000,
)

__all__ = ["build_gt_html_dataset", "elements_to_gt_html", "normalize_bbox_to_1000"]
