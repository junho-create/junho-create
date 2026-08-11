"""report/gallery.py 단위 테스트: gt_score 가 None 인 레코드(GT 없는 신규 원본 PDF 등)에서
`_row_html`/`build_html` 이 죽지 않아야 함 (2026-07-15 실측: TypeError:
unsupported format string passed to NoneType.__format__ 로 갤러리 생성 전체가 죽는 버그 재현·확인).

    pytest tests/test_report_gallery.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from ocr_filter.io.schema import Record, write_jsonl  # noqa: E402
from ocr_filter.report.gallery import _row_html, build_html  # noqa: E402


def _els(text: str = "hi") -> list[dict]:
    return [{"category": "Text", "text": text, "bbox": [0, 0, 100, 20]}]


def test_row_html_handles_none_gt_score():
    row = {
        "id": "x", "gt_score": None, "agreement_score": 0.9,
        "n_elements": {"target": 1, "external_a": 1, "external_b": 1},
        "paddle_text": "hi",
        "panels": {"gt": "data:image/jpeg;base64,", "target": "data:image/jpeg;base64,",
                   "external_a": "data:image/jpeg;base64,", "external_b": "data:image/jpeg;base64,"},
    }
    html = _row_html(row)  # 예전엔 여기서 TypeError
    assert "gt_score=N/A" in html
    assert "agreement=0.900" in html


def test_build_html_end_to_end_with_gt_none_record():
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        img_path = tmp / "page.png"
        Image.new("RGB", (100, 100), "white").save(img_path)

        record = Record(id="p0", image_path=str(img_path), gt=None,
                         source_type="layout", meta={})
        unified_path = tmp / "unified.jsonl"
        write_jsonl([record], unified_path)

        cmcv_row = {
            "id": "p0", "source_type": "layout", "tier": "Easy",
            "gt_score": None, "teds_score": None, "agreement_score": 0.95,
            "pairwise_scores": {"target_dots": 0.95, "target_paddle": 0.9, "dots_paddle": 0.9},
            "pseudo_label": None,
            "elements": {"target": _els(), "external_a": _els(), "external_b": _els()},
            "n_elements": {"target": 1, "external_a": 1, "external_b": 1},
            "errors": None,
        }
        cmcv_results_path = tmp / "cmcv_results.jsonl"
        cmcv_results_path.write_text(json.dumps(cmcv_row) + "\n", encoding="utf-8")

        html = build_html(unified_path, cmcv_results_path, per_tier=5)
        assert "gt_score=N/A" in html
        assert "p0" in html
