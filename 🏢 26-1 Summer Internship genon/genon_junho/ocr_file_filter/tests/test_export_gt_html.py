"""ocr_filter/export/gt_html.py 단위 테스트: 큐레이션 라벨 -> SFT 학습 포맷(gt_html) 변환.

    pytest tests/test_export_gt_html.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from ocr_filter.export.gt_html import (  # noqa: E402
    build_gt_html_dataset,
    elements_to_gt_html,
    normalize_bbox_to_1000,
)


def test_elements_to_gt_html_table_kept_raw():
    elements = [{"category": "Table", "text": "<table><tr><td>a</td></tr></table>",
                 "bbox": [10, 20, 500, 600]}]
    html = elements_to_gt_html(elements)
    assert html == (
        '<div data-bbox="10 20 500 600" data-label="Table">\n'
        "<table><tr><td>a</td></tr></table>\n</div>"
    )


def test_elements_to_gt_html_normalizes_nonstandard_category():
    # 2026-07-20: hardcase judge(397B) 출력에 섞인 비표준 레이블이 gt_html 에 그대로
    #새어나가면 안 된다 -- 여기서도 normalize_category 를 거쳐야 함.
    elements = [{"category": "reference_content", "text": "각주", "bbox": [0, 0, 10, 10]}]
    html = elements_to_gt_html(elements)
    assert 'data-label="Text"' in html
    assert 'data-label="reference_content"' not in html


def test_elements_to_gt_html_picture_content_is_always_empty():
    elements = [{"category": "Picture", "text": "이 텍스트는 무시됨", "bbox": [0, 0, 100, 100]}]
    html = elements_to_gt_html(elements)
    assert html == '<div data-bbox="0 0 100 100" data-label="Picture"></div>'


def test_elements_to_gt_html_wraps_text_in_p_and_escapes():
    elements = [{"category": "Text", "text": "a < b & c", "bbox": [0, 0, 10, 10]}]
    html = elements_to_gt_html(elements)
    assert "<p>a &lt; b &amp; c</p>" in html


def test_elements_to_gt_html_newline_becomes_br():
    elements = [{"category": "Text", "text": "줄1\n줄2", "bbox": [0, 0, 10, 10]}]
    html = elements_to_gt_html(elements)
    assert "<p>줄1<br/>줄2</p>" in html


def test_elements_to_gt_html_strips_only_leading_heading_marker():
    # 맨 앞 마크다운 제목(#)만 제거되고, bold(**)/리스트(-)는 리터럴로 그대로 남아야 한다
    # (build_chandra_dataset.py 원본 동작 그대로 -- 레퍼런스 학습셋과 포맷 일치 목적).
    elements = [{"category": "Section-header", "text": "## 제목 **굵게**", "bbox": [0, 0, 10, 10]}]
    html = elements_to_gt_html(elements)
    assert "<p>제목 **굵게**</p>" in html

    elements2 = [{"category": "Text", "text": "- 리스트 항목", "bbox": [0, 0, 10, 10]}]
    html2 = elements_to_gt_html(elements2)
    assert "<p>- 리스트 항목</p>" in html2


def test_elements_to_gt_html_joins_multiple_elements_in_order():
    elements = [
        {"category": "Title", "text": "제목", "bbox": [0, 0, 10, 10]},
        {"category": "Text", "text": "본문", "bbox": [0, 20, 10, 30]},
    ]
    html = elements_to_gt_html(elements)
    assert html.index('data-label="Title"') < html.index('data-label="Text"')


def test_normalize_bbox_to_1000_norm1000_rounds_and_clamps():
    assert normalize_bbox_to_1000([10.4, 20.6, 999.9, 1000.4], "norm1000") == [10, 21, 1000, 1000]


def test_normalize_bbox_to_1000_pixel_scales_by_image_size():
    # 이미지 200x100 픽셀에서 bbox [100,50,200,100] -> 0-1000 정규화하면 [500,500,1000,1000]
    assert normalize_bbox_to_1000([100, 50, 200, 100], "pixel", img_w=200, img_h=100) == [
        500, 500, 1000, 1000,
    ]


def test_normalize_bbox_to_1000_pixel_requires_dims():
    import pytest
    with pytest.raises(ValueError):
        normalize_bbox_to_1000([0, 0, 10, 10], "pixel")


def test_normalize_bbox_to_1000_unknown_coord_system_raises():
    import pytest
    with pytest.raises(ValueError):
        normalize_bbox_to_1000([0, 0, 10, 10], "weird")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_build_gt_html_dataset_end_to_end(tmp_path):
    img_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    unified = tmp_path / "unified.jsonl"
    _write_jsonl(unified, [
        {"id": "easy1", "image_path": str(img_path), "gt": [], "source_type": "layout"},
        {"id": "medium1", "image_path": str(img_path), "gt": [], "source_type": "layout"},
        {"id": "hard1", "image_path": str(img_path), "gt": [], "source_type": "layout"},
        {"id": "hard_unresolved", "image_path": str(img_path), "gt": [], "source_type": "layout"},
    ])

    final_dataset = tmp_path / "final_dataset.jsonl"
    _write_jsonl(final_dataset, [
        {"id": "easy1", "source_type": "layout", "tier": "Easy", "resolved": True,
         "label_source": "target_auto",
         "label": [{"category": "Title", "text": "제목", "bbox": [10, 10, 500, 60]}]},
        {"id": "medium1", "source_type": "layout", "tier": "Medium", "resolved": True,
         "label_source": "pseudo_label_dots",
         "label": [{"category": "Text", "text": "본문", "bbox": [100, 50, 200, 100]}]},
        {"id": "hard1", "source_type": "layout", "tier": "Hard", "resolved": True,
         "label_source": "hardcase_judge",
         "label": [{"category": "Text", "text": "본문2", "bbox": [0, 0, 100, 50]}]},
        {"id": "hard_unresolved", "source_type": "layout", "tier": "Hard", "resolved": False,
         "label_source": "unprocessed_hard_fallback", "label": []},
    ])

    out_path = tmp_path / "gt_html_dataset.jsonl"
    stats = build_gt_html_dataset(final_dataset, unified, out_path)

    assert stats["n_total"] == 4
    assert stats["n_written"] == 3  # resolved=False 인 hard_unresolved 는 제외
    assert stats["n_skipped_unresolved"] == 1
    assert stats["n_missing_image"] == 0
    assert stats["tier_counts"] == {"Easy": 1, "Medium": 1, "Hard": 1}

    with open(out_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 3
    for rec in lines:
        assert rec["prompt_style"] == "unified_layout"
        assert rec["bbox_scale"] == 1000
        assert rec["output_format"] == "html"
        assert rec["ocr_info"] == []
        assert 'data-bbox="' in rec["gt_html"]

    # Easy 는 이미 0-1000 정규화라 그대로(반올림만), Medium 은 픽셀(200x100 이미지) 기준 정규화됨.
    easy_html = lines[0]["gt_html"]
    assert 'data-bbox="10 10 500 60"' in easy_html
    medium_html = lines[1]["gt_html"]
    assert 'data-bbox="500 500 1000 1000"' in medium_html


def test_build_gt_html_dataset_tiers_filter_excludes_easy(tmp_path):
    # Easy는 target 모델 자기 자신의 출력이라 학습에 안 쓰고 Medium/Hard만 쓰는 경우
    # (2026-07-20, 사용자 확인) -- tiers={"Medium","Hard"} 로 Easy를 걸러낼 수 있어야 한다.
    img_path = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    unified = tmp_path / "unified.jsonl"
    _write_jsonl(unified, [
        {"id": "easy1", "image_path": str(img_path), "gt": [], "source_type": "layout"},
        {"id": "medium1", "image_path": str(img_path), "gt": [], "source_type": "layout"},
        {"id": "hard1", "image_path": str(img_path), "gt": [], "source_type": "layout"},
    ])

    final_dataset = tmp_path / "final_dataset.jsonl"
    _write_jsonl(final_dataset, [
        {"id": "easy1", "source_type": "layout", "tier": "Easy", "resolved": True,
         "label_source": "target_auto",
         "label": [{"category": "Title", "text": "제목", "bbox": [10, 10, 500, 60]}]},
        {"id": "medium1", "source_type": "layout", "tier": "Medium", "resolved": True,
         "label_source": "pseudo_label_dots",
         "label": [{"category": "Text", "text": "본문", "bbox": [100, 50, 200, 100]}]},
        {"id": "hard1", "source_type": "layout", "tier": "Hard", "resolved": True,
         "label_source": "hardcase_judge",
         "label": [{"category": "Text", "text": "본문2", "bbox": [0, 0, 100, 50]}]},
    ])

    out_path = tmp_path / "gt_html_dataset.jsonl"
    stats = build_gt_html_dataset(final_dataset, unified, out_path, tiers={"Medium", "Hard"})

    assert stats["n_written"] == 2
    assert stats["n_skipped_wrong_tier"] == 1
    assert stats["tier_counts"] == {"Medium": 1, "Hard": 1}


def test_build_gt_html_dataset_skips_records_missing_from_unified(tmp_path):
    unified = tmp_path / "unified.jsonl"
    _write_jsonl(unified, [])  # 비어있음 -> 전부 missing_image 로 스킵

    final_dataset = tmp_path / "final_dataset.jsonl"
    _write_jsonl(final_dataset, [
        {"id": "ghost", "source_type": "layout", "tier": "Easy", "resolved": True,
         "label_source": "target_auto", "label": []},
    ])

    out_path = tmp_path / "out.jsonl"
    stats = build_gt_html_dataset(final_dataset, unified, out_path)
    assert stats["n_written"] == 0
    assert stats["n_missing_image"] == 1


def test_build_gt_html_dataset_skips_pages_with_broken_lowercase_table(tmp_path):
    # 2026-07-20 실데이터 버그: category 가 정확히 소문자 "table" 인 요소는 예전 파서
    # 버그로 원본 <table> HTML 이 이미 복구 불가능하게 태그가 벗겨진 상태 -- 그런 페이지는
    # export 단계에서 통째로 스킵해야 한다(가짜 표 HTML 을 학습에 넣지 않기 위해).
    img_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(img_path)

    unified = tmp_path / "unified.jsonl"
    _write_jsonl(unified, [
        {"id": "broken1", "image_path": str(img_path), "gt": [], "source_type": "layout"},
        {"id": "ok1", "image_path": str(img_path), "gt": [], "source_type": "layout"},
    ])

    final_dataset = tmp_path / "final_dataset.jsonl"
    _write_jsonl(final_dataset, [
        {"id": "broken1", "source_type": "layout", "tier": "Hard", "resolved": True,
         "label_source": "hardcase_judge",
         "label": [{"category": "table", "text": "깨진 표 텍스트", "bbox": [0, 0, 10, 10]}]},
        {"id": "ok1", "source_type": "layout", "tier": "Hard", "resolved": True,
         "label_source": "hardcase_judge",
         "label": [{"category": "Table", "text": "<table><tr><td>a</td></tr></table>",
                    "bbox": [0, 0, 10, 10]}]},
    ])

    out_path = tmp_path / "out.jsonl"
    stats = build_gt_html_dataset(final_dataset, unified, out_path)
    assert stats["n_written"] == 1
    assert stats["n_skipped_broken_table"] == 1

    with open(out_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 1
    assert "<table><tr><td>a</td></tr></table>" in lines[0]["gt_html"]
