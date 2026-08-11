"""metrics 단위 테스트: edit_distance / TEDS(-S).

    pytest tests/test_metrics.py
    python tests/test_metrics.py   # 실데이터 gt_html 로 자기 자신과 비교(=1.0) 시연
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.metrics import SCORERS, edit_distance_score, levenshtein, teds_score  # noqa: E402


def test_levenshtein_basic():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("kitten", "sitting") == 3


def test_edit_distance_score_range():
    assert edit_distance_score("", "") == 1.0
    assert edit_distance_score("hello", "hello") == 1.0
    assert edit_distance_score("hello", "world") < 1.0
    assert edit_distance_score("abc", "xyz") == 0.0  # 전부 치환, len 같음 → 0


def test_teds_identical_tables_is_one():
    html = '<table><tr><td colspan="2">a</td><td>b</td></tr></table>'
    assert teds_score(html, html) == 1.0
    assert teds_score(html, html, structure_only=True) == 1.0


def test_teds_penalizes_structure_change():
    gt = '<table><tr><td>a</td><td>b</td></tr></table>'
    pred_text_only_diff = '<table><tr><td>a</td><td>X</td></tr></table>'
    pred_structure_diff = '<table><tr><td colspan="2">a</td></tr></table>'

    s_text = teds_score(gt, pred_text_only_diff)
    s_struct_full = teds_score(gt, pred_structure_diff)
    assert 0.0 < s_text < 1.0
    assert 0.0 < s_struct_full < 1.0
    # 구조만 다른 표를 구조만 보는 TEDS-S 로 재면, 텍스트만 다른 경우의 TEDS-S(=1.0)보다 낮아야.
    assert teds_score(gt, pred_structure_diff, structure_only=True) < \
        teds_score(gt, pred_text_only_diff, structure_only=True)


def test_teds_completely_different_is_low():
    gt = '<table><tr><td>a</td><td>b</td><td>c</td></tr></table>'
    pred = '<div><p>completely unrelated</p></div>'
    assert teds_score(gt, pred) < 0.5


def test_cdm_empty_input_is_none():
    # CDM(수식) 전용 테스트는 tests/test_metrics_cdm.py 참고 -- 여기선 공통 SCORERS 시그니처
    # (빈 입력 → None)만 확인.
    assert SCORERS["cdm"]("", "x") is None


def _demo():
    # 실제 table_src_6902 gt_html 하나를 자기 자신과 비교 → TEDS=1.0 이어야 정상.
    import json

    real_jsonl = (
        "/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/"
        "_train_data/table_src_6902/data/train.jsonl"
    )
    try:
        with open(real_jsonl, encoding="utf-8") as f:
            item = json.loads(f.readline())
    except FileNotFoundError:
        print("실데이터 경로 없음 (다른 서버) — 데모 스킵")
        return
    html = item["gt_html"]
    print(f"gt_html 길이: {len(html)}")
    print(f"self TEDS   = {teds_score(html, html):.4f}")
    print(f"self TEDS-S = {teds_score(html, html, structure_only=True):.4f}")
    mangled = html.replace("<td", "<th", 1)
    print(f"태그 하나 바꾼 TEDS-S = {teds_score(html, mangled, structure_only=True):.4f}")


if __name__ == "__main__":
    _demo()
