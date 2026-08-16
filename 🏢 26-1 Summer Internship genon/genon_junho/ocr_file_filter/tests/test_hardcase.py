"""hardcase 단위 테스트: judge/refine/prelabel 로직을 mock VLM 응답으로 검증
(실제 vLLM 서버 없이 — `call_fn` 주입).

    pytest tests/test_hardcase.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_filter.hardcase.parse import parse_elements_response, parse_judge_verdict  # noqa: E402
from ocr_filter.hardcase.pipeline import (  # noqa: E402
    generate_label,
    judge_and_refine,
    judge_layout_once,
    judge_once,
    prelabel_once,
    refine_once,
    revise_label,
)
from ocr_filter.hardcase.prompts import build_merge_chandra_prompt  # noqa: E402

PASS_VERDICT = json.dumps({"pass": True, "overall_score": 5, "issues": []})
FAIL_VERDICT = json.dumps({
    "pass": False, "overall_score": 2,
    "issues": [{"element_index": 0, "category": "Text", "issue": "bbox_misaligned",
                "description": "box shifted right"}],
})
REFINED_ELEMENTS = json.dumps([{"bbox": [10, 10, 100, 50], "category": "Text", "text": "fixed"}])


def _tiny_png(path: Path) -> None:
    from PIL import Image
    Image.new("RGB", (50, 50), "white").save(path)


def test_parse_judge_verdict_handles_code_fences():
    raw = f"```json\n{PASS_VERDICT}\n```"
    v = parse_judge_verdict(raw)
    assert v["pass"] is True
    assert v["overall_score"] == 5
    assert v["parse_error"] is False


def test_parse_judge_verdict_falls_back_conservatively_on_garbage():
    v = parse_judge_verdict("이건 그냥 잡담")
    assert v["pass"] is False
    assert v["parse_error"] is True


def test_parse_elements_response_extracts_array():
    els = parse_elements_response(f"```json\n{REFINED_ELEMENTS}\n```")
    assert els == [{"category": "Text", "text": "fixed", "bbox": [10, 10, 100, 50]}]


def test_parse_elements_response_empty_on_garbage():
    assert parse_elements_response("no json here") == []


def test_judge_once_calls_with_two_images_and_parses():
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page.png"
        _tiny_png(img)
        calls = []

        def fake_call(cfg, images, prompt):
            calls.append((cfg, len(images), prompt))
            return PASS_VERDICT

        verdict = judge_once(str(img), [{"category": "Text", "bbox": [1, 1, 2, 2], "text": "a"}],
                              {"name": "judge"}, call_fn=fake_call)
        assert verdict["pass"] is True
        assert len(calls) == 1
        assert calls[0][1] == 2  # 원본 + 렌더링 오버레이 2장


def test_refine_once_keeps_previous_on_parse_failure():
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page.png"
        _tiny_png(img)
        original = [{"category": "Text", "bbox": [1, 1, 2, 2], "text": "orig"}]

        result = refine_once(str(img), original, [], {"name": "refine"},
                              call_fn=lambda *a, **k: "이상한 응답")
        assert result == original  # 파싱 실패 → 퇴행 방지, 이전 유지


def test_judge_and_refine_resolves_when_judge_passes_first_round():
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page.png"
        _tiny_png(img)
        result = judge_and_refine(
            str(img), [{"category": "Text", "bbox": [1, 1, 2, 2], "text": "a"}],
            {"name": "judge"}, max_rounds=3, call_fn=lambda *a, **k: PASS_VERDICT,
        )
        assert result["resolved"] is True
        assert result["rounds"] == 1


def test_judge_and_refine_exhausts_rounds_when_always_fails():
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page.png"
        _tiny_png(img)

        # judge 는 항상 FAIL, refine 은 항상 REFINED_ELEMENTS 응답
        calls = {"judge": 0, "refine": 0}

        def fake_call(cfg, images, prompt):
            if len(images) == 2:  # judge (원본+렌더링)
                calls["judge"] += 1
                return FAIL_VERDICT
            calls["refine"] += 1
            return REFINED_ELEMENTS

        result = judge_and_refine(
            str(img), [{"category": "Text", "bbox": [1, 1, 2, 2], "text": "a"}],
            {"name": "judge"}, max_rounds=3, call_fn=fake_call,
        )
        assert result["resolved"] is False
        assert result["rounds"] == 3
        assert calls["judge"] == 3
        assert calls["refine"] == 3
        assert result["final_elements"] == [{"category": "Text", "text": "fixed",
                                              "bbox": [10, 10, 100, 50]}]


def test_prelabel_once_falls_back_to_input_on_parse_failure():
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page.png"
        _tiny_png(img)
        original = [{"category": "Text", "bbox": [1, 1, 2, 2], "text": "orig"}]
        candidate = prelabel_once(str(img), original, [], {"name": "prelabel"},
                                   call_fn=lambda *a, **k: "이상한 응답")
        assert candidate == original


# ── 단일 콜 라벨 생성 (비판적 병합 맥락 + Chandra 지시문을 한 콜에) ────────────────────────
CHANDRA_REPLY = (
    "<think>표가 하나 있고 푸터가 있다</think>\n"
    '<div data-bbox="100 200 500 800" data-label="Table">'
    "<table><tr><td>A</td></tr></table></div>\n"
    '<div data-bbox="0 900 1000 950" data-label="Page-footer"><p>- 11 -</p></div>'
)
REF_A = [{"category": "Table", "bbox": [200, 200, 1000, 800], "text": "A 표"}]
REF_B = [{"category": "Table", "bbox": [210, 205, 990, 795], "text": "B 표"}]


def test_generate_label_converts_chandra_1000_bbox_to_pixel():
    """Chandra 출력은 0-1000 정규화 — 나머지 파이프라인(judge 렌더링/labeler)이 쓰는
    원본 픽셀좌표로 환산돼야 한다."""
    els, _ = generate_label("/x/page.png", REF_A, REF_B, (2000, 1000), {"name": "gen"},
                            call_fn=lambda *a, **k: CHANDRA_REPLY)
    assert [round(v) for v in els[0]["bbox"]] == [200, 200, 1000, 800]   # x*2000/1000, y*1000/1000
    assert [round(v) for v in els[1]["bbox"]] == [0, 900, 2000, 950]


def test_generate_label_strips_thinking_and_keeps_table_html():
    els, _ = generate_label("/x/page.png", REF_A, REF_B, (100, 100), {"name": "gen"},
                            call_fn=lambda *a, **k: CHANDRA_REPLY)
    assert [e["category"] for e in els] == ["Table", "Page-footer"]
    assert els[0]["text"].startswith("<table>")      # 표는 HTML 통째로 보존
    assert "표가 하나" not in json.dumps(els, ensure_ascii=False)  # 사고 블록은 제거


def test_generate_label_uses_chandra_system_and_disables_thinking():
    seen = {}

    def fake(cfg, images, prompt, system=None, enable_thinking=None):
        seen.update(images=images, prompt=prompt, system=system, think=enable_thinking)
        return CHANDRA_REPLY

    generate_label("/x/page.png", REF_A, REF_B, (100, 100), {"name": "gen"}, call_fn=fake)
    assert seen["images"] == ["/x/page.png"]          # 원본 1장만 (judge 처럼 오버레이 안 씀)
    assert "layout analysis model" in seen["system"]  # Chandra system 프롬프트
    assert seen["think"] is False                     # 사고 truncation/비용 방지


def test_merge_chandra_prompt_normalizes_refs_and_drops_bboxless():
    """A/B 는 원본 픽셀좌표 — 출력 좌표계(0-1000)와 맞춰 보여줘야 모델이 안 헷갈린다."""
    prompt = build_merge_chandra_prompt(
        REF_A, [{"category": "Text", "bbox": None, "text": "구조 없음"}], 2000, 2000,
    )
    assert "bbox=[100, 100, 500, 400]" in prompt   # A: 픽셀 → 0-1000 정규화
    assert "구조 없음" not in prompt                # bbox 없는 레퍼런스는 스캐폴딩 가치 없어 제외
    assert "data-bbox" in prompt                   # Chandra 지시문이 같은 콜에 융합됨


def test_run_judge_single_call_then_gate_and_resumes(monkeypatch):
    """Hard 만 골라 라벨 생성 1콜 + 판정 1콜(되먹임 없음), 이미 처리한 id 는 건너뛴다."""
    import ocr_filter.hardcase.run as run_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = tmp / "page.png"
        _tiny_png(img)
        unified = tmp / "unified.jsonl"
        unified.write_text("".join(
            json.dumps({"id": f"p{i}", "image_path": str(img), "source_type": "layout",
                        "meta": {}}, ensure_ascii=False) + "\n" for i in range(3)
        ), encoding="utf-8")
        cmcv = tmp / "cmcv.jsonl"
        cmcv.write_text("".join(
            json.dumps({"id": f"p{i}", "tier": tier,
                        "elements": {"external_a": REF_A, "external_b": REF_B}}) + "\n"
            for i, tier in enumerate(["Hard", "Easy", "Hard"])
        ), encoding="utf-8")

        n_gen = {"n": 0}

        def fake_gen(image_path, a_els, b_els, img_size, gen_cfg, **kw):
            n_gen["n"] += 1
            return [{"category": "Text", "text": "t", "bbox": [1, 2, 3, 4]}], "raw"

        monkeypatch.setattr(run_mod, "generate_label", fake_gen)
        monkeypatch.setattr(run_mod, "judge_layout_once",
                            lambda *a, **k: {"resolved": True, "overall_score": 5,
                                             "element_issues": []})

        out = tmp / "judge.jsonl"
        stats = run_mod.run_judge(unified, cmcv, out, {}, {"name": "j"}, workers=1)
        assert stats["n_done"] == 2 and n_gen["n"] == 2   # Easy 는 제외, Hard 2건만
        rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
        assert {r["id"] for r in rows} == {"p0", "p2"}
        assert rows[0]["resolved"] is True                # PASS → 자동 확정

        # 재개: 이미 done 이라 추가 생성 콜 없음
        stats2 = run_mod.run_judge(unified, cmcv, out, {}, {"name": "j"}, workers=1)
        assert stats2["n_total"] == 0 and n_gen["n"] == 2

        # prelabel 통과: 모델 콜 없이 export 가 읽는 필드로 옮겨준다
        pre = tmp / "prelabel.jsonl"
        run_mod.run_prelabel(out, pre)
        prows = [json.loads(x) for x in pre.read_text(encoding="utf-8").splitlines()]
        assert prows[0]["prelabel_elements"] == rows[0]["final_elements"]


def test_judge_prompt_gets_normalized_bbox_not_raw_pixels():
    """오버레이는 max_side 로 축소되는데 프롬프트에 원본 픽셀 좌표를 주면 judge 가 축소본에
    대조해 멀쩡한 박스를 '오른쪽으로 밀렸다'고 오판한다 → 0-1000 정규화해서 줘야 한다."""
    seen = {}

    def fake(cfg, images, prompt, **k):
        seen["prompt"] = prompt
        return PASS_VERDICT

    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page.png"
        from PIL import Image as _Image
        _Image.new("RGB", (2000, 1000), "white").save(img)   # 1000 초과 → 축소 발생
        els = [{"category": "Text", "bbox": [1000, 500, 2000, 1000], "text": "t"}]
        judge_once(str(img), els, {"name": "j"}, call_fn=fake)

    assert "bbox=[500, 500, 1000, 1000]" in seen["prompt"]   # 픽셀 → 0-1000 정규화
    assert "bbox=[1000, 500, 2000, 1000]" not in seen["prompt"]
    assert "NORMALIZED to a 0-1000" in seen["prompt"]        # 프롬프트가 좌표계를 명시


def test_parse_chandra_label_accepts_class_and_any_attr_order():
    """라벨 생성 모델(397B)은 파인튜닝이 안 돼서 data-label 대신 class 를 쓰고 속성 순서도
    뒤집어 낸다 — 그래도 받아줘야 한다(엄격판은 0개를 뱉어 빈 라벨이 됨)."""
    from ocr_filter.hardcase.parse import parse_chandra_label

    raw = ('```html\n<div class="Section-header" data-bbox="106 51 240 64">머리말</div>\n'
           '<div data-bbox="0 100 1000 900" data-label="Table"><table><tr><td>1</td></tr>'
           "</table></div>\n```")
    els = parse_chandra_label(raw)
    assert [e["category"] for e in els] == ["Section-header", "Table"]
    assert els[0]["bbox"] == [106, 51, 240, 64]
    assert els[1]["text"].startswith("<table>")


def test_parse_chandra_label_normalizes_category_and_preserves_lowercase_table_html():
    # 2026-07-20 실데이터 버그 재현: 397B judge 모델이 표준 11종을 안 지키고 소문자
    # "table" 을 낸 경우, category 정규화 전에 "Table" 과 문자열 비교하면 표로 인식이
    # 안 돼서 <table> 원본 HTML 이 태그 벗겨진 텍스트로 깨졌다(1,494 페이지에서 실측).
    from ocr_filter.hardcase.parse import parse_chandra_label

    raw = ('<div data-label="table" data-bbox="0 100 1000 900">'
           "<table><tr><td>1</td></tr></table></div>\n"
           '<div data-label="reference_content" data-bbox="0 0 10 10">각주</div>')
    els = parse_chandra_label(raw)
    assert els[0]["category"] == "Table"
    assert els[0]["text"] == "<table><tr><td>1</td></tr></table>"  # 태그 안 벗겨짐
    assert els[1]["category"] == "Text"  # 비표준 자유형식 레이블도 표준으로 매핑


def test_run_judge_guards_empty_label_and_skips_judge(monkeypatch):
    """빈 라벨은 judge 에 보내면 PASS(5점) 자동확정되므로, 아예 판정 전에 끊어야 한다."""
    import ocr_filter.hardcase.run as run_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = tmp / "page.png"
        _tiny_png(img)
        unified = tmp / "unified.jsonl"
        unified.write_text(json.dumps(
            {"id": "p0", "image_path": str(img), "source_type": "layout", "meta": {}}) + "\n",
            encoding="utf-8")
        cmcv = tmp / "cmcv.jsonl"
        cmcv.write_text(json.dumps(
            {"id": "p0", "tier": "Hard",
             "elements": {"external_a": REF_A, "external_b": REF_B}}) + "\n", encoding="utf-8")

        judged = {"called": False}

        def spy_judge(*a, **k):
            judged["called"] = True
            return {"resolved": True, "overall_score": 5, "element_issues": []}

        monkeypatch.setattr(run_mod, "generate_label", lambda *a, **k: ([], "raw"))
        monkeypatch.setattr(run_mod, "judge_layout_once", spy_judge)

        out = tmp / "judge.jsonl"
        run_mod.run_judge(unified, cmcv, out, {}, {"name": "j"}, workers=1)
        row = json.loads(out.read_text(encoding="utf-8").strip())

        assert judged["called"] is False        # judge 호출 자체를 안 함 (콜 절약 + 오판 방지)
        assert row["resolved"] is False         # 자동확정되지 않고 사람 검수로
        assert "empty_label" in row["error"]


# ── 단일 종합판정(검증된 스키마) + expected_category/severity 보강 + 단일 revise ──────────
LAYOUT_PASS_JSON = json.dumps({"pass": True, "overall_score": 5.0, "issues": []})
LAYOUT_FAIL_JSON = json.dumps({
    "pass": False, "overall_score": 2.0,
    "issues": [
        {"element_index": 0, "category": "Text", "expected_category": None,
         "issue": "bbox_shifted", "severity": "MAJOR", "description": "오른쪽으로 밀림"},
    ],
})


def test_parse_layout_verdict_extracts_issues_with_severity():
    from ocr_filter.hardcase.parse import parse_layout_verdict

    v = parse_layout_verdict(LAYOUT_FAIL_JSON)
    assert v["resolved"] is False
    assert v["overall_score"] == 2.0
    assert len(v["element_issues"]) == 1
    ei = v["element_issues"][0]
    assert ei["element_index"] == 0 and ei["issue"] == "bbox_shifted"
    assert ei["severity"] == "MAJOR"


def test_parse_layout_verdict_drops_malformed_issues():
    from ocr_filter.hardcase.parse import parse_layout_verdict

    raw = json.dumps({
        "pass": False, "overall_score": 2.0,
        "issues": [
            {"element_index": 0, "category": "Text",
             "issue": "not_a_real_type", "description": "x"},   # 미등록 issue → drop
            {"element_index": -1, "category": "Text",
             "issue": "bbox_shifted", "description": "x"},       # 음수 인덱스 → drop
            {"element_index": 1, "category": "Text",
             "issue": "bbox_shifted", "description": ""},        # 빈 description → drop
        ],
    })
    v = parse_layout_verdict(raw)
    assert v["element_issues"] == []


def test_parse_layout_verdict_conservative_on_garbage():
    from ocr_filter.hardcase.parse import parse_layout_verdict

    v = parse_layout_verdict("이건 그냥 잡담")
    assert v["resolved"] is False and v["parse_error"] is True


def test_judge_layout_once_uses_normalized_bbox():
    seen = {}

    def fake(cfg, images, prompt, **k):
        seen["prompt"] = prompt
        return LAYOUT_PASS_JSON

    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "page.png"
        from PIL import Image as _Image
        _Image.new("RGB", (2000, 1000), "white").save(img)
        els = [{"category": "Text", "bbox": [1000, 500, 2000, 1000], "text": "t"}]
        v = judge_layout_once(str(img), els, {"name": "j"}, call_fn=fake)

    assert v["resolved"] is True
    assert "bbox=[500, 500, 1000, 1000]" in seen["prompt"]   # 픽셀 → 0-1000 정규화 (judge_once 와 동일)


def test_revise_label_targets_flagged_element_only():
    """이전 refine(자유 텍스트 지적)과 달리, revise 는 element_issues 의 정확한 인덱스를
    프롬프트에 그대로 노출해야 한다 — 그래야 모델이 어느 요소를 고칠지 알 수 있다."""
    seen = {}
    FIXED = ('<div data-bbox="0 0 500 500" data-label="Text"><p>fixed</p></div>')

    def fake(cfg, images, prompt, system=None, enable_thinking=None):
        seen["prompt"] = prompt
        return FIXED

    verdict = {"element_issues": [
        {"element_index": 0, "bbox": [100, 200, 300, 400], "category": "Text",
         "expected_category": None, "issue": "bbox_shifted", "severity": "MAJOR",
         "description": "오른쪽으로 밀림"},
    ]}
    prev = [{"category": "Text", "bbox": [100, 200, 300, 400], "text": "orig"}]
    els, _raw = revise_label("/x/p.png", prev, verdict, (1000, 1000), {"name": "gen"}, call_fn=fake)

    assert "bbox_shifted" in seen["prompt"] and "오른쪽으로 밀림" in seen["prompt"]
    assert els[0]["category"] == "Text" and [round(v) for v in els[0]["bbox"]] == [0, 0, 500, 500]


def test_revise_label_falls_back_to_previous_on_parse_failure():
    prev = [{"category": "Text", "bbox": [1, 2, 3, 4], "text": "orig"}]
    verdict = {"element_issues": []}
    els, _raw = revise_label("/x/p.png", prev, verdict, (100, 100), {"name": "gen"},
                             call_fn=lambda *a, **k: "이상한 응답")
    assert els == prev


def test_run_judge_revises_once_on_fail_then_repasses(monkeypatch):
    """FAIL 이고 element_issues 가 있으면 revise 를 정확히 한 번만 거치고 재판정한다."""
    import ocr_filter.hardcase.run as run_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = tmp / "page.png"
        _tiny_png(img)
        unified = tmp / "unified.jsonl"
        unified.write_text(json.dumps(
            {"id": "p0", "image_path": str(img), "source_type": "layout", "meta": {}}) + "\n",
            encoding="utf-8")
        cmcv = tmp / "cmcv.jsonl"
        cmcv.write_text(json.dumps(
            {"id": "p0", "tier": "Hard",
             "elements": {"external_a": REF_A, "external_b": REF_B}}) + "\n", encoding="utf-8")

        judge_calls = {"n": 0}
        revise_calls = {"n": 0}

        def fake_gen(image_path, a_els, b_els, img_size, gen_cfg, **kw):
            return [{"category": "Text", "text": "t", "bbox": [1, 2, 3, 4]}], "raw"

        def fake_judge(image_path, elements, judge_cfg, call_fn=None):
            judge_calls["n"] += 1
            if judge_calls["n"] == 1:
                return {"resolved": False, "overall_score": 2,  
                        "element_issues": [{"element_index": 0, "bbox": [1, 2, 3, 4],
                                            "category": "Text", "expected_category": None,
                                            "issue": "bbox_shifted", "severity": "MAJOR",
                                            "description": "밀림"}]}
            return {"resolved": True, "overall_score": 5, "element_issues": []}

        def fake_revise(image_path, elements, verdict, img_size, gen_cfg, **kw):
            revise_calls["n"] += 1
            return [{"category": "Text", "text": "fixed", "bbox": [5, 6, 7, 8]}], "raw"

        monkeypatch.setattr(run_mod, "generate_label", fake_gen)
        monkeypatch.setattr(run_mod, "judge_layout_once", fake_judge)
        monkeypatch.setattr(run_mod, "revise_label", fake_revise)

        out = tmp / "judge.jsonl"
        run_mod.run_judge(unified, cmcv, out, {}, {"name": "j"}, workers=1)
        row = json.loads(out.read_text(encoding="utf-8").strip())

        assert judge_calls["n"] == 2 and revise_calls["n"] == 1   # 판정 → revise → 재판정, 루프 아님
        assert row["resolved"] is True and row["revised"] is True
        assert row["final_elements"][0]["text"] == "fixed"        # revise 결과가 최종 라벨


def test_run_judge_skips_revise_when_no_actionable_issues(monkeypatch):
    """FAIL 이어도 element_issues 가 없으면(고칠 타겟 불명) revise 를 아예 건너뛴다."""
    import ocr_filter.hardcase.run as run_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = tmp / "page.png"
        _tiny_png(img)
        unified = tmp / "unified.jsonl"
        unified.write_text(json.dumps(
            {"id": "p0", "image_path": str(img), "source_type": "layout", "meta": {}}) + "\n",
            encoding="utf-8")
        cmcv = tmp / "cmcv.jsonl"
        cmcv.write_text(json.dumps(
            {"id": "p0", "tier": "Hard",
             "elements": {"external_a": REF_A, "external_b": REF_B}}) + "\n", encoding="utf-8")

        revise_calls = {"n": 0}
        monkeypatch.setattr(run_mod, "generate_label",
                            lambda *a, **k: ([{"category": "Text", "text": "t",
                                              "bbox": [1, 2, 3, 4]}], "raw"))
        monkeypatch.setattr(run_mod, "judge_layout_once",
                            lambda *a, **k: {"resolved": False, "overall_score": 2,
                                             "element_issues": []})

        def fake_revise(*a, **k):
            revise_calls["n"] += 1
            return [], "raw"

        monkeypatch.setattr(run_mod, "revise_label", fake_revise)

        out = tmp / "judge.jsonl"
        run_mod.run_judge(unified, cmcv, out, {}, {"name": "j"}, workers=1)
        row = json.loads(out.read_text(encoding="utf-8").strip())

        assert revise_calls["n"] == 0
        assert row["resolved"] is False and row["revised"] is False


def test_run_judge_records_error_without_killing_batch(monkeypatch):
    """한 건이 터져도(서버 다운 등) 나머지는 계속 — 실패는 기록만."""
    import ocr_filter.hardcase.run as run_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = tmp / "page.png"
        _tiny_png(img)
        unified = tmp / "unified.jsonl"
        unified.write_text(json.dumps(
            {"id": "p0", "image_path": str(img), "source_type": "layout", "meta": {}}) + "\n",
            encoding="utf-8")
        cmcv = tmp / "cmcv.jsonl"
        cmcv.write_text(json.dumps(
            {"id": "p0", "tier": "Hard",
             "elements": {"external_a": REF_A, "external_b": REF_B}}) + "\n", encoding="utf-8")

        def boom(*a, **k):
            raise ConnectionError("서버 다운")

        monkeypatch.setattr(run_mod, "generate_label", boom)
        out = tmp / "judge.jsonl"
        stats = run_mod.run_judge(unified, cmcv, out, {}, {"name": "j"}, workers=1)
        assert stats["n_done"] == 1 and stats["n_errors"] == 1
        row = json.loads(out.read_text(encoding="utf-8").strip())
        assert row["resolved"] is False and "ConnectionError" in row["error"]
