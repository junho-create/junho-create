"""[6] Hard case 정제 파이프라인.

1단계 Judge-and-Refine: 예측(bbox 오버레이 렌더링) vs 원본을 heavy judge VLM 에 보여줘서
판정 → FAIL 이면 같은(혹은 다른) 모델에게 원본+이슈만 주고 재교정 → 최대 max_rounds 반복.

2단계 Targeted Expert Annotation 사전 라벨링: 1단계에서 안 풀린(unresolved) 건만, "독립"
모델(judge 와 다른 계열이어야 함 — configs 의 prelabel_vlm)이 원본을 보고 처음부터 다시
초안을 만들어서 사람이 그걸 고치기만 하면 되게 해준다.

두 단계 다 `call_fn` 을 주입할 수 있어서(기본은 `ocr_filter.hardcase.client.call_vlm`)
실제 서버 없이도 로직만 단위 테스트 가능.
"""

from __future__ import annotations

from typing import Callable

from PIL import Image

from ocr_filter.cmcv.prompts import CHANDRA_SYSTEM
from ocr_filter.hardcase.client import call_vlm as _default_call_vlm
from ocr_filter.hardcase.parse import (
    parse_chandra_label,
    parse_elements_response,
    parse_judge_verdict,
    parse_layout_verdict,
)
from ocr_filter.hardcase.prompts import (
    build_judge_prompt,
    build_layout_judge_prompt,
    build_merge_chandra_prompt,
    build_prelabel_prompt,
    build_refine_prompt,
    build_revise_chandra_prompt,
)
from ocr_filter.report.render import draw_boxes

CallFn = Callable[..., str]


def _bbox_1000_to_pixel(bbox: list[float] | None, img_w: int, img_h: int) -> list[float] | None:
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    return [x0 * img_w / 1000, y0 * img_h / 1000, x1 * img_w / 1000, y1 * img_h / 1000]


def generate_label(
    image_path: str,
    a_elements: list[dict],
    b_elements: list[dict],
    img_size: tuple[int, int],
    gen_cfg: dict,
    call_fn: CallFn = _default_call_vlm,
    enable_thinking: bool | None = False,
) -> tuple[list[dict], str]:
    """단일 콜 라벨 생성: 원본 이미지 + A/B(dots/paddle) 예측을 레퍼런스로 주고, heavy 모델이
    Chandra div-HTML(bbox 0-1000)로 최종 레이아웃을 한 번에 낸다. 파싱 후 bbox 를 원본 픽셀
    좌표로 환산해 돌려준다(파이프라인 나머지 — judge 렌더링/labeler export — 가 픽셀좌표 기준).
    반환: (픽셀좌표 elements, raw 응답)."""
    img_w, img_h = img_size
    prompt = build_merge_chandra_prompt(a_elements, b_elements, img_w, img_h)
    raw = call_fn(gen_cfg, [image_path], prompt, system=CHANDRA_SYSTEM,
                  enable_thinking=enable_thinking)
    elements = parse_chandra_label(raw)  # bbox 0-1000
    for e in elements:
        e["bbox"] = _bbox_1000_to_pixel(e.get("bbox"), img_w, img_h)
    return elements, raw


def _bbox_pixel_to_1000(bbox: list[float] | None, img_w: int, img_h: int) -> list[int] | None:
    if not bbox or len(bbox) != 4 or not img_w or not img_h:
        return None
    x0, y0, x1, y1 = bbox
    return [round(x0 * 1000 / img_w), round(y0 * 1000 / img_h),
            round(x1 * 1000 / img_w), round(y1 * 1000 / img_h)]


def judge_once(
    image_path: str, elements: list[dict], judge_cfg: dict, call_fn: CallFn = _default_call_vlm,
) -> dict:
    """원본 + (bbox 오버레이 렌더링) 2장을 judge VLM 에 보여 판정만.

    프롬프트에 넣는 bbox 는 **0-1000 정규화**해서 준다: draw_boxes 가 오버레이를 max_side 로
    축소(1654x2339 → 707x1000)하는데, 원본 픽셀 좌표를 그대로 숫자로 주면 judge 가 그 숫자를
    축소된 오버레이에 대조해서 멀쩡한 박스를 죄다 "오른쪽으로 밀렸다/너무 크다"로 오판한다
    (2026-07-16 실측: 정상 라벨 3건이 전부 score=2 로 FAIL, 불평이 x>600 처럼 축소본 폭 기준).
    정규화하면 두 이미지 크기가 뭐든 좌표 해석이 일관된다."""
    rendered = draw_boxes(image_path, elements, coord_system="pixel")
    with Image.open(image_path) as im:
        img_w, img_h = im.size
    norm_elements = [
        {**e, "bbox": _bbox_pixel_to_1000(e.get("bbox"), img_w, img_h)} for e in elements
    ]
    prompt = build_judge_prompt(norm_elements)
    raw = call_fn(judge_cfg, [image_path, rendered], prompt)
    return parse_judge_verdict(raw)


def judge_layout_once(
    image_path: str, elements: list[dict], judge_cfg: dict, call_fn: CallFn = _default_call_vlm,
    render_content: bool = True,
) -> dict:
    """판정 1회. judge 에게 넘기는 이미지는 **최대 4장**이다:

        1. 원본 문서 이미지
        2. bbox 오버레이 (박스 위치/누락 검증용)
        3. 파싱된 표 HTML 을 실제로 렌더한 이미지   ← render-then-verify
        4. 파싱된 수식 LaTeX 를 실제로 렌더한 이미지 ← render-then-verify

    3·4번이 MinerU2.5-Pro §3.3 의 핵심이다. 2번(오버레이)만 주면 judge 는 박스가 맞는지만
    볼 수 있고 표/수식의 **내용과 구조**는 원리적으로 검증할 수 없다 — 실제로 그 상태로
    Hard 24,548건이 텍스트 무검증으로 확정됐다(2026-07-30 진단). 파싱 결과를 렌더해서
    나란히 보여주면 닫히지 않은 태그·병합 오류·빠진 셀 같은 결함이 시각적 붕괴로 드러난다.

    렌더가 실패하거나(표/수식이 없거나 도구 미설치) render_content=False 면 그 이미지는
    빠지고 기존 2장 방식으로 동작한다 — 판정 자체를 못 하게 되지는 않는다.

    bbox 는 judge_once 와 동일하게 0-1000 정규화해서 프롬프트에 준다."""
    rendered = draw_boxes(image_path, elements, coord_system="pixel")
    with Image.open(image_path) as im:
        img_w, img_h = im.size
    norm_elements = [
        {**e, "bbox": _bbox_pixel_to_1000(e.get("bbox"), img_w, img_h)} for e in elements
    ]

    images: list = [image_path, rendered]
    parsed_renders: dict = {}
    if render_content:
        from ocr_filter.report.render_parse import render_parsed_elements
        parsed_renders = render_parsed_elements(elements)
        images.extend(parsed_renders[k] for k in ("tables", "formulas") if k in parsed_renders)

    prompt = build_layout_judge_prompt(norm_elements, rendered_kinds=list(parsed_renders))
    raw = call_fn(judge_cfg, images, prompt)
    return parse_layout_verdict(raw)


def revise_label(
    image_path: str,
    elements: list[dict],
    verdict: dict,
    img_size: tuple[int, int],
    gen_cfg: dict,
    call_fn: CallFn = _default_call_vlm,
    enable_thinking: bool | None = False,
) -> tuple[list[dict], str]:
    """FAIL 판정을 받은 라벨을, judge_layout_once 가 낸 element_issues(정확한 element_index +
    expected_category + 설명)로 **딱 한 번만** 교정한다(루프 아님). 예전 REFINE_PROMPT/
    judge_and_refine 은 "Box 10a" 식 자유 텍스트 지적이라 refine 이 뭘 고칠지 못 찾고 원본을
    거의 그대로 에코했는데(2026-07 관찰), 이번엔 인덱스가 정확해서 타겟팅된 수정이 가능하다.
    파싱 실패면 원본 라벨을 그대로 유지(퇴행 방지)."""
    img_w, img_h = img_size
    norm_elements = [
        {**e, "bbox": _bbox_pixel_to_1000(e.get("bbox"), img_w, img_h)} for e in elements
    ]
    prompt = build_revise_chandra_prompt(norm_elements, verdict.get("element_issues", []))
    raw = call_fn(gen_cfg, [image_path], prompt, system=CHANDRA_SYSTEM,
                  enable_thinking=enable_thinking)
    revised = parse_chandra_label(raw)
    if not revised:
        return elements, raw
    for e in revised:
        e["bbox"] = _bbox_1000_to_pixel(e.get("bbox"), img_w, img_h)
    return revised, raw


def refine_once(
    image_path: str, elements: list[dict], issues: list[dict], refine_cfg: dict,
    call_fn: CallFn = _default_call_vlm,
) -> list[dict]:
    """원본 1장 + 이슈만 주고 새 예측. 파싱 실패하면 이전 예측 그대로 유지(퇴행 방지)."""
    prompt = build_refine_prompt(elements, issues)
    raw = call_fn(refine_cfg, [image_path], prompt)
    new_elements = parse_elements_response(raw)
    return new_elements if new_elements else elements


def judge_and_refine(
    image_path: str,
    initial_elements: list[dict],
    judge_cfg: dict,
    refine_cfg: dict | None = None,
    max_rounds: int = 3,
    call_fn: CallFn = _default_call_vlm,
) -> dict:
    """refine_cfg 를 안 주면 judge_cfg 로 refine 도 겸한다(같은 heavy 모델이 판정+교정 둘 다)."""
    refine_cfg = refine_cfg or judge_cfg
    current = initial_elements
    history = []

    for round_idx in range(1, max_rounds + 1):
        verdict = judge_once(image_path, current, judge_cfg, call_fn)
        history.append({"round": round_idx, "verdict": verdict, "elements": current})
        if verdict["pass"]:
            return {"resolved": True, "rounds": round_idx,
                    "final_elements": current, "history": history}
        current = refine_once(image_path, current, verdict["issues"], refine_cfg, call_fn)

    return {"resolved": False, "rounds": max_rounds,
            "final_elements": current, "history": history}


def prelabel_once(
    image_path: str, elements: list[dict], issues: list[dict], prelabel_cfg: dict,
    call_fn: CallFn = _default_call_vlm,
) -> list[dict]:
    """2단계: 독립 모델이 원본만 보고 초안을 새로 만듦 (사람이 다듬을 시작점)."""
    prompt = build_prelabel_prompt(elements, issues)
    raw = call_fn(prelabel_cfg, [image_path], prompt)
    candidates = parse_elements_response(raw)
    return candidates if candidates else elements
