#!/usr/bin/env python3
"""[merge] cmcv(Medium) + rescue(Hard→Medium-rescued) + hardcase(Hard judge) →
final_dataset.jsonl (export gt-html 의 입력 스키마).

export/gt_html.py 가 요구하는 필드만 정확히 채운다: id / resolved / tier / label.
Easy 는 target 자기 출력이라 여기서 아예 안 받는다(SFT 가치 낮음, README 관례와 동일).

    python3 scripts/build_final_output.py \
        --cmcv-results  /home/jhyeo/ocr_filter_result/batch5400/cmcv_results.jsonl \
        --rescue        /home/jhyeo/ocr_filter_result/batch5400/rescue_results.jsonl \
        --hardcase-judge /home/jhyeo/ocr_filter_result/batch5400/hardcase_judge.jsonl \
        --out           /home/jhyeo/ocr_filter_result/batch5400/final_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ocr_filter.cmcv.normalize import full_text, normalize_category
from ocr_filter.metrics import SCORERS

_STRUCT_CATEGORIES = {"Table", "Formula"}


def _recovered_pseudo_label(d: dict, text_min: float) -> list | None:
    """cmcv._process_one 은 text subtask 가 정확히 'Medium' 일 때만 pseudo_label 을
    채택한다(외부 두 모델만 합의). 하지만 text subtask 'Easy'(target 포함 셋 다 합의,
    Medium 보다 더 강한 합의)인데 다른 subtask(layout 등) 때문에 페이지 tier 가
    Medium 으로 내려간 경우는 이 채택 규칙에 안 걸려 pseudo_label 이 비게 된다.
    실측(2026-07-30 batch5400 라이브 실행 중): tier=Medium 50건 중 29건(58%)이 바로
    이 케이스. GPU 재호출 없이 이미 저장된 elements 로 같은 게이트(text_min)를
    적용해 구제한다."""
    if d.get("tier") != "Medium" or d.get("pseudo_label"):
        return None
    if d.get("subtask_tiers", {}).get("text") != "Easy":
        return None
    elements = d.get("elements") or {}
    a = elements.get("external_a") or []
    b = elements.get("external_b") or []
    a_text = [e for e in a if normalize_category(e.get("category") or "") not in _STRUCT_CATEGORIES]
    b_text = [e for e in b if normalize_category(e.get("category") or "") not in _STRUCT_CATEGORIES]
    text_score = SCORERS["edit_distance"](full_text(a_text), full_text(b_text))
    if text_score >= text_min:
        return a
    return None


def _has_unverified_formula(d: dict) -> bool:
    """CDM 서킷브레이커가 트립되면(연속 8회 렌더 실패) 그 프로세스에서는 영구적으로
    formula subtask 점수가 None 이 된다(cmcv/metrics/cdm.py `_cdm_disabled`). `_page_tier()`는
    None subtask 를 그냥 '해당 없음'으로 제외하고 나머지 중 최악만 보므로, Formula 요소가
    있는 페이지가 수식 정합성은 전혀 검증 안 된 채로 Medium 까지 올라올 수 있다(2026-07-31
    batch5400 라이브 실행 중 실측 확인). Table/Formula 는 text subtask 채점에서도 제외되므로
    (`_STRUCT_CATEGORIES`) 다른 subtask 가 대신 잡아주지도 못한다 — 그대로 두면 무검증 수식이
    GT 로 샌다. 이런 레코드는 직접 export 대신 hardcase 재검증이 필요하므로 여기서 제외한다."""
    fs = d.get("subtask_scores", {}).get("formula", {})
    if fs and any(v is not None for v in fs.values()):
        return False  # 실제로 채점됐음(값이 0.0 이어도 렌더는 성공한 것)
    target_els = (d.get("elements") or {}).get("target") or []
    return any(normalize_category(e.get("category") or "") == "Formula" for e in target_els)


def _read_jsonl(path: str | Path | None):
    if path is None:
        return
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(
    cmcv_results: str, rescue: str | None, hardcase_judge: str | None, out_path: str,
    text_min: float = 0.90,
) -> dict:
    counts = Counter()
    seen_ids: set[str] = set()

    with open(out_path, "w", encoding="utf-8") as fout:
        # 1) cmcv 의 Medium (dots.ocr 원본 pseudo-label, 텍스트 게이트 이미 통과)
        for d in _read_jsonl(cmcv_results):
            if d.get("tier") != "Medium":
                continue
            if _has_unverified_formula(d):
                counts["Medium(skipped_unverified_formula)"] += 1
                continue
            label = d.get("pseudo_label")
            label_source = "cmcv_pseudo_label"
            if not label:
                label = _recovered_pseudo_label(d, text_min)
                label_source = "cmcv_pseudo_label_text_easy_recovered"
            if not label:
                continue
            rid = d["id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            fout.write(json.dumps({
                "id": rid, "resolved": True, "tier": "Medium",
                "label_source": label_source, "label": label,
            }, ensure_ascii=False) + "\n")
            counts["Medium(cmcv)" if label_source == "cmcv_pseudo_label" else "Medium(text_easy_recovered)"] += 1

        # 2) rescue 로 건진 Hard (요소 단위 외부합의, N:M 그룹 매칭)
        for d in _read_jsonl(rescue):
            rid = d["id"]
            if rid in seen_ids:
                continue
            label = d.get("label")
            if not label:
                continue
            seen_ids.add(rid)
            fout.write(json.dumps({
                "id": rid, "resolved": True, "tier": "Medium-rescued",
                "label_source": d.get("label_source", "rescue_external_consensus"),
                "label": label,
            }, ensure_ascii=False) + "\n")
            counts["Medium-rescued"] += 1

        # 3) hardcase judge 로 해결된 Hard (122B 단일콜 생성+판정, resolved=True 만)
        for d in _read_jsonl(hardcase_judge):
            if not d.get("resolved"):
                continue
            rid = d["id"]
            if rid in seen_ids:
                continue
            label = d.get("final_elements")
            if not label:
                continue
            seen_ids.add(rid)
            fout.write(json.dumps({
                "id": rid, "resolved": True, "tier": "Hard",
                "label_source": "hardcase_judge_122b", "label": label,
            }, ensure_ascii=False) + "\n")
            counts["Hard(judge)"] += 1

    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmcv-results", required=True)
    ap.add_argument("--rescue", default=None, help="rescue.py run_rescue 출력 (없으면 스킵)")
    ap.add_argument("--hardcase-judge", default=None, help="hardcase judge 출력 (없으면 스킵)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-min", type=float, default=0.90,
                     help="text subtask 'Easy' 구제 시 dots↔paddle 텍스트 일치 게이트 (cmcv 기본값과 동일)")
    args = ap.parse_args()

    counts = build(args.cmcv_results, args.rescue, args.hardcase_judge, args.out, args.text_min)
    total = sum(counts.values())
    print(f"완료: {counts} 합계={total}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
