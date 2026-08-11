"""[6] Hard case 정제 실행기 — cmcv 결과의 Hard 티어만 대상으로 라벨 생성 + 판정 게이트.

Hard 케이스는 두 외부 모델(dots/paddle)이 불일치한 페이지다. cmcv_results.jsonl 에 이미
저장된 그 두 예측(external_a/external_b)을 레퍼런스로 써서, heavy 모델(397B)이 **단일 콜**로
최종 라벨(Chandra div-HTML → 픽셀좌표 elements)을 낸다. 그 뒤 같은 모델이 판정만 한 번 —
PASS 면 자동 확정, FAIL 이면 사람 검수 큐로 (되먹임 재교정 루프 없음: 안 바뀔 bbox 를 큰
모델로 재확인하고 텍스트를 매번 재-OCR 하던 낭비를 제거).

    _work/hardcase_judge.jsonl     ← run_judge()     (라벨 생성 + 판정 게이트)
    _work/hardcase_prelabel.jsonl  ← run_prelabel()  (judge 라벨을 labeler export 포맷으로 통과)

    python -m ocr_filter.cli hardcase judge      # 라벨 생성 + 판정
    python -m ocr_filter.cli hardcase prelabel   # export 용 통과 (judge 결과 필요)
    python -m ocr_filter.cli hardcase run        # 둘 다 순서대로
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from ocr_filter.hardcase.pipeline import generate_label, judge_layout_once, revise_label
from ocr_filter.hardcase.client import call_vlm
from ocr_filter.io.schema import Record, read_jsonl

_write_lock = threading.Lock()


def _no_think(cfg, images, prompt):
    """judge 게이트 콜용 — 이 모델(397B)도 thinking 모델이라 사고를 꺼서 truncation/비용을 막는다."""
    return call_vlm(cfg, images, prompt, enable_thinking=False)


_TEXTLESS_OK = {"Picture"}  # 텍스트가 비어 있어도 정상인 카테고리


def _label_quality_error(label: list[dict]) -> str | None:
    """생성 라벨의 기본 위생 검사 — judge 에 보내기 전에 걸러야 하는 구조적 결함.

    - bbox 없는 요소만 남으면: 오버레이에 박스가 하나도 안 그려져 judge 가 PASS 를 주는
      empty_label 케이스와 같은 경로로 자동확정될 수 있다.
    - 텍스트 카테고리의 과반이 빈 텍스트면: 박스만 치고 전사를 포기한 출력 — 이대로
      GT 가 되면 "글자가 있는데 빈 라벨"을 학습시킨다. (Picture 등은 빈 텍스트가 정상)
    """
    valid = [e for e in label if e.get("bbox") and len(e["bbox"]) == 4]
    if not valid:
        return "no_bbox: 모든 요소에 bbox 없음"
    text_els = [e for e in valid if e.get("category") not in _TEXTLESS_OK]
    empty = [e for e in text_els if not (e.get("text") or "").strip()]
    if text_els and len(empty) > len(text_els) * 0.5:
        return f"empty_texts: 텍스트 카테고리 {len(text_els)}개 중 {len(empty)}개가 빈 텍스트"
    return None


def _hard_elements(cmcv_results_path: str | Path) -> dict[str, tuple[list, list]]:
    """Hard 티어 id → (external_a=dots, external_b=paddle) 예측. 두 예측 모두 원본 픽셀좌표의
    구조화 출력이라 라벨 생성 콜의 레퍼런스로 그대로 쓴다."""
    out: dict[str, tuple[list, list]] = {}
    with open(cmcv_results_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("tier") == "Hard":
                els = d.get("elements", {}) or {}
                out[d["id"]] = (els.get("external_a", []) or [], els.get("external_b", []) or [])
    return out


def _done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with open(out_path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _run_pool(records: list, out_path: Path, workers: int, process_fn) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total, n_done, n_errors = len(records), 0, 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_fn, r): r for r in records}
        for fut in as_completed(futures):
            result = fut.result()
            with _write_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
            n_done += 1
            if result.get("error"):
                n_errors += 1
            if n_done % 5 == 0 or n_done == n_total:
                print(f"[{n_done}/{n_total}] {time.time() - t0:.0f}s 경과, errors={n_errors}")
    return {"n_total": n_total, "n_done": n_done, "n_errors": n_errors}


def run_judge(
    unified_path: str | Path,
    cmcv_results_path: str | Path,
    out_path: str | Path,
    models_cfg: dict,          # 하위호환용 (더 이상 target 9B 를 안 부른다)
    judge_cfg: dict,
    refine_cfg: dict | None = None,   # 미사용 (재교정 루프 제거)
    max_rounds: int = 3,              # 미사용
    workers: int = 8,
    limit: int | None = None,
    gen_cfg: dict | None = None,
) -> dict:
    """gen_cfg 를 안 주면 judge_cfg(397B)로 라벨 생성도 겸한다."""
    out_path = Path(out_path)
    gen_cfg = gen_cfg or judge_cfg
    hard = _hard_elements(cmcv_results_path)
    done = _done_ids(out_path)
    records = [r for r in read_jsonl(unified_path) if r.id in hard and r.id not in done]
    if limit is not None:
        records = records[:limit]

    def process(record: Record) -> dict:
        try:
            with Image.open(record.image_path) as im:
                img_size = im.size
            a_els, b_els = hard[record.id]
            label, _raw = generate_label(record.image_path, a_els, b_els, img_size, gen_cfg)
            if not label:
                # 생성/파싱 실패(빈 라벨)는 judge 에 보내면 안 된다 — 박스가 하나도 없는
                # 오버레이를 보고 "이상 없음"이라며 PASS(5점)를 주는 게 실측으로 확인됐다
                # (2026-07-16). 쓰레기를 자동확정하느니 여기서 끊고 사람 검수로 보낸다.
                return {"id": record.id, "image_path": record.image_path,
                        "final_elements": [], "resolved": False,
                        "error": "empty_label: 생성 응답에서 요소를 못 뽑음"}
            label = [e for e in label if e.get("bbox") and len(e["bbox"]) == 4]
            quality_error = _label_quality_error(label)
            if quality_error:
                # empty_label 과 같은 이유로 judge 에 보내지 않고 즉시 미해결 처리 —
                # 박스/텍스트가 구조적으로 부실한 라벨은 judge 가 PASS 를 줘도 GT 로 못 쓴다.
                return {"id": record.id, "image_path": record.image_path,
                        "final_elements": label, "resolved": False,
                        "error": quality_error}
            verdict = judge_layout_once(record.image_path, label, judge_cfg, call_fn=_no_think)
            revised = False
            if not verdict["resolved"] and verdict.get("element_issues"):
                # FAIL 이고 고칠 대상이 명확할 때만 **딱 한 번** revise 후 재판정한다(루프
                # 아님) — 정확한 element_index 피드백이면 refine 이 실제로 먹힐 가능성이
                # 이전(자유 텍스트 지적)보다 훨씬 높다. element_issues 가 비어있으면(예:
                # 판정 자체가 애매한 전반적 사유) 고칠 타겟이 없으므로 revise 를 건너뛴다.
                new_label, _raw2 = revise_label(
                    record.image_path, label, verdict, img_size, gen_cfg,
                )
                if new_label:
                    label = new_label
                    revised = True
                    verdict = judge_layout_once(record.image_path, label, judge_cfg,
                                                 call_fn=_no_think)
            return {
                "id": record.id,
                "image_path": record.image_path,
                "final_elements": label,
                "resolved": bool(verdict["resolved"]),  # PASS=자동확정, FAIL=사람 검수로
                "revised": revised,
                "verdict": verdict,
            }
        except Exception as e:  # noqa: BLE001 — 한 건 실패가 전체를 죽이지 않게 기록만
            return {"id": record.id, "image_path": record.image_path,
                    "final_elements": [], "resolved": False,
                    "error": f"{type(e).__name__}: {e}"}

    return _run_pool(records, out_path, workers, process)


def run_prelabel(
    judge_out_path: str | Path,
    out_path: str | Path,
    prelabel_cfg: dict | None = None,   # 미사용 (별도 사전라벨 콜 제거 — judge 라벨을 그대로 씀)
    workers: int = 8,
    limit: int | None = None,
) -> dict:
    """judge 단계가 낸 최종 라벨(final_elements)을 labeler export 가 읽는 필드
    (prelabel_elements)로 통과시킨다. 추가 모델 콜 없음."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)

    rows = []
    with open(judge_out_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["id"] not in done:
                rows.append(d)
    if limit is not None:
        rows = rows[:limit]

    n = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for d in rows:
            out_f.write(json.dumps({
                "id": d["id"],
                "image_path": d["image_path"],
                "prelabel_elements": d.get("final_elements", []),
                "resolved": d.get("resolved", False),
            }, ensure_ascii=False) + "\n")
            n += 1
    return {"n_total": len(rows), "n_done": n}


def run_both(
    unified_path: str | Path, cmcv_results_path: str | Path,
    judge_out_path: str | Path, prelabel_out_path: str | Path,
    models_cfg: dict, judge_cfg: dict, prelabel_cfg: dict,
    refine_cfg: dict | None = None, max_rounds: int = 3, workers: int = 8,
    limit: int | None = None,
) -> dict:
    judge_stats = run_judge(
        unified_path, cmcv_results_path, judge_out_path, models_cfg, judge_cfg,
        refine_cfg, max_rounds, workers, limit,
    )
    prelabel_stats = run_prelabel(judge_out_path, prelabel_out_path, prelabel_cfg, workers)
    return {"judge": judge_stats, "prelabel": prelabel_stats}
