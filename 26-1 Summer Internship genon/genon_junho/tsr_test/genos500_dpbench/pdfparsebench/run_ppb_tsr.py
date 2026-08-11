#!/usr/bin/env python3
"""tsr 200장 평가셋(predictions.jsonl)에 LLM-Judge 표 채점 추가 (이슈 doc_parser#318).

genos 500-set과 달리 TSR 평가는 GT↔pred가 **이미 1:1 정렬된 표 쌍**이므로
LLM 매칭(extract) 단계가 불필요하다 — pdf-parse-bench의 채점(judge)만 붙인다.

입력: eval/evaluate.py 가 남긴 run 디렉터리의 predictions.jsonl
      (필드: index, gt_html[_eval], pred_html[_eval], complexity, teds, ...)
출력: --out_dir/tables.json (pdf-parse-bench TableResult 스키마 + llm_scores)
      --out_dir/llm_vs_teds.csv (항목별 teds·LLM 점수 병렬 — 상관 분석용)

사용:
  python run_ppb_tsr.py --predictions <run>/predictions.jsonl \
    --out_dir <run>/ppb_judge \
    --base_url https://genos.genon.ai/api/gateway/rep/serving/776/v1 \
    --api_key <KEY> --model model
"""

import argparse
import csv
import json
import os


def make_client(base_url, api_key):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def preflight(args):
    c = make_client(args.base_url, args.api_key)
    r = c.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "ping. 'pong'만 답해."}],
        max_tokens=2000,
    )
    if not r.choices:
        err = getattr(r, "error", None) or r.model_dump().get("error")
        raise SystemExit(f"[preflight 실패] {err}")
    print(f"[preflight OK] model={r.model}")


# pdf-parse-bench TableResult 는 complexity 를 simple/moderate/complex 로 제한함.
# TSR 평가셋은 medium 등 다른 라벨을 쓰므로 스키마용으로 매핑한다
# (요약·CSV 는 predictions.jsonl 의 원본 라벨을 사용).
_COMPLEXITY_MAP = {"simple": "simple", "easy": "simple",
                   "moderate": "moderate", "medium": "moderate",
                   "complex": "complex", "hard": "complex"}


def convert(predictions_path, out_dir, field):
    """predictions.jsonl -> pdf-parse-bench tables.json (+ 빈 formulas.json)."""
    suffix = "_eval" if field == "eval" else ""
    items = []
    for line in open(predictions_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        gt = r.get(f"gt_html{suffix}") or r.get("gt_html") or ""
        pred = r.get(f"pred_html{suffix}") or r.get("pred_html") or ""
        items.append({
            "index": r["index"],
            "gt_table": gt,
            "extracted_table": pred,
            "complexity": _COMPLEXITY_MAP.get(
                str(r.get("complexity") or "").lower(), "moderate"),
            "llm_scores": [],
        })
    os.makedirs(out_dir, exist_ok=True)
    tables_path = os.path.join(out_dir, "tables.json")
    if not os.path.exists(tables_path):  # 재실행 시 기존 채점 보존
        with open(tables_path, "w") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
    formulas_path = os.path.join(out_dir, "formulas.json")
    if not os.path.exists(formulas_path):
        with open(formulas_path, "w") as f:
            json.dump([], f)
    print(f"표 {len(items)}쌍 -> {tables_path}")
    return tables_path, formulas_path


def summarize(tables_path, out_dir, predictions_path):
    # judge 저장 시 스키마 밖 필드가 유실되므로 teds 는 원본에서 재조인한다.
    teds_by_idx, cplx_by_idx = {}, {}
    for line in open(predictions_path):
        line = line.strip()
        if line:
            r = json.loads(line)
            teds_by_idx[r["index"]] = (r.get("teds"), r.get("teds_structure"))
            cplx_by_idx[r["index"]] = r.get("complexity") or "unknown"
    items = json.load(open(tables_path))
    rows = []
    for it in items:
        s = (it.get("llm_scores") or [{}])[0]
        teds, teds_s = teds_by_idx.get(it["index"], (None, None))
        rows.append({
            "index": it["index"],
            "complexity": cplx_by_idx.get(it["index"], it.get("complexity")),
            "teds": teds, "teds_structure": teds_s,
            "llm_score": s.get("score"),
            "n_errors": len(s.get("errors") or []),
            "first_error": (s.get("errors") or [""])[0][:150],
        })
    csv_path = os.path.join(out_dir, "llm_vs_teds.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    scored = [r for r in rows if r["llm_score"] is not None]
    if scored:
        avg = sum(r["llm_score"] for r in scored) / len(scored)
        print(f"\nLLM 평균 {avg:.3f} (n={len(scored)})")
        # complexity 별
        import collections
        by_c = collections.defaultdict(list)
        for r in scored:
            by_c[r["complexity"]].append(r["llm_score"])
        for c, v in sorted(by_c.items()):
            print(f"  complexity={c:10s} n={len(v):4d} 평균 {sum(v)/len(v):.3f}")
        # TEDS 구간별 LLM 평균 (metric 정렬 확인)
        bands = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.999), (0.999, 1.01)]
        print("  TEDS 구간별 LLM 평균:")
        for lo, hi in bands:
            v = [r["llm_score"] for r in scored
                 if r["teds"] is not None and lo <= r["teds"] < hi]
            if v:
                print(f"    TEDS [{lo:.1f},{hi:.1f}) n={len(v):4d} -> LLM {sum(v)/len(v):.2f}")
    print(f"항목별 CSV: {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--model", default="model")
    ap.add_argument("--field", default="eval", choices=["eval", "raw"],
                    help="채점에 쓸 HTML: eval(정규화판, TEDS와 동일 입력) 또는 raw")
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--skip_judge", action="store_true", help="변환·요약만")
    args = ap.parse_args()

    tables_path, formulas_path = convert(args.predictions, args.out_dir, args.field)

    if not args.skip_judge:
        preflight(args)
        os.environ.setdefault("OPENROUTER_API_KEY", "DUMMY")
        from pathlib import Path
        from pdf_parse_bench.eval import llm_judge
        llm_judge.LLMEvaluator._client = make_client(args.base_url, args.api_key)
        llm_judge.run_batch_evaluation(
            llm_judge_models=[args.model],
            jobs=[llm_judge.EvalPaths(formulas_path=Path(formulas_path),
                                      tables_path=Path(tables_path))],
            skip_existing=True,
            max_workers=args.max_workers,
        )

    summarize(tables_path, args.out_dir, args.predictions)


if __name__ == "__main__":
    main()
