#!/usr/bin/env python3
"""표 매칭 3방식(bbox IoU / OmniDocBench 텍스트 / pdf-parse-bench LLM) 비교.

이슈 doc_parser#318 Phase 3. 같은 GT 테이블에 대해 각 방식이 고른 pred 를
내용 지문(태그·공백·th/td 무시 sha1)으로 비교해서, 방식 간 일치율과
불일치 목록을 낸다. 불일치 목록은 수동 판정(어느 쪽이 옳았나)의 입력이 된다.

입력:
- --bbox_dump   : upstage-dp-bench evaluate.py --table_match_dump 산출 JSON
- --omni_result : OmniDocBench result/pred_md_<tag>_quick_match_table_result.json
- --ppb_dir     : (선택) run_ppb_genos500.py 산출 폴더 (<stem>/tables.json)
- --output      : (선택) 불일치 상세 JSON 저장 경로

조인 키 = (문서 stem, GT 테이블 순번). 세 소스 모두 GT 테이블을
문서 내 reading order 순으로 세므로 순번이 호환된다:
- bbox: _doc_tables_with_bbox 의 element 순회 순서
- omni: get_sorted_text_list(order 정렬) — order 는 우리 GT 의 id 순서
- ppb : GT 세그먼트 중 type=table 만 뽑은 리스트의 index
"""

import argparse
import glob
import hashlib
import json
import os
import re


def norm_sha1(html_text):
    """upstage-dp-bench src/table_evaluation._table_content_sha1 과 동일 규칙."""
    text = re.sub(r"<[^>]+>", "|", html_text or "")
    text = re.sub(r"[\s|]+", "", text)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _stem(name):
    return os.path.splitext(os.path.basename(str(name)))[0]


def load_bbox(path):
    """{(stem, gt_idx): {"sha1": ..|None, "head": ..}}  (None=미매칭)"""
    out = {}
    for doc_key, doc in json.load(open(path)).items():
        for p in doc["pairs"]:
            matched = p["pred_idx"] is not None
            out[(_stem(doc_key), p["gt_idx"])] = {
                "sha1": p.get("pred_norm_sha1") if matched else None,
                "head": p.get("pred_html_head", ""),
                "gt_head": p.get("gt_html_head", ""),
            }
    return out


def load_omni(path):
    out = {}
    per_doc_counter = {}
    for item in json.load(open(path)):
        stem = _stem(item.get("img_id", ""))
        gt_idx = item.get("gt_idx")
        gt_idx = gt_idx[0] if isinstance(gt_idx, list) and gt_idx else gt_idx
        if gt_idx in ("", None):  # extra pred 항목
            continue
        # gt_idx 가 문서 내 테이블 순번이 아닐 수 있어(전체 element 위치),
        # 문서 내 등장 순서로 재부여한다. (omni 표 매칭은 GT 테이블 순으로 순회)
        seq = per_doc_counter.get(stem, 0)
        per_doc_counter[stem] = seq + 1
        pred = item.get("pred") or ""
        out[(stem, seq)] = {
            "sha1": norm_sha1(pred) if pred else None,
            "head": pred[:120],
            "gt_head": (item.get("gt") or "")[:120],
        }
    return out


def load_ppb(ppb_dir):
    out = {}
    for path in glob.glob(os.path.join(ppb_dir, "*", "tables.json")):
        stem = os.path.basename(os.path.dirname(path))
        for item in json.load(open(path)):
            pred = item.get("extracted_table") or ""
            out[(stem, item["index"])] = {
                "sha1": norm_sha1(pred) if pred else None,
                "head": pred[:120],
                "gt_head": (item.get("gt_table") or "")[:120],
            }
    return out


def compare(name_a, a, name_b, b):
    keys = sorted(set(a) & set(b))
    stats = {"common_gt": len(keys), "both_same": 0, "both_diff": 0,
             f"{name_a}_only": 0, f"{name_b}_only": 0, "both_unmatched": 0}
    disagreements = []
    for k in keys:
        sa, sb = a[k]["sha1"], b[k]["sha1"]
        if sa and sb:
            if sa == sb:
                stats["both_same"] += 1
            else:
                stats["both_diff"] += 1
                disagreements.append({
                    "doc": k[0], "gt_idx": k[1], "type": "different_pred",
                    "gt_head": a[k]["gt_head"],
                    f"{name_a}_head": a[k]["head"], f"{name_b}_head": b[k]["head"],
                })
        elif sa and not sb:
            stats[f"{name_a}_only"] += 1
            disagreements.append({"doc": k[0], "gt_idx": k[1],
                                  "type": f"{name_a}_only",
                                  "gt_head": a[k]["gt_head"],
                                  f"{name_a}_head": a[k]["head"]})
        elif sb and not sa:
            stats[f"{name_b}_only"] += 1
            disagreements.append({"doc": k[0], "gt_idx": k[1],
                                  "type": f"{name_b}_only",
                                  "gt_head": b[k]["gt_head"],
                                  f"{name_b}_head": b[k]["head"]})
        else:
            stats["both_unmatched"] += 1
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        stats["key_mismatch"] = {f"{name_a}_extra_keys": len(only_a),
                                 f"{name_b}_extra_keys": len(only_b)}
    return stats, disagreements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox_dump", required=True)
    ap.add_argument("--omni_result", required=True)
    ap.add_argument("--ppb_dir", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    sources = {"bbox": load_bbox(args.bbox_dump), "omni": load_omni(args.omni_result)}
    if args.ppb_dir:
        sources["llm"] = load_ppb(args.ppb_dir)

    print("GT 테이블 수(소스별):",
          {k: len(v) for k, v in sources.items()},
          "| 매칭 수:",
          {k: sum(1 for x in v.values() if x["sha1"]) for k, v in sources.items()})

    all_disagreements = {}
    names = list(sources)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            stats, dis = compare(na, sources[na], nb, sources[nb])
            print(f"\n== {na} vs {nb}")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            all_disagreements[f"{na}_vs_{nb}"] = dis

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_disagreements, f, ensure_ascii=False, indent=1)
        n = sum(len(v) for v in all_disagreements.values())
        print(f"\n불일치 상세 {n}건 -> {args.output}")


if __name__ == "__main__":
    main()
