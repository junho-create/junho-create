#!/usr/bin/env python3
"""combined_e26 데이터셋을 combined_e25_28791 에서 처음부터 재현한다.

전 단계가 결정적(deterministic)이라 몇 번을 돌려도 같은 결과가 나온다.

파이프라인
---------
[0] 원본            combined_e25_28791                                  28,788
[1] ragged 제거     렌더링하면 사각형이 아닌 표가 있는 페이지 통째 제외    -268
[2] 수식 페이지 교체 Formula 태그 있는 페이지의 gt_html 을 dots.ocr 전체
                    페이지 예측으로 교체(행 수 변화 없음). 단 아래는 제외:
                      - reference_16886 출신(사람 라벨 GT라 신뢰)
                      - dots.ocr parse_note 있는 페이지(추론 자체가 불완전)
[3] 빈 수식 제거     dots.ocr 이 Formula 를 빈 내용으로 뱉은 페이지 제외.
                    단 KEEP_MARKER 페이지는 사용자가 직접 고치기로 해서 보존.

주의(둘 다 과거에 실제로 밟은 지뢰)
  - 컨텐츠 wrapping 은 손으로 짜지 말 것. data.build_chandra_dataset 의
    _content_for/_div 를 그대로 재사용해야 HTML escape(`&`->`&amp;`)와 맨 앞
    markdown heading(`## `) 제거가 GT 관례와 일치한다.
  - dots.ocr parse_note(truncated/unparsable) 페이지를 그냥 쓰면 잘린 예측이
    학습 GT 로 들어간다. 반드시 걸러야 한다.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/home/jhyeo/finetuning/vlm")
from data.build_chandra_dataset import _content_for, _div  # noqa: E402

SRC_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e25_28791")
DOTS_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out")
FORMULA_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out_formula_only")
RAGGED = Path("/tmp/claude-0/-home/37194d69-d7fb-46e5-b454-3e37f1455c1a/scratchpad/ragged_tables_confirmed.jsonl")
OUT_ROOT = Path("/home/jhyeo/finetuning/finetuning_dataset")

SPLITS = ("train", "valid", "test")
# 수식이 latex 없이 마크다운 텍스트로만 나왔지만, 사용자가 직접 고치기로 한 페이지.
KEEP_MARKER = "2025년 지방공무원보수업무 등 처리지침_page_384"


def load_ragged_pages() -> set[str]:
    pages = set()
    with RAGGED.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.add(json.loads(line)["image_path"])
    return pages


def load_dots_pages() -> dict[str, dict]:
    """image_path -> {"pred_blocks": [...], "parse_note": str|None}"""
    out = {}
    for split in SPLITS:
        with (DOTS_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[r["image_path"]] = {
                    "pred_blocks": r.get("pred_blocks", []),
                    "parse_note": r.get("parse_note"),
                }
    return out


def load_formula_pages() -> set[str]:
    pages = set()
    for split in SPLITS:
        with (FORMULA_DIR / f"gt_{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pages.add(json.loads(line)["image_path"])
    return pages


def empty_formula_pages(dots_pages: dict[str, dict]) -> set[str]:
    """dots.ocr 이 Formula 를 빈 내용으로 예측한 페이지."""
    pages = set()
    for img, page in dots_pages.items():
        for b in page["pred_blocks"]:
            if (b.get("label") or "").lower() != "formula":
                continue
            if not (b.get("content") or "").strip():
                pages.add(img)
                break
    return pages


def blocks_to_gt_html(blocks: list[dict]) -> str:
    divs = []
    for b in blocks:
        bbox = b.get("bbox_1000")
        if not bbox or len(bbox) != 4:
            continue
        label_raw = (b.get("label") or "").strip()
        if not label_raw:
            continue
        label = label_raw.capitalize()  # "page-header" -> "Page-header"
        divs.append(_div([int(v) for v in bbox], label, _content_for(label, b.get("content") or "")))
    return "\n".join(divs)


def main():
    ragged = load_ragged_pages()
    dots_pages = load_dots_pages()
    formula_pages = load_formula_pages()
    empty_pages = empty_formula_pages(dots_pages)

    print(f"[입력] ragged 페이지 {len(ragged)} / 수식 페이지 {len(formula_pages)} / "
          f"dots 빈수식 페이지 {len(empty_pages)}")

    result = {s: [] for s in SPLITS}
    tally = {
        "src": 0, "ragged_removed": 0, "replaced": 0,
        "skip_ref": 0, "skip_parse": 0, "skip_nodots": 0,
        "empty_removed": 0, "empty_kept": 0,
    }
    per_split = {}

    for split in SPLITS:
        n_src = n_ragged = n_repl = n_ref = n_parse = n_nodots = n_empty = 0
        with (SRC_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_src += 1
                r = json.loads(line)
                img = r["image_path"]

                # [1] ragged 페이지 제외
                if img in ragged:
                    n_ragged += 1
                    continue

                # [2] 수식 페이지를 dots.ocr 전체 페이지 예측으로 교체
                if img in formula_pages:
                    if "reference_16886" in img:
                        n_ref += 1
                    else:
                        page = dots_pages.get(img)
                        if not page:
                            n_nodots += 1
                        elif page["parse_note"]:
                            n_parse += 1
                        else:
                            # [3] 빈 수식 페이지는 제외(단 KEEP_MARKER 는 보존)
                            if img in empty_pages and KEEP_MARKER not in img:
                                n_empty += 1
                                continue
                            r["gt_html"] = blocks_to_gt_html(page["pred_blocks"])
                            n_repl += 1

                result[split].append(r)

        per_split[split] = {
            "src": n_src, "ragged": n_ragged, "replaced": n_repl,
            "skip_ref": n_ref, "skip_parse": n_parse, "skip_nodots": n_nodots,
            "empty_removed": n_empty, "final": len(result[split]),
        }
        tally["src"] += n_src
        tally["ragged_removed"] += n_ragged
        tally["replaced"] += n_repl
        tally["skip_ref"] += n_ref
        tally["skip_parse"] += n_parse
        tally["skip_nodots"] += n_nodots
        tally["empty_removed"] += n_empty

    total = sum(len(v) for v in result.values())
    out_dir = OUT_ROOT / f"combined_e26_{total}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for split in SPLITS:
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in result[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"{'split':<7} {'원본':>7} {'-ragged':>8} {'-빈수식':>8} {'=최종':>7} | "
          f"{'dots교체':>9} {'ref보존':>8} {'parse보존':>10}")
    for split in SPLITS:
        p = per_split[split]
        print(f"{split:<7} {p['src']:>7} {p['ragged']:>8} {p['empty_removed']:>8} {p['final']:>7} | "
              f"{p['replaced']:>9} {p['skip_ref']:>8} {p['skip_parse']:>10}")
    print(f"{'합계':<7} {tally['src']:>7} {tally['ragged_removed']:>8} "
          f"{tally['empty_removed']:>8} {total:>7} | "
          f"{tally['replaced']:>9} {tally['skip_ref']:>8} {tally['skip_parse']:>10}")

    assert tally["src"] - tally["ragged_removed"] - tally["empty_removed"] == total, "카운트 불일치"
    print(f"\n검산 OK: {tally['src']} - {tally['ragged_removed']} - {tally['empty_removed']} = {total}")
    print(f"-> {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
