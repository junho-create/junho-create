#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Server-only sampler image organizer

What it does (on training server only):
1) Copy sampled JSONL -> <out_dir>/train_raw.server.jsonl
2) Build:
   - <out_dir>/train_raw.jsonl                  (image_path => images/<basename>)
   - <out_dir>/image_paths_from_train_raw.txt
   - <out_dir>/image_paths_from_train_raw.unique.txt
3) Copy matched images into <out_dir>/images/
4) Validate row/image consistency and write <out_dir>/sync_report.json

Usage:
  bash scripts/sync_sampled_images_from_server.sh \
    --project_root /home/vlm_train/qwen3_vl_tsr \
    --input_jsonl data/processed/from_server/<run>/train_raw.jsonl \
    --out_dir _train_data/<target_dir>

Options:
  --project_root   Project root used to resolve relative image paths
                   (default: /home/vlm_train/qwen3_vl_tsr)
  --input_jsonl    Sampled JSONL path (required)
  --out_dir        Output dataset dir (required)
  --overwrite      Overwrite existing files in out_dir/images when names collide
  -h, --help       Show this help
USAGE
}

PROJECT_ROOT="/home/vlm_train/qwen3_vl_tsr"
INPUT_JSONL=""
OUT_DIR=""
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project_root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --input_jsonl)
      INPUT_JSONL="$2"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT_JSONL}" || -z "${OUT_DIR}" ]]; then
  echo "[ERROR] --input_jsonl and --out_dir are required." >&2
  usage
  exit 1
fi

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "[ERROR] Input JSONL not found: ${INPUT_JSONL}" >&2
  exit 1
fi
if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "[ERROR] project_root not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/images"

SERVER_JSONL="${OUT_DIR}/train_raw.server.jsonl"
NORMALIZED_JSONL="${OUT_DIR}/train_raw.jsonl"
ALL_LIST="${OUT_DIR}/image_paths_from_train_raw.txt"
UNIQ_LIST="${OUT_DIR}/image_paths_from_train_raw.unique.txt"
REPORT_JSON="${OUT_DIR}/sync_report.json"

if [[ "$(realpath "${INPUT_JSONL}")" != "$(realpath -m "${SERVER_JSONL}")" ]]; then
  cp -f "${INPUT_JSONL}" "${SERVER_JSONL}"
fi

echo "[INFO] Build manifests + normalized JSONL"
PROJECT_ROOT="${PROJECT_ROOT}" SERVER_JSONL="${SERVER_JSONL}" NORMALIZED_JSONL="${NORMALIZED_JSONL}" \
ALL_LIST="${ALL_LIST}" UNIQ_LIST="${UNIQ_LIST}" python3 - <<'PY'
import json
import os
import re
from pathlib import Path

server_jsonl = Path(os.environ["SERVER_JSONL"])
normalized_jsonl = Path(os.environ["NORMALIZED_JSONL"])
all_list = Path(os.environ["ALL_LIST"])
uniq_list = Path(os.environ["UNIQ_LIST"])

def basename_from_path(path_like: str) -> str:
    s = str(path_like or "").strip().replace("\\", "/")
    if not s:
        return ""
    s = re.split(r"[?#]", s, maxsplit=1)[0]
    return s.rsplit("/", 1)[-1]

seen = set()
basename_to_src = {}
rows = 0
errors = 0

with server_jsonl.open("r", encoding="utf-8") as fi, \
     normalized_jsonl.open("w", encoding="utf-8") as fo, \
     all_list.open("w", encoding="utf-8") as fa, \
     uniq_list.open("w", encoding="utf-8") as fu:
    for raw in fi:
        line = raw.strip()
        if not line:
            continue
        rows += 1
        rec = json.loads(line)
        src = str(rec.get("image_path", "")).strip()
        if not src:
            errors += 1
            continue

        b = basename_from_path(src)
        if not b:
            errors += 1
            continue

        # same basename mapping to different source path is unsafe for flatten layout
        prev = basename_to_src.get(b)
        if prev is not None and prev != src:
            raise RuntimeError(
                f"basename collision: {b}\n  prev={prev}\n  curr={src}"
            )
        basename_to_src[b] = src

        fa.write(src + "\n")
        if src not in seen:
            seen.add(src)
            fu.write(src + "\n")

        rec["image_path"] = f"images/{b}"
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"rows={rows}, unique_paths={len(seen)}, errors={errors}")
print(f"wrote: {normalized_jsonl}")
print(f"wrote: {uniq_list}")
PY

echo "[INFO] Copy matched images into ${OUT_DIR}/images"
PROJECT_ROOT="${PROJECT_ROOT}" OUT_DIR="${OUT_DIR}" UNIQ_LIST="${UNIQ_LIST}" OVERWRITE="${OVERWRITE}" \
REPORT_JSON="${REPORT_JSON}" python3 - <<'PY'
import json
import os
import re
import shutil
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
out_dir = Path(os.environ["OUT_DIR"]).resolve()
uniq_list = Path(os.environ["UNIQ_LIST"]).resolve()
images_dir = out_dir / "images"
overwrite = os.environ["OVERWRITE"] == "1"
report_json = Path(os.environ["REPORT_JSON"]).resolve()

def basename_from_path(path_like: str) -> str:
    s = str(path_like or "").strip().replace("\\", "/")
    if not s:
        return ""
    s = re.split(r"[?#]", s, maxsplit=1)[0]
    return s.rsplit("/", 1)[-1]

def resolve_source(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw
    return project_root / raw

copied = 0
skipped_existing = 0
missing = 0
missing_paths = []

for line in uniq_list.read_text(encoding="utf-8").splitlines():
    src_raw = line.strip()
    if not src_raw:
        continue
    src = resolve_source(src_raw)
    dst = images_dir / basename_from_path(src_raw)

    if not src.exists():
        missing += 1
        missing_paths.append(src_raw)
        continue

    if dst.exists() and not overwrite:
        skipped_existing += 1
        continue

    shutil.copy2(src, dst)
    copied += 1

report = {
    "project_root": str(project_root),
    "out_dir": str(out_dir),
    "images_dir": str(images_dir),
    "copied": copied,
    "skipped_existing": skipped_existing,
    "missing": missing,
    "missing_paths_preview": missing_paths[:30],
}
report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

echo "[INFO] Validate normalized JSONL image paths"
OUT_DIR="${OUT_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
jsonl = out_dir / "train_raw.jsonl"
rows = 0
missing = 0
for raw in jsonl.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line:
        continue
    rows += 1
    rec = json.loads(line)
    p = out_dir / rec.get("image_path", "")
    if not p.exists():
        missing += 1

print(f"rows={rows}, missing_images={missing}")
PY

echo "[DONE] Organized sampled JSONL + matched images in: ${OUT_DIR}"
