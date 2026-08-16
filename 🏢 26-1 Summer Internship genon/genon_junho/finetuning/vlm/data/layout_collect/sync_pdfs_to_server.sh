#!/usr/bin/env bash
# 로컬 Mac에서 실행: layout_data에서 N페이지 분량 PDF만 선별 → 174 서버 전송
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/Users/jae_hyeok_shin/tsr_test/layout_data}"
TARGET_PAGES="${TARGET_PAGES:-10000}"
BUFFER_RATIO="${BUFFER_RATIO:-1.05}"
CLEAR_REMOTE="${CLEAR_REMOTE:-1}"
REMOTE="root@192.168.75.174"
REMOTE_PORT="${REMOTE_PORT:-2222}"
REMOTE_DIR="${REMOTE_DIR:-/home/vlm_train/labeler_feat/input/pdfs_batch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST:-${SCRIPT_DIR}/pdf_subset.manifest.txt}"
SUMMARY="${SUMMARY:-${SCRIPT_DIR}/pdf_subset.summary.json}"

echo "[1/3] Select PDFs for ~${TARGET_PAGES} pages (buffer=${BUFFER_RATIO})"
python3 "${SCRIPT_DIR}/select_pdf_subset.py" \
  --root "${LOCAL_ROOT}" \
  --target-pages "${TARGET_PAGES}" \
  --buffer-ratio "${BUFFER_RATIO}" \
  --manifest "${MANIFEST}" \
  --summary "${SUMMARY}"

SELECTED="$(python3 -c "import json; print(json.load(open('${SUMMARY}'))['selected_pdf_count'])")"
PAGES="$(python3 -c "import json; print(json.load(open('${SUMMARY}'))['selected_page_count'])")"
echo "  → ${SELECTED} PDFs, ${PAGES} pages"

if [[ "${CLEAR_REMOTE}" == "1" ]]; then
  echo "[2/3] Clear remote input dir: ${REMOTE_DIR}"
  ssh -p "${REMOTE_PORT}" "${REMOTE}" "rm -rf '${REMOTE_DIR}' && mkdir -p '${REMOTE_DIR}'"
else
  echo "[2/3] Keep remote dir (CLEAR_REMOTE=0)"
  ssh -p "${REMOTE_PORT}" "${REMOTE}" "mkdir -p '${REMOTE_DIR}'"
fi

echo "[3/3] Rsync selected PDFs only"
rsync -avP \
  --files-from="${MANIFEST}" \
  -e "ssh -p ${REMOTE_PORT}" \
  "${LOCAL_ROOT}/" \
  "${REMOTE}:${REMOTE_DIR}/"

echo "Remote PDF count:"
ssh -p "${REMOTE_PORT}" "${REMOTE}" \
  "find '${REMOTE_DIR}' -type f \\( -iname '*.pdf' \\) | wc -l; du -sh '${REMOTE_DIR}'"
