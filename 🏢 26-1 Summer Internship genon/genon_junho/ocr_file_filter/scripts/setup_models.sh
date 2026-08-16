#!/usr/bin/env bash
# CMCV 3모델 원스톱 브링업.
#
# 인터넷 되는 머신에서:   ./scripts/setup_models.sh download
# 폐쇄망 이동이 필요하면:  ./scripts/setup_models.sh download && \
#                          python -m ocr_filter.cli models bundle _models_bundle.tar.gz
#                          # → scp 로 대상 서버에 옮기고 tar xzf 로 ./_models 복원
# 서버(GPU)에서:          ./scripts/setup_models.sh serve && ./scripts/setup_models.sh status
set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:-help}"; shift || true
case "$CMD" in
  download) python -m ocr_filter.cli models download "$@" ;;
  bundle)   python -m ocr_filter.cli models bundle "${1:-_models_bundle.tar.gz}" ;;
  serve)    python -m ocr_filter.cli models serve "$@" ;;
  status)   python -m ocr_filter.cli models status ;;
  stop)     python -m ocr_filter.cli models stop "$@" ;;
  *) echo "usage: $0 {download|bundle|serve|status|stop} [--only ...]"; exit 1 ;;
esac
