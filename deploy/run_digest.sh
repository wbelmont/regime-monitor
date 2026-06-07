#!/bin/bash
# Wrapper that launchd runs each morning to send the iMessage digest.
# It refreshes data (live), computes today's signal, and sends a change-gated
# digest via iMessage. Output is appended to reports/digest.log for debugging.
#
# Edit PROJECT_DIR if you move the repo. The venv python is used directly so no
# shell activation is needed (launchd runs with a minimal environment).

set -euo pipefail

PROJECT_DIR="${HOME}/Desktop/regime-monitor"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
LOG="${PROJECT_DIR}/reports/digest.log"

cd "${PROJECT_DIR}"

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') : regime digest ====="
  # Live data (drop --no-refresh for the real morning run). The digest is
  # change-gated by default, so it only texts you when something moves.
  PYTHONPATH=. "${PYTHON}" -m regime.cli digest
  # Also refresh the static dashboard so the hosted/iCloud copy is current.
  PYTHONPATH=. "${PYTHON}" -m regime.cli dashboard --no-log
  echo
} >> "${LOG}" 2>&1
