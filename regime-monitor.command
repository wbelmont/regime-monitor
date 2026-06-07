#!/bin/bash
# Double-clickable launcher for the regime monitor (macOS).
# Runs the live source directly with the venv's Python — no install/rebuild
# needed, so your latest code edits always take effect.

# Resolve this script's directory (the project root), regardless of where it's
# launched from.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

# Put the project root on the path so `import regime` always works, then run.
PYTHONPATH="$PROJECT_DIR" "$PROJECT_DIR/.venv/bin/python3" -m regime.cli "${@:-update}"

# Keep the window open when double-clicked from Finder.
if [ -z "$1" ]; then
    echo
    read -n 1 -s -r -p "Press any key to close..."
fi
