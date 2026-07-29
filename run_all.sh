#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
python3 "$PACKAGE_DIR/code/run_all.py"
python3 -m unittest discover -s "$PACKAGE_DIR/tests" -v
python3 "$PACKAGE_DIR/code/verify_hashes.py"
