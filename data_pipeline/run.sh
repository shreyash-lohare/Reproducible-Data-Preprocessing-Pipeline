#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-config.yaml}

# Prefer python3 when plain python is unavailable (common in WSL)
PYTHON_BIN=${PYTHON_BIN:-python}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN=python3

"$PYTHON_BIN" -m src.validate --config "$CONFIG_PATH"
"$PYTHON_BIN" -m src.preprocess --config "$CONFIG_PATH"
