#!/usr/bin/env bash
# Bootstrap the Decipherment Bench environment ON the cluster (run via remote.sh).
set -euo pipefail
cd "${GLYPHOS_CLUSTER_ROOT:-$HOME/glyphos}"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "[setup] installing uv (user-level)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "[setup] python + deps"
uv sync
echo "[setup] verifying (lint + tests + gate + smoke)"
make check
echo "[setup] cluster environment ready: $(pwd)"
