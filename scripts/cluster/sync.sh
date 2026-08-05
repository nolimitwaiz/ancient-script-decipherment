#!/usr/bin/env bash
# Sync repo + data to the cluster (rsync, delete-safe: never deletes remote
# runs/ outputs). Data raw trees go too (they are the reproducibility record);
# the HF cache is excluded (regenerable).
set -euo pipefail
host="${GLYPHOS_CLUSTER_HOST:-your-cluster-login-host}"
user="${GLYPHOS_CLUSTER_USER:-your-username}"
root="${GLYPHOS_CLUSTER_ROOT:-glyphos}"   # relative to remote $HOME unless absolute
repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
data_dir="$(cd "$repo_dir/data" && pwd -P)"   # resolves the symlink

echo "[sync] repo -> ${user}@${host}:${root}"
rsync -az --stats \
    --exclude '.git/' --exclude '.venv' --exclude '/data' \
    --exclude 'runs/smoke/' --exclude '__pycache__' --exclude '.pytest_cache' \
    --exclude '.ruff_cache' \
    "$repo_dir/" "${user}@${host}:${root}/"

echo "[sync] data (${data_dir}) -> ${user}@${host}:${root}-data"
rsync -az --stats \
    --exclude '_hf_cache/' \
    "$data_dir/" "${user}@${host}:${root}-data/"

echo "[sync] linking data dir on cluster"
ssh "${user}@${host}" "cd ${root} && ln -sfn ../${root}-data data && ls -la data"
echo "[sync] pulling cluster ledger shards"
mkdir -p "$repo_dir/runs/ledger.d"
rsync -az --ignore-existing \
    "${user}@${host}:${root}/runs/ledger.d/" "$repo_dir/runs/ledger.d/" 2>/dev/null || true
echo "[sync] done (run 'uv run glyphos-ledger compact' to fold shards in)"
