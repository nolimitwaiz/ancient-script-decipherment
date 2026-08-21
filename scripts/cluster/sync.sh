#!/usr/bin/env bash
# Sync repo + data to the cluster (rsync, delete-safe: never deletes remote
# runs/ outputs). Data raw trees go too (they are the reproducibility record);
# the HF cache is excluded (regenerable).
set -euo pipefail
# local, gitignored overrides for cluster host/user (see README)
[ -f "$(dirname "$0")/../../.env.local" ] && . "$(dirname "$0")/../../.env.local"
host="${GLYPHOS_CLUSTER_HOST:-your-cluster-login-host}"
user="${GLYPHOS_CLUSTER_USER:-your-username}"
root="${GLYPHOS_CLUSTER_ROOT:-glyphos}"   # relative to remote $HOME unless absolute
repo_dir="$(cd "$(dirname "$0")/../.." && pwd)"
data_dir="$(cd "$repo_dir/data" && pwd -P)"   # resolves the symlink

echo "[sync] repo -> ${user}@${host}:${root}"
rsync -az --stats \
    --exclude '.git/' --exclude '.venv' --exclude '/data' \
    --exclude '/runs/' \
    --exclude '__pycache__' --exclude '.pytest_cache' \
    --exclude '.ruff_cache' \
    "$repo_dir/" "${user}@${host}:${root}/"

echo "[sync] data (${data_dir}) -> ${user}@${host}:${root}-data"
rsync -az --stats \
    --exclude '_hf_cache/' \
    "$data_dir/" "${user}@${host}:${root}-data/"

echo "[sync] linking data dir on cluster"
ssh "${user}@${host}" "cd ${root} && ln -sfn ../${root}-data data && ls -la data"
# NEVER push runs/: the cluster ledger is authoritative for cluster runs.
# (A push once overwrote 12 compacted events — 2026-08-06.) Pull only.
echo "[sync] pulling cluster ledger (shards + canonical)"
mkdir -p "$repo_dir/runs/ledger.d"
rsync -az --ignore-existing \
    "${user}@${host}:${root}/runs/ledger.d/" "$repo_dir/runs/ledger.d/" 2>/dev/null || true
ssh "${user}@${host}" "cat ${root}/runs/ledger.jsonl 2>/dev/null" \
    > "$repo_dir/runs/.cluster_ledger.jsonl" 2>/dev/null || true
python3 - "$repo_dir" <<'PY'
import sys, pathlib
repo = pathlib.Path(sys.argv[1]); main = repo/"runs/ledger.jsonl"
remote = repo/"runs/.cluster_ledger.jsonl"
if remote.exists():
    have = set(main.read_text().splitlines()) if main.exists() else set()
    new = [l for l in remote.read_text().splitlines() if l.strip() and l not in have]
    if new:
        with open(main, "a") as f: f.write("\n".join(new) + "\n")
    print(f"[sync] merged {len(new)} cluster ledger event(s)")
    remote.unlink()
PY
echo "[sync] done (run 'uv run glyphos-ledger compact' to fold shards in)"
