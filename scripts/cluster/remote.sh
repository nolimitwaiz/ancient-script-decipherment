#!/usr/bin/env bash
# Run a command on the CLSP cluster: bash scripts/cluster/remote.sh '<command>'
set -euo pipefail
# local, gitignored overrides for cluster host/user (see README)
[ -f "$(dirname "$0")/../../.env.local" ] && . "$(dirname "$0")/../../.env.local"
host="${GLYPHOS_CLUSTER_HOST:-your-cluster-login-host}"
user="${GLYPHOS_CLUSTER_USER:-your-username}"
if [ $# -eq 0 ]; then
    echo "usage: remote.sh '<command to run on cluster>'" >&2
    exit 2
fi
exec ssh -o BatchMode=yes -o ConnectTimeout=10 "${user}@${host}" "$@"
