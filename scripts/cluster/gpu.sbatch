#!/usr/bin/env bash
# GPU job template: sbatch scripts/cluster/gpu.sbatch <command...>
#
# GPU policy (CONVENTIONS.md): TYPELESS request — SLURM allocates whatever GPU is
# free (A100 or otherwise); never queue-wait on a specific card. Scale with
#   sbatch --gres=gpu:4 scripts/cluster/gpu.sbatch ...
# and pin a type ONLY when genuinely required:
#   sbatch --gres=gpu:a100:2 scripts/cluster/gpu.sbatch ...
# Training code must run on any CUDA generation (bf16 with fp32 fallback) and
# any world size — the job may land on anything.
#SBATCH --job-name=glyphos-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
cd "${GLYPHOS_CLUSTER_ROOT:-$HOME/glyphos}"
mkdir -p runs
echo "[job $SLURM_JOB_ID] $(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} :: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
# No silent fallbacks: a GPU job that cannot see CUDA must die, not train on CPU.
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable on this node/driver'" || exit 1
exec "$@"
