#!/usr/bin/env bash
# Multi-CPU job template: sbatch scripts/cluster/cpu.sbatch <command...>
# Data prep, bulk rendering, corpus processing — no GPU.
#SBATCH --job-name=glyphos-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${GLYPHOS_CLUSTER_ROOT:-$HOME/glyphos}"
mkdir -p runs
echo "[job $SLURM_JOB_ID] $(hostname) cpus=$SLURM_CPUS_PER_TASK :: $*"
exec "$@"
