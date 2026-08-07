# Cluster execution (CLSP, SLURM)

Policy (CONVENTIONS.md): the cluster is the DEFAULT for anything heavy — CPU jobs
included. The Mac does editing, unit tests, and `make smoke`; multi-minute
data processing, bulk rendering, and all training go through SLURM.

GPU policy: request GPUs TYPELESS (`--gres=gpu:N`) so SLURM allocates
whatever is free — A100 or otherwise; never queue-wait on a specific card.
Pin a type only when a job genuinely needs it (`--gres=gpu:a100:N`).

## One-time setup

1. Authorize the Mac's key (interactive, run locally so output
   lands in-chat):

       ! ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@your-cluster-login-host

2. Sync repo + data and bootstrap the environment:

       make cluster-sync
       bash scripts/cluster/remote.sh 'bash ~/glyphos/scripts/cluster/setup_env.sh'

Environment knobs (all optional):
- `GLYPHOS_CLUSTER_HOST` (default `your-cluster-login-host`)
- `GLYPHOS_CLUSTER_USER` (default: local username)
- `GLYPHOS_CLUSTER_ROOT` (default `~/glyphos` on the cluster; point the data
  dir at scratch/export storage if home quota is tight — see sync.sh)

## Running work

    # multi-CPU data/render job (16 cores default)
    bash scripts/cluster/remote.sh 'cd ~/glyphos && sbatch scripts/cluster/cpu.sbatch \
        uv run python scripts/prepare_data.py split --corpus greek_first1k --all-schemes'

    # any-free-GPU job (Phase 3+)
    bash scripts/cluster/remote.sh 'cd ~/glyphos && sbatch scripts/cluster/gpu.sbatch \
        uv run python scripts/train.py --config configs/...'

    # watch
    bash scripts/cluster/remote.sh 'squeue -u $USER; tail -5 ~/glyphos/runs/slurm-*.out'

The ledger lives in git (`runs/ledger.jsonl`, union-merged), so cluster runs
register/complete exactly like local ones — commit and pull between machines.
