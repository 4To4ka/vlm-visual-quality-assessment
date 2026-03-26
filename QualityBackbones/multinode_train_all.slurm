#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=cn3,cn18
#SBATCH --gres=gpu:8
#SBATCH --job-name=qb_loo_heads_multi
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

usage() {
  printf 'Usage: %s [--reuse-plan]\n' "$0"
}

REUSE_PLAN=0
while (($#)); do
  case "$1" in
    --reuse-plan)
      REUSE_PLAN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

CONTAINER_IMAGE=/scratch/amoskalenko/users/26f_pri/QualityBackbones/qualitybackbones.sqsh
CONTAINER_MOUNTS=/scratch/amoskalenko/users/26f_pri/:/workdir:rw
REPO=/workdir/QualityBackbones
DATASETS_ROOT=/workdir/datasets/IQA
OUTPUTS_ROOT="${REPO}/outputs"
RUNS_ROOT="${OUTPUTS_ROOT}/training_runs_loo_all"
PLAN_PATH="${RUNS_ROOT}/plan.tsv"
WORKERS_PER_NODE=8
TOTAL_WORKERS="${SLURM_NTASKS:-$(( ${SLURM_JOB_NUM_NODES:-1} * WORKERS_PER_NODE ))}"
NUM_WORKERS=12
BATCH_SIZE=256
MAX_EPOCHS=10
TARGET_FIELD=normalized_score
MLP_DROPOUT=0.15
EARLY_STOPPING_PATIENCE=2
NUM_CHOICES=4
SEED=42

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export REPO DATASETS_ROOT OUTPUTS_ROOT RUNS_ROOT PLAN_PATH
export NUM_WORKERS BATCH_SIZE MAX_EPOCHS TARGET_FIELD MLP_DROPOUT
export EARLY_STOPPING_PATIENCE NUM_CHOICES SEED TOTAL_WORKERS

printf '[launch] nodes=%s total_workers=%s reuse_plan=%s runs_root=%s\n' \
  "${SLURM_JOB_NUM_NODES:-1}" "${TOTAL_WORKERS}" "${REUSE_PLAN}" "${RUNS_ROOT}"

if [[ "${REUSE_PLAN}" == "1" ]]; then
  srun \
    --nodes=1 \
    --ntasks=1 \
    --exact \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    python3 "${REPO}/scripts/train_all_loo.py" check-plan
else
  srun \
    --nodes=1 \
    --ntasks=1 \
    --exact \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    python3 "${REPO}/scripts/train_all_loo.py" build-plan
fi

srun \
  --ntasks="${TOTAL_WORKERS}" \
  --ntasks-per-node="${WORKERS_PER_NODE}" \
  --exact \
  --kill-on-bad-exit=1 \
  --label \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  python3 "${REPO}/scripts/train_all_loo.py" run-shard
