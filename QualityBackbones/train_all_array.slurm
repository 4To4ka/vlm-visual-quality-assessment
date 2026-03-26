#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclude=cn3,cn18
#SBATCH --gres=gpu:8
#SBATCH --job-name=qb_loo_heads_array
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

usage() {
  printf 'Usage: %s [--reuse-plan]\n' "$0"
  printf 'Submit with sbatch, optionally using --array=START-END%%MAX.\n'
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

HOST_WORKDIR=/scratch/amoskalenko/users/26f_pri
HOST_REPO="${HOST_WORKDIR}/QualityBackbones"
CONTAINER_IMAGE="${HOST_REPO}/qualitybackbones.sqsh"
CONTAINER_MOUNTS="${HOST_WORKDIR}:/workdir:rw"
REPO=/workdir/QualityBackbones
DATASETS_ROOT=/workdir/datasets/IQA
OUTPUTS_ROOT="${REPO}/outputs"
RUNS_ROOT="${OUTPUTS_ROOT}/training_runs_loo_all"
PLAN_PATH="${RUNS_ROOT}/plan.tsv"
PLAN_READY_PATH="${RUNS_ROOT}/plan.ready.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
WORKERS_PER_TASK=8
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
TOTAL_WORKERS="$(( ARRAY_TASK_COUNT * WORKERS_PER_TASK ))"
PLAN_WAIT_TIMEOUT=1800
PLAN_WAIT_INTERVAL=5
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
export REPO DATASETS_ROOT OUTPUTS_ROOT RUNS_ROOT PLAN_PATH PLAN_READY_PATH
export WORKERS_PER_TASK ARRAY_TASK_ID ARRAY_TASK_COUNT TOTAL_WORKERS
export PLAN_WAIT_TIMEOUT PLAN_WAIT_INTERVAL
export NUM_WORKERS BATCH_SIZE MAX_EPOCHS TARGET_FIELD MLP_DROPOUT
export EARLY_STOPPING_PATIENCE NUM_CHOICES SEED

printf '[launch] array_task=%s/%s workers_per_task=%s total_workers=%s reuse_plan=%s runs_root=%s\n' \
  "${ARRAY_TASK_ID}" "${ARRAY_TASK_COUNT}" "${WORKERS_PER_TASK}" "${TOTAL_WORKERS}" "${REUSE_PLAN}" "${RUNS_ROOT}"

RUN_CMD=(python3 "${REPO}/scripts/train_all_loo.py" run-array-task)
if [[ "${REUSE_PLAN}" == "1" ]]; then
  RUN_CMD+=(--reuse-plan)
fi

srun \
  --nodes=1 \
  --ntasks=1 \
  --exact \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  "${RUN_CMD[@]}"
