#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclude=cn3,cn18
#SBATCH --gres=gpu:8
#SBATCH --job-name=qb_loo_heads
#SBATCH --output=slurm-%x-%j.out

srun \
  --container-image=/scratch/amoskalenko/users/26f_pri/QualityBackbones/qualitybackbones.sqsh \
  --container-mounts=/scratch/amoskalenko/users/26f_pri/:/workdir:rw \
  bash <<'EOF'
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
REPO=/workdir/QualityBackbones
DATASETS_ROOT=/workdir/datasets/IQA
OUTPUTS_ROOT="${REPO}/outputs"
RUNS_ROOT="${OUTPUTS_ROOT}/training_runs_loo_all"
PLAN_PATH="${RUNS_ROOT}/plan.tsv"
GPUS=8
NUM_WORKERS=12
BATCH_SIZE=256
MAX_EPOCHS=10
TARGET_FIELD=normalized_score
MLP_DROPOUT=0.15
EARLY_STOPPING_PATIENCE=2
NUM_CHOICES=4
SEED=42
mkdir -p "${RUNS_ROOT}"
export REPO DATASETS_ROOT OUTPUTS_ROOT RUNS_ROOT PLAN_PATH
export GPUS NUM_WORKERS BATCH_SIZE MAX_EPOCHS TARGET_FIELD MLP_DROPOUT
export EARLY_STOPPING_PATIENCE NUM_CHOICES SEED
# 1) Build full run plan:
#    - only datasets that exist in DATASETS_ROOT and have embeddings in OUTPUTS_ROOT
#    - only features common across ALL datasets: (model, layer_name, dim)
#    - LOO splits across datasets
#    - head types: linear + mlp
#    - tasks: regression, pairwise(hard/soft), listwise(hard/soft)
python3 - <<'PY'
import csv
import json
import os
import re
from pathlib import Path
import h5py
datasets_root = Path(os.environ["DATASETS_ROOT"])
outputs_root = Path(os.environ["OUTPUTS_ROOT"])
plan_path = Path(os.environ["PLAN_PATH"])
skip_dirs = {"training_runs", "training_runs_loo_all", "maniqa_training_runs"}
dataset_models: dict[str, set[str]] = {}
for ds_dir in sorted(outputs_root.iterdir()):
    if not ds_dir.is_dir() or ds_dir.name in skip_dirs:
        continue
    if not (datasets_root / ds_dir.name / "data.csv").exists():
        continue
    models = set()
    for model_dir in ds_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if (
            (model_dir / "meta.json").exists()
            and (model_dir / "layers.h5").exists()
            and (model_dir / "index.parquet").exists()
        ):
            models.add(model_dir.name)
    if models:
        dataset_models[ds_dir.name] = models
datasets = sorted(dataset_models.keys())
if len(datasets) < 2:
    raise SystemExit(f"Need >=2 datasets with embeddings. Found: {datasets}")
common_models = set.intersection(*(dataset_models[d] for d in datasets))
if not common_models:
    raise SystemExit("No common models across all datasets.")
features: list[tuple[str, str, int]] = []
for model in sorted(common_models):
    per_dataset_dims: dict[str, dict[str, int]] = {}
    for ds in datasets:
        meta_path = outputs_root / ds / model / "meta.json"
        h5_path = outputs_root / ds / model / "layers.h5"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        layer_names = [str(x) for x in meta.get("layer_names", [])]
        dims: dict[str, int] = {}
        with h5py.File(h5_path, "r") as h5f:
            for idx, lname in enumerate(layer_names):
                key = f"layer_{idx:03d}"
                if key in h5f and len(h5f[key].shape) == 2:
                    dims[lname] = int(h5f[key].shape[1])
        per_dataset_dims[ds] = dims
    common_layer_names = set.intersection(*(set(v.keys()) for v in per_dataset_dims.values()))
    for lname in sorted(common_layer_names):
        dim_set = {per_dataset_dims[ds][lname] for ds in datasets}
        if len(dim_set) == 1:
            features.append((model, lname, next(iter(dim_set))))
if not features:
    raise SystemExit("No common (model, layer_name, dim) features across all datasets.")
tasks = [
    ("regression", "na"),
    ("pairwise", "hard"),
    ("pairwise", "soft"),
    ("listwise", "hard"),
    ("listwise", "soft"),
]
head_types = ["linear", "mlp"]


def sanitize_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._")
    return token or "layer"


rows = []
run_idx = 0
for val_ds in datasets:
    train_ds = [d for d in datasets if d != val_ds]
    train_sources = ",".join(train_ds)
    for model, layer_name, in_dim in features:
        layer_dir = sanitize_path_token(layer_name)
        feature_source = f"{model}:{layer_name}"
        for head_type in head_types:
            for task, target_mode in tasks:
                rows.append(
                    {
                        "run_idx": run_idx,
                        "val_dataset": val_ds,
                        "train_sources": train_sources,
                        "model_key": model,
                        "layer_name": layer_name,
                        "layer_dir": layer_dir,
                        "feature_source": feature_source,
                        "in_dim": in_dim,
                        "head_type": head_type,
                        "task": task,
                        "target_mode": target_mode,
                    }
                )
                run_idx += 1
plan_path.parent.mkdir(parents=True, exist_ok=True)
with plan_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"[plan] datasets={datasets}")
print(f"[plan] common_models={len(common_models)}")
print(f"[plan] common_features={len(features)}")
print(f"[plan] total_runs={len(rows)}")
print(f"[plan] plan_path={plan_path}")
PY
# 2) Execute plan on 8 GPUs in parallel (1 worker process per GPU)
python3 - <<'PY'
import csv
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path
repo = Path(os.environ["REPO"])
plan_path = Path(os.environ["PLAN_PATH"])
runs_root = Path(os.environ["RUNS_ROOT"])
datasets_root = os.environ["DATASETS_ROOT"]
outputs_root = os.environ["OUTPUTS_ROOT"]
gpus = int(os.environ["GPUS"])
num_workers = str(os.environ["NUM_WORKERS"])
batch_size = str(os.environ["BATCH_SIZE"])
max_epochs = str(os.environ["MAX_EPOCHS"])
target_field = os.environ["TARGET_FIELD"]
dropout = str(os.environ["MLP_DROPOUT"])
early_stopping_patience = str(os.environ["EARLY_STOPPING_PATIENCE"])
num_choices = str(os.environ["NUM_CHOICES"])
seed = str(os.environ["SEED"])
with plan_path.open("r", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f, delimiter="\t"))
if not all_rows:
    raise SystemExit("Empty plan.tsv")
shards = [all_rows[g::gpus] for g in range(gpus)]
def worker(gpu_id: int, rows: list[dict]) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    fail_path = runs_root / f"fails_gpu{gpu_id}.tsv"
    fails = []
    done = 0
    skipped = 0
    for row in rows:
        run_idx = int(row["run_idx"])
        run_name = f"loo_{run_idx:08d}"
        run_parent = runs_root / row["val_dataset"] / row["model_key"] / row["layer_dir"]
        run_dir = run_parent / run_name
        # idempotent restart: skip if already completed
        if (run_dir / "result.json").exists():
            skipped += 1
            continue
        cmd = [
            sys.executable, str(repo / "scripts" / "train_embeddings.py"),
            "--train-mode", "embeddings",
            "--datasets-root", datasets_root,
            "--outputs-root", outputs_root,
            "--output-dir", str(run_parent),
            "--run-name", run_name,
            "--feature-source", row["feature_source"],
            "--target-field", target_field,
            "--dropout", dropout,
            "--head-type", row["head_type"],
            "--task", row["task"],
            "--ranking-sampling", "within_dataset",  # pairs/lists only inside one dataset
            "--val-source", row["val_dataset"],
            "--batch-size", batch_size,
            "--num-workers", num_workers,
            "--max-epochs", max_epochs,
            "--early-stopping-patience", early_stopping_patience,
            "--accelerator", "gpu",
            "--devices", "1",
            "--precision", "bf16-mixed",
            "--seed", seed,
            "--no-tensorboard",
            "--no-save-val-predictions",
        ]
        for ds in row["train_sources"].split(","):
            ds = ds.strip()
            if ds:
                cmd += ["--train-source", ds]
        if row["task"] == "pairwise":
            cmd += ["--pairwise-target", row["target_mode"]]
        elif row["task"] == "listwise":
            cmd += ["--listwise-target", row["target_mode"], "--num-choices", num_choices]
        print(
            f"[GPU {gpu_id}] run_idx={run_idx} val={row['val_dataset']} "
            f"feat={row['feature_source']} head={row['head_type']} "
            f"task={row['task']} target={row['target_mode']} run_dir={run_dir}",
            flush=True,
        )
        rc = subprocess.call(cmd, env=env)
        done += 1
        if rc != 0:
            fails.append((run_idx, rc, row))
    if fails:
        with fail_path.open("w", encoding="utf-8") as fp:
            fp.write(
                "run_idx\trc\tval_dataset\tmodel_key\tlayer_name\tfeature_source\t"
                "head_type\ttask\ttarget_mode\n"
            )
            for run_idx, rc, row in fails:
                fp.write(
                    f"{run_idx}\t{rc}\t{row['val_dataset']}\t{row['model_key']}\t"
                    f"{row['layer_name']}\t{row['feature_source']}\t"
                    f"{row['head_type']}\t{row['task']}\t{row['target_mode']}\n"
                )
        print(f"[GPU {gpu_id}] done={done}, skipped={skipped}, failed={len(fails)} -> {fail_path}", flush=True)
        raise SystemExit(1)
    print(f"[GPU {gpu_id}] done={done}, skipped={skipped}, failed=0", flush=True)
procs = []
for gpu_id, shard in enumerate(shards):
    p = mp.Process(target=worker, args=(gpu_id, shard))
    p.start()
    procs.append(p)
bad = 0
for p in procs:
    p.join()
    if p.exitcode != 0:
        bad += 1
if bad > 0:
    raise SystemExit(f"{bad} worker(s) reported failures. See fails_gpu*.tsv in {runs_root}")
print("[done] all workers finished without failures")
PY
EOF
