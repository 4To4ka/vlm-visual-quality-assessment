from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


TASKS: list[tuple[str, str]] = [
    ("regression", "na"),
    ("pairwise", "hard"),
    ("pairwise", "soft"),
    ("listwise", "hard"),
    ("listwise", "soft"),
]
HEAD_TYPES = ["linear", "mlp"]
SKIP_DIRS = {"training_runs", "training_runs_loo_all", "maniqa_training_runs"}


@dataclass(frozen=True)
class LauncherConfig:
    repo: Path
    datasets_root: Path
    outputs_root: Path
    runs_root: Path
    plan_path: Path
    plan_ready_path: Path | None
    batch_size: int
    num_workers: int
    max_epochs: int
    target_field: str
    dropout: float
    early_stopping_patience: int
    num_choices: int
    seed: int


def sanitize_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._")
    return token or "layer"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def load_config_from_env() -> LauncherConfig:
    plan_ready_raw = os.environ.get("PLAN_READY_PATH")
    return LauncherConfig(
        repo=Path(_require_env("REPO")),
        datasets_root=Path(_require_env("DATASETS_ROOT")),
        outputs_root=Path(_require_env("OUTPUTS_ROOT")),
        runs_root=Path(_require_env("RUNS_ROOT")),
        plan_path=Path(_require_env("PLAN_PATH")),
        plan_ready_path=Path(plan_ready_raw) if plan_ready_raw else None,
        batch_size=int(_require_env("BATCH_SIZE")),
        num_workers=int(_require_env("NUM_WORKERS")),
        max_epochs=int(_require_env("MAX_EPOCHS")),
        target_field=_require_env("TARGET_FIELD"),
        dropout=float(_require_env("MLP_DROPOUT")),
        early_stopping_patience=int(_require_env("EARLY_STOPPING_PATIENCE")),
        num_choices=int(_require_env("NUM_CHOICES")),
        seed=int(_require_env("SEED")),
    )


def build_plan(config: LauncherConfig) -> None:
    import h5py

    dataset_models: dict[str, set[str]] = {}
    for ds_dir in sorted(config.outputs_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name in SKIP_DIRS:
            continue
        if not (config.datasets_root / ds_dir.name / "data.csv").exists():
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

    datasets = sorted(dataset_models)
    if len(datasets) < 2:
        raise SystemExit(f"Need >=2 datasets with embeddings. Found: {datasets}")

    common_models = set.intersection(*(dataset_models[dataset] for dataset in datasets))
    if not common_models:
        raise SystemExit("No common models across all datasets.")

    features: list[tuple[str, str, int]] = []
    for model in sorted(common_models):
        per_dataset_dims: dict[str, dict[str, int]] = {}
        for dataset in datasets:
            meta_path = config.outputs_root / dataset / model / "meta.json"
            h5_path = config.outputs_root / dataset / model / "layers.h5"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            layer_names = [str(item) for item in meta.get("layer_names", [])]
            dims: dict[str, int] = {}
            with h5py.File(h5_path, "r") as handle:
                for idx, layer_name in enumerate(layer_names):
                    key = f"layer_{idx:03d}"
                    if key in handle and len(handle[key].shape) == 2:
                        dims[layer_name] = int(handle[key].shape[1])
            per_dataset_dims[dataset] = dims

        common_layer_names = set.intersection(*(set(value) for value in per_dataset_dims.values()))
        for layer_name in sorted(common_layer_names):
            dim_set = {per_dataset_dims[dataset][layer_name] for dataset in datasets}
            if len(dim_set) == 1:
                features.append((model, layer_name, next(iter(dim_set))))

    if not features:
        raise SystemExit("No common (model, layer_name, dim) features across all datasets.")

    rows: list[dict[str, str | int]] = []
    run_idx = 0
    for val_dataset in datasets:
        train_datasets = [dataset for dataset in datasets if dataset != val_dataset]
        train_sources = ",".join(train_datasets)
        for model, layer_name, in_dim in features:
            layer_dir = sanitize_path_token(layer_name)
            feature_source = f"{model}:{layer_name}"
            for head_type in HEAD_TYPES:
                for task, target_mode in TASKS:
                    rows.append(
                        {
                            "run_idx": run_idx,
                            "val_dataset": val_dataset,
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

    config.plan_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config.plan_path.with_name(f"{config.plan_path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(config.plan_path)

    print(f"[plan] datasets={datasets}")
    print(f"[plan] common_models={len(common_models)}")
    print(f"[plan] common_features={len(features)}")
    print(f"[plan] total_runs={len(rows)}")
    print(f"[plan] plan_path={config.plan_path}")


def load_plan(plan_path: Path) -> list[dict[str, str]]:
    with plan_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit("Empty plan.tsv")
    return rows


def check_plan(config: LauncherConfig) -> int:
    rows = load_plan(config.plan_path)
    print(f"[plan] reusing existing plan at {config.plan_path}")
    print(f"[plan] total_runs={len(rows)}")
    return len(rows)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def write_plan_ready_marker(
    config: LauncherConfig,
    *,
    source: str,
    total_workers: int,
    array_task_id: int,
    array_task_count: int,
) -> None:
    if config.plan_ready_path is None:
        return
    payload = {
        "source": source,
        "plan_path": str(config.plan_path),
        "total_workers": total_workers,
        "array_task_id": array_task_id,
        "array_task_count": array_task_count,
    }
    _write_text_atomic(config.plan_ready_path, json.dumps(payload, indent=2))


def wait_for_plan(
    config: LauncherConfig,
    *,
    timeout_seconds: int,
    poll_interval_seconds: float,
    require_ready_marker: bool,
) -> int:
    if timeout_seconds <= 0:
        raise SystemExit("plan wait timeout must be > 0 seconds")
    if poll_interval_seconds <= 0:
        raise SystemExit("plan wait interval must be > 0 seconds")

    start_time = time.monotonic()
    waiting_message_printed = False
    ready_target = config.plan_ready_path if require_ready_marker else config.plan_path
    while True:
        plan_exists = config.plan_path.exists() and config.plan_path.stat().st_size > 0
        marker_exists = True
        if require_ready_marker:
            marker_exists = (
                config.plan_ready_path is not None
                and config.plan_ready_path.exists()
                and config.plan_ready_path.stat().st_size > 0
            )
        if plan_exists and marker_exists:
            return len(load_plan(config.plan_path))

        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_seconds:
            raise SystemExit(
                "Timed out waiting for the shared plan to become ready: "
                f"plan={config.plan_path} ready_marker={ready_target}"
            )
        if not waiting_message_printed:
            print(
                f"[plan] waiting for shared plan: plan={config.plan_path} ready_marker={ready_target}",
                flush=True,
            )
            waiting_message_printed = True
        time.sleep(poll_interval_seconds)


def build_train_command(config: LauncherConfig, row: dict[str, str], run_parent: Path) -> list[str]:
    command = [
        sys.executable,
        str(config.repo / "scripts" / "train_embeddings.py"),
        "--train-mode",
        "embeddings",
        "--datasets-root",
        str(config.datasets_root),
        "--outputs-root",
        str(config.outputs_root),
        "--output-dir",
        str(run_parent),
        "--run-name",
        f"loo_{int(row['run_idx']):08d}",
        "--feature-source",
        row["feature_source"],
        "--target-field",
        config.target_field,
        "--dropout",
        str(config.dropout),
        "--head-type",
        row["head_type"],
        "--task",
        row["task"],
        "--ranking-sampling",
        "within_dataset",
        "--val-source",
        row["val_dataset"],
        "--batch-size",
        str(config.batch_size),
        "--num-workers",
        str(config.num_workers),
        "--max-epochs",
        str(config.max_epochs),
        "--early-stopping-patience",
        str(config.early_stopping_patience),
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--precision",
        "bf16-mixed",
        "--seed",
        str(config.seed),
        "--no-tensorboard",
        "--no-save-val-predictions",
    ]

    for dataset in row["train_sources"].split(","):
        dataset = dataset.strip()
        if dataset:
            command += ["--train-source", dataset]

    if row["task"] == "pairwise":
        command += ["--pairwise-target", row["target_mode"]]
    elif row["task"] == "listwise":
        command += [
            "--listwise-target",
            row["target_mode"],
            "--num-choices",
            str(config.num_choices),
        ]

    return command


def write_fail_report(
    fail_path: Path,
    failures: list[tuple[int, int, dict[str, str]]],
) -> None:
    with fail_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "run_idx\trc\tval_dataset\tmodel_key\tlayer_name\tfeature_source\t"
            "head_type\ttask\ttarget_mode\n"
        )
        for run_idx, return_code, row in failures:
            handle.write(
                f"{run_idx}\t{return_code}\t{row['val_dataset']}\t{row['model_key']}\t"
                f"{row['layer_name']}\t{row['feature_source']}\t{row['head_type']}\t"
                f"{row['task']}\t{row['target_mode']}\n"
            )


def run_shard(
    config: LauncherConfig,
    worker_rank: int,
    num_workers: int,
    local_gpu_id: int,
    force_local_cuda_visible_devices: bool = False,
) -> None:
    all_rows = load_plan(config.plan_path)
    shard_rows = all_rows[worker_rank::num_workers]
    hostname = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    fail_path = config.runs_root / (
        f"fails_rank{worker_rank:03d}_{sanitize_path_token(hostname)}.tsv"
    )

    env = os.environ.copy()
    if force_local_cuda_visible_devices or not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print(
        f"[worker {worker_rank}/{num_workers}] host={hostname} local_gpu={local_gpu_id} "
        f"visible_devices={env.get('CUDA_VISIBLE_DEVICES', '')} assigned={len(shard_rows)}",
        flush=True,
    )

    failures: list[tuple[int, int, dict[str, str]]] = []
    done = 0
    skipped = 0
    for row in shard_rows:
        run_idx = int(row["run_idx"])
        run_name = f"loo_{run_idx:08d}"
        run_parent = config.runs_root / row["val_dataset"] / row["model_key"] / row["layer_dir"]
        run_dir = run_parent / run_name
        if (run_dir / "result.json").exists():
            skipped += 1
            continue

        run_parent.mkdir(parents=True, exist_ok=True)
        command = build_train_command(config, row, run_parent)
        print(
            f"[worker {worker_rank}] run_idx={run_idx} val={row['val_dataset']} "
            f"feat={row['feature_source']} head={row['head_type']} "
            f"task={row['task']} target={row['target_mode']} run_dir={run_dir}",
            flush=True,
        )
        return_code = subprocess.call(command, env=env)
        done += 1
        if return_code != 0:
            failures.append((run_idx, return_code, row))

    if failures:
        write_fail_report(fail_path, failures)
        print(
            f"[worker {worker_rank}] done={done}, skipped={skipped}, failed={len(failures)} "
            f"-> {fail_path}",
            flush=True,
        )
        raise SystemExit(1)

    print(
        f"[worker {worker_rank}] done={done}, skipped={skipped}, failed=0",
        flush=True,
    )


def run_array_task(
    config: LauncherConfig,
    array_task_id: int,
    array_task_count: int,
    workers_per_task: int,
    reuse_plan: bool,
    plan_wait_timeout: int,
    plan_wait_interval: float,
) -> None:
    if array_task_count <= 0:
        raise SystemExit("array_task_count must be >= 1")
    if workers_per_task <= 0:
        raise SystemExit("workers_per_task must be >= 1")
    if array_task_id < 0 or array_task_id >= array_task_count:
        raise SystemExit(
            f"array_task_id={array_task_id} must be in [0, {array_task_count - 1}]"
        )

    total_workers = array_task_count * workers_per_task
    rank_offset = array_task_id * workers_per_task
    print(
        f"[array] task_id={array_task_id}/{array_task_count} workers_per_task={workers_per_task} "
        f"total_workers={total_workers} rank_offset={rank_offset} reuse_plan={int(reuse_plan)}",
        flush=True,
    )

    if reuse_plan:
        check_plan(config)
    else:
        if array_task_id == 0:
            build_plan(config)
            write_plan_ready_marker(
                config,
                source="run-array-task",
                total_workers=total_workers,
                array_task_id=array_task_id,
                array_task_count=array_task_count,
            )
        else:
            total_runs = wait_for_plan(
                config,
                timeout_seconds=plan_wait_timeout,
                poll_interval_seconds=plan_wait_interval,
                require_ready_marker=config.plan_ready_path is not None,
            )
            print(
                f"[plan] array_task={array_task_id} detected shared plan with total_runs={total_runs}",
                flush=True,
            )

    processes: list[tuple[int, int, mp.Process]] = []
    for local_gpu_id in range(workers_per_task):
        worker_rank = rank_offset + local_gpu_id
        process = mp.Process(
            target=run_shard,
            args=(config, worker_rank, total_workers, local_gpu_id, True),
        )
        process.start()
        processes.append((local_gpu_id, worker_rank, process))

    failures: list[tuple[int, int, int]] = []
    for local_gpu_id, worker_rank, process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append((local_gpu_id, worker_rank, int(process.exitcode or 1)))

    if failures:
        details = ", ".join(
            f"gpu={local_gpu_id}:rank={worker_rank}:exit={exit_code}"
            for local_gpu_id, worker_rank, exit_code in failures
        )
        raise SystemExit(
            f"Array task {array_task_id} saw {len(failures)} failed local worker(s): {details}"
        )

    print(f"[array] task_id={array_task_id} completed without local worker failures", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and execute the LOO train_all plan")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-plan", help="Generate plan.tsv under RUNS_ROOT")
    subparsers.add_parser("check-plan", help="Validate an existing plan.tsv under RUNS_ROOT")

    run_parser = subparsers.add_parser("run-shard", help="Run one shard of the plan")
    run_parser.add_argument(
        "--worker-rank",
        type=int,
        default=int(os.environ.get("SLURM_PROCID", "0")),
        help="0-based shard index (defaults to SLURM_PROCID)",
    )
    run_parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_NTASKS", "1")),
        help="Total number of shards/workers (defaults to SLURM_NTASKS)",
    )
    run_parser.add_argument(
        "--local-gpu-id",
        type=int,
        default=int(os.environ.get("SLURM_LOCALID", "0")),
        help="Local GPU index used when CUDA_VISIBLE_DEVICES is not preset",
    )

    array_parser = subparsers.add_parser(
        "run-array-task",
        help="Run one job-array task by launching local GPU workers on a single node",
    )
    array_parser.add_argument(
        "--array-task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
        help="0-based array task id (defaults to SLURM_ARRAY_TASK_ID)",
    )
    array_parser.add_argument(
        "--array-task-count",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")),
        help="Total number of array tasks (defaults to SLURM_ARRAY_TASK_COUNT)",
    )
    array_parser.add_argument(
        "--workers-per-task",
        type=int,
        default=int(os.environ.get("WORKERS_PER_TASK", "8")),
        help="Number of local GPU workers launched by each array task",
    )
    array_parser.add_argument(
        "--reuse-plan",
        action="store_true",
        help="Reuse the existing plan.tsv instead of rebuilding it in array task 0",
    )
    array_parser.add_argument(
        "--plan-wait-timeout",
        type=int,
        default=int(os.environ.get("PLAN_WAIT_TIMEOUT", "1800")),
        help="Seconds to wait for the shared plan to appear",
    )
    array_parser.add_argument(
        "--plan-wait-interval",
        type=float,
        default=float(os.environ.get("PLAN_WAIT_INTERVAL", "5")),
        help="Seconds between shared-plan readiness checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_env()
    if args.command == "build-plan":
        build_plan(config)
        return
    if args.command == "check-plan":
        check_plan(config)
        return
    if args.command == "run-shard":
        run_shard(config, args.worker_rank, args.num_workers, args.local_gpu_id)
        return
    if args.command == "run-array-task":
        run_array_task(
            config,
            array_task_id=args.array_task_id,
            array_task_count=args.array_task_count,
            workers_per_task=args.workers_per_task,
            reuse_plan=args.reuse_plan,
            plan_wait_timeout=args.plan_wait_timeout,
            plan_wait_interval=args.plan_wait_interval,
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
