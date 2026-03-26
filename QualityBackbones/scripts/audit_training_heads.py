from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_NAME_RE = re.compile(r"^loo_(\d+)$")


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    run_dir: Path
    status: str
    run_idx: int | None
    val_dataset: str
    train_datasets: str
    feature_source: str
    model_key: str
    layer_name: str
    in_dim: int | None
    head_type: str
    task: str
    target_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit LOO head training runs: completion counts, dataset/layer coverage, and missing heads"
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/training_runs_loo_all"),
        help="Directory with training run folders",
    )
    parser.add_argument(
        "--plan-path",
        type=Path,
        default=None,
        help="Optional plan.tsv from launcher (default: <runs-root>/plan.tsv if present)",
    )
    parser.add_argument(
        "--train-mode",
        choices=["embeddings", "encoder", "any"],
        default="embeddings",
        help="Filter run summaries by train_mode",
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="heads_audit",
        help="Prefix for generated report files",
    )
    parser.add_argument(
        "--print-missing-limit",
        type=int,
        default=20,
        help="How many missing heads to print in console (0 to disable)",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dataset_from_source(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if ":" in text:
        return text.split(":", 1)[0].strip()
    return text


def _split_feature_source(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if not text:
        return "", ""
    if ":" in text:
        model_key, selector = text.split(":", 1)
        return model_key.strip(), selector.strip()
    return text, ""


def _normalize_target_mode(task: str, target_mode: str) -> str:
    task_norm = task.strip().lower()
    mode_norm = target_mode.strip().lower()
    if task_norm == "regression":
        return "na"
    return mode_norm or "unknown"


def _make_head_key(
    *,
    val_dataset: str,
    feature_source: str,
    head_type: str,
    task: str,
    target_mode: str,
) -> tuple[str, str, str, str, str]:
    return (
        val_dataset.strip(),
        feature_source.strip(),
        head_type.strip().lower(),
        task.strip().lower(),
        _normalize_target_mode(task, target_mode),
    )


def _read_plan(plan_path: Path) -> tuple[list[dict[str, str]], dict[int, dict[str, str]]]:
    rows: list[dict[str, str]] = []
    by_idx: dict[int, dict[str, str]] = {}

    with plan_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for raw in reader:
            row = {k: (v.strip() if isinstance(v, str) else "") for k, v in raw.items()}
            rows.append(row)
            idx_text = row.get("run_idx", "")
            if idx_text.isdigit():
                by_idx[int(idx_text)] = row

    return rows, by_idx


def _collect_runs(
    runs_root: Path,
    train_mode: str,
    plan_by_idx: dict[int, dict[str, str]],
) -> list[RunInfo]:
    infos: list[RunInfo] = []
    train_mode_norm = train_mode.strip().lower()

    for summary_path in sorted(runs_root.rglob("run_summary.json")):
        run_dir = summary_path.parent
        if not run_dir.is_dir():
            continue

        summary = _load_json(summary_path)
        summary_train_mode = str(summary.get("train_mode", "embeddings")).strip().lower()
        if train_mode_norm != "any" and summary_train_mode != train_mode_norm:
            continue

        run_name = run_dir.name
        run_match = RUN_NAME_RE.match(run_name)
        run_idx = int(run_match.group(1)) if run_match else None
        plan_row = plan_by_idx.get(run_idx) if run_idx is not None else None

        status = "completed" if (run_dir / "result.json").exists() else "incomplete"

        feature_source = ""
        feature_sources = summary.get("feature_sources")
        if isinstance(feature_sources, list) and feature_sources:
            feature_source = str(feature_sources[0]).strip()
        elif isinstance(feature_sources, str):
            feature_source = feature_sources.strip()
        if not feature_source and plan_row is not None:
            feature_source = plan_row.get("feature_source", "").strip()

        model_key, layer_name = _split_feature_source(feature_source)
        in_dim: int | None = None
        feature_plan = summary.get("feature_plan")
        if isinstance(feature_plan, list) and len(feature_plan) == 1 and isinstance(feature_plan[0], dict):
            first = feature_plan[0]
            if not model_key:
                model_key = str(first.get("model_key", "")).strip()
            if not layer_name:
                layer_name = str(first.get("layer_name", "")).strip()
            raw_dim = first.get("dim")
            if isinstance(raw_dim, int):
                in_dim = raw_dim
            elif isinstance(raw_dim, str) and raw_dim.strip().isdigit():
                in_dim = int(raw_dim.strip())

        if in_dim is None and plan_row is not None:
            in_dim_text = plan_row.get("in_dim", "").strip()
            if in_dim_text.isdigit():
                in_dim = int(in_dim_text)

        val_dataset = ""
        val_sources = summary.get("val_sources")
        if isinstance(val_sources, list) and val_sources:
            val_dataset = _dataset_from_source(str(val_sources[0]))
        elif isinstance(val_sources, str):
            val_dataset = _dataset_from_source(val_sources)
        if not val_dataset and plan_row is not None:
            val_dataset = plan_row.get("val_dataset", "").strip()

        train_dataset_values: list[str] = []
        train_sources = summary.get("train_sources")
        if isinstance(train_sources, list):
            for item in train_sources:
                dataset = _dataset_from_source(str(item))
                if dataset:
                    train_dataset_values.append(dataset)
        elif isinstance(train_sources, str):
            dataset = _dataset_from_source(train_sources)
            if dataset:
                train_dataset_values.append(dataset)
        train_datasets = ",".join(train_dataset_values)
        if not train_datasets and plan_row is not None:
            train_datasets = plan_row.get("train_sources", "").strip()

        task = str(summary.get("task", "")).strip().lower()
        if not task and plan_row is not None:
            task = plan_row.get("task", "").strip().lower()
        if not task:
            task = "unknown"

        head_type = str(summary.get("head_type", "")).strip().lower()
        if not head_type and plan_row is not None:
            head_type = plan_row.get("head_type", "").strip().lower()
        if not head_type:
            head_type = "unknown"

        target_mode = ""
        if task == "pairwise":
            target_mode = str(summary.get("pairwise_target", "")).strip().lower()
        elif task == "listwise":
            target_mode = str(summary.get("listwise_target", "")).strip().lower()
        elif task == "regression":
            target_mode = "na"
        if not target_mode and plan_row is not None:
            target_mode = plan_row.get("target_mode", "").strip().lower()
        target_mode = _normalize_target_mode(task, target_mode)

        infos.append(
            RunInfo(
                run_name=run_name,
                run_dir=run_dir,
                status=status,
                run_idx=run_idx,
                val_dataset=val_dataset,
                train_datasets=train_datasets,
                feature_source=feature_source,
                model_key=model_key,
                layer_name=layer_name,
                in_dim=in_dim,
                head_type=head_type,
                task=task,
                target_mode=target_mode,
            )
        )

    return infos


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_root}")

    plan_path = args.plan_path
    if plan_path is None:
        fallback = runs_root / "plan.tsv"
        plan_path = fallback if fallback.exists() else None

    plan_rows: list[dict[str, str]] = []
    plan_by_idx: dict[int, dict[str, str]] = {}
    if plan_path is not None and plan_path.exists():
        plan_rows, plan_by_idx = _read_plan(plan_path)

    infos = _collect_runs(runs_root=runs_root, train_mode=args.train_mode, plan_by_idx=plan_by_idx)
    completed_infos = [item for item in infos if item.status == "completed"]
    incomplete_infos = [item for item in infos if item.status != "completed"]

    completed_keys: dict[tuple[str, str, str, str, str], RunInfo] = {}
    for item in completed_infos:
        key = _make_head_key(
            val_dataset=item.val_dataset,
            feature_source=item.feature_source,
            head_type=item.head_type,
            task=item.task,
            target_mode=item.target_mode,
        )
        if key not in completed_keys:
            completed_keys[key] = item

    expected_keys: set[tuple[str, str, str, str, str]] = set()
    expected_key_to_run_idx: dict[tuple[str, str, str, str, str], str] = {}
    if plan_rows:
        for row in plan_rows:
            task = row.get("task", "").strip().lower()
            target_mode = row.get("target_mode", "").strip().lower()
            key = _make_head_key(
                val_dataset=row.get("val_dataset", ""),
                feature_source=row.get("feature_source", ""),
                head_type=row.get("head_type", ""),
                task=task,
                target_mode=target_mode,
            )
            expected_keys.add(key)
            expected_key_to_run_idx[key] = row.get("run_idx", "")

    missing_keys = sorted(expected_keys - set(completed_keys.keys())) if expected_keys else []

    dataset_completed_counts: dict[str, int] = defaultdict(int)
    dataset_completed_layers: dict[str, set[tuple[str, str]]] = defaultdict(set)
    layer_completed_counts: dict[tuple[str, str], int] = defaultdict(int)
    layer_completed_datasets: dict[tuple[str, str], set[str]] = defaultdict(set)
    ds_layer_completed_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    ds_layer_dims: dict[tuple[str, str, str], int | None] = {}

    for key, item in completed_keys.items():
        val_dataset, feature_source, _head, _task, _target = key
        model_key, layer_name = _split_feature_source(feature_source)
        layer_key = (model_key, layer_name)
        ds_layer_key = (val_dataset, model_key, layer_name)

        dataset_completed_counts[val_dataset] += 1
        dataset_completed_layers[val_dataset].add(layer_key)
        layer_completed_counts[layer_key] += 1
        layer_completed_datasets[layer_key].add(val_dataset)
        ds_layer_completed_counts[ds_layer_key] += 1
        ds_layer_dims[ds_layer_key] = item.in_dim

    dataset_expected_counts: dict[str, int] = defaultdict(int)
    dataset_expected_layers: dict[str, set[tuple[str, str]]] = defaultdict(set)
    layer_expected_counts: dict[tuple[str, str], int] = defaultdict(int)
    layer_expected_datasets: dict[tuple[str, str], set[str]] = defaultdict(set)
    ds_layer_expected_counts: dict[tuple[str, str, str], int] = defaultdict(int)

    if expected_keys:
        for val_dataset, feature_source, _head, _task, _target in expected_keys:
            model_key, layer_name = _split_feature_source(feature_source)
            layer_key = (model_key, layer_name)
            ds_layer_key = (val_dataset, model_key, layer_name)

            dataset_expected_counts[val_dataset] += 1
            dataset_expected_layers[val_dataset].add(layer_key)
            layer_expected_counts[layer_key] += 1
            layer_expected_datasets[layer_key].add(val_dataset)
            ds_layer_expected_counts[ds_layer_key] += 1

    prefix = args.report_prefix.strip() or "heads_audit"
    runs_tsv = runs_root / f"{prefix}_runs.tsv"
    completed_tsv = runs_root / f"{prefix}_completed.tsv"
    missing_tsv = runs_root / f"{prefix}_missing.tsv"
    by_dataset_tsv = runs_root / f"{prefix}_by_dataset.tsv"
    by_layer_tsv = runs_root / f"{prefix}_by_layer.tsv"
    by_dataset_layer_tsv = runs_root / f"{prefix}_by_dataset_layer.tsv"
    summary_json = runs_root / f"{prefix}_summary.json"

    run_rows: list[dict[str, Any]] = []
    for item in sorted(infos, key=lambda x: (x.run_idx is None, x.run_idx or 0, x.run_name)):
        run_rows.append(
            {
                "run_name": item.run_name,
                "status": item.status,
                "run_idx": "" if item.run_idx is None else item.run_idx,
                "val_dataset": item.val_dataset,
                "train_datasets": item.train_datasets,
                "feature_source": item.feature_source,
                "model_key": item.model_key,
                "layer_name": item.layer_name,
                "in_dim": "" if item.in_dim is None else item.in_dim,
                "head_type": item.head_type,
                "task": item.task,
                "target_mode": item.target_mode,
                "run_dir": str(item.run_dir),
            }
        )

    completed_rows = [row for row in run_rows if row["status"] == "completed"]

    _write_tsv(
        runs_tsv,
        run_rows,
        [
            "run_name",
            "status",
            "run_idx",
            "val_dataset",
            "train_datasets",
            "feature_source",
            "model_key",
            "layer_name",
            "in_dim",
            "head_type",
            "task",
            "target_mode",
            "run_dir",
        ],
    )
    _write_tsv(
        completed_tsv,
        completed_rows,
        [
            "run_name",
            "status",
            "run_idx",
            "val_dataset",
            "train_datasets",
            "feature_source",
            "model_key",
            "layer_name",
            "in_dim",
            "head_type",
            "task",
            "target_mode",
            "run_dir",
        ],
    )

    dataset_rows: list[dict[str, Any]] = []
    all_datasets = sorted(set(dataset_completed_counts.keys()) | set(dataset_expected_counts.keys()))
    for dataset in all_datasets:
        completed_count = dataset_completed_counts.get(dataset, 0)
        expected_count = dataset_expected_counts.get(dataset, 0)
        ratio = (
            float(completed_count) / float(expected_count)
            if expected_count > 0
            else (1.0 if completed_count > 0 else 0.0)
        )
        dataset_rows.append(
            {
                "val_dataset": dataset,
                "completed_heads": completed_count,
                "expected_heads": expected_count,
                "completion_ratio": f"{ratio:.6f}",
                "completed_unique_layers": len(dataset_completed_layers.get(dataset, set())),
                "expected_unique_layers": len(dataset_expected_layers.get(dataset, set())),
            }
        )
    _write_tsv(
        by_dataset_tsv,
        dataset_rows,
        [
            "val_dataset",
            "completed_heads",
            "expected_heads",
            "completion_ratio",
            "completed_unique_layers",
            "expected_unique_layers",
        ],
    )

    layer_rows: list[dict[str, Any]] = []
    all_layers = sorted(set(layer_completed_counts.keys()) | set(layer_expected_counts.keys()))
    for model_key, layer_name in all_layers:
        completed_count = layer_completed_counts.get((model_key, layer_name), 0)
        expected_count = layer_expected_counts.get((model_key, layer_name), 0)
        ratio = (
            float(completed_count) / float(expected_count)
            if expected_count > 0
            else (1.0 if completed_count > 0 else 0.0)
        )
        layer_rows.append(
            {
                "model_key": model_key,
                "layer_name": layer_name,
                "completed_heads": completed_count,
                "expected_heads": expected_count,
                "completion_ratio": f"{ratio:.6f}",
                "completed_val_datasets": len(layer_completed_datasets.get((model_key, layer_name), set())),
                "expected_val_datasets": len(layer_expected_datasets.get((model_key, layer_name), set())),
            }
        )
    _write_tsv(
        by_layer_tsv,
        layer_rows,
        [
            "model_key",
            "layer_name",
            "completed_heads",
            "expected_heads",
            "completion_ratio",
            "completed_val_datasets",
            "expected_val_datasets",
        ],
    )

    ds_layer_rows: list[dict[str, Any]] = []
    all_ds_layers = sorted(set(ds_layer_completed_counts.keys()) | set(ds_layer_expected_counts.keys()))
    for ds_key in all_ds_layers:
        val_dataset, model_key, layer_name = ds_key
        completed_count = ds_layer_completed_counts.get(ds_key, 0)
        expected_count = ds_layer_expected_counts.get(ds_key, 0)
        ratio = (
            float(completed_count) / float(expected_count)
            if expected_count > 0
            else (1.0 if completed_count > 0 else 0.0)
        )
        ds_layer_rows.append(
            {
                "val_dataset": val_dataset,
                "model_key": model_key,
                "layer_name": layer_name,
                "in_dim": "" if ds_layer_dims.get(ds_key) is None else ds_layer_dims[ds_key],
                "completed_heads": completed_count,
                "expected_heads": expected_count,
                "completion_ratio": f"{ratio:.6f}",
            }
        )
    _write_tsv(
        by_dataset_layer_tsv,
        ds_layer_rows,
        [
            "val_dataset",
            "model_key",
            "layer_name",
            "in_dim",
            "completed_heads",
            "expected_heads",
            "completion_ratio",
        ],
    )

    missing_rows: list[dict[str, Any]] = []
    if missing_keys:
        for key in missing_keys:
            val_dataset, feature_source, head_type, task, target_mode = key
            model_key, layer_name = _split_feature_source(feature_source)
            missing_rows.append(
                {
                    "run_idx": expected_key_to_run_idx.get(key, ""),
                    "val_dataset": val_dataset,
                    "feature_source": feature_source,
                    "model_key": model_key,
                    "layer_name": layer_name,
                    "head_type": head_type,
                    "task": task,
                    "target_mode": target_mode,
                }
            )
    _write_tsv(
        missing_tsv,
        missing_rows,
        [
            "run_idx",
            "val_dataset",
            "feature_source",
            "model_key",
            "layer_name",
            "head_type",
            "task",
            "target_mode",
        ],
    )

    summary = {
        "runs_root": str(runs_root.resolve()),
        "plan_path": str(plan_path.resolve()) if plan_path is not None and plan_path.exists() else None,
        "train_mode_filter": args.train_mode,
        "runs_with_summary": len(infos),
        "completed_run_dirs": len(completed_infos),
        "incomplete_run_dirs": len(incomplete_infos),
        "completed_unique_heads": len(completed_keys),
        "expected_unique_heads": len(expected_keys),
        "missing_unique_heads": len(missing_keys),
        "completion_ratio_unique_heads": (
            float(len(completed_keys)) / float(len(expected_keys)) if expected_keys else None
        ),
        "unique_val_datasets_completed": len(dataset_completed_counts),
        "unique_layers_completed": len(layer_completed_counts),
        "reports": {
            "runs_tsv": str(runs_tsv.resolve()),
            "completed_tsv": str(completed_tsv.resolve()),
            "missing_tsv": str(missing_tsv.resolve()),
            "by_dataset_tsv": str(by_dataset_tsv.resolve()),
            "by_layer_tsv": str(by_layer_tsv.resolve()),
            "by_dataset_layer_tsv": str(by_dataset_layer_tsv.resolve()),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[audit] runs_root={runs_root}")
    print(f"[audit] runs_with_summary={len(infos)}")
    print(f"[audit] completed_run_dirs={len(completed_infos)}")
    print(f"[audit] incomplete_run_dirs={len(incomplete_infos)}")
    print(f"[audit] completed_unique_heads={len(completed_keys)}")
    if expected_keys:
        ratio_pct = 100.0 * float(len(completed_keys)) / float(len(expected_keys))
        print(f"[audit] expected_unique_heads={len(expected_keys)}")
        print(f"[audit] missing_unique_heads={len(missing_keys)}")
        print(f"[audit] completion={ratio_pct:.2f}%")
    else:
        print("[audit] plan.tsv not found -> expected/missing coverage is unavailable")

    if args.print_missing_limit > 0 and missing_rows:
        print(f"[audit] first {min(args.print_missing_limit, len(missing_rows))} missing heads:")
        for row in missing_rows[: args.print_missing_limit]:
            print(
                "  - "
                f"run_idx={row['run_idx']} "
                f"val={row['val_dataset']} "
                f"feature={row['feature_source']} "
                f"head={row['head_type']} "
                f"task={row['task']} "
                f"target={row['target_mode']}"
            )

    print(f"[audit] summary_json={summary_json}")
    print(f"[audit] by_dataset_tsv={by_dataset_tsv}")
    print(f"[audit] by_layer_tsv={by_layer_tsv}")
    print(f"[audit] by_dataset_layer_tsv={by_dataset_layer_tsv}")
    print(f"[audit] missing_tsv={missing_tsv}")


if __name__ == "__main__":
    main()
