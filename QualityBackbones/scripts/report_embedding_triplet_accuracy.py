from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.evaluation import parse_layer_selectors, parse_metric_list, parse_name_list, write_results_table
from quality_backbones.triplet_evaluation import TripletEvaluationConfig, run_triplet_evaluation


def _parse_jobs(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text in {"", "auto"}:
        return None
    value = int(text)
    if value <= 0:
        raise ValueError("--jobs must be a positive integer or 'auto'")
    return value


def _default_run_dir(outputs_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return outputs_root / "embedding_triplet_runs" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact embedding triplet-accuracy report for IQA: per-anchor agreement between embedding-distance "
            "order and absolute target-distance order"
        )
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset list")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model-key list")
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help=(
            "Layer selectors applied per model, for example: all, canonical_embedding, 0,3,7, 0-11. "
            "Default: all layers"
        ),
    )
    parser.add_argument("--target-field", type=str, default="normalized_score")
    parser.add_argument(
        "--pair-scope",
        type=str,
        default="auto",
        choices=("auto", "global", "within_ref"),
        help="Pair scope: global all-pairs, within_ref per-reference groups, or auto (FR datasets -> within_ref)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on number of images per dataset. Default: full dataset",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--embedding-distance",
        nargs="*",
        default=("cos,l2,l1",),
        help="Embedding distances: cos, l2, l1 (comma or space separated)",
    )

    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--jobs", type=str, default="auto", help="Worker processes count or 'auto'")
    parser.add_argument("--tmp-dir", type=Path, default=Path("outputs/embedding_triplet_cache"))
    parser.add_argument("--keep-cache", action="store_true", help="Keep intermediate cache files")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory for incremental artifacts/results.jsonl/progress.json/report.json",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing run-dir")
    parser.add_argument(
        "--progress",
        type=str,
        default="auto",
        choices=("auto", "bar", "log", "off"),
        help="Progress display mode",
    )
    parser.add_argument("--heartbeat-sec", type=int, default=30, help="Heartbeat interval for long runs")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on the first task error")

    parser.add_argument(
        "--table-out",
        type=Path,
        default=None,
        help="Optional table output path (.tsv/.csv/.parquet)",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--print-limit", type=int, default=20, help="How many top rows to print")
    return parser.parse_args()


def _print_report(report: dict[str, object], print_limit: int) -> None:
    rows = list(report.get("results", []))
    dataset_summaries = list(report.get("dataset_summaries", []))
    warnings = list(report.get("warnings", []))
    artifacts = dict(report.get("artifacts", {}))
    progress = dict(report.get("progress", {}))

    print("=== Embedding Triplet Accuracy Report ===")
    print(f"run_dir: {report.get('run_dir', '-')}")
    print(f"session_elapsed_sec: {float(report.get('session_elapsed_sec', float('nan'))):.3f}")
    print(f"datasets: {len(dataset_summaries)}")
    print(f"rows: {len(rows)}")
    if progress:
        print(
            "progress: "
            f"completed={progress.get('completed_tasks', 0)}/{progress.get('total_tasks', 0)} "
            f"failed={progress.get('failed_tasks', 0)} "
            f"skipped={progress.get('skipped_tasks', 0)}"
        )
    print()

    if dataset_summaries:
        print("Per-dataset summary:")
        for item in dataset_summaries:
            print(
                "  - "
                f"{item.get('dataset')}: "
                f"models={item.get('n_models')} "
                f"layers={item.get('n_layers')} "
                f"samples={item.get('n_samples')} "
                f"pairs={item.get('n_pairs')} "
                f"triplets={item.get('n_triplets')} "
                f"pair_scope={item.get('pair_scope')} "
                f"groups={item.get('n_groups')} "
                f"jobs={item.get('jobs')}"
            )
        print()

    best_layers = list(report.get("best_layers", []))
    if best_layers:
        print(f"Best layers (first {max(1, print_limit)}):")
        for row in best_layers[: max(1, print_limit)]:
            print(
                "  - "
                f"{row['dataset']} / {row['model_key']} / {row['embedding_distance']}: "
                f"layer={row['layer_name']} ({row['layer_index']}) "
                f"value={row['value']:.6f} pooled={float(row.get('pooled_value', float('nan'))):.6f}"
            )
        print()

    if warnings:
        print("Warnings:")
        for text in warnings:
            print(f"  - {text}")
        print()

    if artifacts:
        print("Artifacts:")
        for key in ["results_jsonl", "errors_jsonl", "progress_json", "report_json", "table_tsv", "run_log"]:
            value = artifacts.get(key)
            if value:
                print(f"  - {key}: {value}")
        print()


def main() -> None:
    args = parse_args()

    embedding_distances = parse_metric_list(args.embedding_distance, kind="embedding")
    datasets = parse_name_list(args.datasets)
    models = parse_name_list(args.models)
    layers = parse_layer_selectors(args.layers)
    jobs = _parse_jobs(args.jobs)
    run_dir = args.run_dir
    if args.resume:
        if run_dir is None:
            raise ValueError("--resume requires --run-dir")
    elif run_dir is None:
        run_dir = _default_run_dir(args.outputs_root)

    if embedding_distances is None:
        raise ValueError("Embedding distance list must not be empty")

    config = TripletEvaluationConfig(
        datasets_root=args.datasets_root,
        outputs_root=args.outputs_root,
        datasets=datasets,
        models=models,
        layer_selectors=layers,
        target_field=args.target_field,
        sample_limit=args.sample_limit,
        seed=args.seed,
        embedding_distances=embedding_distances,
        pair_scope=args.pair_scope,
        block_size=args.block_size,
        jobs=jobs,
        tmp_dir=args.tmp_dir,
        run_dir=run_dir,
        keep_cache=bool(args.keep_cache),
        resume=bool(args.resume),
        progress_mode=args.progress,
        heartbeat_sec=args.heartbeat_sec,
        fail_fast=bool(args.fail_fast),
    )

    print(f"Run dir: {config.run_dir}")
    report = run_triplet_evaluation(config)
    _print_report(report=report, print_limit=max(1, args.print_limit))

    if args.table_out is not None:
        write_results_table(report.get("results", []), args.table_out)
        print(f"Saved table: {args.table_out}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
