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

from quality_backbones.alignment import AlignmentConfig, run_alignment
from quality_backbones.evaluation import parse_layer_selectors, parse_metric_list, parse_name_list, write_results_table


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
    return outputs_root / "embedding_alignment_runs" / stamp


def _parse_families(raw: list[str] | None) -> tuple[str, ...] | None:
    names = parse_name_list(raw)
    if names is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for item in names:
        token = str(item).strip()
        if not token:
            continue
        upper = token.upper()
        if upper in seen:
            continue
        normalized.append(upper)
        seen.add(upper)
    return tuple(normalized) if normalized else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact embedding-alignment report: pairwise distances in candidate embeddings "
            "vs pairwise distances in reference embeddings"
        )
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset list")

    parser.add_argument(
        "--reference-models",
        nargs="+",
        required=True,
        help="Reference model keys. Repeat or pass space-separated values",
    )
    parser.add_argument(
        "--reference-layers",
        nargs="*",
        default=("last",),
        help="Reference layer selectors. Default: last",
    )
    parser.add_argument(
        "--candidate-models",
        nargs="*",
        default=None,
        help="Optional candidate model keys. Default: all completed models except references",
    )
    parser.add_argument(
        "--candidate-layers",
        nargs="*",
        default=None,
        help="Candidate layer selectors. Default: all layers",
    )
    parser.add_argument(
        "--exclude-families",
        nargs="*",
        default=None,
        help="Optional candidate family exclusions, for example: ARNIQA MANIQA",
    )

    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on number of images per dataset. Default: full dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distance-metrics",
        nargs="*",
        default=("cos,l2,l1",),
        help="Distance metrics applied symmetrically to candidate and reference embeddings",
    )
    parser.add_argument(
        "--corr-metrics",
        nargs="*",
        default=("pcc,scc,kcc",),
        help="Correlation metrics: pcc, scc, kcc",
    )
    parser.add_argument(
        "--pair-scope",
        type=str,
        default="auto",
        choices=("auto", "global", "within_ref"),
        help="Pair scope: global all-pairs, within_ref per-reference groups, or auto (FR datasets -> within_ref)",
    )

    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--jobs", type=str, default="auto", help="Worker processes count or 'auto'")
    parser.add_argument("--tmp-dir", type=Path, default=Path("outputs/embedding_alignment_cache"))
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
    parser.add_argument("--print-limit", type=int, default=20, help="How many best-candidate rows to print")
    return parser.parse_args()


def _print_report(report: dict[str, object], print_limit: int) -> None:
    rows = list(report.get("results", []))
    dataset_summaries = list(report.get("dataset_summaries", []))
    warnings = list(report.get("warnings", []))
    artifacts = dict(report.get("artifacts", {}))
    progress = dict(report.get("progress", {}))

    print("=== Embedding Alignment Report ===")
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
                f"refs={item.get('n_reference_models')} "
                f"candidates={item.get('n_candidate_models')} "
                f"ref_layers={item.get('n_reference_layers')} "
                f"candidate_layers={item.get('n_candidate_layers')} "
                f"pair_scope={item.get('pair_scope')} "
                f"groups={item.get('n_groups')} "
                f"tasks={item.get('completed_tasks')}/{item.get('total_tasks')}"
            )
        print()

    best_candidates = list(report.get("best_candidates", []))
    if best_candidates:
        print(f"Best candidates (first {max(1, print_limit)}):")
        for row in best_candidates[: max(1, print_limit)]:
            print(
                "  - "
                f"{row['dataset']} / ref={row['reference_model_key']}:{row['reference_layer_name']} / "
                f"{row['corr_metric']} / {row['distance_metric']}: "
                f"cand={row['candidate_model_key']}:{row['candidate_layer_name']} "
                f"value={row['value']:.6f}"
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

    datasets = parse_name_list(args.datasets)
    reference_models = parse_name_list(args.reference_models)
    reference_layers = parse_layer_selectors(args.reference_layers)
    candidate_models = parse_name_list(args.candidate_models)
    candidate_layers = parse_layer_selectors(args.candidate_layers)
    exclude_families = _parse_families(args.exclude_families)
    distance_metrics = parse_metric_list(args.distance_metrics, kind="embedding")
    corr_metrics = parse_metric_list(args.corr_metrics, kind="corr")
    jobs = _parse_jobs(args.jobs)

    if reference_models is None:
        raise ValueError("--reference-models must not be empty")
    if distance_metrics is None or corr_metrics is None:
        raise ValueError("Metric lists must not be empty")

    run_dir = args.run_dir
    if args.resume:
        if run_dir is None:
            raise ValueError("--resume requires --run-dir")
    elif run_dir is None:
        run_dir = _default_run_dir(args.outputs_root)

    config = AlignmentConfig(
        datasets_root=args.datasets_root,
        outputs_root=args.outputs_root,
        datasets=datasets,
        reference_models=reference_models,
        reference_layer_selectors=reference_layers,
        candidate_models=candidate_models,
        candidate_layer_selectors=candidate_layers,
        exclude_families=exclude_families,
        sample_limit=args.sample_limit,
        seed=args.seed,
        distance_metrics=distance_metrics,
        corr_metrics=corr_metrics,
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
    report = run_alignment(config)
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
