from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.datasets import load_dataset_index
from quality_backbones.profiling import (  # noqa: E402
    build_profile_adapter,
    default_dtype_for_device,
    iter_selected_model_specs,
    managed_extractor,
    parse_profile_layer_selectors,
    profile_adapter_target,
    resolve_profile_target_indices,
    select_device,
    write_profile_results_table,
)


def _default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / "profile_runs" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile per-layer latency/FLOPs for truncated visual encoders")
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--dataset", type=str, default="koniq10k")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--models", nargs="*", default=None, help="Optional explicit model keys")
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--layers", nargs="*", default=None, help="Layer selectors: all, last, 0, 0-3, canonical_embedding")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--verify-parity", action="store_true", help="Verify truncated outputs against extractor.extract outputs")
    parser.add_argument("--parity-atol", type=float, default=1e-4)
    parser.add_argument("--parity-rtol", type=float, default=1e-4)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--table-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _load_images(datasets_root: Path, dataset: str, max_samples: int, batch_size: int) -> list[Image.Image]:
    frame = load_dataset_index(datasets_root, dataset).iloc[: max_samples].copy()
    if frame.empty:
        raise ValueError(f"Dataset {dataset!r} produced no samples")
    take = min(len(frame), batch_size)
    images: list[Image.Image] = []
    for _, row in frame.iloc[:take].iterrows():
        images.append(Image.open(row["abs_image_path"]).convert("RGB"))
    return images


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _task_key(model_key: str, layer_name: str, batch_size: int) -> str:
    return f"{model_key}::{batch_size}::{layer_name}"


def _build_report(results: list[dict], errors: list[dict], run_dir: Path) -> dict:
    by_model: dict[str, int] = {}
    parity_failures = 0
    for row in results:
        by_model[row["model_key"]] = by_model.get(row["model_key"], 0) + 1
        if row.get("parity_ok") is False:
            parity_failures += 1
    return {
        "run_dir": str(run_dir),
        "rows": len(results),
        "errors": len(errors),
        "models": sorted(by_model.items()),
        "parity_failures": parity_failures,
        "results": results,
        "error_rows": errors,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or _default_run_dir()
    if args.resume and not run_dir.exists():
        raise ValueError(f"--resume requested but run directory does not exist: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    selectors = parse_profile_layer_selectors(args.layers)
    images = _load_images(args.datasets_root, args.dataset, args.max_samples, args.batch_size)
    device = select_device(args.device)
    dtype = default_dtype_for_device(device)

    config_path = run_dir / "config.json"
    config_payload = {
        "datasets_root": str(args.datasets_root),
        "dataset": args.dataset,
        "weights_dir": str(args.weights_dir),
        "models": args.models,
        "max_models": args.max_models,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": str(dtype),
        "layers": list(selectors) if selectors is not None else None,
        "warmup": args.warmup,
        "iters": args.iters,
        "verify_parity": bool(args.verify_parity),
        "parity_atol": args.parity_atol,
        "parity_rtol": args.parity_rtol,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    results_jsonl = run_dir / "results.jsonl"
    errors_jsonl = run_dir / "errors.jsonl"
    existing_results = _load_jsonl(results_jsonl) if args.resume else []
    existing_errors = _load_jsonl(errors_jsonl) if args.resume else []
    completed = {
        _task_key(row["model_key"], row["layer_name"], int(row["batch_size"])): row
        for row in existing_results
    }

    specs = iter_selected_model_specs(args.models)
    if args.max_models is not None:
        specs = specs[: args.max_models]

    print(f"Profiling models={len(specs)} batch={len(images)} device={device} dtype={dtype}")
    results = list(existing_results)
    errors = list(existing_errors)

    for spec in specs:
        print(f"=== {spec.key} ===")
        previous_record: dict | None = None
        try:
            with managed_extractor(spec, device=device, dtype=dtype, weights_dir=args.weights_dir) as extractor:
                adapter = build_profile_adapter(spec, extractor)
                batch = adapter.prepare_batch(images)
                targets = adapter.list_targets()
                selected_indices = resolve_profile_target_indices([item.layer_name for item in targets], selectors)
                extract_result_layer_map = None
                if args.verify_parity:
                    extract_result = extractor.extract(images)
                    extract_result_layer_map = dict(zip(extract_result.layer_names, extract_result.per_layer_np))

                for index in selected_indices:
                    target = targets[index]
                    key = _task_key(spec.key, target.layer_name, len(images))
                    if key in completed:
                        previous_record = completed[key]
                        print(f"[SKIP] {target.layer_name}")
                        continue

                    result = profile_adapter_target(
                        adapter,
                        batch,
                        target,
                        batch_size=len(images),
                        warmup=args.warmup,
                        iters=args.iters,
                        extract_result_layer_map=extract_result_layer_map,
                        parity_atol=args.parity_atol,
                        parity_rtol=args.parity_rtol,
                    )
                    row = result.record.to_dict()
                    if previous_record is not None:
                        row["delta_latency_ms"] = float(row["latency_ms_mean"] - previous_record["latency_ms_mean"])
                        row["delta_flops_total"] = int(row["flops_total"] - previous_record["flops_total"])
                    if args.verify_parity and row.get("parity_ok") is False:
                        raise RuntimeError(
                            f"Parity mismatch for {spec.key}/{target.layer_name}: "
                            f"max_abs={row.get('parity_max_abs_error')} mean_abs={row.get('parity_mean_abs_error')}"
                        )
                    results.append(row)
                    completed[key] = row
                    previous_record = row
                    _append_jsonl(results_jsonl, row)
                    print(
                        f"[OK] {target.layer_name} "
                        f"lat={row['latency_ms_mean']:.3f}ms flops={row['flops_total']} parity={row.get('parity_ok')}"
                    )
        except Exception as exc:
            error_row = {"model_key": spec.key, "error": str(exc)}
            errors.append(error_row)
            _append_jsonl(errors_jsonl, error_row)
            print(f"[FAIL] {spec.key}: {exc}")
            if args.fail_fast:
                raise

        progress_path = run_dir / "progress.json"
        progress_path.write_text(
            json.dumps(
                {
                    "completed_rows": len(results),
                    "error_rows": len(errors),
                    "completed_models": sorted({row['model_key'] for row in results}),
                    "failed_models": sorted({row['model_key'] for row in errors}),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    report = _build_report(results, errors, run_dir)
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.table_out is not None:
        write_profile_results_table(results, args.table_out)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved report: {report_path}")
    print(f"Rows={len(results)} errors={len(errors)} parity_failures={report['parity_failures']}")


if __name__ == "__main__":
    main()
