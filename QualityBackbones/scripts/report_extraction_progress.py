from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.manifest import iter_enabled_image_model_specs


REQUIRED_SUCCESS_FILES: tuple[str, ...] = ("meta.json", "index.parquet", "layers.h5")


@dataclass(frozen=True)
class RunStatus:
    dataset: str
    model_key: str
    status: str
    missing_files: list[str]
    error_message: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report extraction coverage/errors across datasets and models")
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names. Default: all subdirs in datasets-root containing data.csv",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output path for JSON report")
    parser.add_argument("--print-missing-limit", type=int, default=20)
    return parser.parse_args()


def discover_datasets(datasets_root: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        missing_csv = [name for name in explicit if not (datasets_root / name / "data.csv").exists()]
        if missing_csv:
            joined = ", ".join(sorted(missing_csv))
            raise FileNotFoundError(f"Missing data.csv for datasets: {joined}")
        return sorted(set(explicit))

    names = [p.name for p in datasets_root.iterdir() if p.is_dir() and (p / "data.csv").exists()]
    return sorted(names)


def _read_error(error_path: Path) -> str | None:
    if not error_path.exists():
        return None
    try:
        payload = json.loads(error_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "[invalid JSON in error.json]"

    if isinstance(payload, dict):
        value = payload.get("error")
        if value is None:
            return "[error.json exists without 'error' field]"
        return str(value)
    return "[error.json is not an object]"


def status_for_pair(outputs_root: Path, dataset: str, model_key: str) -> RunStatus:
    model_dir = outputs_root / dataset / model_key
    if not model_dir.exists():
        return RunStatus(dataset=dataset, model_key=model_key, status="missing", missing_files=[], error_message=None)

    missing_files = [name for name in REQUIRED_SUCCESS_FILES if not (model_dir / name).exists()]
    has_success_files = len(missing_files) == 0
    error_message = _read_error(model_dir / "error.json")

    if has_success_files:
        status = "success"
    elif error_message is not None:
        status = "error"
    else:
        status = "incomplete"

    return RunStatus(
        dataset=dataset,
        model_key=model_key,
        status=status,
        missing_files=missing_files,
        error_message=error_message,
    )


def build_report(datasets: list[str], model_keys: list[str], outputs_root: Path) -> dict:
    rows: list[RunStatus] = []
    for dataset in datasets:
        for model_key in model_keys:
            rows.append(status_for_pair(outputs_root=outputs_root, dataset=dataset, model_key=model_key))

    by_dataset: dict[str, dict[str, list[str]]] = {}
    by_model: dict[str, dict[str, list[str]]] = {}
    errors: list[dict[str, str]] = []
    incomplete: list[dict[str, object]] = []
    stale_error_with_success: list[dict[str, str]] = []

    for ds in datasets:
        by_dataset[ds] = {"success": [], "error": [], "missing": [], "incomplete": []}
    for mk in model_keys:
        by_model[mk] = {"success": [], "error": [], "missing": [], "incomplete": []}

    for row in rows:
        by_dataset[row.dataset][row.status].append(row.model_key)
        by_model[row.model_key][row.status].append(row.dataset)

        if row.status == "error":
            errors.append(
                {
                    "dataset": row.dataset,
                    "model_key": row.model_key,
                    "error": row.error_message or "[unknown error]",
                }
            )
        if row.status == "incomplete":
            incomplete.append(
                {
                    "dataset": row.dataset,
                    "model_key": row.model_key,
                    "missing_files": row.missing_files,
                }
            )
        if row.status == "success" and row.error_message is not None:
            stale_error_with_success.append(
                {
                    "dataset": row.dataset,
                    "model_key": row.model_key,
                    "error": row.error_message,
                }
            )

    success_on_all = sorted([mk for mk in model_keys if len(by_model[mk]["success"]) == len(datasets)])
    missing_on_all = sorted([mk for mk in model_keys if len(by_model[mk]["missing"]) == len(datasets)])

    return {
        "datasets": datasets,
        "num_datasets": len(datasets),
        "num_models": len(model_keys),
        "required_success_files": list(REQUIRED_SUCCESS_FILES),
        "models_success_on_all_datasets": success_on_all,
        "models_missing_on_all_datasets": missing_on_all,
        "errors": errors,
        "incomplete": incomplete,
        "stale_error_with_success": stale_error_with_success,
        "by_dataset": by_dataset,
        "by_model": by_model,
        "raw_status_rows": [asdict(item) for item in rows],
    }


def print_report(report: dict, missing_limit: int) -> None:
    datasets: list[str] = report["datasets"]

    print("=== Extraction Progress Report ===")
    print(f"datasets: {report['num_datasets']} -> {datasets}")
    print(f"enabled image models: {report['num_models']}")
    print()

    success_all: list[str] = report["models_success_on_all_datasets"]
    print("1) Models with successful runs on all datasets")
    print(f"count: {len(success_all)}")
    if success_all:
        print("models:")
        for model_key in success_all:
            print(f"  - {model_key}")
    print()

    print("2) Models not computed yet (missing, not error)")
    missing_all: list[str] = report["models_missing_on_all_datasets"]
    print(f"fully missing on all datasets: {len(missing_all)}")
    if missing_all:
        print("models:")
        for model_key in missing_all:
            print(f"  - {model_key}")
    print("missing by dataset:")
    for dataset in datasets:
        missing = report["by_dataset"][dataset]["missing"]
        preview = ", ".join(missing[:missing_limit])
        suffix = "" if len(missing) <= missing_limit else f" ... (+{len(missing) - missing_limit} more)"
        print(f"  - {dataset}: {len(missing)} missing")
        if preview:
            print(f"      {preview}{suffix}")
    print()

    print("3) Errors (model + dataset)")
    errors: list[dict[str, str]] = report["errors"]
    print(f"count: {len(errors)}")
    if errors:
        for item in errors:
            print(f"  - {item['dataset']} / {item['model_key']}: {item['error']}")
    else:
        print("  - no errors found")
    print()

    incomplete: list[dict[str, object]] = report["incomplete"]
    if incomplete:
        print("Additional diagnostics: incomplete runs")
        print(f"count: {len(incomplete)}")
        for item in incomplete:
            print(
                f"  - {item['dataset']} / {item['model_key']}: "
                f"missing files={', '.join(item['missing_files'])}"
            )
        print()

    stale: list[dict[str, str]] = report["stale_error_with_success"]
    if stale:
        print("Additional diagnostics: stale error.json with successful artifacts")
        print(f"count: {len(stale)}")
        for item in stale:
            print(f"  - {item['dataset']} / {item['model_key']}: {item['error']}")
        print()


def main() -> None:
    args = parse_args()
    datasets = discover_datasets(args.datasets_root, args.datasets)
    model_keys = sorted(spec.key for spec in iter_enabled_image_model_specs())
    report = build_report(datasets=datasets, model_keys=model_keys, outputs_root=args.outputs_root)

    print_report(report=report, missing_limit=max(1, args.print_missing_limit))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
