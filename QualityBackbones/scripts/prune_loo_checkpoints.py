from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PruneStats:
    runs_seen: int = 0
    runs_pruned: int = 0
    runs_already_clean: int = 0
    deleted_files: int = 0
    skipped_missing_checkpoints_dir: int = 0
    skipped_missing_best_path: int = 0
    skipped_missing_best_checkpoint: int = 0
    skipped_no_checkpoint_files: int = 0


def skipped_total(stats: PruneStats) -> int:
    return (
        stats.skipped_missing_checkpoints_dir
        + stats.skipped_missing_best_path
        + stats.skipped_missing_best_checkpoint
        + stats.skipped_no_checkpoint_files
    )


def _log(message: str) -> None:
    print(message, flush=True)


def _print_progress(processed: int, total: int, stats: PruneStats, every: int) -> None:
    if processed < 1 or total < 1:
        return
    if processed != 1 and processed != total and processed % every != 0:
        return
    percent = 100.0 * float(processed) / float(total)
    print(
        f"[progress] {processed}/{total} ({percent:.1f}%) "
        f"pruned={stats.runs_pruned} deleted={stats.deleted_files} "
        f"clean={stats.runs_already_clean} skipped={skipped_total(stats)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete non-best checkpoints from completed LOO training runs"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/training_runs_loo_all"),
        help="Root directory containing loo_* training runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print checkpoints that would be removed without deleting them",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress summary every N completed runs",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_best_checkpoint(checkpoints_dir: Path, raw_best_path: str) -> Path | None:
    best_text = raw_best_path.strip()
    if not best_text:
        return None

    best_name = Path(best_text).name
    if not best_name:
        return None

    candidate = checkpoints_dir / best_name
    if candidate.exists():
        return candidate

    raw_path = Path(best_text)
    if raw_path.exists() and raw_path.parent.resolve() == checkpoints_dir.resolve():
        return raw_path

    return None


def prune_run(result_path: Path, dry_run: bool, stats: PruneStats) -> None:
    run_dir = result_path.parent
    checkpoints_dir = run_dir / "checkpoints"
    stats.runs_seen += 1

    if not checkpoints_dir.is_dir():
        stats.skipped_missing_checkpoints_dir += 1
        _log(f"[skip] missing checkpoints dir: {checkpoints_dir}")
        return

    payload = _load_json(result_path)
    raw_best_path = str(payload.get("best_model_path", "") or "").strip()
    if not raw_best_path:
        stats.skipped_missing_best_path += 1
        _log(f"[skip] missing best_model_path: {result_path}")
        return

    best_checkpoint = _resolve_best_checkpoint(checkpoints_dir, raw_best_path)
    if best_checkpoint is None:
        stats.skipped_missing_best_checkpoint += 1
        _log(
            "[skip] best checkpoint not found under checkpoints dir: "
            f"run={run_dir} best_model_path={raw_best_path}"
        )
        return

    checkpoint_paths = sorted(path for path in checkpoints_dir.rglob("*.ckpt") if path.is_file())
    if not checkpoint_paths:
        stats.skipped_no_checkpoint_files += 1
        _log(f"[skip] no checkpoint files: {checkpoints_dir}")
        return

    best_resolved = best_checkpoint.resolve()
    stale_paths = [path for path in checkpoint_paths if path.resolve() != best_resolved]
    if not stale_paths:
        stats.runs_already_clean += 1
        _log(f"[ok] already clean: {best_checkpoint}")
        return

    action = "would delete" if dry_run else "delete"
    _log(f"[run] keep={best_checkpoint}")
    for checkpoint_path in stale_paths:
        _log(f"[{action}] {checkpoint_path}")
        if not dry_run:
            checkpoint_path.unlink()

    stats.runs_pruned += 1
    stats.deleted_files += len(stale_paths)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be >= 1")
    if not runs_root.exists():
        raise SystemExit(f"Runs root does not exist: {runs_root}")
    if not runs_root.is_dir():
        raise SystemExit(f"Runs root is not a directory: {runs_root}")

    stats = PruneStats()
    result_paths = sorted(runs_root.rglob("result.json"))
    if not result_paths:
        print(f"No completed runs found under: {runs_root}")
        return

    total_runs = len(result_paths)
    print(
        f"[start] runs_root={runs_root} completed_runs={total_runs} dry_run={int(bool(args.dry_run))}",
        flush=True,
    )

    for processed, result_path in enumerate(result_paths, start=1):
        prune_run(result_path, dry_run=bool(args.dry_run), stats=stats)
        _print_progress(processed, total_runs, stats, every=args.progress_every)

    mode = "dry-run" if args.dry_run else "done"
    print(
        f"[{mode}] runs_seen={stats.runs_seen} runs_pruned={stats.runs_pruned} "
        f"runs_already_clean={stats.runs_already_clean} deleted_files={stats.deleted_files} "
        f"skipped_missing_checkpoints_dir={stats.skipped_missing_checkpoints_dir} "
        f"skipped_missing_best_path={stats.skipped_missing_best_path} "
        f"skipped_missing_best_checkpoint={stats.skipped_missing_best_checkpoint} "
        f"skipped_no_checkpoint_files={stats.skipped_no_checkpoint_files}"
    )


if __name__ == "__main__":
    main()
