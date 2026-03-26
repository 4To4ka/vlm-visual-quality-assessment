from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetFixReport:
    dataset: str
    rows: int
    path_updates: int
    filename_updates: int
    unresolved: int
    examples: list[str]
    written: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix case-only path mismatches in dataset data.csv files")
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset directories")
    parser.add_argument("--dry-run", action="store_true", help="Report issues without writing changes")
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".bak",
        help="Suffix for backup files when writing",
    )
    return parser.parse_args()


def _iter_datasets(datasets_root: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        return [datasets_root / name for name in selected]
    return sorted([p for p in datasets_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def _is_blank(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _build_dir_maps(directory: Path) -> tuple[set[str], dict[str, list[str]]]:
    names = {child.name for child in directory.iterdir()}
    lowered: dict[str, list[str]] = defaultdict(list)
    for name in names:
        lowered[name.lower()].append(name)
    return names, dict(lowered)


def _canonicalize_rel_path(
    dataset_dir: Path,
    rel_path: str,
    cache: dict[Path, tuple[set[str], dict[str, list[str]]]],
) -> tuple[str | None, str | None]:
    rel = Path(rel_path)
    if rel.is_absolute():
        return None, "absolute_path"

    current = dataset_dir
    canonical_parts: list[str] = []

    for token in rel.parts:
        if token in {".", ""}:
            continue
        if token == "..":
            return None, f"parent_token:{rel_path}"

        maps = cache.get(current)
        if maps is None:
            try:
                maps = _build_dir_maps(current)
            except Exception:
                return None, f"missing_dir:{current}"
            cache[current] = maps
        exact_names, lowered = maps

        if token in exact_names:
            chosen = token
        else:
            matches = lowered.get(token.lower(), [])
            if len(matches) == 1:
                chosen = matches[0]
            elif len(matches) > 1:
                return None, f"ambiguous_case:{current}/{token}"
            else:
                return None, f"missing_component:{current}/{token}"

        canonical_parts.append(chosen)
        current = current / chosen

    return str(Path(*canonical_parts)), None


def _fix_one_dataset(dataset_dir: Path, dry_run: bool, backup_suffix: str) -> DatasetFixReport:
    csv_path = dataset_dir / "data.csv"
    if not csv_path.exists():
        return DatasetFixReport(
            dataset=dataset_dir.name,
            rows=0,
            path_updates=0,
            filename_updates=0,
            unresolved=1,
            examples=[f"missing_csv:{csv_path}"],
            written=False,
        )

    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        return DatasetFixReport(
            dataset=dataset_dir.name,
            rows=0,
            path_updates=0,
            filename_updates=0,
            unresolved=1,
            examples=[f"invalid_header:{csv_path}"],
            written=False,
        )

    if "path" not in fieldnames:
        fieldnames.append("path")

    path_updates = 0
    filename_updates = 0
    unresolved = 0
    examples: list[str] = []
    cache: dict[Path, tuple[set[str], dict[str, list[str]]]] = {}

    for idx, row in enumerate(rows):
        base_path = row.get("path")
        filename = row.get("filename", "")
        rel_path = filename if _is_blank(base_path) else str(base_path)

        canonical_rel, error = _canonicalize_rel_path(dataset_dir, rel_path, cache)
        if error is not None or canonical_rel is None:
            unresolved += 1
            if len(examples) < 8:
                examples.append(f"row={idx} unresolved={error} path={rel_path}")
            continue

        if rel_path != canonical_rel:
            row["path"] = canonical_rel
            path_updates += 1
            if len(examples) < 8:
                examples.append(f"row={idx} path: {rel_path} -> {canonical_rel}")

        if "filename" in row and not _is_blank(row.get("filename")):
            current_name = str(row["filename"])
            canonical_name = Path(canonical_rel).name
            if current_name != canonical_name and current_name.lower() == canonical_name.lower():
                row["filename"] = canonical_name
                filename_updates += 1
                if len(examples) < 8:
                    examples.append(f"row={idx} filename: {current_name} -> {canonical_name}")

    should_write = (path_updates > 0 or filename_updates > 0) and not dry_run
    if should_write:
        backup_path = csv_path.with_name(csv_path.name + backup_suffix)
        shutil.copy2(csv_path, backup_path)
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return DatasetFixReport(
        dataset=dataset_dir.name,
        rows=len(rows),
        path_updates=path_updates,
        filename_updates=filename_updates,
        unresolved=unresolved,
        examples=examples,
        written=should_write,
    )


def main() -> None:
    args = parse_args()
    reports = [
        _fix_one_dataset(dataset_dir, dry_run=args.dry_run, backup_suffix=args.backup_suffix)
        for dataset_dir in _iter_datasets(args.datasets_root, args.datasets)
    ]

    total_updates = 0
    total_unresolved = 0
    for rep in reports:
        dataset_updates = rep.path_updates + rep.filename_updates
        total_updates += dataset_updates
        total_unresolved += rep.unresolved
        print(
            f"{rep.dataset}: rows={rep.rows} path_updates={rep.path_updates} "
            f"filename_updates={rep.filename_updates} unresolved={rep.unresolved} "
            f"written={rep.written}"
        )
        for ex in rep.examples:
            print(f"  - {ex}")

    print(
        f"TOTAL: datasets={len(reports)} updates={total_updates} unresolved={total_unresolved} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
