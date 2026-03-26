from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency in some setups
    Image = None


@dataclass
class DatasetAuditReport:
    dataset: str
    sampled_rows: int
    missing_paths: int
    case_mismatch_rows: int
    metadata_errors: int
    image_errors: int
    examples: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset data.csv indices")
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset directories")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Random rows per dataset (default: full scan)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Open sampled images with PIL.Image.verify",
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


def _resolve_rel_path(
    dataset_dir: Path,
    rel_path: str,
    cache: dict[Path, tuple[set[str], dict[str, list[str]]]],
) -> tuple[Path | None, bool, str | None]:
    rel = Path(rel_path)
    if rel.is_absolute():
        abs_path = rel
        return (abs_path if abs_path.exists() else None), False, None

    current = dataset_dir
    had_case_mismatch = False

    for token in rel.parts:
        if token in {".", ""}:
            continue
        if token == "..":
            return None, False, f"parent_token:{rel_path}"

        maps = cache.get(current)
        if maps is None:
            try:
                maps = _build_dir_maps(current)
            except Exception:
                return None, False, f"missing_dir:{current}"
            cache[current] = maps
        exact_names, lowered = maps

        if token in exact_names:
            chosen = token
        else:
            matches = lowered.get(token.lower(), [])
            if len(matches) == 1:
                chosen = matches[0]
                had_case_mismatch = True
            elif len(matches) > 1:
                return None, False, f"ambiguous_case:{current}/{token}"
            else:
                return None, False, f"missing_component:{current}/{token}"
        current = current / chosen

    return current, had_case_mismatch, None


def _sample_rows(rows: list[dict[str, str]], sample_size: int | None, rng: random.Random) -> list[tuple[int, dict[str, str]]]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(rows):
        return list(enumerate(rows))
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [(idx, rows[idx]) for idx in indices]


def _audit_one_dataset(
    dataset_dir: Path,
    sample_size: int | None,
    rng: random.Random,
    verify_images: bool,
) -> DatasetAuditReport:
    csv_path = dataset_dir / "data.csv"
    if not csv_path.exists():
        return DatasetAuditReport(
            dataset=dataset_dir.name,
            sampled_rows=0,
            missing_paths=1,
            case_mismatch_rows=0,
            metadata_errors=0,
            image_errors=0,
            examples=[f"missing_csv:{csv_path}"],
        )

    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    sampled_rows = _sample_rows(rows, sample_size, rng)
    cache: dict[Path, tuple[set[str], dict[str, list[str]]]] = {}

    missing_paths = 0
    case_mismatch_rows = 0
    metadata_errors = 0
    image_errors = 0
    examples: list[str] = []

    for idx, row in sampled_rows:
        path_raw = row.get("path")
        filename = row.get("filename", "")
        rel_path = filename if _is_blank(path_raw) else str(path_raw)

        abs_path, had_case_mismatch, resolve_error = _resolve_rel_path(dataset_dir, rel_path, cache)
        if abs_path is None:
            missing_paths += 1
            if len(examples) < 8:
                examples.append(f"row={idx} missing={resolve_error} path={rel_path}")
            continue

        if had_case_mismatch:
            case_mismatch_rows += 1
            if len(examples) < 8:
                examples.append(f"row={idx} case_mismatch path={rel_path} resolved={abs_path.relative_to(dataset_dir)}")

        metadata_raw = row.get("metadata", "")
        try:
            parsed = json.loads(metadata_raw)
            if not isinstance(parsed, dict):
                metadata_errors += 1
                if len(examples) < 8:
                    examples.append(f"row={idx} metadata_not_object")
        except Exception:
            metadata_errors += 1
            if len(examples) < 8:
                examples.append(f"row={idx} metadata_json_error")

        if verify_images:
            if Image is None:
                image_errors += 1
                if len(examples) < 8:
                    examples.append("PIL_unavailable")
            else:
                try:
                    with Image.open(abs_path) as img:
                        img.verify()
                except Exception as exc:
                    image_errors += 1
                    if len(examples) < 8:
                        examples.append(f"row={idx} image_error={type(exc).__name__} path={abs_path}")

    return DatasetAuditReport(
        dataset=dataset_dir.name,
        sampled_rows=len(sampled_rows),
        missing_paths=missing_paths,
        case_mismatch_rows=case_mismatch_rows,
        metadata_errors=metadata_errors,
        image_errors=image_errors,
        examples=examples,
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    reports = [
        _audit_one_dataset(
            dataset_dir=dataset_dir,
            sample_size=args.sample_size,
            rng=rng,
            verify_images=args.verify_images,
        )
        for dataset_dir in _iter_datasets(args.datasets_root, args.datasets)
    ]

    total_missing = 0
    total_case = 0
    total_md = 0
    total_img = 0
    for rep in reports:
        total_missing += rep.missing_paths
        total_case += rep.case_mismatch_rows
        total_md += rep.metadata_errors
        total_img += rep.image_errors
        print(
            f"{rep.dataset}: sampled={rep.sampled_rows} missing={rep.missing_paths} "
            f"case={rep.case_mismatch_rows} metadata={rep.metadata_errors} image={rep.image_errors}"
        )
        for ex in rep.examples:
            print(f"  - {ex}")

    print(
        f"TOTAL: datasets={len(reports)} missing={total_missing} case={total_case} "
        f"metadata={total_md} image={total_img} sample_size={args.sample_size}"
    )


if __name__ == "__main__":
    main()
