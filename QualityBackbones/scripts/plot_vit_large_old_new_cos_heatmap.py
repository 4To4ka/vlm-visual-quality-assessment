#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cosine-similarity heatmap between old/new vit_large layers"
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("/mnt/l/QualityBackbones/outputs/kadid10k/vit_large"),
        help="Directory with new-format embeddings (index.parquet + layers.h5)",
    )
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("/mnt/c/Users/fyodo/Desktop/Encoders/embeddings_final/vit_large"),
        help="Directory with old per-image NPZ embeddings",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="How many common images to use (<=0 means all common images)",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("/mnt/l/QualityBackbones/outputs/kadid10k/vit_large/old_new_cos_heatmap.pdf"),
        help="Output PDF path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/mnt/l/QualityBackbones/outputs/kadid10k/vit_large/old_new_cos_heatmap.csv"),
        help="Output CSV path (matrix values)",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimals for numeric annotations in cells",
    )
    return parser.parse_args()


def _validate_input_paths(new_dir: Path, old_dir: Path) -> tuple[Path, Path]:
    index_path = new_dir / "index.parquet"
    h5_path = new_dir / "layers.h5"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing H5 file: {h5_path}")
    if not old_dir.exists():
        raise FileNotFoundError(f"Missing old embeddings dir: {old_dir}")
    return index_path, h5_path


def _pick_common_images(index_path: Path, old_dir: Path, sample_size: int) -> tuple[np.ndarray, list[str], int]:
    idx = pd.read_parquet(index_path)
    required_cols = {"row_id", "filename"}
    missing = required_cols.difference(idx.columns)
    if missing:
        raise ValueError(f"Missing required columns in index.parquet: {sorted(missing)}")

    idx = idx.sort_values("row_id").reset_index(drop=True)
    idx["stem"] = idx["filename"].astype(str).map(lambda x: Path(x).stem)

    old_stems = {p.name[: -len("_embeddings.npz")] for p in old_dir.glob("*_embeddings.npz")}
    common = idx[idx["stem"].isin(old_stems)].copy()
    common_count = len(common)
    if common_count == 0:
        raise RuntimeError("No common images between new index and old NPZ directory")

    if sample_size > 0:
        common = common.iloc[:sample_size].copy()

    row_ids = common["row_id"].to_numpy(dtype=np.int64)
    stems = common["stem"].tolist()
    return row_ids, stems, common_count


def _load_new_layers(h5_path: Path, row_ids: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    with h5py.File(h5_path, "r") as fp:
        layer_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in fp.attrs["layer_names"]]
        per_layer: dict[str, np.ndarray] = {}
        for i, layer_name in enumerate(layer_names):
            ds_name = f"layer_{i:03d}"
            arr = fp[ds_name][row_ids].astype(np.float32)
            per_layer[layer_name] = arr
    return layer_names, per_layer


def _load_old_layers(old_dir: Path, stems: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    if not stems:
        raise ValueError("Empty image list")

    first_path = old_dir / f"{stems[0]}_embeddings.npz"
    first_npz = np.load(first_path)
    old_layer_names = list(first_npz.files)

    buffers: dict[str, list[np.ndarray]] = {name: [] for name in old_layer_names}

    for stem in stems:
        npz_path = old_dir / f"{stem}_embeddings.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing old NPZ: {npz_path}")
        data = np.load(npz_path)
        for layer_name in old_layer_names:
            if layer_name not in data.files:
                raise KeyError(f"Layer {layer_name} missing in {npz_path}")
            arr = data[layer_name]
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            buffers[layer_name].append(arr.astype(np.float32))

    per_layer = {name: np.stack(values, axis=0) for name, values in buffers.items()}
    return old_layer_names, per_layer


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _compute_cos_matrix(
    old_layer_names: list[str],
    old_layers: dict[str, np.ndarray],
    new_layer_names: list[str],
    new_layers: dict[str, np.ndarray],
) -> np.ndarray:
    old_norm = {k: _normalize_rows(v) for k, v in old_layers.items()}
    new_norm = {k: _normalize_rows(v) for k, v in new_layers.items()}

    matrix = np.full((len(old_layer_names), len(new_layer_names)), np.nan, dtype=np.float64)

    for i, old_name in enumerate(old_layer_names):
        a = old_norm[old_name]
        for j, new_name in enumerate(new_layer_names):
            b = new_norm[new_name]
            if a.shape != b.shape:
                continue
            cos_vals = np.sum(a * b, axis=1)
            matrix[i, j] = float(np.mean(cos_vals))
    return matrix


def _plot_heatmap(
    matrix: np.ndarray,
    old_layer_names: list[str],
    new_layer_names: list[str],
    out_pdf: Path,
    n_used: int,
    n_common: int,
    decimals: int,
) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(16.0, len(new_layer_names) * 0.8)
    fig_h = max(10.0, len(old_layer_names) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    valid_vals = matrix[np.isfinite(matrix)]
    vmin = float(np.min(valid_vals)) if valid_vals.size else -1.0
    vmax = float(np.max(valid_vals)) if valid_vals.size else 1.0
    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean cosine similarity", rotation=90)

    ax.set_xticks(np.arange(len(new_layer_names)))
    ax.set_xticklabels(new_layer_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(old_layer_names)))
    ax.set_yticklabels(old_layer_names, fontsize=8)
    ax.set_xlabel("New method layers")
    ax.set_ylabel("Old method layers")
    ax.set_title(
        "vit_large: cosine similarity between all old/new layers\n"
        f"images used: {n_used} (common available: {n_common})"
    )

    mid = (vmin + vmax) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isfinite(val):
                text = "nan"
                text_color = "black"
            else:
                text = f"{val:.{decimals}f}"
                text_color = "white" if val < mid else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    index_path, h5_path = _validate_input_paths(args.new_dir, args.old_dir)
    row_ids, stems, common_count = _pick_common_images(index_path, args.old_dir, args.sample_size)

    new_layer_names, new_layers = _load_new_layers(h5_path, row_ids)
    old_layer_names, old_layers = _load_old_layers(args.old_dir, stems)

    matrix = _compute_cos_matrix(old_layer_names, old_layers, new_layer_names, new_layers)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=old_layer_names, columns=new_layer_names).to_csv(args.output_csv)

    _plot_heatmap(
        matrix=matrix,
        old_layer_names=old_layer_names,
        new_layer_names=new_layer_names,
        out_pdf=args.output_pdf,
        n_used=len(stems),
        n_common=common_count,
        decimals=args.decimals,
    )

    print(f"Saved PDF: {args.output_pdf}")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Used images: {len(stems)}")
    print(f"Old layers: {len(old_layer_names)} | New layers: {len(new_layer_names)}")


if __name__ == "__main__":
    main()
