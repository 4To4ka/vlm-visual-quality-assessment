#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.manifest import iter_enabled_image_model_specs


BACKGROUND = "#ffffff"
PANEL = "#ffffff"
INK = "#111111"
MUTED = "#5f6368"
GRID = "#d9d9d9"
TEAL = "#1f77b4"
GOLD = "#c69214"
CORAL = "#c75d3b"
NAVY = "#1f3b73"
SAND = "#f4f4f4"
MISSING = "#efefef"
HIGHLIGHT = "#eab308"

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "supplementary_heatmap",
    ["#f7fbff", "#d6e6f4", "#9ecae1", "#4f93c5", "#0b4f8a"],
)

FAMILY_PALETTE = [
    "#264653",
    "#2a9d8f",
    "#d9a441",
    "#d06b4d",
    "#7f5539",
    "#669bbc",
    "#588157",
    "#8d99ae",
    "#bc6c25",
    "#3d405b",
    "#6a994e",
    "#b56576",
    "#457b9d",
    "#9c6644",
    "#4d908e",
    "#ffb703",
]


@dataclass(frozen=True)
class QualityArtifacts:
    rows: pd.DataFrame
    layer_scores: pd.DataFrame
    dataset_best: pd.DataFrame
    model_ranking: pd.DataFrame
    top_models: pd.DataFrame
    dataset_winners: pd.DataFrame
    family_summary: pd.DataFrame
    layer_profiles: pd.DataFrame
    duplicate_groups: list[list[tuple[str, str, str]]]
    dataset_order: list[str]
    total_datasets: int
    nan_rows: int
    unique_slice_count: int
    total_slice_count: int


@dataclass(frozen=True)
class AlignmentArtifacts:
    rows: pd.DataFrame
    layer_scores: pd.DataFrame
    dataset_best: pd.DataFrame
    model_ranking: pd.DataFrame
    reference_ranking: pd.DataFrame
    top_models: pd.DataFrame
    family_summary: pd.DataFrame
    reference_global: pd.DataFrame
    dataset_order: list[str]
    reference_order: list[str]
    total_datasets: int
    nan_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a supplementary PDF and summary tables for embedding quality/alignment reports"
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("outputs/embedding_quality_runs/stage1_without_flive_pipal.tsv"),
        help="Quality report path (.tsv/.csv/.json or run directory)",
    )
    parser.add_argument(
        "--alignment-report",
        type=Path,
        default=Path("outputs/embedding_alignment_runs/stage1_kadid_refs"),
        help="Alignment report path (.tsv/.csv/.json or run directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embedding_supplementary_report"),
        help="Directory for PDF and summary tables",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many headline models to report in the main leaderboard",
    )
    parser.add_argument(
        "--exclude-families",
        nargs="*",
        default=("ARNIQA", "MANIQA"),
        help="Families excluded from headline ranking",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Embedding Quality and Alignment Supplementary Analysis",
        help="Title shown in the combined PDF",
    )
    parser.add_argument(
        "--complete-intersection-only",
        action="store_true",
        help=(
            "Keep only models that have complete embedding-metric coverage on every retained dataset, "
            "and keep only datasets that contain all such models in both quality and alignment reports"
        ),
    )
    return parser.parse_args()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "grid.color": GRID,
            "font.family": "DejaVu Serif",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 18,
        }
    )


def _resolve_report_path(path: Path, *, preferred_names: tuple[str, ...]) -> Path:
    if path.is_dir():
        for name in preferred_names:
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No supported report files inside {path}; tried: {preferred_names}")
    if path.exists():
        return path
    raise FileNotFoundError(f"Report path does not exist: {path}")


def _load_report_table(path: Path, *, preferred_names: tuple[str, ...], results_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = _resolve_report_path(path, preferred_names=preferred_names)
    suffix = resolved.suffix.lower()
    meta: dict[str, Any] = {"source_path": str(resolved)}
    if suffix == ".tsv":
        return pd.read_csv(resolved, sep="\t"), meta
    if suffix == ".csv":
        return pd.read_csv(resolved), meta
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            meta.update(payload)
            if results_key in payload:
                return pd.DataFrame(payload[results_key]), meta
        if isinstance(payload, list):
            return pd.DataFrame(payload), meta
        raise ValueError(f"Unsupported JSON payload in {resolved}")
    raise ValueError(f"Unsupported report format: {resolved}")


def _build_model_metadata() -> pd.DataFrame:
    rows = []
    for spec in iter_enabled_image_model_specs():
        rows.append(
            {
                "model_key": spec.key,
                "family": spec.family,
                "size": spec.size,
                "source": spec.source,
                "loader": spec.loader,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("model_key")


def _infer_family(model_key: str) -> str:
    if model_key.startswith("arniqa_"):
        return "ARNIQA"
    if model_key.startswith("maniqa_"):
        return "MANIQA"
    if model_key.startswith("clip_"):
        return "CLIP"
    if model_key.startswith("dinov2_"):
        return "DINOv2"
    if model_key.startswith("dinov3_"):
        return "DINOv3"
    if model_key.startswith("siglip2_"):
        return "SigLIP2"
    if model_key.startswith("siglip_"):
        return "SigLIP"
    if model_key.startswith("internvit_"):
        return "InternViT"
    if model_key.startswith("vit_mae_"):
        return "MAE"
    if model_key.startswith("vit_"):
        return "ViT"
    return model_key.split("_", maxsplit=1)[0]


def _enrich_with_metadata(df: pd.DataFrame, model_col: str, model_meta: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(model_meta, left_on=model_col, right_on="model_key", how="left")
    merged["family"] = merged["family"].fillna(merged[model_col].map(_infer_family))
    merged["size"] = merged["size"].fillna("-")
    return merged


def _short_layer_name(name: str) -> str:
    text = str(name)
    replacements = {
        "canonical_embedding": "canon",
        "pooler_output": "pool",
        "image_embeds_l2": "img_l2",
        "image_embeds": "img",
        "arniqa_embedding": "arniqa",
        "maniqa_embedding": "maniqa",
    }
    if text in replacements:
        return replacements[text]
    if text.startswith("hidden_state_"):
        return f"hs{text.rsplit('_', maxsplit=1)[-1]}"
    if text.startswith("block_"):
        return f"b{text.rsplit('_', maxsplit=1)[-1]}"
    return text.replace("_", " ")[:12]


def _format_score(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    value_f = float(value)
    if math.isnan(value_f):
        return "--"
    return f"{value_f:.{digits}f}"


def _preprocess_quality_base(
    quality_rows: pd.DataFrame,
    *,
    model_meta: pd.DataFrame,
    excluded_families: set[str],
) -> pd.DataFrame:
    df = quality_rows.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna()].copy()
    df["layer_index"] = df["layer_index"].astype(int)
    df = _enrich_with_metadata(df, model_col="model_key", model_meta=model_meta)
    return df[~df["family"].isin(excluded_families)].copy()


def _preprocess_alignment_base(
    alignment_rows: pd.DataFrame,
    *,
    model_meta: pd.DataFrame,
    excluded_families: set[str],
) -> pd.DataFrame:
    df = alignment_rows.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna()].copy()
    df["candidate_layer_index"] = df["candidate_layer_index"].astype(int)
    df = _enrich_with_metadata(df, model_col="candidate_model_key", model_meta=model_meta)
    return df[~df["family"].isin(excluded_families)].copy()


def _rank_to_percentile(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.ones(len(values), dtype=np.float64), index=values.index)
    ranks = values.rank(method="average", ascending=False)
    return 1.0 - (ranks - 1.0) / (len(values) - 1.0)


def _slice_fingerprint(df: pd.DataFrame, key_cols: list[str]) -> str:
    ordered = df[key_cols + ["value"]].sort_values(key_cols).reset_index(drop=True).copy()
    ordered["value"] = ordered["value"].round(12)
    hashed = pd.util.hash_pandas_object(ordered, index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _choose_slice_representative(group: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    return sorted(group, key=lambda item: (item[0], item[1], 0 if item[2] == "abs" else 1, item[2]))[0]


def _deduplicate_quality_slices(df: pd.DataFrame) -> tuple[pd.DataFrame, list[list[tuple[str, str, str]]]]:
    key_cols = ["dataset", "model_key", "layer_index", "layer_name"]
    slice_cols = ["corr_metric", "embedding_distance", "score_distance"]
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for slice_key, sub in df.groupby(slice_cols, sort=True):
        fingerprint = _slice_fingerprint(sub, key_cols=key_cols)
        normalized_key = (str(slice_key[0]), str(slice_key[1]), str(slice_key[2]))
        groups.setdefault(fingerprint, []).append(normalized_key)
    duplicate_groups = sorted((sorted(items) for items in groups.values()), key=lambda items: items[0])
    keep = [_choose_slice_representative(group) for group in duplicate_groups]
    keep_df = pd.DataFrame(keep, columns=slice_cols)
    deduped = df.merge(keep_df, on=slice_cols, how="inner")
    return deduped, duplicate_groups


def _compute_complete_intersection(
    quality_base: pd.DataFrame,
    alignment_base: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, Any]]:
    quality_deduped, duplicate_groups = _deduplicate_quality_slices(quality_base)
    quality_slice_cols = ["corr_metric", "embedding_distance", "score_distance"]
    alignment_slice_cols = ["reference_model_key", "corr_metric", "distance_metric"]

    quality_expected = max(1, quality_deduped[quality_slice_cols].drop_duplicates().shape[0])
    alignment_expected = max(1, alignment_base[alignment_slice_cols].drop_duplicates().shape[0])

    quality_presence = (
        quality_deduped.drop_duplicates(["dataset", "model_key"] + quality_slice_cols)
        .groupby(["dataset", "model_key"], as_index=False)
        .size()
        .rename(columns={"size": "quality_slice_count"})
    )
    quality_complete = quality_presence[quality_presence["quality_slice_count"] == quality_expected].copy()

    alignment_presence = (
        alignment_base.drop_duplicates(["dataset", "candidate_model_key"] + alignment_slice_cols)
        .groupby(["dataset", "candidate_model_key"], as_index=False)
        .size()
        .rename(columns={"size": "alignment_slice_count", "candidate_model_key": "model_key"})
    )
    alignment_complete = alignment_presence[alignment_presence["alignment_slice_count"] == alignment_expected].copy()

    quality_pairs = {(str(row.dataset), str(row.model_key)) for row in quality_complete.itertuples(index=False)}
    alignment_pairs = {(str(row.dataset), str(row.model_key)) for row in alignment_complete.itertuples(index=False)}
    complete_pairs = quality_pairs & alignment_pairs

    dataset_order = list(dict.fromkeys(quality_base["dataset"].tolist()))
    dataset_candidates = [dataset for dataset in dataset_order if dataset in set(alignment_base["dataset"].tolist())]

    quality_models = set(quality_base["model_key"].tolist())
    alignment_models = set(alignment_base["candidate_model_key"].tolist())
    model_candidates = sorted(quality_models & alignment_models)

    datasets = dataset_candidates
    models = model_candidates
    changed = True
    while changed and datasets and models:
        next_models = [model_key for model_key in models if all((dataset, model_key) in complete_pairs for dataset in datasets)]
        next_datasets = [dataset for dataset in datasets if all((dataset, model_key) in complete_pairs for model_key in next_models)]
        changed = next_models != models or next_datasets != datasets
        models = next_models
        datasets = next_datasets

    if not datasets or not models:
        raise ValueError("Complete intersection is empty; no datasets/models satisfy the requested completeness rule")

    summary = {
        "mode": "complete_intersection_only",
        "quality_unique_slices": quality_expected,
        "alignment_unique_slices": alignment_expected,
        "quality_duplicate_groups": len(duplicate_groups),
        "retained_dataset_count": len(datasets),
        "retained_model_count": len(models),
        "retained_datasets": datasets,
        "retained_models": models,
    }
    return datasets, models, summary


def _prepare_quality_artifacts(
    quality_rows: pd.DataFrame,
    quality_meta: dict[str, Any],
    *,
    model_meta: pd.DataFrame,
    excluded_families: set[str],
    top_k: int,
) -> QualityArtifacts:
    df = quality_rows.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    nan_rows = int(df["value"].isna().sum())
    df = df[df["value"].notna()].copy()
    df["layer_index"] = df["layer_index"].astype(int)
    df = _enrich_with_metadata(df, model_col="model_key", model_meta=model_meta)
    df = df[~df["family"].isin(excluded_families)].copy()

    present_datasets = set(df["dataset"].unique().tolist())
    dataset_order = [dataset for dataset in list(quality_meta.get("config", {}).get("datasets") or []) if dataset in present_datasets]
    if not dataset_order:
        dataset_order = list(dict.fromkeys(df["dataset"].tolist()))

    deduped, duplicate_groups = _deduplicate_quality_slices(df)
    deduped["quality_percentile"] = deduped.groupby(
        ["dataset", "corr_metric", "embedding_distance", "score_distance"]
    )["value"].transform(_rank_to_percentile)

    layer_scores = (
        deduped.groupby(["dataset", "model_key", "family", "layer_index", "layer_name"], as_index=False)
        .agg(
            quality_score=("quality_percentile", "mean"),
            slice_count=("quality_percentile", "count"),
            raw_value_mean=("value", "mean"),
        )
        .sort_values(["dataset", "model_key", "quality_score", "raw_value_mean", "layer_index"], ascending=[True, True, False, False, True])
    )

    dataset_best = layer_scores.drop_duplicates(["dataset", "model_key"]).copy()
    dataset_best["dataset_rank"] = dataset_best.groupby("dataset")["quality_score"].rank(
        method="average", ascending=False
    )
    dataset_best["dataset_rank_int"] = dataset_best.groupby("dataset")["quality_score"].rank(
        method="first", ascending=False
    ).astype(int)

    dataset_winners = (
        dataset_best.sort_values(["dataset", "quality_score", "raw_value_mean", "model_key"], ascending=[True, False, False, True])
        .drop_duplicates("dataset")
        .loc[:, ["dataset", "model_key", "family", "layer_name", "layer_index", "quality_score"]]
        .rename(columns={"layer_name": "best_layer", "layer_index": "best_layer_index"})
    )
    winner_counts = dataset_winners["model_key"].value_counts().rename_axis("model_key").reset_index(name="dataset_wins")

    full_coverage = len(dataset_order)
    best_layer_global = (
        layer_scores.groupby(["model_key", "family", "layer_index", "layer_name"], as_index=False)
        .agg(
            global_layer_quality=("quality_score", "mean"),
            layer_dataset_count=("dataset", "nunique"),
        )
        .sort_values(
            ["model_key", "global_layer_quality", "layer_dataset_count", "layer_index"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("model_key")
        .rename(columns={"layer_name": "global_best_layer", "layer_index": "global_best_layer_index"})
    )

    model_ranking = (
        dataset_best.groupby(["model_key", "family"], as_index=False)
        .agg(
            global_quality_score=("quality_score", "mean"),
            quality_score_std=("quality_score", lambda values: float(np.std(values, ddof=0))),
            mean_dataset_rank=("dataset_rank", "mean"),
            dataset_count=("dataset", "nunique"),
        )
        .merge(best_layer_global, on=["model_key", "family"], how="left")
        .merge(winner_counts, on="model_key", how="left")
        .fillna({"dataset_wins": 0})
    )
    model_ranking["dataset_wins"] = model_ranking["dataset_wins"].astype(int)
    model_ranking["coverage_frac"] = model_ranking["dataset_count"] / max(1, full_coverage)
    model_ranking["headline_eligible"] = model_ranking["dataset_count"] == full_coverage
    model_ranking = model_ranking.sort_values(
        ["headline_eligible", "global_quality_score", "mean_dataset_rank", "dataset_wins", "model_key"],
        ascending=[False, False, True, False, True],
    ).reset_index(drop=True)
    model_ranking.insert(0, "quality_rank", np.arange(1, len(model_ranking) + 1))

    eligible = model_ranking[model_ranking["headline_eligible"]].copy()
    top_models = eligible.head(top_k).copy()
    if len(top_models) < top_k:
        fallback = model_ranking.loc[~model_ranking["model_key"].isin(top_models["model_key"])].head(top_k - len(top_models))
        top_models = pd.concat([top_models, fallback], ignore_index=True)

    family_summary = (
        model_ranking.groupby("family", as_index=False)
        .agg(
            family_quality_median=("global_quality_score", "median"),
            family_quality_mean=("global_quality_score", "mean"),
            model_count=("model_key", "nunique"),
        )
        .sort_values(["family_quality_median", "family_quality_mean", "model_count"], ascending=[False, False, False])
    )

    layer_profiles = (
        layer_scores.groupby(["model_key", "family", "layer_index", "layer_name"], as_index=False)
        .agg(
            mean_quality=("quality_score", "mean"),
            min_quality=("quality_score", "min"),
            max_quality=("quality_score", "max"),
            dataset_count=("dataset", "nunique"),
        )
        .sort_values(["model_key", "layer_index"])
    )

    return QualityArtifacts(
        rows=df,
        layer_scores=layer_scores,
        dataset_best=dataset_best,
        model_ranking=model_ranking,
        top_models=top_models,
        dataset_winners=dataset_winners,
        family_summary=family_summary,
        layer_profiles=layer_profiles,
        duplicate_groups=duplicate_groups,
        dataset_order=dataset_order,
        total_datasets=full_coverage,
        nan_rows=nan_rows,
        unique_slice_count=len(duplicate_groups),
        total_slice_count=df[["corr_metric", "embedding_distance", "score_distance"]].drop_duplicates().shape[0],
    )


def _prepare_alignment_artifacts(
    alignment_rows: pd.DataFrame,
    alignment_meta: dict[str, Any],
    *,
    model_meta: pd.DataFrame,
    excluded_families: set[str],
    top_k: int,
) -> AlignmentArtifacts:
    df = alignment_rows.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    nan_rows = int(df["value"].isna().sum())
    df = df[df["value"].notna()].copy()
    df["candidate_layer_index"] = df["candidate_layer_index"].astype(int)
    df = _enrich_with_metadata(df, model_col="candidate_model_key", model_meta=model_meta)
    df = df[~df["family"].isin(excluded_families)].copy()

    present_datasets = set(df["dataset"].unique().tolist())
    dataset_order = [dataset for dataset in list(alignment_meta.get("config", {}).get("datasets") or []) if dataset in present_datasets]
    if not dataset_order:
        dataset_order = list(dict.fromkeys(df["dataset"].tolist()))
    present_refs = set(df["reference_model_key"].unique().tolist())
    reference_order = [reference for reference in list(alignment_meta.get("config", {}).get("reference_models") or []) if reference in present_refs]
    if not reference_order:
        reference_order = sorted(df["reference_model_key"].unique().tolist())

    df["alignment_percentile"] = df.groupby(
        ["dataset", "reference_model_key", "corr_metric", "distance_metric"]
    )["value"].transform(_rank_to_percentile)

    layer_scores = (
        df.groupby(
            [
                "dataset",
                "candidate_model_key",
                "family",
                "candidate_layer_index",
                "candidate_layer_name",
            ],
            as_index=False,
        )
        .agg(
            alignment_score=("alignment_percentile", "mean"),
            slice_count=("alignment_percentile", "count"),
            raw_value_mean=("value", "mean"),
        )
        .sort_values(
            ["dataset", "candidate_model_key", "alignment_score", "raw_value_mean", "candidate_layer_index"],
            ascending=[True, True, False, False, True],
        )
    )

    dataset_best = layer_scores.drop_duplicates(["dataset", "candidate_model_key"]).copy()
    dataset_best["dataset_rank"] = dataset_best.groupby("dataset")["alignment_score"].rank(
        method="average", ascending=False
    )

    full_coverage = len(dataset_order)
    best_layer_global = (
        layer_scores.groupby(
            ["candidate_model_key", "family", "candidate_layer_index", "candidate_layer_name"],
            as_index=False,
        )
        .agg(
            global_best_alignment=("alignment_score", "mean"),
            layer_dataset_count=("dataset", "nunique"),
        )
        .sort_values(
            ["candidate_model_key", "global_best_alignment", "layer_dataset_count", "candidate_layer_index"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("candidate_model_key")
        .rename(
            columns={
                "candidate_model_key": "model_key",
                "candidate_layer_index": "global_best_layer_index",
                "candidate_layer_name": "global_best_layer",
            }
        )
    )

    model_ranking = (
        dataset_best.groupby(["candidate_model_key", "family"], as_index=False)
        .agg(
            global_alignment_score=("alignment_score", "mean"),
            alignment_score_std=("alignment_score", lambda values: float(np.std(values, ddof=0))),
            mean_dataset_rank=("dataset_rank", "mean"),
            dataset_count=("dataset", "nunique"),
        )
        .rename(columns={"candidate_model_key": "model_key"})
        .merge(best_layer_global, on=["model_key", "family"], how="left")
    )
    model_ranking["coverage_frac"] = model_ranking["dataset_count"] / max(1, full_coverage)
    model_ranking["headline_eligible"] = model_ranking["dataset_count"] == full_coverage
    model_ranking = model_ranking.sort_values(
        ["headline_eligible", "global_alignment_score", "mean_dataset_rank", "model_key"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    model_ranking.insert(0, "alignment_rank", np.arange(1, len(model_ranking) + 1))

    top_models = model_ranking[model_ranking["headline_eligible"]].head(top_k).copy()
    if len(top_models) < top_k:
        fallback = model_ranking.loc[~model_ranking["model_key"].isin(top_models["model_key"])].head(top_k - len(top_models))
        top_models = pd.concat([top_models, fallback], ignore_index=True)

    family_summary = (
        model_ranking.groupby("family", as_index=False)
        .agg(
            family_alignment_median=("global_alignment_score", "median"),
            family_alignment_mean=("global_alignment_score", "mean"),
            model_count=("model_key", "nunique"),
        )
        .sort_values(["family_alignment_median", "family_alignment_mean", "model_count"], ascending=[False, False, False])
    )

    ref_layer_scores = (
        df.groupby(
            [
                "reference_model_key",
                "dataset",
                "candidate_model_key",
                "family",
                "candidate_layer_index",
                "candidate_layer_name",
            ],
            as_index=False,
        )
        .agg(reference_alignment_score=("alignment_percentile", "mean"))
        .sort_values(
            [
                "reference_model_key",
                "dataset",
                "candidate_model_key",
                "reference_alignment_score",
                "candidate_layer_index",
            ],
            ascending=[True, True, True, False, True],
        )
    )
    ref_best = ref_layer_scores.drop_duplicates(["reference_model_key", "dataset", "candidate_model_key"]).copy()
    reference_ranking = (
        ref_best.groupby(["reference_model_key", "candidate_model_key", "family"], as_index=False)
        .agg(
            reference_alignment_score=("reference_alignment_score", "mean"),
            dataset_count=("dataset", "nunique"),
        )
        .sort_values(["reference_model_key", "reference_alignment_score", "dataset_count", "candidate_model_key"], ascending=[True, False, False, True])
        .rename(columns={"candidate_model_key": "model_key"})
    )

    reference_global = reference_ranking.pivot_table(
        index=["model_key", "family"],
        columns="reference_model_key",
        values="reference_alignment_score",
    ).reset_index()

    return AlignmentArtifacts(
        rows=df,
        layer_scores=layer_scores,
        dataset_best=dataset_best,
        model_ranking=model_ranking,
        reference_ranking=reference_ranking,
        top_models=top_models,
        family_summary=family_summary,
        reference_global=reference_global,
        dataset_order=dataset_order,
        reference_order=reference_order,
        total_datasets=full_coverage,
        nan_rows=nan_rows,
    )


def _build_quality_metric_family_scores(quality: QualityArtifacts) -> pd.DataFrame:
    df, _ = _deduplicate_quality_slices(quality.rows)
    df = df[df["corr_metric"].isin(["scc", "pcc"])].copy()
    df["metric_percentile"] = df.groupby(
        ["dataset", "corr_metric", "embedding_distance", "score_distance"]
    )["value"].transform(_rank_to_percentile)

    layer_scores = (
        df.groupby(["dataset", "family", "model_key", "corr_metric", "layer_index", "layer_name"], as_index=False)
        .agg(metric_score=("metric_percentile", "mean"), raw_value_mean=("value", "mean"))
        .sort_values(
            ["dataset", "corr_metric", "model_key", "metric_score", "raw_value_mean", "layer_index"],
            ascending=[True, True, True, False, False, True],
        )
    )
    dataset_best = layer_scores.drop_duplicates(["dataset", "corr_metric", "model_key"]).copy()
    dataset_best["dataset_rank"] = dataset_best.groupby(["dataset", "corr_metric"])["metric_score"].rank(
        method="average", ascending=False
    )

    best_layer_global = (
        layer_scores.groupby(["family", "model_key", "corr_metric", "layer_index", "layer_name"], as_index=False)
        .agg(global_best_layer_score=("metric_score", "mean"), dataset_count=("dataset", "nunique"))
        .sort_values(
            ["family", "model_key", "corr_metric", "global_best_layer_score", "dataset_count", "layer_index"],
            ascending=[True, True, True, False, False, True],
        )
        .drop_duplicates(["family", "model_key", "corr_metric"])
        .rename(columns={"layer_name": "global_best_layer", "layer_index": "global_best_layer_index"})
    )

    model_scores = (
        dataset_best.groupby(["family", "model_key", "corr_metric"], as_index=False)
        .agg(
            global_metric_score=("metric_score", "mean"),
            metric_score_std=("metric_score", lambda values: float(np.std(values, ddof=0))),
            mean_dataset_rank=("dataset_rank", "mean"),
            dataset_count=("dataset", "nunique"),
        )
        .merge(best_layer_global, on=["family", "model_key", "corr_metric", "dataset_count"], how="left")
    )
    model_scores["family_rank"] = model_scores.groupby(["corr_metric", "family"])["global_metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    model_scores = model_scores[model_scores["dataset_count"] == quality.total_datasets].copy()
    model_scores["family_rank"] = model_scores.groupby(["corr_metric", "family"])["global_metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    model_scores = model_scores.sort_values(
        ["corr_metric", "family", "family_rank", "global_metric_score", "model_key"],
        ascending=[True, True, True, False, True],
    )
    return model_scores


def _build_alignment_metric_family_scores(alignment: AlignmentArtifacts) -> pd.DataFrame:
    df = alignment.rows[alignment.rows["corr_metric"].isin(["scc", "pcc"])].copy()
    df["metric_percentile"] = df.groupby(
        ["dataset", "reference_model_key", "corr_metric", "distance_metric"]
    )["value"].transform(_rank_to_percentile)

    layer_scores = (
        df.groupby(
            [
                "dataset",
                "family",
                "candidate_model_key",
                "corr_metric",
                "candidate_layer_index",
                "candidate_layer_name",
            ],
            as_index=False,
        )
        .agg(metric_score=("metric_percentile", "mean"), raw_value_mean=("value", "mean"))
        .sort_values(
            [
                "dataset",
                "corr_metric",
                "candidate_model_key",
                "metric_score",
                "raw_value_mean",
                "candidate_layer_index",
            ],
            ascending=[True, True, True, False, False, True],
        )
    )
    dataset_best = layer_scores.drop_duplicates(["dataset", "corr_metric", "candidate_model_key"]).copy()
    dataset_best["dataset_rank"] = dataset_best.groupby(["dataset", "corr_metric"])["metric_score"].rank(
        method="average", ascending=False
    )

    best_layer_global = (
        layer_scores.groupby(
            ["family", "candidate_model_key", "corr_metric", "candidate_layer_index", "candidate_layer_name"],
            as_index=False,
        )
        .agg(global_best_layer_score=("metric_score", "mean"), dataset_count=("dataset", "nunique"))
        .sort_values(
            [
                "family",
                "candidate_model_key",
                "corr_metric",
                "global_best_layer_score",
                "dataset_count",
                "candidate_layer_index",
            ],
            ascending=[True, True, True, False, False, True],
        )
        .drop_duplicates(["family", "candidate_model_key", "corr_metric"])
        .rename(
            columns={
                "candidate_model_key": "model_key",
                "candidate_layer_name": "global_best_layer",
                "candidate_layer_index": "global_best_layer_index",
            }
        )
    )

    model_scores = (
        dataset_best.groupby(["family", "candidate_model_key", "corr_metric"], as_index=False)
        .agg(
            global_metric_score=("metric_score", "mean"),
            metric_score_std=("metric_score", lambda values: float(np.std(values, ddof=0))),
            mean_dataset_rank=("dataset_rank", "mean"),
            dataset_count=("dataset", "nunique"),
        )
        .rename(columns={"candidate_model_key": "model_key"})
        .merge(best_layer_global, on=["family", "model_key", "corr_metric", "dataset_count"], how="left")
    )
    model_scores["family_rank"] = model_scores.groupby(["corr_metric", "family"])["global_metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    model_scores = model_scores[model_scores["dataset_count"] == alignment.total_datasets].copy()
    model_scores["family_rank"] = model_scores.groupby(["corr_metric", "family"])["global_metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    model_scores = model_scores.sort_values(
        ["corr_metric", "family", "family_rank", "global_metric_score", "model_key"],
        ascending=[True, True, True, False, True],
    )
    return model_scores


def _build_quality_dataset_metric_family_scores(quality: QualityArtifacts) -> pd.DataFrame:
    df, _ = _deduplicate_quality_slices(quality.rows)
    df = df[df["corr_metric"].isin(["scc", "pcc"])].copy()
    df["metric_percentile"] = df.groupby(
        ["dataset", "corr_metric", "embedding_distance", "score_distance"]
    )["value"].transform(_rank_to_percentile)
    layer_scores = (
        df.groupby(["dataset", "family", "model_key", "corr_metric", "layer_index", "layer_name"], as_index=False)
        .agg(metric_score=("metric_percentile", "mean"), raw_value_mean=("value", "mean"))
        .sort_values(
            ["dataset", "corr_metric", "model_key", "metric_score", "raw_value_mean", "layer_index"],
            ascending=[True, True, True, False, False, True],
        )
    )
    dataset_best = layer_scores.drop_duplicates(["dataset", "corr_metric", "model_key"]).copy()
    dataset_best["family_rank"] = dataset_best.groupby(["dataset", "corr_metric", "family"])["metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return dataset_best.sort_values(["dataset", "corr_metric", "family", "family_rank", "metric_score"], ascending=[True, True, True, True, False])


def _build_alignment_dataset_metric_family_scores(alignment: AlignmentArtifacts) -> pd.DataFrame:
    df = alignment.rows[alignment.rows["corr_metric"].isin(["scc", "pcc"])].copy()
    df["metric_percentile"] = df.groupby(
        ["dataset", "reference_model_key", "corr_metric", "distance_metric"]
    )["value"].transform(_rank_to_percentile)
    layer_scores = (
        df.groupby(
            [
                "dataset",
                "family",
                "candidate_model_key",
                "corr_metric",
                "candidate_layer_index",
                "candidate_layer_name",
            ],
            as_index=False,
        )
        .agg(metric_score=("metric_percentile", "mean"), raw_value_mean=("value", "mean"))
        .sort_values(
            [
                "dataset",
                "corr_metric",
                "candidate_model_key",
                "metric_score",
                "raw_value_mean",
                "candidate_layer_index",
            ],
            ascending=[True, True, True, False, False, True],
        )
    )
    dataset_best = (
        layer_scores.drop_duplicates(["dataset", "corr_metric", "candidate_model_key"])
        .rename(
            columns={
                "candidate_model_key": "model_key",
                "candidate_layer_index": "layer_index",
                "candidate_layer_name": "layer_name",
            }
        )
        .copy()
    )
    dataset_best["family_rank"] = dataset_best.groupby(["dataset", "corr_metric", "family"])["metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return dataset_best.sort_values(["dataset", "corr_metric", "family", "family_rank", "metric_score"], ascending=[True, True, True, True, False])


def _build_family_color_map(families: list[str]) -> dict[str, str]:
    unique_families = []
    for family in families:
        if family not in unique_families:
            unique_families.append(family)
    colors = {}
    for index, family in enumerate(unique_families):
        colors[family] = FAMILY_PALETTE[index % len(FAMILY_PALETTE)]
    return colors


def _soften_color(color: str, amount: float = 0.78) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(color), dtype=np.float64)
    return tuple((1.0 - amount) * rgb + amount * np.ones(3, dtype=np.float64))


def _apply_card(ax: plt.Axes, *, title: str | None = None, subtitle: str | None = None) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, loc="left", pad=8)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=MUTED)


def _draw_table(ax: plt.Axes, table_df: pd.DataFrame, *, title: str, family_color_map: dict[str, str], row_height: float = 1.5) -> None:
    _apply_card(ax, title=title)
    ax.axis("off")
    display = table_df.copy()
    if "Family" in display.columns:
        row_colors = [_soften_color(family_color_map.get(family, SAND)) for family in display["Family"]]
    else:
        row_colors = [PANEL for _ in range(len(display))]
    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=display.columns.tolist(),
        colLoc="left",
        cellLoc="left",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, row_height)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.get_text().set_color("white")
            cell.get_text().set_color(INK)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(row_colors[row - 1])
            if col == 0:
                cell.get_text().set_weight("bold")


def _plot_ranked_bars(
    ax: plt.Axes,
    *,
    data: pd.DataFrame,
    value_col: str,
    label_col: str,
    family_col: str,
    family_color_map: dict[str, str],
    title: str,
    subtitle: str,
    xlabel: str,
) -> None:
    _apply_card(ax, title=title, subtitle=subtitle)
    ordered = data.iloc[::-1].copy()
    colors = [family_color_map.get(family, TEAL) for family in ordered[family_col]]
    ax.barh(ordered[label_col], ordered[value_col], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.35)
    xmin = max(0.0, float(ordered[value_col].min()) - 0.04)
    ax.set_xlim(xmin, 1.0)
    for ypos, value in enumerate(ordered[value_col]):
        ax.text(float(value) + 0.006, ypos, _format_score(value), va="center", fontsize=8.5)


def _plot_cover_page(
    pdf: PdfPages,
    *,
    title: str,
    quality: QualityArtifacts,
    alignment: AlignmentArtifacts,
    quality_source: str,
    alignment_source: str,
) -> None:
    fig = plt.figure(figsize=(12.5, 8.8))
    fig.patch.set_facecolor(BACKGROUND)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.9])
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.78),
            1.0,
            0.20,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor=NAVY,
            edgecolor=NAVY,
            transform=ax.transAxes,
        )
    )
    ax.text(0.04, 0.91, title, fontsize=24, fontweight="bold", color="white", ha="left", va="center")
    ax.text(
        0.04,
        0.84,
        "Non-IQA headline ranking from embedding_quality; embedding_alignment used as supporting geometry analysis.",
        fontsize=12,
        color="#e8f1f2",
        ha="left",
        va="center",
    )

    top_models = quality.top_models.reset_index(drop=True)
    top3_text = ", ".join(top_models.loc[:2, "model_key"].tolist())
    alignment_top = alignment.top_models.iloc[0]["model_key"] if not alignment.top_models.empty else "-"
    overview = [
        f"Datasets in scope: {quality.total_datasets} ({', '.join(quality.dataset_order)})",
        f"Quality candidates after IQA exclusion: {quality.model_ranking['model_key'].nunique()} models",
        f"Alignment candidates after IQA exclusion: {alignment.model_ranking['model_key'].nunique()} models",
        f"Deduplicated quality slices: {quality.unique_slice_count} unique from {quality.total_slice_count} raw metric slices",
        f"Removed NaN rows: quality={quality.nan_rows}, alignment={alignment.nan_rows}",
    ]
    findings = [
        f"Headline top-3 by robust quality score: {top3_text}",
        f"Best overall geometric alignment model: {alignment_top}",
        "Late-middle transformer layers dominate the leaderboard more often than canonical heads.",
        "SPAQ has partial model coverage; headline ranking keeps only full-coverage models.",
    ]
    methodology = [
        "Quality score = average percentile rank of a layer across unique metric slices per dataset.",
        "Model ranking = average dataset score of the model's best layer, then averaged across datasets.",
        "Global best layer = layer with the best mean robust quality score across all available datasets.",
        "Alignment score = average percentile rank across both references and all alignment metrics.",
    ]
    sources = [
        f"Quality source: {quality_source}",
        f"Alignment source: {alignment_source}",
        "Current saved stage covers AGIQA-3K, CSIQ, LIVEC, PieAPP, SPAQ, TID2013, kadid10k, koniq10k.",
    ]

    y = 0.72
    blocks = [
        ("Overview", overview),
        ("Methodology", methodology),
        ("Key Findings", findings),
        ("Sources And Caveats", sources),
    ]
    for header, items in blocks:
        ax.text(0.04, y, header, fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
        y -= 0.035
        for item in items:
            ax.text(0.05, y, f"- {item}", fontsize=11, color=INK, ha="left", va="top")
            y -= 0.045
        y -= 0.02

    ax.text(
        0.04,
        0.04,
        "Interpret scores as normalized percentiles inside each dataset/metric slice; higher is better.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_report_tables(
    pdf: PdfPages,
    *,
    quality: QualityArtifacts,
    alignment: AlignmentArtifacts,
    family_color_map: dict[str, str],
    note: str | None = None,
) -> None:
    fig = plt.figure(figsize=(13.0, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)

    top_bars = quality.top_models.copy()
    ax_left = fig.add_subplot(gs[0, 0])
    _plot_ranked_bars(
        ax_left,
        data=top_bars,
        value_col="global_quality_score",
        label_col="model_key",
        family_col="family",
        family_color_map=family_color_map,
        title="Top-10 Models by Embedding Quality",
        subtitle="Non-IQA models with full coverage across all 8 datasets.",
        xlabel="Global quality score",
    )

    top_table = quality.top_models.copy()
    top_table["global_best_layer"] = top_table["global_best_layer"].map(_short_layer_name)
    top_table["global_quality_score"] = top_table["global_quality_score"].map(_format_score)
    top_table = top_table.rename(
        columns={
            "quality_rank": "#",
            "model_key": "Model",
            "family": "Family",
            "global_quality_score": "Score",
            "global_best_layer": "Best Layer",
        }
    )[["#", "Model", "Family", "Best Layer", "Score"]]
    _draw_table(
        fig.add_subplot(gs[0, 1]),
        top_table,
        title="Best Models Report",
        family_color_map=family_color_map,
        row_height=1.25,
    )

    fig.text(
        0.02,
        0.02,
        "Ranking = best layer per dataset, aggregated across deduplicated quality metric slices, then averaged across datasets.",
        fontsize=9,
        color=MUTED,
    )
    if note:
        fig.text(0.98, 0.02, note, fontsize=9, color=MUTED, ha="right")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_support_tables(
    pdf: PdfPages,
    *,
    quality: QualityArtifacts,
    alignment: AlignmentArtifacts,
    family_color_map: dict[str, str],
    note: str | None = None,
) -> None:
    fig = plt.figure(figsize=(13.0, 8.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.2)

    winners = quality.dataset_winners.copy()
    winners["quality_score"] = winners["quality_score"].map(_format_score)
    winners["best_layer"] = winners["best_layer"].map(_short_layer_name)
    winners_table = winners.rename(
        columns={
            "dataset": "Dataset",
            "model_key": "Winner",
            "family": "Family",
            "best_layer": "Layer",
            "quality_score": "Score",
        }
    )[["Dataset", "Winner", "Family", "Layer", "Score"]]
    _draw_table(
        fig.add_subplot(gs[0, 0]),
        winners_table,
        title="Per-Dataset Winners",
        family_color_map=family_color_map,
        row_height=1.32,
    )

    align_top = alignment.top_models.head(8).copy()
    ax_right = fig.add_subplot(gs[0, 1])
    _plot_ranked_bars(
        ax_right,
        data=align_top,
        value_col="global_alignment_score",
        label_col="model_key",
        family_col="family",
        family_color_map=family_color_map,
        title="Top Models by Embedding Alignment",
        subtitle="Supporting geometry analysis against ARNIQA-KADID10K and MANIQA-KADID10K references.",
        xlabel="Global alignment score",
    )

    fig.text(
        0.02,
        0.02,
        "SPAQ remains partial-coverage in the saved runs, so the headline leaderboard keeps only full-coverage models.",
        fontsize=9,
        color=MUTED,
    )
    if note:
        fig.text(0.98, 0.02, note, fontsize=9, color=MUTED, ha="right")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_heatmap_payload(
    dataset_best: pd.DataFrame,
    *,
    value_col: str,
    layer_col: str,
    model_order: list[str],
    dataset_order: list[str],
    model_col: str,
) -> tuple[np.ndarray, list[list[str]], list[str], list[str]]:
    _ = layer_col
    pivot = dataset_best.pivot_table(index=model_col, columns="dataset", values=value_col, aggfunc="first")
    matrix = np.full((len(model_order), len(dataset_order)), np.nan, dtype=np.float64)
    labels: list[list[str]] = []
    for i, model in enumerate(model_order):
        row_labels: list[str] = []
        for j, dataset in enumerate(dataset_order):
            if model in pivot.index and dataset in pivot.columns:
                value = pivot.loc[model, dataset]
                if pd.notna(value):
                    matrix[i, j] = float(value)
            if np.isfinite(matrix[i, j]):
                row_labels.append(f"{matrix[i, j]:.2f}")
            else:
                row_labels.append("")
        labels.append(row_labels)
    return matrix, labels, model_order, dataset_order


def _plot_heatmap(
    ax: plt.Axes,
    *,
    matrix: np.ndarray,
    annotations: list[list[str]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    subtitle: str,
    colorbar_label: str,
    annotate: bool = True,
) -> None:
    _apply_card(ax, title=title, subtitle=subtitle)
    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad(MISSING)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=BACKGROUND, linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)

    norm = Normalize(vmin=0.0, vmax=1.0)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    text_color = "white" if norm(value) > 0.62 else INK
                    ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=7.5, color=text_color)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(colorbar_label, fontsize=9)


def _plot_quality_heatmap(pdf: PdfPages, *, quality: QualityArtifacts) -> None:
    model_order = quality.top_models["model_key"].tolist()
    matrix, labels, rows, cols = _build_heatmap_payload(
        quality.dataset_best,
        value_col="quality_score",
        layer_col="layer_name",
        model_order=model_order,
        dataset_order=quality.dataset_order,
        model_col="model_key",
    )
    fig, ax = plt.subplots(figsize=(13.0, 8.2))
    _plot_heatmap(
        ax,
        matrix=matrix,
        annotations=labels,
        row_labels=rows,
        col_labels=cols,
        title="Best-Layer Quality Across Datasets",
        subtitle="Each cell shows robust quality score and the best layer for that dataset/model pair.",
        colorbar_label="Robust quality score",
    )
    fig.suptitle("Embedding Quality Overview", x=0.02, y=0.98, ha="left", fontsize=19, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_alignment_reference_heatmaps(pdf: PdfPages, *, alignment: AlignmentArtifacts) -> None:
    for reference_key in alignment.reference_order:
        reference_rank = alignment.reference_ranking[alignment.reference_ranking["reference_model_key"] == reference_key].copy()
        model_order = reference_rank.head(10)["model_key"].tolist()
        ref_rows = alignment.rows[alignment.rows["reference_model_key"] == reference_key].copy()
        ref_rows["reference_alignment_percentile"] = ref_rows.groupby(
            ["dataset", "corr_metric", "distance_metric"]
        )["value"].transform(_rank_to_percentile)
        ref_layer_scores = (
            ref_rows.groupby(
                ["dataset", "candidate_model_key", "candidate_layer_index", "candidate_layer_name"], as_index=False
            )
            .agg(alignment_score=("reference_alignment_percentile", "mean"))
            .sort_values(
                ["dataset", "candidate_model_key", "alignment_score", "candidate_layer_index"],
                ascending=[True, True, False, True],
            )
            .drop_duplicates(["dataset", "candidate_model_key"])
            .rename(columns={"candidate_model_key": "model_key", "candidate_layer_name": "layer_name"})
        )
        matrix, labels, rows, cols = _build_heatmap_payload(
            ref_layer_scores,
            value_col="alignment_score",
            layer_col="layer_name",
            model_order=model_order,
            dataset_order=alignment.dataset_order,
            model_col="model_key",
        )
        fig, ax = plt.subplots(figsize=(13.0, 6.2))
        _plot_heatmap(
            ax,
            matrix=matrix,
            annotations=labels,
            row_labels=rows,
            col_labels=cols,
            title=f"Alignment to {reference_key}",
            subtitle="Best candidate layer per dataset after averaging across correlation and distance metrics.",
            colorbar_label="Robust alignment score",
            annotate=False,
        )
        fig.suptitle(f"Reference-Specific Alignment: {reference_key}", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _select_models_for_annotations(merged: pd.DataFrame, quality_top: pd.DataFrame, alignment_top: pd.DataFrame) -> list[str]:
    selected = list(quality_top["model_key"].head(4))
    for model_key in alignment_top["model_key"].head(4):
        if model_key not in selected:
            selected.append(model_key)
    for model_key in merged.sort_values(["global_quality_score", "global_alignment_score"], ascending=[False, False]).head(6)["model_key"]:
        if model_key not in selected:
            selected.append(model_key)
    return selected[:6]


def _plot_scatter_pages(
    pdf: PdfPages,
    *,
    quality: QualityArtifacts,
    alignment: AlignmentArtifacts,
    family_color_map: dict[str, str],
) -> None:
    merged = quality.model_ranking.merge(
        alignment.model_ranking[["model_key", "global_alignment_score"]], on="model_key", how="inner"
    )
    merged = merged.merge(alignment.reference_global, on=["model_key", "family"], how="left")

    fig = plt.figure(figsize=(13.0, 8.4))
    gs = fig.add_gridspec(1, 2, wspace=0.18)
    selected = _select_models_for_annotations(merged, quality.top_models, alignment.top_models)

    ax_left = fig.add_subplot(gs[0, 0])
    _apply_card(ax_left, title="Quality vs Alignment", subtitle="All non-IQA models with full global summaries.")
    ax_left.scatter(
        merged["global_quality_score"],
        merged["global_alignment_score"],
        s=55,
        color="#c1c7cd",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    legend_handles = {}
    offsets = [(0.004, 0.006), (0.005, -0.01), (-0.04, 0.008), (0.006, 0.014), (-0.038, -0.012), (0.006, -0.014)]
    for idx, (_, row) in enumerate(merged[merged["model_key"].isin(selected)].iterrows()):
        color = family_color_map.get(row["family"], TEAL)
        legend_handles[row["family"]] = color
        ax_left.scatter(
            row["global_quality_score"],
            row["global_alignment_score"],
            s=120,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        dx, dy = offsets[idx % len(offsets)]
        ax_left.text(
            row["global_quality_score"] + dx,
            row["global_alignment_score"] + dy,
            row["model_key"],
            fontsize=7.5,
            color=INK,
        )
    ax_left.set_xlabel("Global quality score")
    ax_left.set_ylabel("Global alignment score")
    ax_left.grid(alpha=0.35)
    ax_left.set_xlim(0.0, 1.02)
    ax_left.set_ylim(0.0, 1.02)

    ax_right = fig.add_subplot(gs[0, 1])
    ref_cols = [col for col in alignment.reference_order if col in merged.columns]
    if len(ref_cols) >= 2:
        x_col, y_col = ref_cols[0], ref_cols[1]
        sizes = 40 + 120 * merged["global_quality_score"].fillna(0.0)
        scatter = ax_right.scatter(
            merged[x_col],
            merged[y_col],
            s=sizes,
            c=merged["global_quality_score"],
            cmap=HEATMAP_CMAP,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        max_lim = float(max(merged[x_col].max(), merged[y_col].max(), 1.0))
        ax_right.plot([0.0, max_lim], [0.0, max_lim], linestyle="--", color=MUTED, linewidth=1.0)
        for idx, (_, row) in enumerate(merged[merged["model_key"].isin(selected)].iterrows()):
            dx, dy = offsets[idx % len(offsets)]
            ax_right.text(row[x_col] + dx, row[y_col] + dy, row["model_key"], fontsize=7.5)
        ax_right.set_xlabel(f"Alignment to {x_col}")
        ax_right.set_ylabel(f"Alignment to {y_col}")
        ax_right.grid(alpha=0.35)
        ax_right.set_xlim(0.0, max_lim + 0.03)
        ax_right.set_ylim(0.0, max_lim + 0.03)
        _apply_card(ax_right, title="Reference Agreement", subtitle="Marker size scales with global quality score.")
        cbar = fig.colorbar(scatter, ax=ax_right, fraction=0.046, pad=0.04)
        cbar.set_label("Global quality score", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    else:
        _apply_card(ax_right, title="Reference Agreement")
        ax_right.axis("off")
        ax_right.text(0.5, 0.5, "At least two references are required for this panel.", ha="center", va="center")

    if legend_handles:
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=9, label=family)
            for family, color in legend_handles.items()
        ]
        ax_left.legend(handles=handles, title="Highlighted families", frameon=False, loc="lower right", fontsize=7.5, title_fontsize=8)

    fig.suptitle("Cross-Report Consistency", x=0.02, y=0.99, ha="left", fontsize=19, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_family_summary(
    pdf: PdfPages,
    *,
    quality: QualityArtifacts,
    alignment: AlignmentArtifacts,
    family_color_map: dict[str, str],
) -> None:
    quality_fam = quality.family_summary.copy()
    alignment_fam = alignment.family_summary.copy()
    families = sorted(set(quality_fam["family"]).union(alignment_fam["family"]))
    family_order = (
        quality_fam.merge(alignment_fam, on="family", how="outer")
        .fillna(0.0)
        .assign(joint=lambda x: x.get("family_quality_median", 0.0) + x.get("family_alignment_median", 0.0))
        .sort_values("joint", ascending=False)["family"]
        .tolist()
    )

    fig = plt.figure(figsize=(13.0, 8.6))
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    for ax, table, value_col, title in [
        (fig.add_subplot(gs[0, 0]), quality_fam, "family_quality_median", "Family Median Quality"),
        (fig.add_subplot(gs[0, 1]), alignment_fam, "family_alignment_median", "Family Median Alignment"),
    ]:
        _apply_card(ax, title=title, subtitle="Median global score across models in the family.")
        ordered = table.set_index("family").reindex(family_order).dropna(subset=[value_col]).reset_index()
        y_pos = np.arange(len(ordered))
        colors = [family_color_map.get(family, TEAL) for family in ordered["family"]]
        ax.barh(y_pos, ordered[value_col], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{family} ({count})" for family, count in zip(ordered["family"], ordered["model_count"])])
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Median robust score")
        ax.grid(axis="x", alpha=0.35)
        for pos, value in enumerate(ordered[value_col]):
            ax.text(float(value) + 0.01, pos, _format_score(value), va="center", fontsize=9)
    fig.suptitle("Family-Level Summary", x=0.02, y=0.99, ha="left", fontsize=19, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_profiles(
    pdf: PdfPages,
    *,
    quality: QualityArtifacts,
    family_color_map: dict[str, str],
    models: list[str],
    page_title: str,
) -> None:
    fig = plt.figure(figsize=(13.0, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)
    for index in range(4):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        if index >= len(models):
            ax.axis("off")
            continue
        model_key = models[index]
        model_profile = quality.layer_profiles[quality.layer_profiles["model_key"] == model_key].copy()
        if model_profile.empty:
            ax.axis("off")
            continue
        family = str(model_profile.iloc[0]["family"])
        color = family_color_map.get(family, TEAL)
        best_row = model_profile.sort_values(["mean_quality", "layer_index"], ascending=[False, True]).iloc[0]
        _apply_card(ax, title=model_key, subtitle=None)
        ax.fill_between(
            model_profile["layer_index"],
            model_profile["min_quality"],
            model_profile["max_quality"],
            color=color,
            alpha=0.16,
        )
        ax.plot(model_profile["layer_index"], model_profile["mean_quality"], color=color, linewidth=2.2)
        ax.scatter(model_profile["layer_index"], model_profile["mean_quality"], color=color, s=18, zorder=3)
        ax.scatter([best_row["layer_index"]], [best_row["mean_quality"]], color=GOLD, edgecolor="white", s=90, zorder=4)
        ax.axvline(best_row["layer_index"], color=GOLD, linestyle="--", linewidth=1.1, alpha=0.8)
        ax.text(
            0.02,
            0.92,
            f"{family} | best {best_row['layer_index']}: {_short_layer_name(str(best_row['layer_name']))}",
            transform=ax.transAxes,
            fontsize=8.2,
            color=MUTED,
        )
        ax.text(
            0.02,
            0.05,
            f"score={_format_score(float(best_row['mean_quality']))}",
            transform=ax.transAxes,
            fontsize=8.2,
            color=MUTED,
        )
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Robust quality score")
        ax.grid(alpha=0.28)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    fig.suptitle(page_title, x=0.02, y=0.99, ha="left", fontsize=19, fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_family_metric_pages(
    pdf: PdfPages,
    *,
    scores: pd.DataFrame,
    family_color_map: dict[str, str],
    section_title: str,
    note: str | None = None,
) -> None:
    if scores.empty:
        return
    for corr_metric in ["scc", "pcc"]:
        metric_df = scores[scores["corr_metric"] == corr_metric].copy()
        if metric_df.empty:
            continue
        family_order = (
            metric_df.groupby("family")["global_metric_score"].max().sort_values(ascending=False).index.tolist()
        )
        metric_min = max(0.0, float(metric_df["global_metric_score"].min()) - 0.05)
        per_page = 6
        for start in range(0, len(family_order), per_page):
            chunk = family_order[start : start + per_page]
            fig = plt.figure(figsize=(13.0, 8.8))
            gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.22)
            for index in range(per_page):
                ax = fig.add_subplot(gs[index // 2, index % 2])
                if index >= len(chunk):
                    ax.axis("off")
                    continue
                family = chunk[index]
                fam = metric_df[metric_df["family"] == family].sort_values(
                    ["family_rank", "global_metric_score", "model_key"],
                    ascending=[True, False, True],
                ).head(3)
                base_color = family_color_map.get(family, TEAL)
                bar_colors = [_soften_color(base_color, amount) for amount in (0.15, 0.35, 0.55)[: len(fam)]]
                ordered = fam.iloc[::-1].copy()
                _apply_card(
                    ax,
                    title=family,
                    subtitle=f"Top {len(fam)} models | {corr_metric.upper()} | best layer in table files",
                )
                ax.barh(ordered["model_key"], ordered["global_metric_score"], color=bar_colors[::-1], edgecolor="white", linewidth=0.8)
                ax.set_xlim(metric_min, 1.0)
                ax.grid(axis="x", alpha=0.35)
                ax.tick_params(axis="y", labelsize=8)
                ax.tick_params(axis="x", labelsize=8)
                for ypos, (_, row) in enumerate(ordered.iterrows()):
                    ax.text(float(row["global_metric_score"]) + 0.006, ypos, _format_score(row["global_metric_score"]), va="center", fontsize=8)
                if index // 2 == 2:
                    ax.set_xlabel(f"{corr_metric.upper()} robust score")
            fig.suptitle(
                f"{section_title}: {corr_metric.upper()} grouped by family",
                x=0.02,
                y=0.99,
                ha="left",
                fontsize=18,
                fontweight="bold",
            )
            if note:
                fig.text(0.98, 0.02, note, fontsize=9, color=MUTED, ha="right")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _plot_dataset_family_metric_pages(
    pdf: PdfPages,
    *,
    scores: pd.DataFrame,
    family_color_map: dict[str, str],
    section_title: str,
    note: str | None = None,
    top_n: int = 3,
) -> None:
    if scores.empty:
        return
    dataset_order = list(dict.fromkeys(scores["dataset"].tolist()))
    for dataset in dataset_order:
        dataset_df = scores[scores["dataset"] == dataset].copy()
        for corr_metric in ["scc", "pcc"]:
            metric_df = dataset_df[dataset_df["corr_metric"] == corr_metric].copy()
            if metric_df.empty:
                continue
            family_order = (
                metric_df.groupby("family")["metric_score"].max().sort_values(ascending=False).index.tolist()
            )
            metric_min = max(0.0, float(metric_df["metric_score"].min()) - 0.05)
            per_page = 6
            for start in range(0, len(family_order), per_page):
                chunk = family_order[start : start + per_page]
                fig = plt.figure(figsize=(13.0, 8.8))
                gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.22)
                for index in range(per_page):
                    ax = fig.add_subplot(gs[index // 2, index % 2])
                    if index >= len(chunk):
                        ax.axis("off")
                        continue
                    family = chunk[index]
                    fam = metric_df[metric_df["family"] == family].sort_values(
                        ["family_rank", "metric_score", "model_key"],
                        ascending=[True, False, True],
                    ).head(top_n)
                    base_color = family_color_map.get(family, TEAL)
                    bar_colors = [_soften_color(base_color, amount) for amount in (0.15, 0.35, 0.55)[: len(fam)]]
                    ordered = fam.iloc[::-1].copy()
                    _apply_card(ax, title=family, subtitle=f"{dataset} | top {len(fam)} models")
                    ax.barh(ordered["model_key"], ordered["metric_score"], color=bar_colors[::-1], edgecolor="white", linewidth=0.8)
                    ax.set_xlim(metric_min, 1.0)
                    ax.grid(axis="x", alpha=0.35)
                    ax.tick_params(axis="y", labelsize=8)
                    ax.tick_params(axis="x", labelsize=8)
                    top_row = fam.iloc[0]
                    ax.text(
                        0.02,
                        0.05,
                        f"best layer: {_short_layer_name(str(top_row['layer_name']))}",
                        transform=ax.transAxes,
                        fontsize=7.8,
                        color=MUTED,
                    )
                    for ypos, (_, row) in enumerate(ordered.iterrows()):
                        ax.text(float(row["metric_score"]) + 0.006, ypos, _format_score(row["metric_score"]), va="center", fontsize=8)
                    if index // 2 == 2:
                        ax.set_xlabel(f"{corr_metric.upper()} robust score")
                fig.suptitle(
                    f"{section_title}: {dataset} ({corr_metric.upper()}) grouped by family",
                    x=0.02,
                    y=0.99,
                    ha="left",
                    fontsize=18,
                    fontweight="bold",
                )
                if note:
                    fig.text(0.98, 0.02, note, fontsize=9, color=MUTED, ha="right")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def _write_summary_tables(
    output_dir: Path,
    *,
    quality: QualityArtifacts,
    alignment: AlignmentArtifacts,
    quality_family_metric_scores: pd.DataFrame,
    alignment_family_metric_scores: pd.DataFrame,
    quality_dataset_family_metric_scores: pd.DataFrame,
    alignment_dataset_family_metric_scores: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality.model_ranking.to_csv(output_dir / "quality_model_ranking.tsv", sep="\t", index=False)
    quality.top_models.to_csv(output_dir / "quality_top10.tsv", sep="\t", index=False)
    quality.dataset_best.to_csv(output_dir / "quality_best_layers_by_dataset.tsv", sep="\t", index=False)
    quality.dataset_winners.to_csv(output_dir / "quality_dataset_winners.tsv", sep="\t", index=False)
    alignment.model_ranking.to_csv(output_dir / "alignment_model_ranking.tsv", sep="\t", index=False)
    alignment.reference_ranking.to_csv(output_dir / "alignment_reference_ranking.tsv", sep="\t", index=False)
    alignment.dataset_best.to_csv(output_dir / "alignment_best_layers_by_dataset.tsv", sep="\t", index=False)

    merged = quality.model_ranking.merge(
        alignment.model_ranking[["model_key", "global_alignment_score"]],
        on="model_key",
        how="inner",
    )
    merged["joint_score"] = (merged["global_quality_score"] + merged["global_alignment_score"]) / 2.0
    merged.sort_values(["joint_score", "global_quality_score", "global_alignment_score"], ascending=[False, False, False]).to_csv(
        output_dir / "quality_alignment_joint.tsv", sep="\t", index=False
    )
    quality_family_metric_scores.to_csv(output_dir / "quality_family_scc_pcc.tsv", sep="\t", index=False)
    alignment_family_metric_scores.to_csv(output_dir / "alignment_family_scc_pcc.tsv", sep="\t", index=False)
    quality_dataset_family_metric_scores.to_csv(output_dir / "quality_family_scc_pcc_by_dataset.tsv", sep="\t", index=False)
    alignment_dataset_family_metric_scores.to_csv(output_dir / "alignment_family_scc_pcc_by_dataset.tsv", sep="\t", index=False)


def _write_text_report(
    output_dir: Path,
    *,
    quality: QualityArtifacts,
    alignment: AlignmentArtifacts,
    quality_source: str,
    alignment_source: str,
    filter_summary: dict[str, Any] | None = None,
) -> None:
    top_lines = []
    top_lines.append("Embedding Quality And Alignment Report")
    top_lines.append("")
    top_lines.append(f"Quality source: {quality_source}")
    top_lines.append(f"Alignment source: {alignment_source}")
    top_lines.append(f"Datasets: {', '.join(quality.dataset_order)}")
    if filter_summary is not None:
        top_lines.append(
            f"Completeness filter: retained {filter_summary['retained_dataset_count']} datasets and {filter_summary['retained_model_count']} models"
        )
    top_lines.append("")
    top_lines.append("Top-10 models by embedding_quality (non-IQA, full coverage)")
    for _, row in quality.top_models.iterrows():
        top_lines.append(
            f"{int(row['quality_rank']):2d}. {row['model_key']} | family={row['family']} | "
            f"score={float(row['global_quality_score']):.4f} | best_layer={row['global_best_layer']}"
        )
    top_lines.append("")
    top_lines.append("Per-dataset winners")
    for _, row in quality.dataset_winners.iterrows():
        top_lines.append(
            f"- {row['dataset']}: {row['model_key']} ({row['family']}) | "
            f"score={float(row['quality_score']):.4f} | layer={row['best_layer']}"
        )
    top_lines.append("")
    top_lines.append("Top-5 models by embedding_alignment")
    for _, row in alignment.top_models.head(5).iterrows():
        top_lines.append(
            f"- {row['model_key']} | family={row['family']} | "
            f"score={float(row['global_alignment_score']):.4f} | best_layer={row['global_best_layer']}"
        )
    (output_dir / "best_models_report.txt").write_text("\n".join(top_lines) + "\n", encoding="utf-8")


def build_visual_report(args: argparse.Namespace) -> Path:
    _configure_style()
    model_meta = _build_model_metadata()
    excluded_families = {str(item).strip() for item in args.exclude_families if str(item).strip()}

    quality_rows, quality_meta = _load_report_table(
        args.quality_report,
        preferred_names=("report.json", "results.tsv", "results.csv"),
        results_key="results",
    )
    alignment_rows, alignment_meta = _load_report_table(
        args.alignment_report,
        preferred_names=("report.json", "results.tsv", "results.csv"),
        results_key="results",
    )

    filter_summary: dict[str, Any] | None = None
    if args.complete_intersection_only:
        quality_base = _preprocess_quality_base(
            quality_rows,
            model_meta=model_meta,
            excluded_families=excluded_families,
        )
        alignment_base = _preprocess_alignment_base(
            alignment_rows,
            model_meta=model_meta,
            excluded_families=excluded_families,
        )
        retained_datasets, retained_models, filter_summary = _compute_complete_intersection(
            quality_base,
            alignment_base,
        )
        quality_rows = quality_rows[
            quality_rows["dataset"].isin(retained_datasets) & quality_rows["model_key"].isin(retained_models)
        ].copy()
        alignment_rows = alignment_rows[
            alignment_rows["dataset"].isin(retained_datasets)
            & alignment_rows["candidate_model_key"].isin(retained_models)
        ].copy()

    quality = _prepare_quality_artifacts(
        quality_rows,
        quality_meta,
        model_meta=model_meta,
        excluded_families=excluded_families,
        top_k=max(1, args.top_k),
    )
    alignment = _prepare_alignment_artifacts(
        alignment_rows,
        alignment_meta,
        model_meta=model_meta,
        excluded_families=excluded_families,
        top_k=max(1, args.top_k),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    family_color_map = _build_family_color_map(
        quality.model_ranking["family"].tolist() + alignment.model_ranking["family"].tolist()
    )
    quality_family_metric_scores = _build_quality_metric_family_scores(quality)
    alignment_family_metric_scores = _build_alignment_metric_family_scores(alignment)
    quality_dataset_family_metric_scores = _build_quality_dataset_metric_family_scores(quality)
    alignment_dataset_family_metric_scores = _build_alignment_dataset_metric_family_scores(alignment)
    note = None
    if filter_summary is not None:
        note = (
            f"complete intersection: {filter_summary['retained_dataset_count']} datasets, "
            f"{filter_summary['retained_model_count']} models"
        )

    pdf_path = output_dir / "embedding_quality_alignment_supplementary.pdf"
    with PdfPages(pdf_path) as pdf:
        _plot_report_tables(pdf, quality=quality, alignment=alignment, family_color_map=family_color_map, note=note)
        _plot_support_tables(pdf, quality=quality, alignment=alignment, family_color_map=family_color_map, note=note)
        _plot_quality_heatmap(pdf, quality=quality)
        _plot_alignment_reference_heatmaps(pdf, alignment=alignment)
        _plot_family_metric_pages(
            pdf,
            scores=quality_family_metric_scores,
            family_color_map=family_color_map,
            section_title="Embedding quality",
            note=note,
        )
        _plot_family_metric_pages(
            pdf,
            scores=alignment_family_metric_scores,
            family_color_map=family_color_map,
            section_title="Embedding alignment",
            note=note,
        )
        _plot_scatter_pages(pdf, quality=quality, alignment=alignment, family_color_map=family_color_map)
        _plot_family_summary(pdf, quality=quality, alignment=alignment, family_color_map=family_color_map)
        top_models = quality.top_models["model_key"].tolist()
        for start in range(0, len(top_models), 4):
            end = min(len(top_models), start + 4)
            _plot_layer_profiles(
                pdf,
                quality=quality,
                family_color_map=family_color_map,
                models=top_models[start:end],
                page_title=f"Layer Profiles For Top Models ({start + 1}-{end})",
            )

    family_dataset_pdf_path = output_dir / "embedding_family_by_dataset_scc_pcc.pdf"
    with PdfPages(family_dataset_pdf_path) as pdf:
        _plot_dataset_family_metric_pages(
            pdf,
            scores=quality_dataset_family_metric_scores,
            family_color_map=family_color_map,
            section_title="Embedding quality",
            note=note,
        )
        _plot_dataset_family_metric_pages(
            pdf,
            scores=alignment_dataset_family_metric_scores,
            family_color_map=family_color_map,
            section_title="Embedding alignment",
            note=note,
        )

    _write_summary_tables(
        output_dir,
        quality=quality,
        alignment=alignment,
        quality_family_metric_scores=quality_family_metric_scores,
        alignment_family_metric_scores=alignment_family_metric_scores,
        quality_dataset_family_metric_scores=quality_dataset_family_metric_scores,
        alignment_dataset_family_metric_scores=alignment_dataset_family_metric_scores,
    )
    _write_text_report(
        output_dir,
        quality=quality,
        alignment=alignment,
        quality_source=str(quality_meta.get("source_path", args.quality_report)),
        alignment_source=str(alignment_meta.get("source_path", args.alignment_report)),
        filter_summary=filter_summary,
    )

    manifest = {
        "quality_source": str(quality_meta.get("source_path", args.quality_report)),
        "alignment_source": str(alignment_meta.get("source_path", args.alignment_report)),
        "pdf": str(pdf_path),
        "family_dataset_pdf": str(family_dataset_pdf_path),
        "tables": {
            "quality_model_ranking": str(output_dir / "quality_model_ranking.tsv"),
            "quality_top10": str(output_dir / "quality_top10.tsv"),
            "quality_best_layers_by_dataset": str(output_dir / "quality_best_layers_by_dataset.tsv"),
            "quality_dataset_winners": str(output_dir / "quality_dataset_winners.tsv"),
            "alignment_model_ranking": str(output_dir / "alignment_model_ranking.tsv"),
            "alignment_reference_ranking": str(output_dir / "alignment_reference_ranking.tsv"),
            "alignment_best_layers_by_dataset": str(output_dir / "alignment_best_layers_by_dataset.tsv"),
            "quality_alignment_joint": str(output_dir / "quality_alignment_joint.tsv"),
            "quality_family_scc_pcc": str(output_dir / "quality_family_scc_pcc.tsv"),
            "alignment_family_scc_pcc": str(output_dir / "alignment_family_scc_pcc.tsv"),
            "quality_family_scc_pcc_by_dataset": str(output_dir / "quality_family_scc_pcc_by_dataset.tsv"),
            "alignment_family_scc_pcc_by_dataset": str(output_dir / "alignment_family_scc_pcc_by_dataset.tsv"),
        },
        "text_report": str(output_dir / "best_models_report.txt"),
        "excluded_families": sorted(excluded_families),
    }
    if filter_summary is not None:
        manifest["filter_summary"] = filter_summary
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pdf_path


def main() -> None:
    args = parse_args()
    pdf_path = build_visual_report(args)
    print(f"Saved supplementary PDF: {pdf_path}")
    print(f"Saved family-by-dataset PDF: {args.output_dir / 'embedding_family_by_dataset_scc_pcc.pdf'}")
    print(f"Saved summary tables in: {args.output_dir}")


if __name__ == "__main__":
    main()
