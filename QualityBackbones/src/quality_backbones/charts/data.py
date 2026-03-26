from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quality_backbones.evaluation import FULL_REFERENCE_DATASETS
from quality_backbones.manifest import iter_enabled_image_model_specs


FLOPS_LOG_DIR = Path(__file__).resolve().parents[3] / "logs" / "full_flops_20260324_v2"


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
    layer_profiles: pd.DataFrame
    dataset_order: list[str]
    reference_order: list[str]
    total_datasets: int
    nan_rows: int


@dataclass(frozen=True)
class ReportExperimentArtifacts:
    slug: str
    title: str
    slice_name: str
    quality_source: str
    alignment_source: str
    excluded_families: tuple[str, ...]
    quality: QualityArtifacts
    alignment: AlignmentArtifacts
    joint_ranking: pd.DataFrame
    joint_layer_profiles: pd.DataFrame
    quality_family_metric_scores: pd.DataFrame
    alignment_family_metric_scores: pd.DataFrame
    quality_dataset_family_metric_scores: pd.DataFrame
    alignment_dataset_family_metric_scores: pd.DataFrame
    family_color_map: dict[str, str]


@dataclass(frozen=True)
class DeltaExperimentArtifacts:
    slug: str
    title: str
    quality_gap: pd.DataFrame
    alignment_gap: pd.DataFrame
    joint_gap: pd.DataFrame
    fr: ReportExperimentArtifacts
    nr: ReportExperimentArtifacts


@dataclass(frozen=True)
class QualityScopeComparisonArtifacts:
    slug: str
    title: str
    slice_name: str
    correct_source: str
    wrong_source: str
    excluded_families: tuple[str, ...]
    correct: QualityArtifacts
    wrong: QualityArtifacts
    model_comparison: pd.DataFrame
    dataset_model_comparison: pd.DataFrame
    dataset_summary: pd.DataFrame
    family_shift: pd.DataFrame
    layer_changes: pd.DataFrame
    summary_metrics: dict[str, Any]
    family_color_map: dict[str, str]


@dataclass(frozen=True)
class TripletArtifacts:
    rows: pd.DataFrame
    layer_scores: pd.DataFrame
    dataset_best: pd.DataFrame
    model_ranking: pd.DataFrame
    top_models: pd.DataFrame
    dataset_winners: pd.DataFrame
    family_summary: pd.DataFrame
    layer_profiles: pd.DataFrame
    dataset_order: list[str]
    total_datasets: int
    nan_rows: int


@dataclass(frozen=True)
class TripletExperimentArtifacts:
    slug: str
    title: str
    slice_name: str
    triplet_source: str
    excluded_families: tuple[str, ...]
    triplet: TripletArtifacts
    triplet_family_scores: pd.DataFrame
    triplet_dataset_family_scores: pd.DataFrame
    family_color_map: dict[str, str]


@dataclass(frozen=True)
class TripletDeltaExperimentArtifacts:
    slug: str
    title: str
    triplet_gap: pd.DataFrame
    fr: TripletExperimentArtifacts
    nr: TripletExperimentArtifacts


def resolve_report_path(path: Path, *, preferred_names: tuple[str, ...]) -> Path:
    if path.is_dir():
        for name in preferred_names:
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No supported report files inside {path}; tried: {preferred_names}")
    if path.exists():
        return path
    raise FileNotFoundError(f"Report path does not exist: {path}")


def load_report_table(
    path: Path,
    *,
    preferred_names: tuple[str, ...],
    results_key: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = resolve_report_path(path, preferred_names=preferred_names)
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


def build_model_metadata() -> pd.DataFrame:
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


@lru_cache(maxsize=1)
def _load_flops_profile_table_cached() -> pd.DataFrame:
    profile_paths = sorted(FLOPS_LOG_DIR.glob("*.tsv"))
    if not profile_paths:
        raise FileNotFoundError(f"No FLOPs profiling tables found under {FLOPS_LOG_DIR}")

    frames = [pd.read_csv(path, sep="\t") for path in profile_paths]
    table = pd.concat(frames, ignore_index=True)
    for column in ("layer_index", "flops_total", "input_height", "input_width"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    table = table[
        table["model_key"].notna() & table["layer_index"].notna() & table["flops_total"].notna()
    ].copy()
    table["layer_index"] = table["layer_index"].astype(int)
    table["layer_name"] = table["layer_name"].astype(str)
    table = (
        table.sort_values(["model_key", "layer_index", "flops_total", "layer_name"])
        .drop_duplicates(["model_key", "layer_index"], keep="last")
        .reset_index(drop=True)
    )
    return table


def load_flops_profile_table() -> pd.DataFrame:
    return _load_flops_profile_table_cached().copy()


def build_selected_layer_flops_table(
    model_ranking: pd.DataFrame,
    *,
    score_col: str,
    rank_col: str,
) -> pd.DataFrame:
    required = {
        "model_key",
        "family",
        "global_best_layer_index",
        "global_best_layer",
        score_col,
        rank_col,
    }
    missing = sorted(required - set(model_ranking.columns))
    if missing:
        raise KeyError(f"Missing columns for FLOPs join: {missing}")

    ranking = model_ranking.copy().reset_index(drop=True)
    ranking["global_best_layer_index"] = pd.to_numeric(
        ranking["global_best_layer_index"], errors="coerce"
    ).astype("Int64")

    profiles = _load_flops_profile_table_cached()
    profile_cols = ["model_key", "layer_index", "layer_name", "flops_total", "input_height", "input_width"]
    profile_table = profiles.loc[:, profile_cols].copy()

    match_by_index = ranking.merge(
        profile_table.rename(
            columns={
                "layer_index": "matched_layer_index",
                "layer_name": "matched_layer_name",
                "flops_total": "selected_flops",
                "input_height": "profile_input_height",
                "input_width": "profile_input_width",
            }
        ),
        left_on=["model_key", "global_best_layer_index"],
        right_on=["model_key", "matched_layer_index"],
        how="left",
    )
    match_by_name = ranking.merge(
        profile_table.rename(
            columns={
                "layer_index": "fallback_layer_index",
                "layer_name": "fallback_layer_name",
                "flops_total": "fallback_selected_flops",
                "input_height": "fallback_input_height",
                "input_width": "fallback_input_width",
            }
        ),
        left_on=["model_key", "global_best_layer"],
        right_on=["model_key", "fallback_layer_name"],
        how="left",
    )

    joined = match_by_index.copy()
    joined["profile_layer_index"] = joined["matched_layer_index"].combine_first(match_by_name["fallback_layer_index"])
    joined["profile_layer_name"] = joined["matched_layer_name"].combine_first(match_by_name["fallback_layer_name"])
    joined["selected_flops"] = joined["selected_flops"].combine_first(match_by_name["fallback_selected_flops"])
    joined["profile_input_height"] = joined["profile_input_height"].combine_first(
        match_by_name["fallback_input_height"]
    )
    joined["profile_input_width"] = joined["profile_input_width"].combine_first(
        match_by_name["fallback_input_width"]
    )
    joined["flops_join_mode"] = np.where(
        joined["matched_layer_index"].notna(),
        "index",
        np.where(match_by_name["fallback_layer_index"].notna(), "name", "missing"),
    )

    full_flops = (
        profiles.groupby("model_key", as_index=False)
        .agg(
            full_flops=("flops_total", "max"),
            profiled_layer_count=("layer_index", "nunique"),
        )
        .sort_values("model_key")
        .reset_index(drop=True)
    )
    joined = joined.merge(full_flops, on="model_key", how="left")
    joined["selected_to_full_ratio"] = joined["selected_flops"] / joined["full_flops"]
    joined["flops_saved_ratio"] = 1.0 - joined["selected_to_full_ratio"]
    joined["selected_flops_g"] = joined["selected_flops"] / 1e9
    joined["full_flops_g"] = joined["full_flops"] / 1e9
    joined["layer_name_match"] = joined["profile_layer_name"].eq(joined["global_best_layer"])
    return joined.sort_values([rank_col, score_col, "model_key"], ascending=[True, False, True]).reset_index(
        drop=True
    )


def build_quality_flops_comparison(quality: QualityArtifacts) -> pd.DataFrame:
    return build_selected_layer_flops_table(
        quality.model_ranking,
        score_col="global_quality_score",
        rank_col="quality_rank",
    )


def build_alignment_flops_comparison(alignment: AlignmentArtifacts) -> pd.DataFrame:
    return build_selected_layer_flops_table(
        alignment.model_ranking,
        score_col="global_alignment_score",
        rank_col="alignment_rank",
    )


def build_triplet_flops_comparison(triplet: TripletArtifacts) -> pd.DataFrame:
    return build_selected_layer_flops_table(
        triplet.model_ranking,
        score_col="global_triplet_score",
        rank_col="triplet_rank",
    )


def infer_family(model_key: str) -> str:
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


def enrich_with_metadata(df: pd.DataFrame, model_col: str, model_meta: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(model_meta, left_on=model_col, right_on="model_key", how="left")
    merged["family"] = merged["family"].fillna(merged[model_col].map(infer_family))
    merged["size"] = merged["size"].fillna("-")
    return merged


def rank_to_percentile(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.ones(len(values), dtype=np.float64), index=values.index)
    ranks = values.rank(method="average", ascending=False)
    return 1.0 - (ranks - 1.0) / (len(values) - 1.0)


def slice_fingerprint(df: pd.DataFrame, key_cols: list[str]) -> str:
    ordered = df[key_cols + ["value"]].sort_values(key_cols).reset_index(drop=True).copy()
    ordered["value"] = ordered["value"].round(12)
    hashed = pd.util.hash_pandas_object(ordered, index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def choose_slice_representative(group: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    return sorted(group, key=lambda item: (item[0], item[1], 0 if item[2] == "abs" else 1, item[2]))[0]


def deduplicate_quality_slices(df: pd.DataFrame) -> tuple[pd.DataFrame, list[list[tuple[str, str, str]]]]:
    key_cols = ["dataset", "model_key", "layer_index", "layer_name"]
    slice_cols = ["corr_metric", "embedding_distance", "score_distance"]
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for slice_key, sub in df.groupby(slice_cols, sort=True):
        fingerprint = slice_fingerprint(sub, key_cols=key_cols)
        normalized_key = (str(slice_key[0]), str(slice_key[1]), str(slice_key[2]))
        groups.setdefault(fingerprint, []).append(normalized_key)
    duplicate_groups = sorted((sorted(items) for items in groups.values()), key=lambda items: items[0])
    keep = [choose_slice_representative(group) for group in duplicate_groups]
    keep_df = pd.DataFrame(keep, columns=slice_cols)
    deduped = df.merge(keep_df, on=slice_cols, how="inner")
    return deduped, duplicate_groups


def filter_rows_by_slice(df: pd.DataFrame, dataset_col: str, slice_name: str) -> pd.DataFrame:
    if slice_name == "overall":
        return df.copy()
    fr_datasets = set(FULL_REFERENCE_DATASETS)
    if slice_name == "fr":
        return df[df[dataset_col].isin(fr_datasets)].copy()
    if slice_name == "nr":
        return df[~df[dataset_col].isin(fr_datasets)].copy()
    raise ValueError(f"Unsupported slice: {slice_name}")


def prepare_quality_artifacts(
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
    df = enrich_with_metadata(df, model_col="model_key", model_meta=model_meta)
    df = df[~df["family"].isin(excluded_families)].copy()

    present_datasets = set(df["dataset"].unique().tolist())
    dataset_order = [
        dataset
        for dataset in list(quality_meta.get("config", {}).get("datasets") or [])
        if dataset in present_datasets
    ]
    if not dataset_order:
        dataset_order = list(dict.fromkeys(df["dataset"].tolist()))

    deduped, duplicate_groups = deduplicate_quality_slices(df)
    deduped["quality_percentile"] = deduped.groupby(
        ["dataset", "corr_metric", "embedding_distance", "score_distance"]
    )["value"].transform(rank_to_percentile)

    layer_scores = (
        deduped.groupby(["dataset", "model_key", "family", "layer_index", "layer_name"], as_index=False)
        .agg(
            quality_score=("quality_percentile", "mean"),
            slice_count=("quality_percentile", "count"),
            raw_value_mean=("value", "mean"),
        )
        .sort_values(
            ["dataset", "model_key", "quality_score", "raw_value_mean", "layer_index"],
            ascending=[True, True, False, False, True],
        )
    )

    dataset_best = layer_scores.drop_duplicates(["dataset", "model_key"]).copy()
    dataset_best["dataset_rank"] = dataset_best.groupby("dataset")["quality_score"].rank(
        method="average", ascending=False
    )
    dataset_best["dataset_rank_int"] = dataset_best.groupby("dataset")["quality_score"].rank(
        method="first", ascending=False
    ).astype(int)

    dataset_winners = (
        dataset_best.sort_values(
            ["dataset", "quality_score", "raw_value_mean", "model_key"],
            ascending=[True, False, False, True],
        )
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


def prepare_triplet_artifacts(
    triplet_rows: pd.DataFrame,
    triplet_meta: dict[str, Any],
    *,
    model_meta: pd.DataFrame,
    excluded_families: set[str],
    top_k: int,
) -> TripletArtifacts:
    df = triplet_rows.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    nan_rows = int(df["value"].isna().sum())
    df = df[df["value"].notna()].copy()
    df["layer_index"] = df["layer_index"].astype(int)
    df = enrich_with_metadata(df, model_col="model_key", model_meta=model_meta)
    df = df[~df["family"].isin(excluded_families)].copy()

    present_datasets = set(df["dataset"].unique().tolist())
    dataset_order = [
        dataset
        for dataset in list(triplet_meta.get("config", {}).get("datasets") or [])
        if dataset in present_datasets
    ]
    if not dataset_order:
        dataset_order = list(dict.fromkeys(df["dataset"].tolist()))

    df["triplet_percentile"] = df.groupby(
        ["dataset", "corr_metric", "embedding_distance", "score_distance"]
    )["value"].transform(rank_to_percentile)

    layer_scores = (
        df.groupby(["dataset", "model_key", "family", "layer_index", "layer_name"], as_index=False)
        .agg(
            triplet_score=("triplet_percentile", "mean"),
            slice_count=("triplet_percentile", "count"),
            raw_value_mean=("value", "mean"),
        )
        .sort_values(
            ["dataset", "model_key", "triplet_score", "raw_value_mean", "layer_index"],
            ascending=[True, True, False, False, True],
        )
    )

    dataset_best = layer_scores.drop_duplicates(["dataset", "model_key"]).copy()
    dataset_best["dataset_rank"] = dataset_best.groupby("dataset")["triplet_score"].rank(
        method="average", ascending=False
    )
    dataset_best["dataset_rank_int"] = dataset_best.groupby("dataset")["triplet_score"].rank(
        method="first", ascending=False
    ).astype(int)

    dataset_winners = (
        dataset_best.sort_values(
            ["dataset", "triplet_score", "raw_value_mean", "model_key"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("dataset")
        .loc[:, ["dataset", "model_key", "family", "layer_name", "layer_index", "triplet_score"]]
        .rename(columns={"layer_name": "best_layer", "layer_index": "best_layer_index"})
    )
    winner_counts = dataset_winners["model_key"].value_counts().rename_axis("model_key").reset_index(name="dataset_wins")

    full_coverage = len(dataset_order)
    best_layer_global = (
        layer_scores.groupby(["model_key", "family", "layer_index", "layer_name"], as_index=False)
        .agg(
            global_layer_triplet=("triplet_score", "mean"),
            layer_dataset_count=("dataset", "nunique"),
        )
        .sort_values(
            ["model_key", "global_layer_triplet", "layer_dataset_count", "layer_index"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("model_key")
        .rename(columns={"layer_name": "global_best_layer", "layer_index": "global_best_layer_index"})
    )
    max_layer_index = (
        df.groupby("model_key", as_index=False)["layer_index"]
        .max()
        .rename(columns={"layer_index": "max_layer_index"})
    )

    model_ranking = (
        dataset_best.groupby(["model_key", "family"], as_index=False)
        .agg(
            global_triplet_score=("triplet_score", "mean"),
            triplet_score_std=("triplet_score", lambda values: float(np.std(values, ddof=0))),
            mean_dataset_rank=("dataset_rank", "mean"),
            dataset_count=("dataset", "nunique"),
        )
        .merge(best_layer_global, on=["model_key", "family"], how="left")
        .merge(max_layer_index, on="model_key", how="left")
        .merge(winner_counts, on="model_key", how="left")
        .fillna({"dataset_wins": 0})
    )
    model_ranking["dataset_wins"] = model_ranking["dataset_wins"].astype(int)
    model_ranking["coverage_frac"] = model_ranking["dataset_count"] / max(1, full_coverage)
    model_ranking["headline_eligible"] = model_ranking["dataset_count"] == full_coverage
    denominator = model_ranking["max_layer_index"].replace(0, np.nan)
    model_ranking["global_best_layer_fraction"] = (
        model_ranking["global_best_layer_index"] / denominator
    ).fillna(1.0)
    model_ranking = model_ranking.sort_values(
        ["headline_eligible", "global_triplet_score", "mean_dataset_rank", "dataset_wins", "model_key"],
        ascending=[False, False, True, False, True],
    ).reset_index(drop=True)
    model_ranking.insert(0, "triplet_rank", np.arange(1, len(model_ranking) + 1))

    eligible = model_ranking[model_ranking["headline_eligible"]].copy()
    top_models = eligible.head(top_k).copy()
    if len(top_models) < top_k:
        fallback = model_ranking.loc[~model_ranking["model_key"].isin(top_models["model_key"])].head(top_k - len(top_models))
        top_models = pd.concat([top_models, fallback], ignore_index=True)

    family_summary = (
        model_ranking.groupby("family", as_index=False)
        .agg(
            family_triplet_median=("global_triplet_score", "median"),
            family_triplet_mean=("global_triplet_score", "mean"),
            mean_best_layer_fraction=("global_best_layer_fraction", "mean"),
            model_count=("model_key", "nunique"),
        )
        .sort_values(["family_triplet_median", "family_triplet_mean", "model_count"], ascending=[False, False, False])
    )

    layer_profiles = (
        layer_scores.groupby(["model_key", "family", "layer_index", "layer_name"], as_index=False)
        .agg(
            mean_triplet=("triplet_score", "mean"),
            min_triplet=("triplet_score", "min"),
            max_triplet=("triplet_score", "max"),
            dataset_count=("dataset", "nunique"),
        )
        .sort_values(["model_key", "layer_index"])
    )

    return TripletArtifacts(
        rows=df,
        layer_scores=layer_scores,
        dataset_best=dataset_best,
        model_ranking=model_ranking,
        top_models=top_models,
        dataset_winners=dataset_winners,
        family_summary=family_summary,
        layer_profiles=layer_profiles,
        dataset_order=dataset_order,
        total_datasets=full_coverage,
        nan_rows=nan_rows,
    )


def prepare_alignment_artifacts(
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
    df = enrich_with_metadata(df, model_col="candidate_model_key", model_meta=model_meta)
    df = df[~df["family"].isin(excluded_families)].copy()

    present_datasets = set(df["dataset"].unique().tolist())
    dataset_order = [
        dataset
        for dataset in list(alignment_meta.get("config", {}).get("datasets") or [])
        if dataset in present_datasets
    ]
    if not dataset_order:
        dataset_order = list(dict.fromkeys(df["dataset"].tolist()))

    present_refs = set(df["reference_model_key"].unique().tolist())
    reference_order = [
        reference
        for reference in list(alignment_meta.get("config", {}).get("reference_models") or [])
        if reference in present_refs
    ]
    if not reference_order:
        reference_order = sorted(df["reference_model_key"].unique().tolist())

    df["alignment_percentile"] = df.groupby(
        ["dataset", "reference_model_key", "corr_metric", "distance_metric"]
    )["value"].transform(rank_to_percentile)

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
        .sort_values(
            ["reference_model_key", "reference_alignment_score", "dataset_count", "candidate_model_key"],
            ascending=[True, False, False, True],
        )
        .rename(columns={"candidate_model_key": "model_key"})
    )

    reference_global = reference_ranking.pivot_table(
        index=["model_key", "family"],
        columns="reference_model_key",
        values="reference_alignment_score",
    ).reset_index()

    layer_profiles = (
        layer_scores.groupby(["candidate_model_key", "family", "candidate_layer_index", "candidate_layer_name"], as_index=False)
        .agg(
            mean_alignment=("alignment_score", "mean"),
            min_alignment=("alignment_score", "min"),
            max_alignment=("alignment_score", "max"),
            dataset_count=("dataset", "nunique"),
        )
        .rename(
            columns={
                "candidate_model_key": "model_key",
                "candidate_layer_index": "layer_index",
                "candidate_layer_name": "layer_name",
            }
        )
        .sort_values(["model_key", "layer_index"])
    )

    return AlignmentArtifacts(
        rows=df,
        layer_scores=layer_scores,
        dataset_best=dataset_best,
        model_ranking=model_ranking,
        reference_ranking=reference_ranking,
        top_models=top_models,
        family_summary=family_summary,
        reference_global=reference_global,
        layer_profiles=layer_profiles,
        dataset_order=dataset_order,
        reference_order=reference_order,
        total_datasets=full_coverage,
        nan_rows=nan_rows,
    )


def build_quality_metric_family_scores(quality: QualityArtifacts) -> pd.DataFrame:
    df, _ = deduplicate_quality_slices(quality.rows)
    df = df[df["corr_metric"].isin(["scc", "pcc"])].copy()
    df["metric_percentile"] = df.groupby(
        ["dataset", "corr_metric", "embedding_distance", "score_distance"]
    )["value"].transform(rank_to_percentile)

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
    model_scores = model_scores[model_scores["dataset_count"] == quality.total_datasets].copy()
    model_scores["family_rank"] = model_scores.groupby(["corr_metric", "family"])["global_metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return model_scores.sort_values(
        ["corr_metric", "family", "family_rank", "global_metric_score", "model_key"],
        ascending=[True, True, True, False, True],
    )


def build_alignment_metric_family_scores(alignment: AlignmentArtifacts) -> pd.DataFrame:
    df = alignment.rows[alignment.rows["corr_metric"].isin(["scc", "pcc"])].copy()
    df["metric_percentile"] = df.groupby(
        ["dataset", "reference_model_key", "corr_metric", "distance_metric"]
    )["value"].transform(rank_to_percentile)

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
    model_scores = model_scores[model_scores["dataset_count"] == alignment.total_datasets].copy()
    model_scores["family_rank"] = model_scores.groupby(["corr_metric", "family"])["global_metric_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return model_scores.sort_values(
        ["corr_metric", "family", "family_rank", "global_metric_score", "model_key"],
        ascending=[True, True, True, False, True],
    )


def build_quality_dataset_metric_family_scores(quality: QualityArtifacts) -> pd.DataFrame:
    df, _ = deduplicate_quality_slices(quality.rows)
    df = df[df["corr_metric"].isin(["scc", "pcc"])].copy()
    df["metric_percentile"] = df.groupby(
        ["dataset", "corr_metric", "embedding_distance", "score_distance"]
    )["value"].transform(rank_to_percentile)
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
    return dataset_best.sort_values(
        ["dataset", "corr_metric", "family", "family_rank", "metric_score"],
        ascending=[True, True, True, True, False],
    )


def build_alignment_dataset_metric_family_scores(alignment: AlignmentArtifacts) -> pd.DataFrame:
    df = alignment.rows[alignment.rows["corr_metric"].isin(["scc", "pcc"])].copy()
    df["metric_percentile"] = df.groupby(
        ["dataset", "reference_model_key", "corr_metric", "distance_metric"]
    )["value"].transform(rank_to_percentile)
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
    return dataset_best.sort_values(
        ["dataset", "corr_metric", "family", "family_rank", "metric_score"],
        ascending=[True, True, True, True, False],
    )


def build_triplet_family_scores(triplet: TripletArtifacts) -> pd.DataFrame:
    family_scores = triplet.model_ranking[
        [
            "family",
            "model_key",
            "global_triplet_score",
            "triplet_rank",
            "global_best_layer",
            "global_best_layer_index",
            "global_best_layer_fraction",
            "dataset_count",
            "dataset_wins",
        ]
    ].copy()
    family_scores["family_rank"] = family_scores.groupby("family")["global_triplet_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return family_scores.sort_values(
        ["family", "family_rank", "global_triplet_score", "model_key"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def build_triplet_dataset_family_scores(triplet: TripletArtifacts) -> pd.DataFrame:
    dataset_scores = triplet.dataset_best[
        ["dataset", "family", "model_key", "triplet_score", "dataset_rank_int", "layer_name", "layer_index"]
    ].copy()
    dataset_scores["family_rank"] = dataset_scores.groupby(["dataset", "family"])["triplet_score"].rank(
        method="first", ascending=False
    ).astype(int)
    return dataset_scores.sort_values(
        ["dataset", "family", "family_rank", "triplet_score", "model_key"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)


def build_joint_ranking(quality: QualityArtifacts, alignment: AlignmentArtifacts) -> pd.DataFrame:
    merged = quality.model_ranking.merge(
        alignment.model_ranking[["model_key", "global_alignment_score", "global_best_layer"]],
        on="model_key",
        how="inner",
        suffixes=("", "_alignment"),
    )
    merged["joint_score"] = (merged["global_quality_score"] + merged["global_alignment_score"]) / 2.0
    merged = merged.sort_values(
        ["joint_score", "global_quality_score", "global_alignment_score", "model_key"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    merged.insert(0, "joint_rank", np.arange(1, len(merged) + 1))
    return merged


def build_joint_layer_profiles(quality: QualityArtifacts, alignment: AlignmentArtifacts) -> pd.DataFrame:
    merged = quality.layer_profiles.merge(
        alignment.layer_profiles[["model_key", "layer_index", "layer_name", "mean_alignment", "min_alignment", "max_alignment"]],
        on=["model_key", "layer_index", "layer_name"],
        how="inner",
    )
    merged["joint_score"] = (merged["mean_quality"] + merged["mean_alignment"]) / 2.0
    return merged.sort_values(["model_key", "layer_index"]).reset_index(drop=True)


def build_gap_table(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    score_col: str,
    rank_col: str,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    merged = left[["model_key", "family", score_col, rank_col]].merge(
        right[["model_key", score_col, rank_col]],
        on="model_key",
        how="inner",
        suffixes=(f"_{left_label}", f"_{right_label}"),
    )
    merged[f"delta_{score_col}"] = merged[f"{score_col}_{left_label}"] - merged[f"{score_col}_{right_label}"]
    merged[f"delta_{rank_col}"] = merged[f"{rank_col}_{right_label}"] - merged[f"{rank_col}_{left_label}"]
    merged["abs_score_delta"] = merged[f"delta_{score_col}"].abs()
    return merged.sort_values(["abs_score_delta", f"delta_{score_col}", "model_key"], ascending=[False, False, True])


def build_joint_gap_table(fr_joint: pd.DataFrame, nr_joint: pd.DataFrame) -> pd.DataFrame:
    merged = fr_joint[["model_key", "family", "joint_score", "joint_rank", "global_quality_score", "global_alignment_score"]].merge(
        nr_joint[["model_key", "joint_score", "joint_rank", "global_quality_score", "global_alignment_score"]],
        on="model_key",
        how="inner",
        suffixes=("_fr", "_nr"),
    )
    merged["delta_joint_score"] = merged["joint_score_fr"] - merged["joint_score_nr"]
    merged["delta_joint_rank"] = merged["joint_rank_nr"] - merged["joint_rank_fr"]
    merged["delta_quality_score"] = merged["global_quality_score_fr"] - merged["global_quality_score_nr"]
    merged["delta_alignment_score"] = merged["global_alignment_score_fr"] - merged["global_alignment_score_nr"]
    merged["abs_joint_delta"] = merged["delta_joint_score"].abs()
    return merged.sort_values(["abs_joint_delta", "delta_joint_score", "model_key"], ascending=[False, False, True])


def _dataset_metadata_frame(meta: dict[str, Any], *, default_pair_scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in list(meta.get("dataset_summaries", [])):
        pair_scope = item.get("pair_scope")
        group_field = item.get("group_field")
        n_groups = item.get("n_groups")
        rows.append(
            {
                "dataset": str(item.get("dataset")),
                "pair_scope": str(pair_scope or default_pair_scope),
                "group_field": group_field,
                "n_groups": int(n_groups) if n_groups is not None else (1 if default_pair_scope == "global" else None),
                "n_pairs": int(item.get("n_pairs", 0)),
                "n_samples": int(item.get("n_samples", 0)),
                "metadata_has_grouping": bool(pair_scope is not None or group_field is not None or n_groups is not None),
            }
        )
    return pd.DataFrame(rows)


def build_quality_scope_model_comparison(correct: QualityArtifacts, wrong: QualityArtifacts) -> pd.DataFrame:
    merged = correct.model_ranking[
        [
            "model_key",
            "family",
            "quality_rank",
            "global_quality_score",
            "global_best_layer",
            "global_best_layer_index",
            "dataset_count",
            "headline_eligible",
        ]
    ].merge(
        wrong.model_ranking[
            [
                "model_key",
                "quality_rank",
                "global_quality_score",
                "global_best_layer",
                "global_best_layer_index",
                "dataset_count",
                "headline_eligible",
            ]
        ],
        on="model_key",
        how="inner",
        suffixes=("_correct", "_wrong"),
    )
    merged["delta_quality_score"] = merged["global_quality_score_correct"] - merged["global_quality_score_wrong"]
    merged["abs_quality_delta"] = merged["delta_quality_score"].abs()
    merged["delta_rank"] = merged["quality_rank_wrong"] - merged["quality_rank_correct"]
    merged["abs_rank_delta"] = merged["delta_rank"].abs()
    merged["layer_changed"] = merged["global_best_layer_correct"] != merged["global_best_layer_wrong"]
    merged["headline_eligible"] = merged["headline_eligible_correct"] & merged["headline_eligible_wrong"]
    return merged.sort_values(
        ["abs_quality_delta", "abs_rank_delta", "delta_quality_score", "model_key"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_quality_scope_dataset_model_comparison(correct: QualityArtifacts, wrong: QualityArtifacts) -> pd.DataFrame:
    merged = correct.dataset_best[
        [
            "dataset",
            "model_key",
            "family",
            "quality_score",
            "dataset_rank_int",
            "layer_name",
            "layer_index",
        ]
    ].merge(
        wrong.dataset_best[
            [
                "dataset",
                "model_key",
                "quality_score",
                "dataset_rank_int",
                "layer_name",
                "layer_index",
            ]
        ],
        on=["dataset", "model_key"],
        how="inner",
        suffixes=("_correct", "_wrong"),
    )
    merged["delta_quality_score"] = merged["quality_score_correct"] - merged["quality_score_wrong"]
    merged["abs_quality_delta"] = merged["delta_quality_score"].abs()
    merged["delta_rank"] = merged["dataset_rank_int_wrong"] - merged["dataset_rank_int_correct"]
    merged["abs_rank_delta"] = merged["delta_rank"].abs()
    merged["layer_changed"] = merged["layer_name_correct"] != merged["layer_name_wrong"]
    return merged.sort_values(
        ["dataset", "abs_quality_delta", "abs_rank_delta", "model_key"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def build_quality_scope_dataset_summary(
    correct_meta: dict[str, Any],
    wrong_meta: dict[str, Any],
    dataset_model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    correct_meta_df = _dataset_metadata_frame(correct_meta, default_pair_scope="within_ref")
    wrong_meta_df = _dataset_metadata_frame(wrong_meta, default_pair_scope="global")
    datasets = dataset_model_comparison["dataset"].drop_duplicates().tolist()
    correct_meta_df = correct_meta_df[correct_meta_df["dataset"].isin(datasets)].copy()
    wrong_meta_df = wrong_meta_df[wrong_meta_df["dataset"].isin(datasets)].copy()

    stats_rows: list[dict[str, Any]] = []
    for dataset, group in dataset_model_comparison.groupby("dataset", sort=False):
        top5_correct = set(
            group.sort_values(["dataset_rank_int_correct", "quality_score_correct", "model_key"], ascending=[True, False, True])
            .head(5)["model_key"]
            .tolist()
        )
        top5_wrong = set(
            group.sort_values(["dataset_rank_int_wrong", "quality_score_wrong", "model_key"], ascending=[True, False, True])
            .head(5)["model_key"]
            .tolist()
        )
        stats_rows.append(
            {
                "dataset": dataset,
                "mean_delta_quality": float(group["delta_quality_score"].mean()),
                "mean_abs_delta_quality": float(group["abs_quality_delta"].mean()),
                "max_abs_delta_quality": float(group["abs_quality_delta"].max()),
                "mean_abs_rank_delta": float(group["abs_rank_delta"].mean()),
                "layer_changed_rate": float(group["layer_changed"].mean()),
                "rank_spearman": float(group["dataset_rank_int_correct"].corr(group["dataset_rank_int_wrong"], method="spearman")),
                "top5_overlap": int(len(top5_correct & top5_wrong)),
            }
        )
    stats_df = pd.DataFrame(stats_rows)

    merged = correct_meta_df.merge(wrong_meta_df, on="dataset", how="inner", suffixes=("_correct", "_wrong"))
    merged = merged.merge(stats_df, on="dataset", how="left")
    merged["pair_count_ratio_wrong_over_correct"] = merged["n_pairs_wrong"] / merged["n_pairs_correct"].replace(0, np.nan)
    return merged.sort_values(["mean_abs_delta_quality", "pair_count_ratio_wrong_over_correct", "dataset"], ascending=[False, False, True]).reset_index(drop=True)


def build_quality_scope_family_shift(model_comparison: pd.DataFrame) -> pd.DataFrame:
    family_shift = (
        model_comparison.groupby("family", as_index=False)
        .agg(
            model_count=("model_key", "nunique"),
            median_quality_correct=("global_quality_score_correct", "median"),
            median_quality_wrong=("global_quality_score_wrong", "median"),
            mean_delta_quality=("delta_quality_score", "mean"),
            median_delta_quality=("delta_quality_score", "median"),
            mean_abs_delta_quality=("abs_quality_delta", "mean"),
            median_abs_rank_delta=("abs_rank_delta", "median"),
            layer_changed_rate=("layer_changed", "mean"),
        )
        .sort_values(["mean_abs_delta_quality", "layer_changed_rate", "model_count"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return family_shift


def build_quality_scope_layer_changes(
    model_comparison: pd.DataFrame,
    dataset_model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    changed_by_dataset = (
        dataset_model_comparison.groupby("model_key", as_index=False)
        .agg(
            changed_datasets=("layer_changed", "sum"),
            dataset_layer_change_rate=("layer_changed", "mean"),
        )
    )
    layer_changes = (
        model_comparison[
            [
                "model_key",
                "family",
                "global_best_layer_correct",
                "global_best_layer_wrong",
                "global_best_layer_index_correct",
                "global_best_layer_index_wrong",
                "delta_quality_score",
                "abs_quality_delta",
                "delta_rank",
                "abs_rank_delta",
                "layer_changed",
            ]
        ]
        .merge(changed_by_dataset, on="model_key", how="left")
        .sort_values(["changed_datasets", "abs_quality_delta", "abs_rank_delta", "model_key"], ascending=[False, False, False, True])
        .reset_index(drop=True)
    )
    return layer_changes


def build_quality_scope_summary_metrics(
    model_comparison: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    *,
    top_k: int,
) -> dict[str, Any]:
    top_correct = model_comparison.sort_values(["quality_rank_correct", "model_key"], ascending=[True, True]).head(top_k)
    top_wrong = model_comparison.sort_values(["quality_rank_wrong", "model_key"], ascending=[True, True]).head(top_k)
    top_correct_set = set(top_correct["model_key"].tolist())
    top_wrong_set = set(top_wrong["model_key"].tolist())
    spearman_rank = model_comparison["quality_rank_correct"].corr(model_comparison["quality_rank_wrong"], method="spearman")
    return {
        "models_compared": int(model_comparison.shape[0]),
        "datasets_compared": int(dataset_summary.shape[0]),
        "top_overlap_count": int(len(top_correct_set & top_wrong_set)),
        "top_overlap_models": sorted(top_correct_set & top_wrong_set),
        "spearman_rank": None if pd.isna(spearman_rank) else float(spearman_rank),
        "layer_changed_models": int(model_comparison["layer_changed"].sum()),
        "mean_abs_delta_quality": float(model_comparison["abs_quality_delta"].mean()),
        "max_abs_delta_quality": float(model_comparison["abs_quality_delta"].max()),
        "mean_abs_rank_delta": float(model_comparison["abs_rank_delta"].mean()),
        "largest_positive_model": str(model_comparison.sort_values("delta_quality_score", ascending=False).iloc[0]["model_key"]),
        "largest_negative_model": str(model_comparison.sort_values("delta_quality_score", ascending=True).iloc[0]["model_key"]),
    }


def safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if math.isnan(value_f):
        return None
    return value_f
