from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
import psutil

from quality_backbones.evaluation import (
    REQUIRED_SUCCESS_FILES,
    RunArtifacts,
    RunLogger,
    SUPPORTED_CORR_METRICS,
    SUPPORTED_EMBEDDING_DISTANCES,
    SUPPORTED_PAIR_SCOPES,
    LayerSpec,
    PairGroup,
    RankCacheInfo,
    ScoreVectorInfo,
    _acquire_run_lock,
    _append_jsonl,
    _atomic_write_json,
    _build_embedding_distance_vectors,
    _build_rank_cache,
    _build_run_artifacts,
    _build_y_sequence_and_ties,
    _cleanup_paths,
    _count_inversions_int32,
    _flatten_result_entries,
    _json_dumps,
    _limit_worker_threads,
    _load_completed_task_keys,
    _load_layer_names,
    _load_task_entries,
    _pair_count,
    _pearson_from_sums,
    _pearson_from_vectors,
    _prepare_existing_outputs_for_resume,
    _read_rows,
    _release_run_lock,
    _resolve_jobs,
    _resolve_layer_indices,
    _row_offsets,
    _safe_token,
    _scan_groups_scc,
    _task_key,
    _timestamp_now,
    _update_grouping_digest,
    _validate_row_id_alignment,
    discover_datasets,
    discover_models,
    load_dataset_table,
    resolve_pair_groups,
    resolve_pair_scope,
    write_results_table,
)
from quality_backbones.manifest import get_model_spec


ALIGNMENT_TABLE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "reference_model_key",
    "reference_layer_index",
    "reference_layer_name",
    "reference_layer_dim",
    "candidate_model_key",
    "candidate_layer_index",
    "candidate_layer_name",
    "candidate_layer_dim",
    "distance_metric",
    "corr_metric",
    "value",
    "n_samples",
    "n_pairs",
    "pair_scope",
    "group_field",
    "n_groups_total",
    "n_groups_used",
    "exact",
    "elapsed_sec",
)


@dataclass(frozen=True)
class AlignmentConfig:
    datasets_root: Path
    outputs_root: Path
    datasets: tuple[str, ...] | None
    reference_models: tuple[str, ...]
    reference_layer_selectors: tuple[str, ...] | None
    candidate_models: tuple[str, ...] | None
    candidate_layer_selectors: tuple[str, ...] | None
    exclude_families: tuple[str, ...] | None
    sample_limit: int | None
    seed: int
    distance_metrics: tuple[str, ...]
    corr_metrics: tuple[str, ...]
    pair_scope: str
    block_size: int
    jobs: int | None
    tmp_dir: Path
    run_dir: Path
    keep_cache: bool
    resume: bool
    progress_mode: str
    heartbeat_sec: int
    fail_fast: bool


@dataclass(frozen=True)
class DistanceMetricCache:
    vector_info: ScoreVectorInfo
    rank_cache: RankCacheInfo | None


@dataclass(frozen=True)
class AlignmentGroupPlan:
    dataset: str
    reference_spec: LayerSpec
    candidate_specs: tuple[LayerSpec, ...]


@dataclass(frozen=True)
class AlignmentDatasetPlan:
    dataset: str
    available_models: tuple[str, ...]
    reference_models: tuple[str, ...]
    candidate_models: tuple[str, ...]
    row_ids: tuple[int, ...]
    n_samples: int
    n_pairs: int
    pair_scope: str
    group_field: str | None
    pair_groups: tuple[PairGroup, ...]
    reference_specs: tuple[LayerSpec, ...]
    candidate_specs: tuple[LayerSpec, ...]
    groups: tuple[AlignmentGroupPlan, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentPairGroupCache:
    group_key: str
    positions: tuple[int, ...]
    n_samples: int
    n_pairs: int
    reference_vectors: dict[str, ScoreVectorInfo]
    reference_rank_caches: dict[str, RankCacheInfo | None]


@dataclass(frozen=True)
class AlignmentTask:
    task_key: str
    dataset: str
    reference_model_key: str
    reference_layer_index: int
    reference_layer_name: str
    reference_layer_dim: int
    candidate_model_key: str
    candidate_layer_index: int
    candidate_layer_name: str
    candidate_layer_dim: int
    candidate_h5_path: str
    row_ids: tuple[int, ...]
    n_samples: int
    n_pairs: int
    pair_scope: str
    group_field: str | None
    distance_metrics: tuple[str, ...]
    corr_metrics: tuple[str, ...]
    block_size: int
    tmp_dir: str
    keep_cache: bool
    task_log_path: str
    pair_groups: tuple[AlignmentPairGroupCache, ...]


def _normalize_family_token(token: str) -> str:
    return token.strip().upper().replace("-", "").replace("_", "")


def _model_family(model_key: str) -> str:
    try:
        spec = get_model_spec(model_key)
    except Exception:
        return ""
    return _normalize_family_token(spec.family)


def _task_key_alignment(reference_spec: LayerSpec, candidate_spec: LayerSpec) -> str:
    return (
        f"{reference_spec.dataset}::ref::{reference_spec.model_key}::{reference_spec.layer_index:03d}::"
        f"{reference_spec.layer_name}::cand::{candidate_spec.model_key}::{candidate_spec.layer_index:03d}::"
        f"{candidate_spec.layer_name}"
    )


def _sample_row_ids(index_path: Path, sample_limit: int | None, seed: int) -> np.ndarray:
    row_ids = pd.read_parquet(index_path, columns=["row_id"])["row_id"].to_numpy(dtype=np.int64)
    if row_ids.size == 0:
        return row_ids
    if np.unique(row_ids).size != row_ids.size:
        raise ValueError(f"row_id is not unique in {index_path}")
    row_ids = np.sort(row_ids)

    if sample_limit is None or sample_limit <= 0 or sample_limit >= row_ids.size:
        return row_ids

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(row_ids.size, size=sample_limit, replace=False).astype(np.int64))
    return row_ids[chosen]


def _resolve_model_sets(
    available_models: Sequence[str],
    reference_models: Sequence[str],
    candidate_models: Sequence[str] | None,
    exclude_families: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], list[str]]:
    warnings: list[str] = []
    available_set = set(available_models)

    refs: list[str] = []
    missing_refs: list[str] = []
    for model_key in reference_models:
        if model_key in available_set:
            refs.append(model_key)
        else:
            missing_refs.append(model_key)
    if missing_refs:
        warnings.append(f"missing reference models: {', '.join(sorted(set(missing_refs)))}")

    if candidate_models is None:
        candidates = [model_key for model_key in available_models if model_key not in set(refs)]
    else:
        candidates = []
        missing_candidates: list[str] = []
        for model_key in candidate_models:
            if model_key in available_set:
                candidates.append(model_key)
            else:
                missing_candidates.append(model_key)
        if missing_candidates:
            warnings.append(f"missing candidate models: {', '.join(sorted(set(missing_candidates)))}")

    if exclude_families:
        exclude_set = {_normalize_family_token(token) for token in exclude_families if token.strip()}
        filtered: list[str] = []
        excluded_models: list[str] = []
        for model_key in candidates:
            family = _model_family(model_key)
            if family and family in exclude_set:
                excluded_models.append(model_key)
                continue
            filtered.append(model_key)
        candidates = filtered
        if excluded_models:
            warnings.append(
                f"excluded candidate models by family: {', '.join(sorted(excluded_models))}"
            )

    def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        items: list[str] = []
        for value in values:
            if value in seen:
                continue
            items.append(value)
            seen.add(value)
        return tuple(items)

    return _dedupe(refs), _dedupe(candidates), warnings


def _normalize_alignment_layer_selectors(
    selectors: tuple[str, ...] | None,
    layer_names: list[str],
    *,
    default_last: bool,
) -> list[int]:
    if not layer_names:
        raise ValueError("layer_names must not be empty")

    if selectors is None:
        if default_last:
            return [len(layer_names) - 1]
        return list(range(len(layer_names)))

    normalized: list[str] = []
    for token in selectors:
        value = token.strip()
        lowered = value.lower()
        if lowered == "last":
            normalized.append(str(len(layer_names) - 1))
        elif lowered == "first":
            normalized.append("0")
        else:
            normalized.append(value)
    return _resolve_layer_indices(layer_names, tuple(normalized))


def _resolve_alignment_layer_specs(
    outputs_root: Path,
    dataset: str,
    model_keys: Sequence[str],
    selectors: tuple[str, ...] | None,
    *,
    default_last: bool,
) -> list[LayerSpec]:
    specs: list[LayerSpec] = []
    for model_key in model_keys:
        model_dir = outputs_root / dataset / model_key
        meta_path = model_dir / "meta.json"
        h5_path = model_dir / "layers.h5"
        if not meta_path.exists() or not h5_path.exists():
            continue

        layer_names = _load_layer_names(meta_path)
        selected = _normalize_alignment_layer_selectors(selectors, layer_names, default_last=default_last)

        with h5py.File(h5_path, "r") as fp:
            for idx in selected:
                ds_name = f"layer_{idx:03d}"
                if ds_name not in fp:
                    raise KeyError(f"Missing dataset {ds_name} in {h5_path}")
                dim = int(fp[ds_name].shape[1])
                specs.append(
                    LayerSpec(
                        dataset=dataset,
                        model_key=model_key,
                        layer_index=idx,
                        layer_name=layer_names[idx],
                        layer_dim=dim,
                        h5_path=str(h5_path),
                    )
                )
    return specs


def _build_dataset_plan(config: AlignmentConfig, dataset: str) -> AlignmentDatasetPlan:
    warnings: list[str] = []
    resolved_pair_scope = resolve_pair_scope(config.pair_scope, dataset)
    available_models = discover_models(config.outputs_root, dataset, explicit=None)
    if not available_models:
        warnings.append(f"dataset={dataset}: no completed model outputs found")
        return AlignmentDatasetPlan(
            dataset=dataset,
            available_models=tuple(),
            reference_models=tuple(),
            candidate_models=tuple(),
            row_ids=tuple(),
            n_samples=0,
            n_pairs=0,
            pair_scope=resolved_pair_scope,
            group_field=None,
            pair_groups=tuple(),
            reference_specs=tuple(),
            candidate_specs=tuple(),
            groups=tuple(),
            warnings=tuple(warnings),
        )

    refs, candidates, scope_warnings = _resolve_model_sets(
        available_models=available_models,
        reference_models=config.reference_models,
        candidate_models=config.candidate_models,
        exclude_families=config.exclude_families,
    )
    warnings.extend(scope_warnings)

    if not refs:
        warnings.append(f"dataset={dataset}: no reference models available after filtering")
        return AlignmentDatasetPlan(
            dataset=dataset,
            available_models=tuple(available_models),
            reference_models=tuple(),
            candidate_models=tuple(candidates),
            row_ids=tuple(),
            n_samples=0,
            n_pairs=0,
            pair_scope=resolved_pair_scope,
            group_field=None,
            pair_groups=tuple(),
            reference_specs=tuple(),
            candidate_specs=tuple(),
            groups=tuple(),
            warnings=tuple(warnings),
        )

    if not candidates:
        warnings.append(f"dataset={dataset}: no candidate models available after filtering")
        return AlignmentDatasetPlan(
            dataset=dataset,
            available_models=tuple(available_models),
            reference_models=refs,
            candidate_models=tuple(),
            row_ids=tuple(),
            n_samples=0,
            n_pairs=0,
            pair_scope=resolved_pair_scope,
            group_field=None,
            pair_groups=tuple(),
            reference_specs=tuple(),
            candidate_specs=tuple(),
            groups=tuple(),
            warnings=tuple(warnings),
        )

    involved_models = sorted(set(refs).union(candidates))
    _validate_row_id_alignment(config.outputs_root, dataset, involved_models)

    anchor_model = refs[0]
    index_path = config.outputs_root / dataset / anchor_model / "index.parquet"
    row_ids = _sample_row_ids(index_path, config.sample_limit, config.seed)
    n_samples = int(row_ids.shape[0])
    if n_samples < 2:
        warnings.append(f"dataset={dataset}: not enough samples after sampling (n={n_samples})")
        return AlignmentDatasetPlan(
            dataset=dataset,
            available_models=tuple(available_models),
            reference_models=refs,
            candidate_models=candidates,
            row_ids=tuple(int(v) for v in row_ids.tolist()),
            n_samples=n_samples,
            n_pairs=_pair_count(n_samples),
            pair_scope=resolved_pair_scope,
            group_field=None,
            pair_groups=tuple(),
            reference_specs=tuple(),
            candidate_specs=tuple(),
            groups=tuple(),
            warnings=tuple(warnings),
        )

    table = load_dataset_table(config.datasets_root, config.outputs_root, dataset, anchor_model)
    group_field, pair_groups, group_warnings = resolve_pair_groups(
        table,
        row_ids,
        dataset=dataset,
        datasets_root=config.datasets_root,
        pair_scope=resolved_pair_scope,
    )
    warnings.extend(group_warnings)
    n_pairs = sum(group.n_pairs for group in pair_groups)
    if not pair_groups:
        warnings.append(f"dataset={dataset}: no runnable pair groups for pair_scope={resolved_pair_scope}")
        return AlignmentDatasetPlan(
            dataset=dataset,
            available_models=tuple(available_models),
            reference_models=refs,
            candidate_models=candidates,
            row_ids=tuple(int(v) for v in row_ids.tolist()),
            n_samples=n_samples,
            n_pairs=0,
            pair_scope=resolved_pair_scope,
            group_field=group_field,
            pair_groups=tuple(),
            reference_specs=tuple(),
            candidate_specs=tuple(),
            groups=tuple(),
            warnings=tuple(warnings),
        )

    reference_specs = _resolve_alignment_layer_specs(
        outputs_root=config.outputs_root,
        dataset=dataset,
        model_keys=refs,
        selectors=config.reference_layer_selectors,
        default_last=True,
    )
    candidate_specs = _resolve_alignment_layer_specs(
        outputs_root=config.outputs_root,
        dataset=dataset,
        model_keys=candidates,
        selectors=config.candidate_layer_selectors,
        default_last=False,
    )

    if not reference_specs:
        warnings.append(f"dataset={dataset}: no reference layers resolved")
    if not candidate_specs:
        warnings.append(f"dataset={dataset}: no candidate layers resolved")

    groups = tuple(
        AlignmentGroupPlan(dataset=dataset, reference_spec=reference_spec, candidate_specs=tuple(candidate_specs))
        for reference_spec in reference_specs
    )

    return AlignmentDatasetPlan(
        dataset=dataset,
        available_models=tuple(available_models),
        reference_models=refs,
        candidate_models=candidates,
        row_ids=tuple(int(v) for v in row_ids.tolist()),
        n_samples=n_samples,
        n_pairs=n_pairs,
        pair_scope=resolved_pair_scope,
        group_field=group_field,
        pair_groups=pair_groups,
        reference_specs=tuple(reference_specs),
        candidate_specs=tuple(candidate_specs),
        groups=groups,
        warnings=tuple(warnings),
    )


def _compute_vector_sums(path: Path, pair_total: int, *, chunk_size: int = 2_000_000) -> tuple[float, float]:
    arr = np.memmap(path, dtype=np.float32, mode="r", shape=(pair_total,))
    sum_value = 0.0
    sum_sq_value = 0.0
    for start in range(0, pair_total, chunk_size):
        end = min(pair_total, start + chunk_size)
        chunk = np.asarray(arr[start:end], dtype=np.float64)
        sum_value += float(chunk.sum(dtype=np.float64))
        sum_sq_value += float(np.dot(chunk, chunk))
    del arr
    return sum_value, sum_sq_value


def _load_metric_cache_from_meta(
    payload: dict[str, Any],
    metric: str,
    *,
    need_ranks: bool,
    need_avg: bool,
    need_dense: bool,
) -> DistanceMetricCache | None:
    vector_payload = payload.get("vector")
    if not isinstance(vector_payload, dict):
        return None
    vector_path = Path(str(vector_payload.get("path", "")))
    if not vector_path.exists():
        return None

    vector_info = ScoreVectorInfo(
        metric=metric,
        path=str(vector_path),
        sum_value=float(vector_payload.get("sum_value", 0.0)),
        sum_sq_value=float(vector_payload.get("sum_sq_value", 0.0)),
    )

    rank_cache: RankCacheInfo | None = None
    if need_ranks:
        rank_payload = payload.get("rank_cache")
        if not isinstance(rank_payload, dict):
            return None
        avg_path_text = rank_payload.get("avg_rank_path")
        dense_path_text = rank_payload.get("dense_rank_path")
        if need_avg and (not isinstance(avg_path_text, str) or not Path(avg_path_text).exists()):
            return None
        if need_dense and (not isinstance(dense_path_text, str) or not Path(dense_path_text).exists()):
            return None
        rank_cache = RankCacheInfo(
            source_metric=metric,
            avg_rank_path=str(avg_path_text) if isinstance(avg_path_text, str) else None,
            dense_rank_path=str(dense_path_text) if isinstance(dense_path_text, str) else None,
            tie_pairs=int(rank_payload.get("tie_pairs", 0)),
            unique_count=int(rank_payload.get("unique_count", 0)),
            rank_sum=float(rank_payload.get("rank_sum", 0.0)),
            rank_sum_sq=(
                float(rank_payload["rank_sum_sq"])
                if rank_payload.get("rank_sum_sq") is not None
                else None
            ),
        )
    return DistanceMetricCache(vector_info=vector_info, rank_cache=rank_cache)


def _build_reference_distance_cache(
    reference_spec: LayerSpec,
    *,
    row_ids: Sequence[int],
    pair_group: PairGroup,
    config: AlignmentConfig,
    cache_dir: Path,
) -> AlignmentPairGroupCache:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_meta.json"
    n_samples = pair_group.n_samples
    n_pairs = pair_group.n_pairs
    group_positions = np.asarray(pair_group.positions, dtype=np.int64)
    group_row_ids = np.asarray(row_ids, dtype=np.int64)[group_positions]
    row_ids_digest = hashlib.sha1(group_row_ids.tobytes()).hexdigest()[:16]

    need_scc = "scc" in config.corr_metrics
    need_kcc = "kcc" in config.corr_metrics
    need_ranks = need_scc or need_kcc

    if meta_path.exists():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta_payload = {}
        if isinstance(meta_payload, dict):
            if (
                meta_payload.get("dataset") == reference_spec.dataset
                and meta_payload.get("reference_model_key") == reference_spec.model_key
                and int(meta_payload.get("reference_layer_index", -1)) == reference_spec.layer_index
                and str(meta_payload.get("group_key", "")) == pair_group.group_key
                and str(meta_payload.get("row_ids_digest", "")) == row_ids_digest
                and int(meta_payload.get("n_samples", -1)) == n_samples
                and int(meta_payload.get("n_pairs", -1)) == n_pairs
            ):
                metric_payloads = meta_payload.get("metrics")
                if isinstance(metric_payloads, dict):
                    loaded: dict[str, DistanceMetricCache] = {}
                    for metric in config.distance_metrics:
                        payload = metric_payloads.get(metric)
                        if not isinstance(payload, dict):
                            loaded = {}
                            break
                        cache = _load_metric_cache_from_meta(
                            payload,
                            metric,
                            need_ranks=need_ranks,
                            need_avg=need_scc,
                            need_dense=need_kcc,
                        )
                        if cache is None:
                            loaded = {}
                            break
                        loaded[metric] = cache
                    if loaded:
                        return AlignmentPairGroupCache(
                            group_key=pair_group.group_key,
                            positions=pair_group.positions,
                            n_samples=n_samples,
                            n_pairs=n_pairs,
                            reference_vectors={metric: cache.vector_info for metric, cache in loaded.items()},
                            reference_rank_caches={metric: cache.rank_cache for metric, cache in loaded.items()},
                        )

    row_offsets = _row_offsets(n_samples)
    with h5py.File(reference_spec.h5_path, "r") as fp:
        ds_name = f"layer_{reference_spec.layer_index:03d}"
        embeddings = _read_rows(fp[ds_name], group_row_ids).astype(np.float32, copy=False)

    vector_paths, _ = _build_embedding_distance_vectors(
        embeddings=embeddings,
        metrics=config.distance_metrics,
        row_offsets=row_offsets,
        block_size=config.block_size,
        out_dir=cache_dir,
        prefix=(
            f"reference_{_safe_token(reference_spec.model_key)}_"
            f"{reference_spec.layer_index:03d}_{_safe_token(reference_spec.layer_name)}"
        ),
    )

    metric_caches: dict[str, DistanceMetricCache] = {}
    meta_metrics: dict[str, Any] = {}
    for metric in config.distance_metrics:
        path = Path(vector_paths[metric])
        sum_value, sum_sq_value = _compute_vector_sums(path, n_pairs)
        vector_info = ScoreVectorInfo(
            metric=metric,
            path=str(path),
            sum_value=float(sum_value),
            sum_sq_value=float(sum_sq_value),
        )

        rank_cache: RankCacheInfo | None = None
        if need_ranks:
            metric_rank_dir = cache_dir / metric
            metric_rank_dir.mkdir(parents=True, exist_ok=True)
            base_rank_cache, _ = _build_rank_cache(
                values_path=path,
                pair_total=n_pairs,
                out_dir=metric_rank_dir,
                need_avg=need_scc,
                need_dense=need_kcc,
            )
            rank_cache = RankCacheInfo(
                source_metric=metric,
                avg_rank_path=base_rank_cache.avg_rank_path,
                dense_rank_path=base_rank_cache.dense_rank_path,
                tie_pairs=base_rank_cache.tie_pairs,
                unique_count=base_rank_cache.unique_count,
                rank_sum=base_rank_cache.rank_sum,
                rank_sum_sq=base_rank_cache.rank_sum_sq,
            )

        metric_caches[metric] = DistanceMetricCache(vector_info=vector_info, rank_cache=rank_cache)
        meta_metrics[metric] = {
            "vector": asdict(vector_info),
            "rank_cache": asdict(rank_cache) if rank_cache is not None else None,
        }

    meta_payload = {
        "schema_version": 1,
        "dataset": reference_spec.dataset,
        "reference_model_key": reference_spec.model_key,
        "reference_layer_index": reference_spec.layer_index,
        "reference_layer_name": reference_spec.layer_name,
        "group_key": pair_group.group_key,
        "row_ids_digest": row_ids_digest,
        "n_samples": n_samples,
        "n_pairs": n_pairs,
        "distance_metrics": list(config.distance_metrics),
        "corr_metrics": list(config.corr_metrics),
        "metrics": meta_metrics,
        "updated_at": _timestamp_now(),
    }
    _atomic_write_json(meta_path, meta_payload)
    return AlignmentPairGroupCache(
        group_key=pair_group.group_key,
        positions=pair_group.positions,
        n_samples=n_samples,
        n_pairs=n_pairs,
        reference_vectors={metric: cache.vector_info for metric, cache in metric_caches.items()},
        reference_rank_caches={metric: cache.rank_cache for metric, cache in metric_caches.items()},
    )


def _prepare_alignment_tasks(
    reference_spec: LayerSpec,
    candidate_specs: Sequence[LayerSpec],
    *,
    row_ids: Sequence[int],
    n_samples: int,
    n_pairs: int,
    pair_scope: str,
    group_field: str | None,
    config: AlignmentConfig,
    pair_groups: Sequence[AlignmentPairGroupCache],
    dataset_tmp_dir: Path,
    task_logs_root: Path,
) -> list[AlignmentTask]:
    row_tuple = tuple(int(v) for v in row_ids)
    tasks: list[AlignmentTask] = []

    for candidate_spec in candidate_specs:
        task_key = _task_key_alignment(reference_spec, candidate_spec)
        task_log_dir = (
            task_logs_root
            / _safe_token(reference_spec.dataset)
            / _safe_token(reference_spec.model_key)
            / f"ref_{reference_spec.layer_index:03d}_{_safe_token(reference_spec.layer_name)}"
        )
        task_log_path = (
            task_log_dir
            / f"cand_{_safe_token(candidate_spec.model_key)}_{candidate_spec.layer_index:03d}_"
            f"{_safe_token(candidate_spec.layer_name)}.log"
        )
        tasks.append(
            AlignmentTask(
                task_key=task_key,
                dataset=reference_spec.dataset,
                reference_model_key=reference_spec.model_key,
                reference_layer_index=reference_spec.layer_index,
                reference_layer_name=reference_spec.layer_name,
                reference_layer_dim=reference_spec.layer_dim,
                candidate_model_key=candidate_spec.model_key,
                candidate_layer_index=candidate_spec.layer_index,
                candidate_layer_name=candidate_spec.layer_name,
                candidate_layer_dim=candidate_spec.layer_dim,
                candidate_h5_path=candidate_spec.h5_path,
                row_ids=row_tuple,
                n_samples=n_samples,
                n_pairs=n_pairs,
                pair_scope=pair_scope,
                group_field=group_field,
                distance_metrics=config.distance_metrics,
                corr_metrics=config.corr_metrics,
                block_size=config.block_size,
                tmp_dir=str(dataset_tmp_dir),
                keep_cache=config.keep_cache,
                task_log_path=str(task_log_path),
                pair_groups=tuple(pair_groups),
            )
        )
    return tasks


def _evaluate_alignment_task(task: AlignmentTask) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _timestamp_now()
    row_ids = np.asarray(task.row_ids, dtype=np.int64)
    layer_tmp = (
        Path(task.tmp_dir)
        / f"{task.candidate_model_key}_{task.candidate_layer_index:03d}_{os.getpid()}_{time.time_ns()}"
    )
    layer_tmp.mkdir(parents=True, exist_ok=True)
    task_log_path = Path(task.task_log_path)
    task_log_path.parent.mkdir(parents=True, exist_ok=True)

    phase_timings: dict[str, float] = {
        "read_candidate_sec": 0.0,
        "build_candidate_sec": 0.0,
        "pcc_sec": 0.0,
        "sort_sec": 0.0,
        "scc_sec": 0.0,
        "kcc_sec": 0.0,
    }
    cleanup_paths: list[str] = []
    rows: list[dict[str, Any]] = []

    def _task_log(message: str) -> None:
        with task_log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"[{_timestamp_now()}] {message}\n")

    _task_log(
        "START "
        f"task_key={task.task_key} "
        f"reference={task.reference_model_key}:{task.reference_layer_name} "
        f"candidate={task.candidate_model_key}:{task.candidate_layer_name}"
    )

    try:
        phase_started = time.perf_counter()
        _task_log("phase=read_candidate start")
        with h5py.File(task.candidate_h5_path, "r") as fp:
            ds_name = f"layer_{task.candidate_layer_index:03d}"
            if ds_name not in fp:
                raise KeyError(f"Missing dataset {ds_name} in {task.candidate_h5_path}")
            embeddings = _read_rows(fp[ds_name], row_ids).astype(np.float32, copy=False)
        phase_timings["read_candidate_sec"] += float(time.perf_counter() - phase_started)
        _task_log(f"phase=read_candidate done sec={phase_timings['read_candidate_sec']:.3f}")

        metric_sum: dict[tuple[str, str], float] = {}
        metric_count: dict[tuple[str, str], int] = {}
        metric_pairs: dict[tuple[str, str], int] = {}
        metric_elapsed: dict[tuple[str, str], float] = {}

        def _accumulate(distance_metric: str, corr_metric: str, value: float, elapsed_sec: float, n_pairs: int) -> None:
            if math.isnan(value):
                return
            key = (distance_metric, corr_metric)
            metric_sum[key] = metric_sum.get(key, 0.0) + value
            metric_count[key] = metric_count.get(key, 0) + 1
            metric_pairs[key] = metric_pairs.get(key, 0) + n_pairs
            metric_elapsed[key] = metric_elapsed.get(key, 0.0) + elapsed_sec

        for group_index, group in enumerate(task.pair_groups):
            if group.n_pairs <= 0:
                continue
            group_tmp = layer_tmp / f"group_{group_index:04d}_{_safe_token(group.group_key)}"
            group_tmp.mkdir(parents=True, exist_ok=True)
            group_positions = np.asarray(group.positions, dtype=np.int64)
            group_row_offsets = _row_offsets(group.n_samples)

            phase_started = time.perf_counter()
            vector_paths, vector_cleanup = _build_embedding_distance_vectors(
                embeddings=embeddings[group_positions],
                metrics=task.distance_metrics,
                row_offsets=group_row_offsets,
                block_size=task.block_size,
                out_dir=group_tmp,
                prefix=(
                    f"candidate_{_safe_token(task.candidate_model_key)}_"
                    f"{task.candidate_layer_index:03d}_{_safe_token(task.candidate_layer_name)}_group_{group_index:04d}"
                ),
            )
            phase_timings["build_candidate_sec"] += float(time.perf_counter() - phase_started)

            reference_arrays: dict[str, np.memmap] = {}
            reference_avg_ranks: dict[str, np.memmap] = {}
            reference_dense_ranks: dict[str, np.memmap] = {}
            for metric, vector_info in group.reference_vectors.items():
                reference_arrays[metric] = np.memmap(vector_info.path, dtype=np.float32, mode="r", shape=(group.n_pairs,))
                rank_cache = group.reference_rank_caches.get(metric)
                if rank_cache is not None and rank_cache.avg_rank_path is not None:
                    reference_avg_ranks[metric] = np.memmap(
                        rank_cache.avg_rank_path,
                        dtype=np.float64,
                        mode="r",
                        shape=(group.n_pairs,),
                    )
                if rank_cache is not None and rank_cache.dense_rank_path is not None:
                    reference_dense_ranks[metric] = np.memmap(
                        rank_cache.dense_rank_path,
                        dtype=np.int32,
                        mode="r",
                        shape=(group.n_pairs,),
                    )

            try:
                for distance_metric in task.distance_metrics:
                    x_values = np.memmap(vector_paths[distance_metric], dtype=np.float32, mode="r", shape=(group.n_pairs,))
                    order: np.ndarray | None = None
                    try:
                        reference_info = group.reference_vectors[distance_metric]
                        rank_cache = group.reference_rank_caches.get(distance_metric)

                        if "pcc" in task.corr_metrics:
                            corr_started = time.perf_counter()
                            corr_value = _pearson_from_vectors(
                                x_values=x_values,
                                y_values=reference_arrays[distance_metric],
                                y_sum=reference_info.sum_value,
                                y_sum_sq=reference_info.sum_sq_value,
                            )
                            elapsed = float(time.perf_counter() - corr_started)
                            phase_timings["pcc_sec"] += elapsed
                            _accumulate(distance_metric, "pcc", float(corr_value), elapsed, group.n_pairs)

                        need_scc = "scc" in task.corr_metrics
                        need_kcc = "kcc" in task.corr_metrics
                        if need_scc or need_kcc:
                            sort_started = time.perf_counter()
                            if need_kcc:
                                y_dense = reference_dense_ranks.get(distance_metric)
                                if y_dense is None:
                                    raise RuntimeError(f"KCC requested but dense ranks are missing for {distance_metric}")
                                order = np.lexsort((y_dense, x_values))
                            else:
                                order = np.argsort(x_values, kind="mergesort")
                            phase_timings["sort_sec"] += float(time.perf_counter() - sort_started)

                        if need_scc:
                            y_avg = reference_avg_ranks.get(distance_metric)
                            if y_avg is None or rank_cache is None or rank_cache.rank_sum_sq is None:
                                raise RuntimeError(f"SCC requested but average ranks are missing for {distance_metric}")
                            corr_started = time.perf_counter()
                            sum_x, sum_x2, sum_xy = _scan_groups_scc(order, x_values, y_avg)
                            corr_value = _pearson_from_sums(
                                n=group.n_pairs,
                                sum_x=float(sum_x),
                                sum_y=float(rank_cache.rank_sum),
                                sum_x2=float(sum_x2),
                                sum_y2=float(rank_cache.rank_sum_sq),
                                sum_xy=float(sum_xy),
                            )
                            elapsed = float(time.perf_counter() - corr_started)
                            phase_timings["scc_sec"] += elapsed
                            _accumulate(distance_metric, "scc", float(corr_value), elapsed, group.n_pairs)

                        if need_kcc:
                            y_dense = reference_dense_ranks.get(distance_metric)
                            if y_dense is None or rank_cache is None:
                                raise RuntimeError(f"KCC requested but dense ranks are missing for {distance_metric}")
                            corr_started = time.perf_counter()
                            y_seq, ties_x, ties_xy = _build_y_sequence_and_ties(order, x_values, y_dense)
                            discordant = _count_inversions_int32(y_seq)
                            del y_seq
                            n0 = (group.n_pairs * (group.n_pairs - 1)) // 2
                            ties_y = int(rank_cache.tie_pairs)
                            p_plus_q = n0 - ties_x - ties_y + ties_xy
                            numerator = p_plus_q - (2 * discordant)
                            den_left = n0 - ties_x
                            den_right = n0 - ties_y
                            if den_left <= 0 or den_right <= 0:
                                corr_value = float("nan")
                            else:
                                corr_value = float(numerator / math.sqrt(float(den_left) * float(den_right)))
                            elapsed = float(time.perf_counter() - corr_started)
                            phase_timings["kcc_sec"] += elapsed
                            _accumulate(distance_metric, "kcc", float(corr_value), elapsed, group.n_pairs)
                    finally:
                        del x_values
                        if order is not None:
                            del order
            finally:
                for arr in reference_arrays.values():
                    del arr
                for arr in reference_avg_ranks.values():
                    del arr
                for arr in reference_dense_ranks.values():
                    del arr
                if not task.keep_cache:
                    _cleanup_paths(vector_cleanup)
                    shutil.rmtree(group_tmp, ignore_errors=True)

        n_groups_total = len(task.pair_groups)
        rows = []
        for distance_metric in task.distance_metrics:
            for corr_metric in task.corr_metrics:
                key = (distance_metric, corr_metric)
                used_groups = int(metric_count.get(key, 0))
                value = float("nan") if used_groups == 0 else float(metric_sum[key] / used_groups)
                rows.append(
                    {
                        "dataset": task.dataset,
                        "reference_model_key": task.reference_model_key,
                        "reference_layer_index": task.reference_layer_index,
                        "reference_layer_name": task.reference_layer_name,
                        "reference_layer_dim": task.reference_layer_dim,
                        "candidate_model_key": task.candidate_model_key,
                        "candidate_layer_index": task.candidate_layer_index,
                        "candidate_layer_name": task.candidate_layer_name,
                        "candidate_layer_dim": task.candidate_layer_dim,
                        "distance_metric": distance_metric,
                        "corr_metric": corr_metric,
                        "value": value,
                        "n_samples": task.n_samples,
                        "n_pairs": int(metric_pairs.get(key, 0)),
                        "pair_scope": task.pair_scope,
                        "group_field": task.group_field,
                        "n_groups_total": n_groups_total,
                        "n_groups_used": used_groups,
                        "exact": True,
                        "elapsed_sec": float(metric_elapsed.get(key, 0.0)),
                    }
                )

        if not task.keep_cache:
            shutil.rmtree(layer_tmp, ignore_errors=True)

        elapsed = float(time.perf_counter() - started)
        _task_log(f"DONE status=completed elapsed_sec={elapsed:.3f}")
        return {
            "status": "completed",
            "task_key": task.task_key,
            "started_at": started_at,
            "finished_at": _timestamp_now(),
            "dataset": task.dataset,
            "reference_model_key": task.reference_model_key,
            "reference_layer_index": task.reference_layer_index,
            "reference_layer_name": task.reference_layer_name,
            "candidate_model_key": task.candidate_model_key,
            "candidate_layer_index": task.candidate_layer_index,
            "candidate_layer_name": task.candidate_layer_name,
            "rows": rows,
            "elapsed_sec": elapsed,
            "phase_timings": phase_timings,
            "task_log_path": str(task_log_path),
        }
    except Exception as exc:
        if not task.keep_cache:
            shutil.rmtree(layer_tmp, ignore_errors=True)
        elapsed = float(time.perf_counter() - started)
        _task_log(f"DONE status=error elapsed_sec={elapsed:.3f} error={type(exc).__name__}: {exc}")
        return {
            "status": "error",
            "task_key": task.task_key,
            "started_at": started_at,
            "finished_at": _timestamp_now(),
            "dataset": task.dataset,
            "reference_model_key": task.reference_model_key,
            "reference_layer_index": task.reference_layer_index,
            "reference_layer_name": task.reference_layer_name,
            "candidate_model_key": task.candidate_model_key,
            "candidate_layer_index": task.candidate_layer_index,
            "candidate_layer_name": task.candidate_layer_name,
            "elapsed_sec": elapsed,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "task_log_path": str(task_log_path),
        }
    finally:
        if not task.keep_cache and layer_tmp.exists() and not any(layer_tmp.iterdir()):
            layer_tmp.rmdir()


def _result_config_payload(config: AlignmentConfig) -> dict[str, Any]:
    def _sorted_or_none(values: tuple[str, ...] | None) -> list[str] | None:
        if values is None:
            return None
        return sorted(set(values))

    return {
        "datasets": _sorted_or_none(config.datasets),
        "reference_models": sorted(set(config.reference_models)),
        "reference_layer_selectors": _sorted_or_none(config.reference_layer_selectors),
        "candidate_models": _sorted_or_none(config.candidate_models),
        "candidate_layer_selectors": _sorted_or_none(config.candidate_layer_selectors),
        "exclude_families": _sorted_or_none(config.exclude_families),
        "sample_limit": config.sample_limit,
        "seed": config.seed,
        "distance_metrics": sorted(set(config.distance_metrics)),
        "corr_metrics": sorted(set(config.corr_metrics)),
        "pair_scope": config.pair_scope,
    }


def _runtime_config_payload(config: AlignmentConfig) -> dict[str, Any]:
    return {
        "datasets_root": str(config.datasets_root),
        "outputs_root": str(config.outputs_root),
        "tmp_dir": str(config.tmp_dir),
        "run_dir": str(config.run_dir),
        "block_size": config.block_size,
        "jobs": config.jobs,
        "keep_cache": config.keep_cache,
        "resume": config.resume,
        "progress_mode": config.progress_mode,
        "heartbeat_sec": config.heartbeat_sec,
        "fail_fast": config.fail_fast,
        "pair_scope": config.pair_scope,
    }


def _config_fingerprint(config: AlignmentConfig) -> str:
    payload = _result_config_payload(config)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _initialize_run_config(config: AlignmentConfig, artifacts: RunArtifacts) -> dict[str, Any]:
    fingerprint = _config_fingerprint(config)
    if config.resume:
        if not artifacts.config_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing run_config.json in {artifacts.run_dir}")
        payload = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
        existing = str(payload.get("fingerprint", ""))
        if existing != fingerprint:
            raise ValueError(
                f"Run config fingerprint mismatch for resume: existing={existing} current={fingerprint}"
            )
        payload["resume_count"] = int(payload.get("resume_count", 0)) + 1
    else:
        artifacts.run_dir.mkdir(parents=True, exist_ok=True)
        allowed_existing = {artifacts.lock_path.name}
        has_existing_files = any(path.name not in allowed_existing for path in artifacts.run_dir.iterdir())
        if has_existing_files:
            raise FileExistsError(
                f"Run directory already exists and is not empty: {artifacts.run_dir}. Use --resume or a new --run-dir"
            )
        payload = {
            "created_at": _timestamp_now(),
            "resume_count": 0,
            "fingerprint": fingerprint,
            "result_config": _result_config_payload(config),
        }

    payload["fingerprint"] = fingerprint
    payload["runtime_config"] = _runtime_config_payload(config)
    payload["last_started_at"] = _timestamp_now()
    _atomic_write_json(artifacts.config_path, payload)
    return payload


def _prepare_existing_outputs_for_resume_alignment(artifacts: RunArtifacts) -> set[str]:
    completed_keys = _load_completed_task_keys(artifacts.results_jsonl_path)
    if artifacts.results_jsonl_path.exists() and not artifacts.table_tsv_path.exists():
        rows = _flatten_result_entries(_load_task_entries(artifacts.results_jsonl_path))
        if rows:
            write_results_table(rows, artifacts.table_tsv_path)
    return completed_keys


def _append_alignment_rows_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(ALIGNMENT_TABLE_COLUMNS), delimiter="\t")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in ALIGNMENT_TABLE_COLUMNS})


def _initial_progress_state(
    config: AlignmentConfig,
    artifacts: RunArtifacts,
    run_config_payload: dict[str, Any],
    plans: Sequence[AlignmentDatasetPlan],
    completed_task_keys: set[str],
) -> dict[str, Any]:
    datasets_payload: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    total_tasks = 0
    completed_total = 0

    for plan in plans:
        task_keys = [
            _task_key_alignment(group.reference_spec, candidate_spec)
            for group in plan.groups
            for candidate_spec in group.candidate_specs
        ]
        completed_dataset = sum(1 for key in task_keys if key in completed_task_keys)
        total = len(task_keys)
        total_tasks += total
        completed_total += completed_dataset
        warnings.extend(plan.warnings)

        datasets_payload[plan.dataset] = {
            "dataset": plan.dataset,
            "n_available_models": len(plan.available_models),
            "n_reference_models": len(plan.reference_models),
            "n_candidate_models": len(plan.candidate_models),
            "n_reference_layers": len(plan.reference_specs),
            "n_candidate_layers": len(plan.candidate_specs),
            "n_samples": plan.n_samples,
            "n_pairs": plan.n_pairs,
            "pair_scope": plan.pair_scope,
            "group_field": plan.group_field,
            "n_groups": len(plan.pair_groups),
            "total_tasks": total,
            "completed_tasks": completed_dataset,
            "skipped_tasks": completed_dataset if config.resume else 0,
            "failed_tasks": 0,
            "jobs": 0,
            "status": (
                "completed"
                if total > 0 and completed_dataset == total
                else ("skipped" if total == 0 else "pending")
            ),
            "warnings": list(plan.warnings),
            "cache_dir": None,
        }

    return {
        "run_dir": str(artifacts.run_dir),
        "status": "running",
        "created_at": run_config_payload.get("created_at", _timestamp_now()),
        "last_started_at": _timestamp_now(),
        "last_updated_at": _timestamp_now(),
        "resume_count": int(run_config_payload.get("resume_count", 0)),
        "config_fingerprint": str(run_config_payload.get("fingerprint", "")),
        "result_config": run_config_payload.get("result_config", {}),
        "runtime_config": _runtime_config_payload(config),
        "artifacts": {
            "results_jsonl": str(artifacts.results_jsonl_path),
            "errors_jsonl": str(artifacts.errors_jsonl_path),
            "progress_json": str(artifacts.progress_json_path),
            "report_json": str(artifacts.report_json_path),
            "table_tsv": str(artifacts.table_tsv_path),
            "run_log": str(artifacts.run_log_path),
            "cache_root": str(artifacts.cache_root),
            "task_logs_root": str(artifacts.task_logs_root),
        },
        "total_tasks": total_tasks,
        "completed_tasks": completed_total,
        "completed_tasks_this_session": 0,
        "failed_tasks": 0,
        "skipped_tasks": completed_total if config.resume else 0,
        "remaining_tasks": max(0, total_tasks - completed_total),
        "current_dataset": None,
        "running_tasks": {},
        "warnings": warnings,
        "datasets": datasets_payload,
    }


def _write_progress_snapshot(artifacts: RunArtifacts, state: dict[str, Any]) -> None:
    payload = {
        **state,
        "last_updated_at": _timestamp_now(),
        "remaining_tasks": max(0, state["total_tasks"] - state["completed_tasks"]),
        "running_tasks": sorted(state["running_tasks"].values(), key=lambda item: item["task_key"]),
    }
    _atomic_write_json(artifacts.progress_json_path, payload)


def _mark_task_running(state: dict[str, Any], task: AlignmentTask) -> None:
    state["running_tasks"][task.task_key] = {
        "task_key": task.task_key,
        "dataset": task.dataset,
        "reference_model_key": task.reference_model_key,
        "reference_layer_index": task.reference_layer_index,
        "reference_layer_name": task.reference_layer_name,
        "candidate_model_key": task.candidate_model_key,
        "candidate_layer_index": task.candidate_layer_index,
        "candidate_layer_name": task.candidate_layer_name,
        "started_at": _timestamp_now(),
        "task_log_path": task.task_log_path,
    }


def _handle_task_result(
    result: dict[str, Any],
    *,
    state: dict[str, Any],
    artifacts: RunArtifacts,
    logger: RunLogger,
    completed_task_keys: set[str],
) -> None:
    task_key = str(result.get("task_key", ""))
    dataset = str(result.get("dataset", ""))
    ds_state = state["datasets"].get(dataset, {})
    state["running_tasks"].pop(task_key, None)

    if result.get("status") == "completed":
        _append_jsonl(artifacts.results_jsonl_path, result)
        _append_alignment_rows_tsv(artifacts.table_tsv_path, result.get("rows", []))
        completed_task_keys.add(task_key)
        state["completed_tasks"] += 1
        state["completed_tasks_this_session"] += 1
        if ds_state:
            ds_state["completed_tasks"] += 1
            if ds_state["completed_tasks"] >= ds_state["total_tasks"]:
                ds_state["status"] = "completed"
        logger.log(
            f"task completed {task_key} elapsed_sec={float(result.get('elapsed_sec', 0.0)):.3f} "
            f"dataset_progress={ds_state.get('completed_tasks', 0)}/{ds_state.get('total_tasks', 0)}"
        )
        return

    _append_jsonl(artifacts.errors_jsonl_path, result)
    state["failed_tasks"] += 1
    if ds_state:
        ds_state["failed_tasks"] += 1
        ds_state["status"] = "failed"
    logger.log(f"task failed {task_key}: {result.get('error', '[unknown error]')}")


def _build_best_candidates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        value = row.get("value")
        if value is None:
            continue
        value_f = float(value)
        if math.isnan(value_f):
            continue
        key = (
            str(row["dataset"]),
            str(row["reference_model_key"]),
            int(row["reference_layer_index"]),
            str(row["reference_layer_name"]),
            str(row["distance_metric"]),
            str(row["corr_metric"]),
        )
        current = best.get(key)
        if current is None or value_f > float(current["value"]):
            best[key] = {
                "dataset": key[0],
                "reference_model_key": key[1],
                "reference_layer_index": key[2],
                "reference_layer_name": key[3],
                "distance_metric": key[4],
                "corr_metric": key[5],
                "candidate_model_key": str(row["candidate_model_key"]),
                "candidate_layer_index": int(row["candidate_layer_index"]),
                "candidate_layer_name": str(row["candidate_layer_name"]),
                "candidate_layer_dim": int(row["candidate_layer_dim"]),
                "value": value_f,
                "n_samples": int(row["n_samples"]),
                "n_pairs": int(row["n_pairs"]),
                "pair_scope": str(row.get("pair_scope", "global")),
                "group_field": row.get("group_field"),
                "n_groups_total": int(row.get("n_groups_total", 1) or 1),
                "n_groups_used": int(row.get("n_groups_used", 1) or 1),
            }
    return sorted(
        best.values(),
        key=lambda item: (
            item["dataset"],
            item["reference_model_key"],
            item["reference_layer_index"],
            item["corr_metric"],
            item["distance_metric"],
            -item["value"],
        ),
    )


def _finalize_report(
    config: AlignmentConfig,
    artifacts: RunArtifacts,
    state: dict[str, Any],
    *,
    session_elapsed_sec: float,
) -> dict[str, Any]:
    entries = _load_task_entries(artifacts.results_jsonl_path)
    rows = sorted(
        _flatten_result_entries(entries),
        key=lambda row: (
            row["dataset"],
            row["reference_model_key"],
            int(row["reference_layer_index"]),
            row["candidate_model_key"],
            int(row["candidate_layer_index"]),
            row["distance_metric"],
            row["corr_metric"],
        ),
    )
    errors = _load_task_entries(artifacts.errors_jsonl_path)
    dataset_summaries = sorted(state["datasets"].values(), key=lambda item: item["dataset"])

    report = {
        "created_at": _timestamp_now(),
        "session_elapsed_sec": session_elapsed_sec,
        "run_dir": str(artifacts.run_dir),
        "config": {
            **asdict(config),
            "datasets_root": str(config.datasets_root),
            "outputs_root": str(config.outputs_root),
            "tmp_dir": str(config.tmp_dir),
            "run_dir": str(config.run_dir),
        },
        "system": {
            "cpu_count": os.cpu_count(),
            "memory_total_bytes": int(psutil.virtual_memory().total),
            "memory_available_bytes": int(psutil.virtual_memory().available),
        },
        "dataset_summaries": dataset_summaries,
        "warnings": state["warnings"],
        "results": rows,
        "best_candidates": _build_best_candidates(rows),
        "errors": errors,
        "artifacts": state["artifacts"],
        "progress": {
            "total_tasks": state["total_tasks"],
            "completed_tasks": state["completed_tasks"],
            "completed_tasks_this_session": state["completed_tasks_this_session"],
            "failed_tasks": state["failed_tasks"],
            "skipped_tasks": state["skipped_tasks"],
        },
    }
    _atomic_write_json(artifacts.report_json_path, report)
    return report


def _validate_config(config: AlignmentConfig) -> None:
    if not config.datasets_root.exists():
        raise FileNotFoundError(f"datasets-root does not exist: {config.datasets_root}")
    if not config.outputs_root.exists():
        raise FileNotFoundError(f"outputs-root does not exist: {config.outputs_root}")
    if not config.reference_models:
        raise ValueError("At least one reference model is required")
    if not config.distance_metrics:
        raise ValueError("At least one distance metric is required")
    if not config.corr_metrics:
        raise ValueError("At least one correlation metric is required")
    if config.jobs is not None and config.jobs < 1:
        raise ValueError("jobs must be >= 1 or omitted")
    if config.sample_limit is not None and config.sample_limit < 2:
        raise ValueError("sample-limit must be >= 2 or omitted")
    if config.block_size < 2:
        raise ValueError("block-size must be >= 2")
    if config.heartbeat_sec < 1:
        raise ValueError("heartbeat-sec must be >= 1")
    if config.progress_mode not in {"auto", "bar", "log", "off"}:
        raise ValueError("progress_mode must be one of: auto, bar, log, off")
    if config.pair_scope not in SUPPORTED_PAIR_SCOPES:
        raise ValueError(f"pair_scope must be one of: {SUPPORTED_PAIR_SCOPES}")

    bad_distance = sorted(set(config.distance_metrics).difference(SUPPORTED_EMBEDDING_DISTANCES))
    if bad_distance:
        raise ValueError(
            f"Unsupported distance metrics: {bad_distance}; supported={SUPPORTED_EMBEDDING_DISTANCES}"
        )

    bad_corr = sorted(set(config.corr_metrics).difference(SUPPORTED_CORR_METRICS))
    if bad_corr:
        raise ValueError(f"Unsupported corr metrics: {bad_corr}; supported={SUPPORTED_CORR_METRICS}")


def run_alignment(config: AlignmentConfig) -> dict[str, Any]:
    _validate_config(config)
    session_started = time.perf_counter()

    datasets = discover_datasets(config.datasets_root, config.outputs_root, config.datasets)
    if not datasets:
        raise RuntimeError("No datasets discovered. Check --datasets-root/--outputs-root")

    artifacts = _build_run_artifacts(config.run_dir)
    _acquire_run_lock(artifacts.lock_path)
    logger: RunLogger | None = None

    try:
        run_config_payload = _initialize_run_config(config, artifacts)
        artifacts.cache_root.mkdir(parents=True, exist_ok=True)
        artifacts.task_logs_root.mkdir(parents=True, exist_ok=True)
        completed_task_keys = _prepare_existing_outputs_for_resume_alignment(artifacts) if config.resume else set()

        plans: list[AlignmentDatasetPlan] = []
        planning_errors: list[dict[str, Any]] = []
        for dataset in datasets:
            try:
                plans.append(_build_dataset_plan(config, dataset))
            except Exception as exc:
                planning_errors.append(
                    {
                        "status": "error",
                        "task_key": f"{dataset}::__dataset_plan__",
                        "dataset": dataset,
                        "reference_model_key": "__dataset_plan__",
                        "reference_layer_index": -1,
                        "reference_layer_name": "__dataset_plan__",
                        "candidate_model_key": "__dataset_plan__",
                        "candidate_layer_index": -1,
                        "candidate_layer_name": "__dataset_plan__",
                        "started_at": _timestamp_now(),
                        "finished_at": _timestamp_now(),
                        "elapsed_sec": 0.0,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "task_log_path": None,
                    }
                )
                if config.fail_fast:
                    raise

        state = _initial_progress_state(config, artifacts, run_config_payload, plans, completed_task_keys)
        for error_entry in planning_errors:
            state["failed_tasks"] += 1
            state["warnings"].append(
                f"dataset={error_entry['dataset']}: planning failed -> {error_entry['error']}"
            )
            _append_jsonl(artifacts.errors_jsonl_path, error_entry)

        logger = RunLogger(
            artifacts,
            mode=config.progress_mode,
            heartbeat_sec=config.heartbeat_sec,
            total_tasks=state["total_tasks"],
            initial=state["completed_tasks"],
        )
        logger.log(f"run_dir={artifacts.run_dir}")
        logger.log(
            f"planned datasets={len(plans)} tasks={state['total_tasks']} completed_on_resume={state['completed_tasks']}"
        )
        _write_progress_snapshot(artifacts, state)

        for plan in plans:
            dataset = plan.dataset
            ds_state = state["datasets"][dataset]
            state["current_dataset"] = dataset
            _write_progress_snapshot(artifacts, state)
            logger.refresh(
                state["completed_tasks"],
                state["total_tasks"],
                dataset=dataset,
                dataset_completed=ds_state["completed_tasks"],
                dataset_total=ds_state["total_tasks"],
            )
            logger.log(
                f"dataset start {dataset} refs={len(plan.reference_models)} candidates={len(plan.candidate_models)} "
                f"ref_layers={len(plan.reference_specs)} candidate_layers={len(plan.candidate_specs)} "
                f"samples={plan.n_samples} pairs={plan.n_pairs}"
            )

            if not plan.groups:
                ds_state["status"] = "skipped"
                state["current_dataset"] = None
                _write_progress_snapshot(artifacts, state)
                logger.log(f"dataset skip {dataset}: no runnable alignment tasks")
                continue

            dataset_cache_root = artifacts.cache_root / _safe_token(dataset)
            ds_state["cache_dir"] = str(dataset_cache_root)
            subset_digest = hashlib.sha1(np.asarray(plan.row_ids, dtype=np.int64).tobytes())
            _update_grouping_digest(
                subset_digest,
                pair_scope=plan.pair_scope,
                group_field=plan.group_field,
                groups=plan.pair_groups,
            )
            subset_hash = subset_digest.hexdigest()[:16]

            for group in plan.groups:
                pending_candidates = [
                    candidate_spec
                    for candidate_spec in group.candidate_specs
                    if _task_key_alignment(group.reference_spec, candidate_spec) not in completed_task_keys
                ]
                if not pending_candidates:
                    logger.log(
                        f"reference resume-skip {dataset} {group.reference_spec.model_key}:{group.reference_spec.layer_name}"
                    )
                    continue

                jobs = _resolve_jobs(config.jobs, len(pending_candidates), plan.n_pairs, config.corr_metrics)
                ds_state["jobs"] = jobs
                ds_state["status"] = "running"

                reference_cache_dir = (
                    dataset_cache_root
                    / subset_hash
                    / _safe_token(group.reference_spec.model_key)
                    / f"{group.reference_spec.layer_index:03d}_{_safe_token(group.reference_spec.layer_name)}"
                )
                logger.log(
                    f"reference cache {dataset} {group.reference_spec.model_key}:{group.reference_spec.layer_name} -> "
                    f"{reference_cache_dir}"
                )
                reference_pair_groups: list[AlignmentPairGroupCache] = []
                for pair_group in plan.pair_groups:
                    group_cache_dir = reference_cache_dir / f"group_{_safe_token(pair_group.group_key)}"
                    reference_pair_groups.append(
                        _build_reference_distance_cache(
                            group.reference_spec,
                            row_ids=plan.row_ids,
                            pair_group=pair_group,
                            config=config,
                            cache_dir=group_cache_dir,
                        )
                    )

                tasks = _prepare_alignment_tasks(
                    group.reference_spec,
                    pending_candidates,
                    row_ids=plan.row_ids,
                    n_samples=plan.n_samples,
                    n_pairs=plan.n_pairs,
                    pair_scope=plan.pair_scope,
                    group_field=plan.group_field,
                    config=config,
                    pair_groups=reference_pair_groups,
                    dataset_tmp_dir=(
                        config.tmp_dir
                        / _safe_token(dataset)
                        / _safe_token(group.reference_spec.model_key)
                        / f"{group.reference_spec.layer_index:03d}_{_safe_token(group.reference_spec.layer_name)}"
                    ),
                    task_logs_root=artifacts.task_logs_root,
                )

                logger.log(
                    f"reference execute {dataset} {group.reference_spec.model_key}:{group.reference_spec.layer_name} "
                    f"pending_tasks={len(tasks)} jobs={jobs}"
                )
                _write_progress_snapshot(artifacts, state)

                if jobs <= 1:
                    for task in tasks:
                        _mark_task_running(state, task)
                        _write_progress_snapshot(artifacts, state)
                        logger.log(f"task started {task.task_key}")
                        result = _evaluate_alignment_task(task)
                        _handle_task_result(
                            result,
                            state=state,
                            artifacts=artifacts,
                            logger=logger,
                            completed_task_keys=completed_task_keys,
                        )
                        _write_progress_snapshot(artifacts, state)
                        logger.refresh(
                            state["completed_tasks"],
                            state["total_tasks"],
                            dataset=dataset,
                            dataset_completed=ds_state["completed_tasks"],
                            dataset_total=ds_state["total_tasks"],
                        )
                        if result.get("status") == "error" and config.fail_fast:
                            raise RuntimeError(f"Task failed in fail-fast mode: {task.task_key}")
                else:
                    _limit_worker_threads()
                    with ProcessPoolExecutor(max_workers=jobs) as pool:
                        future_to_task = {}
                        for task in tasks:
                            future = pool.submit(_evaluate_alignment_task, task)
                            future_to_task[future] = task
                            _mark_task_running(state, task)
                            logger.log(f"task submitted {task.task_key}")
                        _write_progress_snapshot(artifacts, state)

                        while future_to_task:
                            done, _ = wait(
                                set(future_to_task.keys()),
                                timeout=max(1, config.heartbeat_sec),
                                return_when=FIRST_COMPLETED,
                            )
                            if not done:
                                logger.maybe_heartbeat(state)
                                _write_progress_snapshot(artifacts, state)
                                continue

                            for future in done:
                                task = future_to_task.pop(future)
                                try:
                                    result = future.result()
                                except Exception as exc:
                                    result = {
                                        "status": "error",
                                        "task_key": task.task_key,
                                        "started_at": state["running_tasks"].get(task.task_key, {}).get("started_at"),
                                        "finished_at": _timestamp_now(),
                                        "dataset": task.dataset,
                                        "reference_model_key": task.reference_model_key,
                                        "reference_layer_index": task.reference_layer_index,
                                        "reference_layer_name": task.reference_layer_name,
                                        "candidate_model_key": task.candidate_model_key,
                                        "candidate_layer_index": task.candidate_layer_index,
                                        "candidate_layer_name": task.candidate_layer_name,
                                        "elapsed_sec": 0.0,
                                        "error": str(exc),
                                        "traceback": traceback.format_exc(),
                                        "task_log_path": task.task_log_path,
                                    }
                                _handle_task_result(
                                    result,
                                    state=state,
                                    artifacts=artifacts,
                                    logger=logger,
                                    completed_task_keys=completed_task_keys,
                                )
                                _write_progress_snapshot(artifacts, state)
                                logger.refresh(
                                    state["completed_tasks"],
                                    state["total_tasks"],
                                    dataset=dataset,
                                    dataset_completed=ds_state["completed_tasks"],
                                    dataset_total=ds_state["total_tasks"],
                                )
                                if result.get("status") == "error" and config.fail_fast:
                                    raise RuntimeError(f"Task failed in fail-fast mode: {task.task_key}")

            if ds_state["completed_tasks"] >= ds_state["total_tasks"]:
                ds_state["status"] = "completed"
            elif ds_state["failed_tasks"] > 0:
                ds_state["status"] = "partial"
            state["current_dataset"] = None
            _write_progress_snapshot(artifacts, state)
            logger.log(
                f"dataset done {dataset} completed={ds_state['completed_tasks']}/{ds_state['total_tasks']} "
                f"failed={ds_state['failed_tasks']}"
            )

        report = _finalize_report(
            config,
            artifacts,
            state,
            session_elapsed_sec=float(time.perf_counter() - session_started),
        )
        state["status"] = "completed" if state["failed_tasks"] == 0 else "completed_with_errors"
        state["current_dataset"] = None
        _write_progress_snapshot(artifacts, state)

        if state["failed_tasks"] == 0 and not config.keep_cache and artifacts.cache_root.exists():
            shutil.rmtree(artifacts.cache_root, ignore_errors=True)
            logger.log(f"removed cache root: {artifacts.cache_root}")

        logger.log(f"run finished status={state['status']} report={artifacts.report_json_path}")
        return report
    finally:
        if logger is not None:
            logger.close()
        _release_run_lock(artifacts.lock_path)
