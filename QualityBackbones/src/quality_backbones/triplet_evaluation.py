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
import psutil
from scipy.spatial.distance import cdist

from quality_backbones.evaluation import (
    LayerSpec,
    PairGroup,
    RunArtifacts,
    RunLogger,
    SUPPORTED_EMBEDDING_DISTANCES,
    SUPPORTED_PAIR_SCOPES,
    _acquire_run_lock,
    _append_jsonl,
    _assign_dense_ranks,
    _atomic_write_json,
    _build_run_artifacts,
    _build_y_sequence_and_ties,
    _count_inversions_int32,
    _flatten_result_entries,
    _json_dumps,
    _limit_worker_threads,
    _load_completed_task_keys,
    _load_task_entries,
    _pair_count,
    _prepare_existing_outputs_for_resume,
    _read_rows,
    _release_run_lock,
    _resolve_jobs,
    _resolve_layer_specs,
    _safe_token,
    _select_rows,
    _task_key,
    _timestamp_now,
    _update_grouping_digest,
    _validate_row_id_alignment,
    discover_datasets,
    discover_models,
    extract_field_values,
    load_dataset_table,
    resolve_pair_groups,
    resolve_pair_scope,
    write_results_table,
)


TRIPLET_CORR_METRIC = "triplet_acc"
TRIPLET_SCORE_DISTANCE = "abs"
TRIPLET_TIE_POLICY = "strict_iff"
TRIPLET_AGGREGATION = "mean_over_groups"
TRIPLET_TABLE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model_key",
    "layer_index",
    "layer_name",
    "layer_dim",
    "embedding_distance",
    "score_distance",
    "corr_metric",
    "value",
    "pooled_value",
    "aggregation",
    "tie_policy",
    "n_samples",
    "n_pairs",
    "n_triplets_total",
    "n_triplets_correct",
    "n_ties_x",
    "n_ties_y",
    "n_ties_xy",
    "n_discordant",
    "pair_scope",
    "group_field",
    "n_groups_total",
    "n_groups_used",
    "exact",
    "elapsed_sec",
)


@dataclass(frozen=True)
class TripletEvaluationConfig:
    datasets_root: Path
    outputs_root: Path
    datasets: tuple[str, ...] | None
    models: tuple[str, ...] | None
    layer_selectors: tuple[str, ...] | None
    target_field: str
    sample_limit: int | None
    seed: int
    embedding_distances: tuple[str, ...]
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
class TripletGroupCache:
    group_key: str
    positions: tuple[int, ...]
    n_samples: int
    n_pairs: int
    n_triplets: int
    scores: np.ndarray


@dataclass(frozen=True)
class TripletDatasetPlan:
    dataset: str
    models: tuple[str, ...]
    row_ids: tuple[int, ...]
    scores_subset: np.ndarray
    layer_specs: tuple[LayerSpec, ...]
    n_samples: int
    n_pairs: int
    n_triplets: int
    pair_scope: str
    group_field: str | None
    pair_groups: tuple[PairGroup, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TripletTask:
    task_key: str
    dataset: str
    model_key: str
    layer_index: int
    layer_name: str
    layer_dim: int
    h5_path: str
    row_ids: tuple[int, ...]
    n_samples: int
    n_pairs: int
    n_triplets: int
    pair_scope: str
    group_field: str | None
    embedding_distances: tuple[str, ...]
    block_size: int
    tmp_dir: str
    keep_cache: bool
    task_log_path: str
    pair_groups: tuple[TripletGroupCache, ...]


@dataclass(frozen=True)
class TripletCountSummary:
    total: int
    correct: int
    ties_x: int
    ties_y: int
    ties_xy: int
    discordant: int


def _triplet_count(num_samples: int) -> int:
    if num_samples < 3:
        return 0
    return num_samples * (num_samples - 1) * (num_samples - 2)


def _dataset_subset_hash(
    dataset: str,
    target_field: str,
    row_ids: np.ndarray,
    *,
    pair_scope: str,
    group_field: str | None,
    groups: Sequence[PairGroup],
) -> str:
    digest = hashlib.sha1()
    digest.update(dataset.encode("utf-8"))
    digest.update(b"\0")
    digest.update(target_field.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(row_ids, dtype=np.int64).tobytes())
    digest.update(b"\0")
    _update_grouping_digest(digest, pair_scope=pair_scope, group_field=group_field, groups=groups)
    return digest.hexdigest()[:16]


def _result_config_payload(config: TripletEvaluationConfig) -> dict[str, Any]:
    def _sorted_or_none(values: tuple[str, ...] | None) -> list[str] | None:
        if values is None:
            return None
        return sorted(set(values))

    return {
        "datasets": _sorted_or_none(config.datasets),
        "models": _sorted_or_none(config.models),
        "layer_selectors": _sorted_or_none(config.layer_selectors),
        "target_field": config.target_field,
        "sample_limit": config.sample_limit,
        "seed": config.seed,
        "embedding_distances": sorted(set(config.embedding_distances)),
        "pair_scope": config.pair_scope,
        "score_distance": TRIPLET_SCORE_DISTANCE,
        "corr_metric": TRIPLET_CORR_METRIC,
        "tie_policy": TRIPLET_TIE_POLICY,
        "aggregation": TRIPLET_AGGREGATION,
    }


def _runtime_config_payload(config: TripletEvaluationConfig) -> dict[str, Any]:
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


def _config_fingerprint(config: TripletEvaluationConfig) -> str:
    payload = _result_config_payload(config)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _append_rows_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(TRIPLET_TABLE_COLUMNS), delimiter="\t")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in TRIPLET_TABLE_COLUMNS})


def _build_best_layers(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["model_key"]),
            str(row["embedding_distance"]),
            str(row["score_distance"]),
            str(row["corr_metric"]),
        )
        value = row.get("value")
        if value is None:
            continue
        value_f = float(value)
        if math.isnan(value_f):
            continue
        current = best.get(key)
        if current is None or float(current["value"]) < value_f:
            best[key] = {
                "dataset": key[0],
                "model_key": key[1],
                "embedding_distance": key[2],
                "score_distance": key[3],
                "corr_metric": key[4],
                "layer_index": int(row["layer_index"]),
                "layer_name": str(row["layer_name"]),
                "layer_dim": int(row["layer_dim"]),
                "value": value_f,
                "pooled_value": float(row.get("pooled_value", float("nan"))),
                "n_samples": int(row["n_samples"]),
                "n_pairs": int(row["n_pairs"]),
                "n_triplets_total": int(row.get("n_triplets_total", 0)),
                "n_triplets_correct": int(row.get("n_triplets_correct", 0)),
                "pair_scope": str(row.get("pair_scope", "global")),
                "group_field": row.get("group_field"),
                "n_groups_total": int(row.get("n_groups_total", 1) or 1),
                "n_groups_used": int(row.get("n_groups_used", 1) or 1),
            }
    return sorted(
        best.values(),
        key=lambda item: (
            item["dataset"],
            item["model_key"],
            item["corr_metric"],
            item["embedding_distance"],
            item["score_distance"],
            -item["value"],
        ),
    )


def _count_anchor_triplets(x_values: np.ndarray, y_values: np.ndarray) -> TripletCountSummary:
    m = int(x_values.shape[0])
    if m < 2:
        return TripletCountSummary(total=0, correct=0, ties_x=0, ties_y=0, ties_xy=0, discordant=0)

    total_unordered = (m * (m - 1)) // 2
    total = m * (m - 1)

    y_order = np.argsort(y_values, kind="mergesort")
    y_dense = np.empty(m, dtype=np.int32)
    ties_y, _ = _assign_dense_ranks(y_values, y_order, y_dense)

    order = np.lexsort((y_dense, x_values))
    y_seq, ties_x, ties_xy = _build_y_sequence_and_ties(order, x_values, y_dense)
    discordant = int(_count_inversions_int32(y_seq))
    correct = int(total - (2 * discordant) - ties_x - ties_y + (2 * ties_xy))
    if correct < 0 or correct > total:
        raise RuntimeError(
            "Invalid triplet count state: "
            f"total={total} total_unordered={total_unordered} correct={correct} ties_x={ties_x} "
            f"ties_y={ties_y} ties_xy={ties_xy} discordant={discordant}"
        )

    return TripletCountSummary(
        total=int(total),
        correct=correct,
        ties_x=int(ties_x),
        ties_y=int(ties_y),
        ties_xy=int(ties_xy),
        discordant=discordant,
    )


def _compute_group_triplet_summary(
    embeddings: np.ndarray,
    scores: np.ndarray,
    *,
    embedding_distance: str,
    block_size: int,
) -> TripletCountSummary:
    n_samples = int(scores.shape[0])
    if n_samples < 3:
        return TripletCountSummary(total=0, correct=0, ties_x=0, ties_y=0, ties_xy=0, discordant=0)

    if embedding_distance not in SUPPORTED_EMBEDDING_DISTANCES:
        raise ValueError(
            f"Unsupported embedding distance {embedding_distance!r}; supported={SUPPORTED_EMBEDDING_DISTANCES}"
        )

    embeddings_f32 = np.asarray(embeddings, dtype=np.float32)
    scores_f64 = np.asarray(scores, dtype=np.float64)

    sq_norms: np.ndarray | None = None
    norms: np.ndarray | None = None
    if embedding_distance in {"cos", "l2"}:
        sq_norms = np.sum(embeddings_f32 * embeddings_f32, axis=1, dtype=np.float32)
    if embedding_distance == "cos":
        norms = np.sqrt(np.clip(sq_norms, 1e-12, None))

    total = 0
    correct = 0
    ties_x = 0
    ties_y = 0
    ties_xy = 0
    discordant = 0

    for anchor_start in range(0, n_samples, block_size):
        anchor_stop = min(n_samples, anchor_start + block_size)
        anchor_embeddings = embeddings_f32[anchor_start:anchor_stop]
        anchor_scores = scores_f64[anchor_start:anchor_stop]
        score_block = np.abs(anchor_scores[:, None] - scores_f64[None, :])

        if embedding_distance in {"cos", "l2"}:
            gram = np.asarray(anchor_embeddings @ embeddings_f32.T, dtype=np.float32)
        else:
            gram = None

        if embedding_distance == "cos":
            if norms is None:
                raise RuntimeError("Internal error: cosine norms are missing")
            denom = norms[anchor_start:anchor_stop, None] * norms[None, :]
            denom = np.clip(denom, 1e-12, None)
            dist_block = 1.0 - (gram / denom)
        elif embedding_distance == "l2":
            if sq_norms is None:
                raise RuntimeError("Internal error: l2 squared norms are missing")
            dist_block = sq_norms[anchor_start:anchor_stop, None] + sq_norms[None, :] - (2.0 * gram)
            np.maximum(dist_block, 0.0, out=dist_block)
            np.sqrt(dist_block, out=dist_block)
        else:
            dist_block = np.asarray(cdist(anchor_embeddings, embeddings_f32, metric="cityblock"), dtype=np.float32)

        for local_anchor, anchor_idx in enumerate(range(anchor_start, anchor_stop)):
            x_values = np.empty(n_samples - 1, dtype=np.float32)
            y_values = np.empty(n_samples - 1, dtype=np.float64)
            if anchor_idx > 0:
                x_values[:anchor_idx] = dist_block[local_anchor, :anchor_idx]
                y_values[:anchor_idx] = score_block[local_anchor, :anchor_idx]
            if anchor_idx + 1 < n_samples:
                x_values[anchor_idx:] = dist_block[local_anchor, anchor_idx + 1 :]
                y_values[anchor_idx:] = score_block[local_anchor, anchor_idx + 1 :]

            summary = _count_anchor_triplets(x_values, y_values)
            total += summary.total
            correct += summary.correct
            ties_x += summary.ties_x
            ties_y += summary.ties_y
            ties_xy += summary.ties_xy
            discordant += summary.discordant

    return TripletCountSummary(
        total=total,
        correct=correct,
        ties_x=ties_x,
        ties_y=ties_y,
        ties_xy=ties_xy,
        discordant=discordant,
    )


def _build_pair_group_caches(scores: np.ndarray, pair_groups: Sequence[PairGroup]) -> tuple[TripletGroupCache, ...]:
    caches: list[TripletGroupCache] = []
    for group in pair_groups:
        group_positions = np.asarray(group.positions, dtype=np.int64)
        group_scores = np.asarray(scores[group_positions], dtype=np.float64)
        caches.append(
            TripletGroupCache(
                group_key=group.group_key,
                positions=group.positions,
                n_samples=group.n_samples,
                n_pairs=group.n_pairs,
                n_triplets=_triplet_count(group.n_samples),
                scores=group_scores,
            )
        )
    return tuple(caches)


def _resolve_triplet_jobs(jobs: int | None, task_count: int, n_samples: int, block_size: int) -> int:
    if task_count <= 1:
        return 1
    if jobs is not None:
        return max(1, min(task_count, jobs))

    cpu_count = os.cpu_count() or 1
    upper = min(task_count, cpu_count)

    available = int(psutil.virtual_memory().available * 0.8)
    per_worker = max(1, n_samples) * max(1, block_size) * (4 + 8)
    per_worker += max(1, n_samples) * (4 + 8)
    by_memory = max(1, available // max(1, per_worker))
    return max(1, min(upper, by_memory))


def _evaluate_layer_task(task: TripletTask) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _timestamp_now()
    row_ids = np.asarray(task.row_ids, dtype=np.int64)
    layer_tmp = Path(task.tmp_dir) / f"{task.model_key}_{task.layer_index:03d}_{os.getpid()}_{time.time_ns()}"
    layer_tmp.mkdir(parents=True, exist_ok=True)
    task_log_path = Path(task.task_log_path)
    task_log_path.parent.mkdir(parents=True, exist_ok=True)

    phase_timings: dict[str, float] = {
        "read_embeddings_sec": 0.0,
        "triplet_sec": 0.0,
    }

    def _task_log(message: str) -> None:
        with task_log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"[{_timestamp_now()}] {message}\n")

    _task_log(f"START task_key={task.task_key} dataset={task.dataset} model={task.model_key} layer={task.layer_name}")
    try:
        phase_started = time.perf_counter()
        _task_log("phase=read_embeddings start")
        with h5py.File(task.h5_path, "r") as fp:
            ds_name = f"layer_{task.layer_index:03d}"
            if ds_name not in fp:
                raise KeyError(f"Missing dataset {ds_name} in {task.h5_path}")
            embeddings = _read_rows(fp[ds_name], row_ids).astype(np.float32, copy=False)
        phase_timings["read_embeddings_sec"] += float(time.perf_counter() - phase_started)
        _task_log(f"phase=read_embeddings done sec={phase_timings['read_embeddings_sec']:.3f}")

        metric_sum: dict[str, float] = {}
        metric_count: dict[str, int] = {}
        metric_pairs: dict[str, int] = {}
        metric_triplets_total: dict[str, int] = {}
        metric_triplets_correct: dict[str, int] = {}
        metric_ties_x: dict[str, int] = {}
        metric_ties_y: dict[str, int] = {}
        metric_ties_xy: dict[str, int] = {}
        metric_discordant: dict[str, int] = {}
        metric_elapsed: dict[str, float] = {}

        def _accumulate(metric: str, summary: TripletCountSummary, elapsed_sec: float, n_pairs: int) -> None:
            if summary.total <= 0:
                return
            key = metric
            metric_sum[key] = metric_sum.get(key, 0.0) + (summary.correct / summary.total)
            metric_count[key] = metric_count.get(key, 0) + 1
            metric_pairs[key] = metric_pairs.get(key, 0) + n_pairs
            metric_triplets_total[key] = metric_triplets_total.get(key, 0) + summary.total
            metric_triplets_correct[key] = metric_triplets_correct.get(key, 0) + summary.correct
            metric_ties_x[key] = metric_ties_x.get(key, 0) + summary.ties_x
            metric_ties_y[key] = metric_ties_y.get(key, 0) + summary.ties_y
            metric_ties_xy[key] = metric_ties_xy.get(key, 0) + summary.ties_xy
            metric_discordant[key] = metric_discordant.get(key, 0) + summary.discordant
            metric_elapsed[key] = metric_elapsed.get(key, 0.0) + elapsed_sec

        for group_index, group in enumerate(task.pair_groups):
            if group.n_triplets <= 0:
                _task_log(
                    f"group={group_index:04d} key={group.group_key} skip n_samples={group.n_samples} n_triplets={group.n_triplets}"
                )
                continue

            group_positions = np.asarray(group.positions, dtype=np.int64)
            group_embeddings = embeddings[group_positions]
            for emb_metric in task.embedding_distances:
                metric_started = time.perf_counter()
                _task_log(
                    f"group={group_index:04d} key={group.group_key} embedding_metric={emb_metric} start n_samples={group.n_samples}"
                )
                summary = _compute_group_triplet_summary(
                    group_embeddings,
                    group.scores,
                    embedding_distance=emb_metric,
                    block_size=task.block_size,
                )
                elapsed = float(time.perf_counter() - metric_started)
                phase_timings["triplet_sec"] += elapsed
                _accumulate(emb_metric, summary, elapsed, group.n_pairs)
                _task_log(
                    f"group={group_index:04d} key={group.group_key} embedding_metric={emb_metric} done "
                    f"triplets={summary.total} correct={summary.correct} sec={elapsed:.3f}"
                )

        per_metric_rows: list[dict[str, Any]] = []
        n_groups_total = len(task.pair_groups)
        for emb_metric in task.embedding_distances:
            used_groups = int(metric_count.get(emb_metric, 0))
            triplet_total = int(metric_triplets_total.get(emb_metric, 0))
            triplet_correct = int(metric_triplets_correct.get(emb_metric, 0))
            pooled_value = float("nan") if triplet_total <= 0 else float(triplet_correct / triplet_total)
            value = float("nan") if used_groups == 0 else float(metric_sum[emb_metric] / used_groups)
            per_metric_rows.append(
                {
                    "dataset": task.dataset,
                    "model_key": task.model_key,
                    "layer_index": task.layer_index,
                    "layer_name": task.layer_name,
                    "layer_dim": task.layer_dim,
                    "embedding_distance": emb_metric,
                    "score_distance": TRIPLET_SCORE_DISTANCE,
                    "corr_metric": TRIPLET_CORR_METRIC,
                    "value": value,
                    "pooled_value": pooled_value,
                    "aggregation": TRIPLET_AGGREGATION,
                    "tie_policy": TRIPLET_TIE_POLICY,
                    "n_samples": task.n_samples,
                    "n_pairs": int(metric_pairs.get(emb_metric, 0)),
                    "n_triplets_total": triplet_total,
                    "n_triplets_correct": triplet_correct,
                    "n_ties_x": int(metric_ties_x.get(emb_metric, 0)),
                    "n_ties_y": int(metric_ties_y.get(emb_metric, 0)),
                    "n_ties_xy": int(metric_ties_xy.get(emb_metric, 0)),
                    "n_discordant": int(metric_discordant.get(emb_metric, 0)),
                    "pair_scope": task.pair_scope,
                    "group_field": task.group_field,
                    "n_groups_total": n_groups_total,
                    "n_groups_used": used_groups,
                    "exact": True,
                    "elapsed_sec": float(metric_elapsed.get(emb_metric, 0.0)),
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
            "model_key": task.model_key,
            "layer_index": task.layer_index,
            "layer_name": task.layer_name,
            "rows": per_metric_rows,
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
            "model_key": task.model_key,
            "layer_index": task.layer_index,
            "layer_name": task.layer_name,
            "elapsed_sec": elapsed,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "task_log_path": str(task_log_path),
        }


def _prepare_layer_tasks(
    layer_specs: Sequence[LayerSpec],
    row_ids: Sequence[int],
    n_samples: int,
    n_pairs: int,
    n_triplets: int,
    pair_scope: str,
    group_field: str | None,
    config: TripletEvaluationConfig,
    pair_groups: Sequence[TripletGroupCache],
    dataset_tmp_dir: Path,
    task_logs_root: Path,
) -> list[TripletTask]:
    tasks: list[TripletTask] = []
    row_tuple = tuple(int(v) for v in row_ids)
    for spec in layer_specs:
        task_key = _task_key(spec.dataset, spec.model_key, spec.layer_index, spec.layer_name)
        task_log_dir = task_logs_root / _safe_token(spec.dataset) / _safe_token(spec.model_key)
        task_log_path = task_log_dir / f"layer_{spec.layer_index:03d}_{_safe_token(spec.layer_name)}.log"
        tasks.append(
            TripletTask(
                task_key=task_key,
                dataset=spec.dataset,
                model_key=spec.model_key,
                layer_index=spec.layer_index,
                layer_name=spec.layer_name,
                layer_dim=spec.layer_dim,
                h5_path=spec.h5_path,
                row_ids=row_tuple,
                n_samples=n_samples,
                n_pairs=n_pairs,
                n_triplets=n_triplets,
                pair_scope=pair_scope,
                group_field=group_field,
                embedding_distances=config.embedding_distances,
                block_size=config.block_size,
                tmp_dir=str(dataset_tmp_dir),
                keep_cache=config.keep_cache,
                task_log_path=str(task_log_path),
                pair_groups=tuple(pair_groups),
            )
        )
    return tasks


def _build_dataset_plan(config: TripletEvaluationConfig, dataset: str) -> TripletDatasetPlan:
    warnings: list[str] = []
    resolved_pair_scope = resolve_pair_scope(config.pair_scope, dataset)
    models = discover_models(config.outputs_root, dataset, config.models)
    if not models:
        warnings.append(f"dataset={dataset}: no completed model outputs found")
        return TripletDatasetPlan(
            dataset=dataset,
            models=tuple(),
            row_ids=tuple(),
            scores_subset=np.empty((0,), dtype=np.float64),
            layer_specs=tuple(),
            n_samples=0,
            n_pairs=0,
            n_triplets=0,
            pair_scope=resolved_pair_scope,
            group_field=None,
            pair_groups=tuple(),
            warnings=tuple(warnings),
        )

    _validate_row_id_alignment(config.outputs_root, dataset, models)

    anchor_model = models[0]
    table = load_dataset_table(config.datasets_root, config.outputs_root, dataset, anchor_model)
    raw_scores = extract_field_values(table, config.target_field, numeric=True)
    valid_mask = np.isfinite(raw_scores)
    if not np.all(valid_mask):
        dropped = int((~valid_mask).sum())
        warnings.append(f"dataset={dataset}: dropped {dropped} rows with non-finite target values")
    table = table.loc[valid_mask].copy().reset_index(drop=True)
    scores = raw_scores[valid_mask]

    row_ids, scores_subset = _select_rows(table, scores, config.sample_limit, config.seed)
    n_samples = int(row_ids.shape[0])
    if n_samples < 2:
        warnings.append(f"dataset={dataset}: not enough samples after filtering/sampling (n={n_samples})")
        return TripletDatasetPlan(
            dataset=dataset,
            models=tuple(models),
            row_ids=tuple(int(v) for v in row_ids.tolist()),
            scores_subset=np.asarray(scores_subset, dtype=np.float64),
            layer_specs=tuple(),
            n_samples=n_samples,
            n_pairs=_pair_count(n_samples),
            n_triplets=_triplet_count(n_samples),
            pair_scope=resolved_pair_scope,
            group_field=None,
            pair_groups=tuple(),
            warnings=tuple(warnings),
        )

    group_field, pair_groups, group_warnings = resolve_pair_groups(
        table,
        row_ids,
        dataset=dataset,
        datasets_root=config.datasets_root,
        pair_scope=resolved_pair_scope,
    )
    warnings.extend(group_warnings)
    n_pairs = sum(group.n_pairs for group in pair_groups)
    n_triplets = sum(_triplet_count(group.n_samples) for group in pair_groups)
    if not pair_groups:
        warnings.append(f"dataset={dataset}: no runnable pair groups for pair_scope={resolved_pair_scope}")
        return TripletDatasetPlan(
            dataset=dataset,
            models=tuple(models),
            row_ids=tuple(int(v) for v in row_ids.tolist()),
            scores_subset=np.asarray(scores_subset, dtype=np.float64),
            layer_specs=tuple(),
            n_samples=n_samples,
            n_pairs=0,
            n_triplets=0,
            pair_scope=resolved_pair_scope,
            group_field=group_field,
            pair_groups=tuple(),
            warnings=tuple(warnings),
        )

    layer_specs = _resolve_layer_specs(
        outputs_root=config.outputs_root,
        dataset=dataset,
        model_keys=models,
        selectors=config.layer_selectors,
    )

    return TripletDatasetPlan(
        dataset=dataset,
        models=tuple(models),
        row_ids=tuple(int(v) for v in row_ids.tolist()),
        scores_subset=np.asarray(scores_subset, dtype=np.float64),
        layer_specs=tuple(layer_specs),
        n_samples=n_samples,
        n_pairs=n_pairs,
        n_triplets=n_triplets,
        pair_scope=resolved_pair_scope,
        group_field=group_field,
        pair_groups=pair_groups,
        warnings=tuple(warnings),
    )


def _initial_progress_state(
    config: TripletEvaluationConfig,
    artifacts: RunArtifacts,
    run_config_payload: dict[str, Any],
    plans: Sequence[TripletDatasetPlan],
    completed_task_keys: set[str],
) -> dict[str, Any]:
    datasets_payload: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    total_tasks = 0
    completed_total = 0

    for plan in plans:
        task_keys = [_task_key(spec.dataset, spec.model_key, spec.layer_index, spec.layer_name) for spec in plan.layer_specs]
        completed_dataset = sum(1 for key in task_keys if key in completed_task_keys)
        total = len(task_keys)
        total_tasks += total
        completed_total += completed_dataset
        warnings.extend(plan.warnings)

        datasets_payload[plan.dataset] = {
            "dataset": plan.dataset,
            "n_models": len(plan.models),
            "n_layers": len(plan.layer_specs),
            "n_samples": plan.n_samples,
            "n_pairs": plan.n_pairs,
            "n_triplets": plan.n_triplets,
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


def _initialize_run_config(config: TripletEvaluationConfig, artifacts: RunArtifacts) -> dict[str, Any]:
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


def _mark_task_running(state: dict[str, Any], task: TripletTask) -> None:
    state["running_tasks"][task.task_key] = {
        "task_key": task.task_key,
        "dataset": task.dataset,
        "model_key": task.model_key,
        "layer_index": task.layer_index,
        "layer_name": task.layer_name,
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
        _append_rows_tsv(artifacts.table_tsv_path, result.get("rows", []))
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


def _finalize_report(
    config: TripletEvaluationConfig,
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
            row["model_key"],
            int(row["layer_index"]),
            row["embedding_distance"],
            row["score_distance"],
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
            "score_distance": TRIPLET_SCORE_DISTANCE,
            "corr_metric": TRIPLET_CORR_METRIC,
            "tie_policy": TRIPLET_TIE_POLICY,
            "aggregation": TRIPLET_AGGREGATION,
        },
        "system": {
            "cpu_count": os.cpu_count(),
            "memory_total_bytes": int(psutil.virtual_memory().total),
            "memory_available_bytes": int(psutil.virtual_memory().available),
        },
        "dataset_summaries": dataset_summaries,
        "warnings": state["warnings"],
        "results": rows,
        "best_layers": _build_best_layers(rows),
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


def run_triplet_evaluation(config: TripletEvaluationConfig) -> dict[str, Any]:
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
        completed_task_keys = _prepare_existing_outputs_for_resume(artifacts) if config.resume else set()

        plans = [_build_dataset_plan(config, dataset) for dataset in datasets]
        state = _initial_progress_state(config, artifacts, run_config_payload, plans, completed_task_keys)
        _write_progress_snapshot(artifacts, state)

        logger = RunLogger(
            artifacts,
            mode=config.progress_mode,
            heartbeat_sec=config.heartbeat_sec,
            total_tasks=state["total_tasks"],
            initial=state["completed_tasks"],
        )
        logger.log(
            f"run start datasets={len(datasets)} total_tasks={state['total_tasks']} completed={state['completed_tasks']}"
        )

        for plan in plans:
            dataset = plan.dataset
            state["current_dataset"] = dataset
            ds_state = state["datasets"][dataset]
            logger.refresh(
                state["completed_tasks"],
                state["total_tasks"],
                dataset=dataset,
                dataset_completed=ds_state["completed_tasks"],
                dataset_total=ds_state["total_tasks"],
            )
            logger.log(
                f"dataset start {dataset} models={len(plan.models)} layers={len(plan.layer_specs)} "
                f"samples={plan.n_samples} pairs={plan.n_pairs} triplets={plan.n_triplets}"
            )

            if not plan.layer_specs:
                ds_state["status"] = "skipped"
                state["current_dataset"] = None
                _write_progress_snapshot(artifacts, state)
                logger.log(f"dataset skip {dataset}: no runnable layer tasks")
                continue

            pending_specs = [
                spec
                for spec in plan.layer_specs
                if _task_key(spec.dataset, spec.model_key, spec.layer_index, spec.layer_name) not in completed_task_keys
            ]

            jobs = 0 if not pending_specs else _resolve_triplet_jobs(
                config.jobs,
                len(pending_specs),
                plan.n_samples,
                config.block_size,
            )
            ds_state["jobs"] = jobs

            if not pending_specs:
                ds_state["status"] = "completed"
                state["current_dataset"] = None
                _write_progress_snapshot(artifacts, state)
                logger.log(f"dataset resume-skip {dataset}: all tasks already completed")
                continue

            cache_hash = _dataset_subset_hash(
                plan.dataset,
                config.target_field,
                np.asarray(plan.row_ids, dtype=np.int64),
                pair_scope=plan.pair_scope,
                group_field=plan.group_field,
                groups=plan.pair_groups,
            )
            ds_state["cache_dir"] = str(artifacts.cache_root / _safe_token(plan.dataset) / cache_hash)

            pair_group_caches = _build_pair_group_caches(plan.scores_subset, plan.pair_groups)
            tasks = _prepare_layer_tasks(
                layer_specs=pending_specs,
                row_ids=plan.row_ids,
                n_samples=plan.n_samples,
                n_pairs=plan.n_pairs,
                n_triplets=plan.n_triplets,
                pair_scope=plan.pair_scope,
                group_field=plan.group_field,
                config=config,
                pair_groups=pair_group_caches,
                dataset_tmp_dir=config.tmp_dir / _safe_token(dataset),
                task_logs_root=artifacts.task_logs_root,
            )

            ds_state["status"] = "running"
            _write_progress_snapshot(artifacts, state)
            logger.log(f"dataset execute {dataset}: pending_tasks={len(tasks)} jobs={jobs}")

            if jobs <= 1:
                for task in tasks:
                    _mark_task_running(state, task)
                    _write_progress_snapshot(artifacts, state)
                    logger.log(f"task started {task.task_key}")
                    result = _evaluate_layer_task(task)
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
                        future = pool.submit(_evaluate_layer_task, task)
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
                                    "model_key": task.model_key,
                                    "layer_index": task.layer_index,
                                    "layer_name": task.layer_name,
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


def _validate_config(config: TripletEvaluationConfig) -> None:
    if not config.datasets_root.exists():
        raise FileNotFoundError(f"datasets-root does not exist: {config.datasets_root}")
    if not config.outputs_root.exists():
        raise FileNotFoundError(f"outputs-root does not exist: {config.outputs_root}")
    if config.jobs is not None and config.jobs < 1:
        raise ValueError("jobs must be >= 1 or omitted")
    if config.heartbeat_sec < 1:
        raise ValueError("heartbeat-sec must be >= 1")
    if config.progress_mode not in {"auto", "bar", "log", "off"}:
        raise ValueError("progress_mode must be one of: auto, bar, log, off")
    if config.pair_scope not in SUPPORTED_PAIR_SCOPES:
        raise ValueError(f"pair_scope must be one of: {SUPPORTED_PAIR_SCOPES}")
    if config.sample_limit is not None and config.sample_limit < 2:
        raise ValueError("sample-limit must be >= 2 or omitted")
    if config.block_size < 2:
        raise ValueError("block-size must be >= 2")

    bad_embedding = sorted(set(config.embedding_distances).difference(SUPPORTED_EMBEDDING_DISTANCES))
    if bad_embedding:
        raise ValueError(
            f"Unsupported embedding distances: {bad_embedding}; supported={SUPPORTED_EMBEDDING_DISTANCES}"
        )
    if not config.embedding_distances:
        raise ValueError("At least one embedding distance is required")
