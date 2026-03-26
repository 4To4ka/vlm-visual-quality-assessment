from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import traceback
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
import psutil
from numba import njit
from scipy.spatial.distance import cdist

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional runtime dependency
    tqdm = None


SUPPORTED_EMBEDDING_DISTANCES: tuple[str, ...] = ("cos", "l2", "l1")
SUPPORTED_SCORE_DISTANCES: tuple[str, ...] = ("abs", "sq")
SUPPORTED_CORR_METRICS: tuple[str, ...] = ("pcc", "scc", "kcc")
SUPPORTED_PAIR_SCOPES: tuple[str, ...] = ("auto", "global", "within_ref")
FULL_REFERENCE_DATASETS: tuple[str, ...] = ("CSIQ", "kadid10k", "TID2013", "PieAPP", "PIPAL")
REQUIRED_SUCCESS_FILES: tuple[str, ...] = ("meta.json", "index.parquet", "layers.h5")
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
TABLE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model_key",
    "layer_index",
    "layer_name",
    "layer_dim",
    "embedding_distance",
    "score_distance",
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
class EvaluationConfig:
    datasets_root: Path
    outputs_root: Path
    datasets: tuple[str, ...] | None
    models: tuple[str, ...] | None
    layer_selectors: tuple[str, ...] | None
    target_field: str
    sample_limit: int | None
    seed: int
    embedding_distances: tuple[str, ...]
    score_distances: tuple[str, ...]
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
class ScoreVectorInfo:
    metric: str
    path: str
    sum_value: float
    sum_sq_value: float


@dataclass(frozen=True)
class RankCacheInfo:
    source_metric: str
    avg_rank_path: str | None
    dense_rank_path: str | None
    tie_pairs: int
    unique_count: int
    rank_sum: float
    rank_sum_sq: float | None


@dataclass(frozen=True)
class LayerSpec:
    dataset: str
    model_key: str
    layer_index: int
    layer_name: str
    layer_dim: int
    h5_path: str


@dataclass(frozen=True)
class LayerTask:
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
    pair_scope: str
    group_field: str | None
    embedding_distances: tuple[str, ...]
    score_distances: tuple[str, ...]
    corr_metrics: tuple[str, ...]
    block_size: int
    tmp_dir: str
    keep_cache: bool
    task_log_path: str
    pair_groups: tuple["PairGroupCache", ...]


@dataclass(frozen=True)
class DatasetScoreCache:
    score_vectors: dict[str, ScoreVectorInfo]
    rank_cache: RankCacheInfo | None
    cleanup_paths: tuple[str, ...]


@dataclass(frozen=True)
class PairGroup:
    group_key: str
    positions: tuple[int, ...]
    n_samples: int
    n_pairs: int


@dataclass(frozen=True)
class PairGroupCache:
    group_key: str
    positions: tuple[int, ...]
    n_samples: int
    n_pairs: int
    score_vectors: dict[str, ScoreVectorInfo]
    rank_cache: RankCacheInfo | None


@dataclass(frozen=True)
class DatasetPlan:
    dataset: str
    models: tuple[str, ...]
    row_ids: tuple[int, ...]
    scores_subset: np.ndarray
    layer_specs: tuple[LayerSpec, ...]
    n_samples: int
    n_pairs: int
    pair_scope: str
    group_field: str | None
    pair_groups: tuple[PairGroup, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    config_path: Path
    lock_path: Path
    results_jsonl_path: Path
    errors_jsonl_path: Path
    progress_json_path: Path
    report_json_path: Path
    table_tsv_path: Path
    run_log_path: Path
    cache_root: Path
    task_logs_root: Path


@njit(cache=True)
def _fill_score_pairs_abs(scores: np.ndarray, out: np.ndarray) -> tuple[float, float]:
    n = int(scores.shape[0])
    offset = 0
    sum_value = 0.0
    sum_sq_value = 0.0
    for i in range(n - 1):
        si = scores[i]
        for j in range(i + 1, n):
            diff = si - scores[j]
            if diff < 0:
                diff = -diff
            out[offset] = diff
            sum_value += diff
            sum_sq_value += diff * diff
            offset += 1
    return sum_value, sum_sq_value


@njit(cache=True)
def _fill_score_pairs_sq(scores: np.ndarray, out: np.ndarray) -> tuple[float, float]:
    n = int(scores.shape[0])
    offset = 0
    sum_value = 0.0
    sum_sq_value = 0.0
    for i in range(n - 1):
        si = scores[i]
        for j in range(i + 1, n):
            diff = si - scores[j]
            if diff < 0:
                diff = -diff
            value = diff * diff
            out[offset] = value
            sum_value += value
            sum_sq_value += value * value
            offset += 1
    return sum_value, sum_sq_value


@njit(cache=True)
def _assign_avg_dense_ranks(
    values: np.ndarray,
    order: np.ndarray,
    avg_rank: np.ndarray,
    dense_rank: np.ndarray,
) -> tuple[int, int, float]:
    m = int(order.shape[0])
    if m == 0:
        return 0, 0, 0.0

    tie_pairs = 0
    unique_count = 0
    sum_rank_sq = 0.0
    start = 0

    while start < m:
        first_idx = order[start]
        first_value = values[first_idx]
        end = start + 1
        while end < m:
            idx = order[end]
            if values[idx] != first_value:
                break
            end += 1

        unique_count += 1
        group_size = end - start
        avg_value = 0.5 * (start + end - 1) + 1.0
        sum_rank_sq += group_size * avg_value * avg_value

        for pos in range(start, end):
            idx = order[pos]
            avg_rank[idx] = avg_value
            dense_rank[idx] = unique_count

        tie_pairs += (group_size * (group_size - 1)) // 2
        start = end

    return tie_pairs, unique_count, sum_rank_sq


@njit(cache=True)
def _assign_avg_ranks(values: np.ndarray, order: np.ndarray, avg_rank: np.ndarray) -> float:
    m = int(order.shape[0])
    if m == 0:
        return 0.0

    sum_rank_sq = 0.0
    start = 0
    while start < m:
        first_idx = order[start]
        first_value = values[first_idx]
        end = start + 1
        while end < m:
            idx = order[end]
            if values[idx] != first_value:
                break
            end += 1

        group_size = end - start
        avg_value = 0.5 * (start + end - 1) + 1.0
        sum_rank_sq += group_size * avg_value * avg_value

        for pos in range(start, end):
            idx = order[pos]
            avg_rank[idx] = avg_value

        start = end

    return sum_rank_sq


@njit(cache=True)
def _assign_dense_ranks(values: np.ndarray, order: np.ndarray, dense_rank: np.ndarray) -> tuple[int, int]:
    m = int(order.shape[0])
    if m == 0:
        return 0, 0

    tie_pairs = 0
    unique_count = 0
    start = 0
    while start < m:
        first_idx = order[start]
        first_value = values[first_idx]
        end = start + 1
        while end < m:
            idx = order[end]
            if values[idx] != first_value:
                break
            end += 1

        unique_count += 1
        group_size = end - start
        tie_pairs += (group_size * (group_size - 1)) // 2

        for pos in range(start, end):
            idx = order[pos]
            dense_rank[idx] = unique_count

        start = end

    return tie_pairs, unique_count


@njit(cache=True)
def _scan_groups_scc(order: np.ndarray, x_values: np.ndarray, y_avg_rank: np.ndarray) -> tuple[float, float, float]:
    m = int(order.shape[0])
    if m == 0:
        return 0.0, 0.0, 0.0

    sum_x = 0.0
    sum_x2 = 0.0
    sum_xy = 0.0

    start = 0
    while start < m:
        first_idx = order[start]
        first_x = x_values[first_idx]
        end = start + 1
        while end < m:
            idx = order[end]
            if x_values[idx] != first_x:
                break
            end += 1

        avg_x_rank = 0.5 * (start + end - 1) + 1.0
        group_size = end - start

        sum_y_group = 0.0
        for pos in range(start, end):
            idx = order[pos]
            sum_y_group += y_avg_rank[idx]

        sum_x += avg_x_rank * group_size
        sum_x2 += avg_x_rank * avg_x_rank * group_size
        sum_xy += avg_x_rank * sum_y_group

        start = end

    return sum_x, sum_x2, sum_xy


@njit(cache=True)
def _build_y_sequence_and_ties(
    order: np.ndarray,
    x_values: np.ndarray,
    y_dense_rank: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    m = int(order.shape[0])
    y_seq = np.empty(m, dtype=np.int32)
    if m == 0:
        return y_seq, 0, 0

    ties_x = 0
    ties_xy = 0
    start = 0

    while start < m:
        first_idx = order[start]
        first_x = x_values[first_idx]
        end = start + 1
        while end < m:
            idx = order[end]
            if x_values[idx] != first_x:
                break
            end += 1

        group_size = end - start
        ties_x += (group_size * (group_size - 1)) // 2

        first_y = y_dense_rank[order[start]]
        run_start = start
        prev_y = first_y

        for pos in range(start, end):
            idx = order[pos]
            y_val = y_dense_rank[idx]
            y_seq[pos] = y_val
            if pos == start:
                continue
            if y_val != prev_y:
                run_size = pos - run_start
                ties_xy += (run_size * (run_size - 1)) // 2
                run_start = pos
                prev_y = y_val

        run_size = end - run_start
        ties_xy += (run_size * (run_size - 1)) // 2

        start = end

    return y_seq, ties_x, ties_xy


@njit(cache=True)
def _count_inversions_int32(values: np.ndarray) -> int:
    n = int(values.shape[0])
    if n <= 1:
        return 0

    work = np.empty_like(values)
    src = values
    dst = work
    width = 1
    inversions = 0

    while width < n:
        left = 0
        while left < n:
            mid = left + width
            right = left + 2 * width
            if mid > n:
                mid = n
            if right > n:
                right = n

            i = left
            j = mid
            k = left

            while i < mid and j < right:
                if src[i] <= src[j]:
                    dst[k] = src[i]
                    i += 1
                else:
                    dst[k] = src[j]
                    j += 1
                    inversions += mid - i
                k += 1

            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1

            while j < right:
                dst[k] = src[j]
                j += 1
                k += 1

            left += 2 * width

        tmp = src
        src = dst
        dst = tmp
        width *= 2

    if src is not values:
        values[:] = src[:]

    return inversions


def _pair_count(num_samples: int) -> int:
    if num_samples < 2:
        return 0
    return (num_samples * (num_samples - 1)) // 2


def _row_offsets(num_samples: int) -> np.ndarray:
    idx = np.arange(num_samples, dtype=np.int64)
    return idx * (2 * num_samples - idx - 1) // 2


def _flatten_tokens(values: Sequence[str] | None, *, lowercase: bool = False) -> tuple[str, ...] | None:
    if values is None:
        return None
    items: list[str] = []
    for raw in values:
        for token in str(raw).split(","):
            cleaned = token.strip()
            if not cleaned:
                continue
            items.append(cleaned.lower() if lowercase else cleaned)
    if not items:
        return None
    return tuple(items)


def _safe_json_loads(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _extract_nested(payload: dict[str, Any] | None, path: list[str]) -> Any:
    current: Any = payload
    for token in path:
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _json_dumps(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, indent=indent, default=_json_default, ensure_ascii=True, sort_keys=indent is None)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, _json_dumps(payload, indent=2))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(_json_dumps(payload))
        fp.write("\n")


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def _safe_token(value: str) -> str:
    chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    result = "".join(chars).strip("_")
    return result or "item"


def _task_key(dataset: str, model_key: str, layer_index: int, layer_name: str) -> str:
    return f"{dataset}::{model_key}::{layer_index:03d}::{layer_name}"


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


def _result_config_payload(config: EvaluationConfig) -> dict[str, Any]:
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
        "score_distances": sorted(set(config.score_distances)),
        "corr_metrics": sorted(set(config.corr_metrics)),
        "pair_scope": config.pair_scope,
    }


def _runtime_config_payload(config: EvaluationConfig) -> dict[str, Any]:
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


def _config_fingerprint(config: EvaluationConfig) -> str:
    payload = _result_config_payload(config)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_field_values(df: pd.DataFrame, field: str, *, numeric: bool) -> np.ndarray:
    field_name = field.strip()
    if not field_name:
        raise ValueError("Field name must not be empty")

    direct_candidates = [field_name, f"{field_name}__csv"]
    for candidate in direct_candidates:
        if candidate in df.columns:
            series = df[candidate]
            if numeric:
                return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            return series.astype(str).to_numpy(dtype=object)

    primary = df["metadata"] if "metadata" in df.columns else pd.Series([None] * len(df))
    fallback = df["metadata__csv"] if "metadata__csv" in df.columns else pd.Series([None] * len(df))

    payloads: list[dict[str, Any] | None] = []
    for p, f in zip(primary.tolist(), fallback.tolist(), strict=False):
        payload = _safe_json_loads(p)
        if payload is None:
            payload = _safe_json_loads(f)
        payloads.append(payload)

    if field_name.startswith("metadata."):
        path = [token for token in field_name[len("metadata.") :].split(".") if token]
    else:
        path = [token for token in field_name.split(".") if token]
    if not path:
        raise ValueError(f"Invalid metadata field path: {field!r}")

    values = [_extract_nested(payload, path) for payload in payloads]
    if all(value is None for value in values):
        raise KeyError(
            f"Field {field_name!r} was not found in dataframe columns or metadata JSON payload"
        )

    if numeric:
        return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)

    normalized: list[str] = []
    for idx, value in enumerate(values):
        if value is None:
            normalized.append(f"__missing_row_{idx}")
        else:
            normalized.append(str(value))
    return np.asarray(normalized, dtype=object)


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _metadata_payloads(df: pd.DataFrame) -> list[dict[str, Any] | None]:
    primary = df["metadata"] if "metadata" in df.columns else pd.Series([None] * len(df))
    fallback = df["metadata__csv"] if "metadata__csv" in df.columns else pd.Series([None] * len(df))
    payloads: list[dict[str, Any] | None] = []
    for p_value, f_value in zip(primary.tolist(), fallback.tolist(), strict=False):
        payload = _safe_json_loads(p_value)
        if payload is None:
            payload = _safe_json_loads(f_value)
        payloads.append(payload)
    return payloads


def _coalesce_text_columns(df: pd.DataFrame, candidates: Sequence[str]) -> list[str | None]:
    values: list[str | None] = [None] * len(df)
    for candidate in candidates:
        if candidate not in df.columns:
            continue
        series = df[candidate].tolist()
        for idx, raw in enumerate(series):
            if values[idx] is not None:
                continue
            values[idx] = _normalize_optional_text(raw)
    return values


def _tid2013_reference_filename_map(datasets_root: Path) -> dict[str, str]:
    ref_dir = datasets_root / "TID2013" / "reference_images"
    mapping: dict[str, str] = {}
    if not ref_dir.exists():
        return mapping
    for path in ref_dir.iterdir():
        if path.is_file():
            mapping[path.stem.upper()] = path.name
    return mapping


def resolve_reference_filenames(df: pd.DataFrame, dataset: str, datasets_root: Path) -> np.ndarray:
    values = _coalesce_text_columns(df, ("ref_filename", "ref_filename__csv"))
    payloads = _metadata_payloads(df)

    if any(value is None for value in values):
        for idx, payload in enumerate(payloads):
            if values[idx] is not None:
                continue
            values[idx] = _normalize_optional_text(_extract_nested(payload, ["ref_filename"]))

    path_values = _coalesce_text_columns(df, ("path", "path__csv", "filename", "filename__csv"))

    if dataset == "TID2013":
        ref_map = _tid2013_reference_filename_map(datasets_root)
        for idx, payload in enumerate(payloads):
            if values[idx] is not None:
                continue
            ref_id = _normalize_optional_text(_extract_nested(payload, ["ref_id"]))
            if ref_id is None:
                path_value = path_values[idx]
                if path_value is not None:
                    stem = Path(path_value).stem
                    if "_" in stem:
                        ref_id = stem.split("_", 1)[0]
            if ref_id is None:
                continue
            ref_key = ref_id.upper()
            values[idx] = ref_map.get(ref_key, f"{ref_key}.BMP")
    elif dataset == "PieAPP":
        for idx, path_value in enumerate(path_values):
            if values[idx] is not None or path_value is None:
                continue
            match = re.search(r"/(ref_[^/]+?)(?:/|$)", path_value)
            if match:
                values[idx] = f"{match.group(1)}.png"
                continue
            filename = Path(path_value).name
            if filename.startswith("distort_"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    values[idx] = f"ref_{parts[1]}.png"
    elif dataset == "PIPAL":
        for idx, payload in enumerate(payloads):
            if values[idx] is not None:
                continue
            ref_id = _normalize_optional_text(_extract_nested(payload, ["ref_id"]))
            if ref_id is None:
                path_value = path_values[idx]
                if path_value is not None:
                    filename = Path(path_value).name
                    if "_" in filename:
                        ref_id = filename.split("_", 1)[0]
            if ref_id is not None:
                values[idx] = f"{ref_id}.bmp"
    elif dataset == "kadid10k":
        for idx, path_value in enumerate(path_values):
            if values[idx] is not None or path_value is None:
                continue
            stem = Path(path_value).stem
            if "_" in stem:
                values[idx] = f"{stem.split('_', 1)[0]}.png"
    elif dataset == "CSIQ":
        for idx, path_value in enumerate(path_values):
            if values[idx] is not None or path_value is None:
                continue
            stem = Path(path_value).name.split(".", 1)[0]
            if stem:
                values[idx] = f"{stem}.png"

    missing = [idx for idx, value in enumerate(values) if value is None]
    if missing:
        raise ValueError(
            f"Could not resolve ref_filename for dataset={dataset}; missing rows={missing[:10]}"
        )
    return np.asarray([str(value) for value in values], dtype=object)


def resolve_pair_scope(mode: str, dataset: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "auto":
        return "within_ref" if dataset in FULL_REFERENCE_DATASETS else "global"
    return normalized


def _subset_table_for_row_ids(df: pd.DataFrame, row_ids: np.ndarray) -> pd.DataFrame:
    if row_ids.size == 0:
        return df.iloc[0:0].copy()
    subset = df.set_index("row_id", drop=False).loc[row_ids.tolist()].copy()
    subset = subset.reset_index(drop=True)
    return subset


def resolve_pair_groups(
    table: pd.DataFrame,
    row_ids: np.ndarray,
    *,
    dataset: str,
    datasets_root: Path,
    pair_scope: str,
) -> tuple[str | None, tuple[PairGroup, ...], list[str]]:
    warnings: list[str] = []
    if pair_scope == "global":
        n_samples = int(row_ids.shape[0])
        if n_samples < 2:
            return None, tuple(), warnings
        return (
            None,
            (
                PairGroup(
                    group_key="__global__",
                    positions=tuple(range(n_samples)),
                    n_samples=n_samples,
                    n_pairs=_pair_count(n_samples),
                ),
            ),
            warnings,
        )

    subset = _subset_table_for_row_ids(table, row_ids)
    ref_values = resolve_reference_filenames(subset, dataset=dataset, datasets_root=datasets_root)
    members: dict[str, list[int]] = {}
    group_order: list[str] = []
    for position, ref_filename in enumerate(ref_values.tolist()):
        key = str(ref_filename)
        if key not in members:
            members[key] = []
            group_order.append(key)
        members[key].append(position)

    groups: list[PairGroup] = []
    dropped = 0
    for key in group_order:
        positions = tuple(members[key])
        n_samples = len(positions)
        if n_samples < 2:
            dropped += 1
            continue
        groups.append(
            PairGroup(
                group_key=key,
                positions=positions,
                n_samples=n_samples,
                n_pairs=_pair_count(n_samples),
            )
        )

    if dropped > 0:
        warnings.append(
            f"dataset={dataset}: dropped {dropped} ref groups with fewer than 2 samples after sampling"
        )
    return "ref_filename", tuple(groups), warnings


def _update_grouping_digest(
    digest: "hashlib._Hash",
    *,
    pair_scope: str,
    group_field: str | None,
    groups: Sequence[PairGroup],
) -> None:
    digest.update(pair_scope.encode("utf-8"))
    digest.update(b"\0")
    digest.update((group_field or "").encode("utf-8"))
    digest.update(b"\0")
    for group in groups:
        digest.update(group.group_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(group.positions, dtype=np.int64).tobytes())
        digest.update(b"\0")


def _has_success_files(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in REQUIRED_SUCCESS_FILES)


def discover_datasets(datasets_root: Path, outputs_root: Path, explicit: tuple[str, ...] | None) -> list[str]:
    if explicit is not None:
        missing_csv = [name for name in explicit if not (datasets_root / name / "data.csv").exists()]
        if missing_csv:
            joined = ", ".join(sorted(missing_csv))
            raise FileNotFoundError(f"Missing data.csv for datasets: {joined}")
        return sorted(set(explicit))

    names: list[str] = []
    for path in datasets_root.iterdir():
        if not path.is_dir():
            continue
        if not (path / "data.csv").exists():
            continue
        if (outputs_root / path.name).exists():
            names.append(path.name)
    if names:
        return sorted(set(names))

    fallback = [p.name for p in datasets_root.iterdir() if p.is_dir() and (p / "data.csv").exists()]
    return sorted(set(fallback))


def discover_models(outputs_root: Path, dataset: str, explicit: tuple[str, ...] | None) -> list[str]:
    dataset_dir = outputs_root / dataset
    if explicit is not None:
        missing: list[str] = []
        incomplete: list[str] = []
        for model_key in explicit:
            model_dir = dataset_dir / model_key
            if not model_dir.exists():
                missing.append(model_key)
                continue
            if not _has_success_files(model_dir):
                incomplete.append(model_key)
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(f"Missing model outputs for dataset={dataset}: {joined}")
        if incomplete:
            joined = ", ".join(sorted(incomplete))
            raise FileNotFoundError(
                f"Incomplete model outputs for dataset={dataset} (need meta.json/index.parquet/layers.h5): {joined}"
            )
        return sorted(set(explicit))

    if not dataset_dir.exists():
        return []

    models: list[str] = []
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        if _has_success_files(child):
            models.append(child.name)
    return sorted(models)


def _load_layer_names(meta_path: Path) -> list[str]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    layer_names = payload.get("layer_names")
    if not isinstance(layer_names, list) or not layer_names:
        raise ValueError(f"Invalid or empty layer_names in {meta_path}")
    return [str(name) for name in layer_names]


def _normalize_index(raw: int, layer_count: int) -> int:
    idx = raw
    if idx < 0:
        idx = layer_count + idx
    if idx < 0 or idx >= layer_count:
        raise IndexError(f"Layer index out of range: {raw} for layer_count={layer_count}")
    return idx


def _parse_layer_selector(selector: str, layer_names: list[str]) -> list[int]:
    token = selector.strip()
    if token in {"all", "*"}:
        return list(range(len(layer_names)))

    parts = [part.strip() for part in token.split(",") if part.strip()]
    selected: list[int] = []
    seen: set[int] = set()

    for part in parts:
        if ":" in part:
            raise ValueError(f"Layer selector should not include model prefix: {part!r}")
        if "-" in part and all(piece.strip().lstrip("-").isdigit() for piece in part.split("-", 1)):
            start_text, end_text = part.split("-", 1)
            start = _normalize_index(int(start_text), len(layer_names))
            end = _normalize_index(int(end_text), len(layer_names))
            step = 1 if start <= end else -1
            for idx in range(start, end + step, step):
                if idx not in seen:
                    selected.append(idx)
                    seen.add(idx)
            continue

        if part.lstrip("-").isdigit():
            idx = _normalize_index(int(part), len(layer_names))
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            continue

        if part not in layer_names:
            raise KeyError(f"Unknown layer token {part!r}. Example available layers: {layer_names[:6]}")
        idx = layer_names.index(part)
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)

    return selected


def _resolve_layer_indices(layer_names: list[str], selectors: tuple[str, ...] | None) -> list[int]:
    if selectors is None:
        return list(range(len(layer_names)))

    selected: list[int] = []
    seen: set[int] = set()
    for selector in selectors:
        for idx in _parse_layer_selector(selector, layer_names):
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)

    if not selected:
        raise ValueError("Layer selectors resolved to an empty layer set")
    return selected


def load_dataset_table(
    datasets_root: Path,
    outputs_root: Path,
    dataset: str,
    model_key: str,
) -> pd.DataFrame:
    index_path = outputs_root / dataset / model_key / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index parquet: {index_path}")

    anchor_df = pd.read_parquet(index_path)
    if "row_id" not in anchor_df.columns:
        raise ValueError(f"row_id column missing in {index_path}")
    anchor_df = anchor_df.sort_values("row_id").reset_index(drop=True)
    if not anchor_df["row_id"].is_unique:
        raise ValueError(f"row_id is not unique in {index_path}")

    data_csv_path = datasets_root / dataset / "data.csv"
    if not data_csv_path.exists():
        return anchor_df

    csv_df = pd.read_csv(data_csv_path)
    merge_keys: list[str] = []
    if "path" in anchor_df.columns and "path" in csv_df.columns:
        if not csv_df.duplicated(subset=["path"], keep=False).any():
            merge_keys = ["path"]
    elif "filename" in anchor_df.columns and "filename" in csv_df.columns:
        merge_keys = ["filename"]

    if not merge_keys and "filename" in anchor_df.columns and "filename" in csv_df.columns:
        merge_keys = ["filename"]
    if not merge_keys:
        return anchor_df

    if csv_df.duplicated(subset=merge_keys, keep=False).any():
        raise ValueError(f"Duplicate merge keys {merge_keys} in dataset CSV: {data_csv_path}")

    rename_map = {
        col: f"{col}__csv"
        for col in csv_df.columns
        if col not in merge_keys and col in anchor_df.columns
    }
    csv_df = csv_df.rename(columns=rename_map)
    merged = anchor_df.merge(csv_df, on=merge_keys, how="left", sort=False, validate="m:1")
    return merged


def _validate_row_id_alignment(outputs_root: Path, dataset: str, model_keys: Sequence[str]) -> None:
    if not model_keys:
        return

    anchor_key = model_keys[0]
    anchor_index = outputs_root / dataset / anchor_key / "index.parquet"
    anchor_row_ids = pd.read_parquet(anchor_index, columns=["row_id"])["row_id"].to_numpy(dtype=np.int64)

    for model_key in model_keys[1:]:
        index_path = outputs_root / dataset / model_key / "index.parquet"
        row_ids = pd.read_parquet(index_path, columns=["row_id"])["row_id"].to_numpy(dtype=np.int64)
        if row_ids.shape[0] != anchor_row_ids.shape[0]:
            raise ValueError(
                f"row count mismatch for dataset={dataset}: {model_key} has {row_ids.shape[0]} rows, "
                f"anchor {anchor_key} has {anchor_row_ids.shape[0]}"
            )
        if not np.array_equal(row_ids, anchor_row_ids):
            raise ValueError(f"row_id alignment mismatch for dataset={dataset}: anchor={anchor_key} model={model_key}")


def _select_rows(df: pd.DataFrame, scores: np.ndarray, sample_limit: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(df) != scores.shape[0]:
        raise ValueError("DataFrame and score vector length mismatch")
    if sample_limit is None or sample_limit <= 0 or sample_limit >= len(df):
        picked = np.arange(len(df), dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        picked = np.sort(rng.choice(len(df), size=sample_limit, replace=False).astype(np.int64))

    subset = df.iloc[picked].copy().reset_index(drop=True)
    row_ids = subset["row_id"].to_numpy(dtype=np.int64)
    subset_scores = scores[picked].astype(np.float64, copy=False)

    order = np.argsort(row_ids)
    row_ids = row_ids[order]
    subset_scores = subset_scores[order]
    return row_ids, subset_scores


def _read_rows(dset: h5py.Dataset, row_ids: np.ndarray) -> np.ndarray:
    if row_ids.size == 0:
        return np.empty((0, int(dset.shape[1])), dtype=dset.dtype)

    order = np.argsort(row_ids)
    sorted_ids = row_ids[order]
    rows = np.asarray(dset[sorted_ids])

    if not np.all(order == np.arange(order.size)):
        restore = np.empty_like(order)
        restore[order] = np.arange(order.size)
        rows = rows[restore]
    return rows


def _pearson_from_sums(
    n: int,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float:
    if n < 2:
        return float("nan")
    num = (n * sum_xy) - (sum_x * sum_y)
    den_x = (n * sum_x2) - (sum_x * sum_x)
    den_y = (n * sum_y2) - (sum_y * sum_y)
    if den_x <= 0.0 or den_y <= 0.0:
        return float("nan")
    den = math.sqrt(den_x * den_y)
    if den == 0.0:
        return float("nan")
    return float(num / den)


def _pearson_from_vectors(
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_sum: float,
    y_sum_sq: float,
    *,
    chunk_size: int = 2_000_000,
) -> float:
    n = int(x_values.shape[0])
    if n != int(y_values.shape[0]):
        raise ValueError("x/y length mismatch for PCC")

    sum_x = 0.0
    sum_x2 = 0.0
    sum_xy = 0.0
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        x_chunk = np.asarray(x_values[start:end], dtype=np.float64)
        y_chunk = np.asarray(y_values[start:end], dtype=np.float64)
        sum_x += float(x_chunk.sum(dtype=np.float64))
        sum_x2 += float(np.dot(x_chunk, x_chunk))
        sum_xy += float(np.dot(x_chunk, y_chunk))

    return _pearson_from_sums(
        n=n,
        sum_x=sum_x,
        sum_y=y_sum,
        sum_x2=sum_x2,
        sum_y2=y_sum_sq,
        sum_xy=sum_xy,
    )


def _build_score_vector(scores: np.ndarray, metric: str, out_path: Path) -> ScoreVectorInfo:
    pair_total = _pair_count(int(scores.shape[0]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(pair_total,))
    if metric == "abs":
        sum_value, sum_sq_value = _fill_score_pairs_abs(scores, arr)
    elif metric == "sq":
        sum_value, sum_sq_value = _fill_score_pairs_sq(scores, arr)
    else:
        raise ValueError(f"Unsupported score distance: {metric}")
    arr.flush()
    del arr

    return ScoreVectorInfo(
        metric=metric,
        path=str(out_path),
        sum_value=float(sum_value),
        sum_sq_value=float(sum_sq_value),
    )


def _build_rank_cache(
    values_path: Path,
    pair_total: int,
    out_dir: Path,
    *,
    need_avg: bool,
    need_dense: bool,
) -> tuple[RankCacheInfo, tuple[str, ...]]:
    values = np.memmap(values_path, dtype=np.float32, mode="r", shape=(pair_total,))
    order = np.argsort(values, kind="mergesort")

    cleanup_paths: list[str] = []
    avg_rank_path: Path | None = None
    dense_rank_path: Path | None = None
    tie_pairs = 0
    unique_count = 0
    rank_sum_sq: float | None = None

    if need_avg and need_dense:
        avg_rank_path = out_dir / "score_rank_avg.f64.bin"
        dense_rank_path = out_dir / "score_rank_dense.i32.bin"
        avg_rank = np.memmap(avg_rank_path, dtype=np.float64, mode="w+", shape=(pair_total,))
        dense_rank = np.memmap(dense_rank_path, dtype=np.int32, mode="w+", shape=(pair_total,))
        tie_pairs, unique_count, rank_sum_sq_value = _assign_avg_dense_ranks(values, order, avg_rank, dense_rank)
        avg_rank.flush()
        dense_rank.flush()
        del avg_rank
        del dense_rank
        rank_sum_sq = float(rank_sum_sq_value)
        cleanup_paths.extend([str(avg_rank_path), str(dense_rank_path)])
    elif need_avg:
        avg_rank_path = out_dir / "score_rank_avg.f64.bin"
        avg_rank = np.memmap(avg_rank_path, dtype=np.float64, mode="w+", shape=(pair_total,))
        rank_sum_sq_value = _assign_avg_ranks(values, order, avg_rank)
        avg_rank.flush()
        del avg_rank
        rank_sum_sq = float(rank_sum_sq_value)
        cleanup_paths.append(str(avg_rank_path))
    elif need_dense:
        dense_rank_path = out_dir / "score_rank_dense.i32.bin"
        dense_rank = np.memmap(dense_rank_path, dtype=np.int32, mode="w+", shape=(pair_total,))
        tie_pairs, unique_count = _assign_dense_ranks(values, order, dense_rank)
        dense_rank.flush()
        del dense_rank
        cleanup_paths.append(str(dense_rank_path))

    del order
    del values

    rank_sum = float(pair_total * (pair_total + 1) / 2.0)
    info = RankCacheInfo(
        source_metric="abs",
        avg_rank_path=str(avg_rank_path) if avg_rank_path is not None else None,
        dense_rank_path=str(dense_rank_path) if dense_rank_path is not None else None,
        tie_pairs=int(tie_pairs),
        unique_count=int(unique_count),
        rank_sum=rank_sum,
        rank_sum_sq=rank_sum_sq,
    )
    return info, tuple(cleanup_paths)


def _write_block_to_vector(
    out_vec: np.ndarray,
    dist_block: np.ndarray,
    row_offsets: np.ndarray,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
) -> None:
    bi = i1 - i0
    bj = j1 - j0
    if i0 == j0:
        if bi <= 1:
            return
        for local_i in range(bi - 1):
            global_i = i0 + local_i
            values = dist_block[local_i, local_i + 1 : bj]
            if values.size == 0:
                continue
            start = int(row_offsets[global_i])
            stop = start + int(values.size)
            out_vec[start:stop] = values
        return

    for local_i in range(bi):
        global_i = i0 + local_i
        start = int(row_offsets[global_i] + (j0 - global_i - 1))
        stop = start + bj
        out_vec[start:stop] = dist_block[local_i]


def _build_embedding_distance_vectors(
    embeddings: np.ndarray,
    metrics: tuple[str, ...],
    row_offsets: np.ndarray,
    block_size: int,
    out_dir: Path,
    prefix: str,
) -> tuple[dict[str, str], tuple[str, ...]]:
    num_samples = int(embeddings.shape[0])
    pair_total = _pair_count(num_samples)

    out_dir.mkdir(parents=True, exist_ok=True)
    vector_paths: dict[str, str] = {}
    vectors: dict[str, np.memmap] = {}
    cleanup_paths: list[str] = []
    for metric in metrics:
        path = out_dir / f"{prefix}_{metric}.f32.bin"
        vector_paths[metric] = str(path)
        vectors[metric] = np.memmap(path, dtype=np.float32, mode="w+", shape=(pair_total,))
        cleanup_paths.append(str(path))

    need_cos = "cos" in metrics
    need_l2 = "l2" in metrics
    need_l1 = "l1" in metrics

    sq_norms: np.ndarray | None = None
    norms: np.ndarray | None = None
    if need_cos or need_l2:
        sq_norms = np.sum(embeddings * embeddings, axis=1, dtype=np.float32)
    if need_cos:
        norms = np.sqrt(np.clip(sq_norms, 1e-12, None))

    for i0 in range(0, num_samples, block_size):
        i1 = min(num_samples, i0 + block_size)
        block_a = embeddings[i0:i1]

        for j0 in range(i0, num_samples, block_size):
            j1 = min(num_samples, j0 + block_size)
            block_b = embeddings[j0:j1]

            gram: np.ndarray | None = None
            if need_cos or need_l2:
                gram = np.asarray(block_a @ block_b.T, dtype=np.float32)

            if need_l2 and gram is not None and sq_norms is not None:
                dist_l2 = (
                    sq_norms[i0:i1, None]
                    + sq_norms[j0:j1][None, :]
                    - (2.0 * gram)
                )
                np.maximum(dist_l2, 0.0, out=dist_l2)
                np.sqrt(dist_l2, out=dist_l2)
                _write_block_to_vector(vectors["l2"], dist_l2, row_offsets, i0, i1, j0, j1)

            if need_cos and gram is not None and norms is not None:
                denom = norms[i0:i1, None] * norms[j0:j1][None, :]
                denom = np.clip(denom, 1e-12, None)
                dist_cos = 1.0 - (gram / denom)
                _write_block_to_vector(vectors["cos"], dist_cos, row_offsets, i0, i1, j0, j1)

            if need_l1:
                dist_l1 = cdist(block_a, block_b, metric="cityblock")
                _write_block_to_vector(vectors["l1"], dist_l1, row_offsets, i0, i1, j0, j1)

    for vec in vectors.values():
        vec.flush()
    for vec in vectors.values():
        del vec

    return vector_paths, tuple(cleanup_paths)


def _build_dataset_score_cache(
    scores: np.ndarray,
    score_distances: tuple[str, ...],
    corr_metrics: tuple[str, ...],
    cache_dir: Path,
) -> DatasetScoreCache:
    pair_total = _pair_count(int(scores.shape[0]))
    score_vectors: dict[str, ScoreVectorInfo] = {}
    rank_cache: RankCacheInfo | None = None

    need_pcc = "pcc" in corr_metrics
    need_scc = "scc" in corr_metrics
    need_kcc = "kcc" in corr_metrics

    need_abs_for_pcc = need_pcc and ("abs" in score_distances)
    need_abs_for_rank = need_scc or need_kcc
    need_abs_vector = need_abs_for_pcc or need_abs_for_rank
    need_sq_vector = need_pcc and ("sq" in score_distances)

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_meta.json"
    meta_payload: dict[str, Any] = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta_payload = loaded
        except Exception:
            meta_payload = {}

    vectors_payload = meta_payload.get("score_vectors")
    if not isinstance(vectors_payload, dict):
        vectors_payload = {}

    if need_abs_vector:
        abs_path = cache_dir / "score_abs.f32.bin"
        abs_meta = vectors_payload.get("abs") if isinstance(vectors_payload.get("abs"), dict) else None
        if abs_meta is not None and abs_path.exists():
            score_vectors["abs"] = ScoreVectorInfo(
                metric="abs",
                path=str(abs_path),
                sum_value=float(abs_meta["sum_value"]),
                sum_sq_value=float(abs_meta["sum_sq_value"]),
            )
        else:
            score_vectors["abs"] = _build_score_vector(scores, "abs", abs_path)

    if need_sq_vector:
        sq_path = cache_dir / "score_sq.f32.bin"
        sq_meta = vectors_payload.get("sq") if isinstance(vectors_payload.get("sq"), dict) else None
        if sq_meta is not None and sq_path.exists():
            score_vectors["sq"] = ScoreVectorInfo(
                metric="sq",
                path=str(sq_path),
                sum_value=float(sq_meta["sum_value"]),
                sum_sq_value=float(sq_meta["sum_sq_value"]),
            )
        else:
            score_vectors["sq"] = _build_score_vector(scores, "sq", sq_path)

    if need_scc or need_kcc:
        abs_info = score_vectors.get("abs")
        if abs_info is None:
            raise RuntimeError("Internal error: abs score vector is required for SCC/KCC rank cache")

        rank_meta = meta_payload.get("rank_cache")
        rank_meta_dict: dict[str, Any] = rank_meta if isinstance(rank_meta, dict) else {}
        avg_path = cache_dir / "score_rank_avg.f64.bin"
        dense_path = cache_dir / "score_rank_dense.i32.bin"

        can_reuse = bool(rank_meta_dict)
        if need_scc and not avg_path.exists():
            can_reuse = False
        if need_kcc and not dense_path.exists():
            can_reuse = False

        if can_reuse:
            rank_cache = RankCacheInfo(
                source_metric=str(rank_meta_dict.get("source_metric", "abs")),
                avg_rank_path=str(avg_path) if avg_path.exists() else None,
                dense_rank_path=str(dense_path) if dense_path.exists() else None,
                tie_pairs=int(rank_meta_dict.get("tie_pairs", 0)),
                unique_count=int(rank_meta_dict.get("unique_count", 0)),
                rank_sum=float(rank_meta_dict.get("rank_sum", 0.0)),
                rank_sum_sq=(
                    float(rank_meta_dict["rank_sum_sq"])
                    if rank_meta_dict.get("rank_sum_sq") is not None
                    else None
                ),
            )
        else:
            rank_cache, _ = _build_rank_cache(
                values_path=Path(abs_info.path),
                pair_total=pair_total,
                out_dir=cache_dir,
                need_avg=need_scc,
                need_dense=need_kcc,
            )

    meta_payload = {
        "schema_version": 1,
        "n_samples": int(scores.shape[0]),
        "n_pairs": int(pair_total),
        "score_vectors": {
            metric: {
                "metric": info.metric,
                "path": info.path,
                "sum_value": info.sum_value,
                "sum_sq_value": info.sum_sq_value,
            }
            for metric, info in score_vectors.items()
        },
        "rank_cache": asdict(rank_cache) if rank_cache is not None else None,
        "updated_at": _timestamp_now(),
    }
    _atomic_write_json(meta_path, meta_payload)

    return DatasetScoreCache(score_vectors=score_vectors, rank_cache=rank_cache, cleanup_paths=tuple())


def _build_pair_group_caches(
    scores: np.ndarray,
    pair_groups: Sequence[PairGroup],
    *,
    score_distances: tuple[str, ...],
    corr_metrics: tuple[str, ...],
    cache_dir: Path,
) -> tuple[PairGroupCache, ...]:
    caches: list[PairGroupCache] = []
    for index, group in enumerate(pair_groups):
        group_positions = np.asarray(group.positions, dtype=np.int64)
        group_scores = scores[group_positions].astype(np.float64, copy=False)
        group_cache_dir = cache_dir / f"group_{index:04d}_{_safe_token(group.group_key)}"
        cache = _build_dataset_score_cache(
            scores=group_scores,
            score_distances=score_distances,
            corr_metrics=corr_metrics,
            cache_dir=group_cache_dir,
        )
        caches.append(
            PairGroupCache(
                group_key=group.group_key,
                positions=group.positions,
                n_samples=group.n_samples,
                n_pairs=group.n_pairs,
                score_vectors=cache.score_vectors,
                rank_cache=cache.rank_cache,
            )
        )
    return tuple(caches)


def _limit_worker_threads() -> None:
    for env_var in THREAD_ENV_VARS:
        os.environ.setdefault(env_var, "1")


def _resolve_jobs(jobs: int | None, task_count: int, pair_total: int, corr_metrics: tuple[str, ...]) -> int:
    if task_count <= 1:
        return 1

    if jobs is not None:
        return max(1, min(task_count, jobs))

    cpu_count = os.cpu_count() or 1
    upper = min(task_count, cpu_count)

    available = int(psutil.virtual_memory().available * 0.8)
    # x vector (float32) + sort index (int64) + optional temp int32 buffer(s)
    per_worker = max(1, pair_total) * (4 + 8)
    if "kcc" in corr_metrics:
        per_worker += max(1, pair_total) * 4
    if "scc" in corr_metrics:
        per_worker += max(1, pair_total) * 8
    by_memory = max(1, available // max(1, per_worker))

    return max(1, min(upper, by_memory))


def _evaluate_layer_task(task: LayerTask) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _timestamp_now()
    row_ids = np.asarray(task.row_ids, dtype=np.int64)
    layer_tmp = Path(task.tmp_dir) / f"{task.model_key}_{task.layer_index:03d}_{os.getpid()}_{time.time_ns()}"
    layer_tmp.mkdir(parents=True, exist_ok=True)
    task_log_path = Path(task.task_log_path)
    task_log_path.parent.mkdir(parents=True, exist_ok=True)

    phase_timings: dict[str, float] = {
        "read_embeddings_sec": 0.0,
        "build_x_sec": 0.0,
        "pcc_sec": 0.0,
        "sort_sec": 0.0,
        "scc_sec": 0.0,
        "kcc_sec": 0.0,
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

        metric_sum: dict[tuple[str, str, str], float] = {}
        metric_count: dict[tuple[str, str, str], int] = {}
        metric_pairs: dict[tuple[str, str, str], int] = {}
        metric_elapsed: dict[tuple[str, str, str], float] = {}

        def _accumulate(
            embedding_distance: str,
            score_distance: str,
            corr_metric: str,
            value: float,
            elapsed_sec: float,
            n_pairs: int,
        ) -> None:
            if math.isnan(value):
                return
            key = (embedding_distance, score_distance, corr_metric)
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
                metrics=task.embedding_distances,
                row_offsets=group_row_offsets,
                block_size=task.block_size,
                out_dir=group_tmp,
                prefix=f"layer_{task.layer_index:03d}_group_{group_index:04d}",
            )
            phase_timings["build_x_sec"] += float(time.perf_counter() - phase_started)

            score_arrays: dict[str, np.memmap] = {}
            for metric, info in group.score_vectors.items():
                score_arrays[metric] = np.memmap(info.path, dtype=np.float32, mode="r", shape=(group.n_pairs,))

            y_avg: np.memmap | None = None
            y_dense: np.memmap | None = None
            if group.rank_cache is not None:
                if group.rank_cache.avg_rank_path is not None:
                    y_avg = np.memmap(group.rank_cache.avg_rank_path, dtype=np.float64, mode="r", shape=(group.n_pairs,))
                if group.rank_cache.dense_rank_path is not None:
                    y_dense = np.memmap(group.rank_cache.dense_rank_path, dtype=np.int32, mode="r", shape=(group.n_pairs,))

            try:
                for emb_metric in task.embedding_distances:
                    x_values = np.memmap(vector_paths[emb_metric], dtype=np.float32, mode="r", shape=(group.n_pairs,))
                    order: np.ndarray | None = None
                    try:
                        if "pcc" in task.corr_metrics:
                            for score_metric in task.score_distances:
                                if score_metric not in group.score_vectors:
                                    continue
                                score_info = group.score_vectors[score_metric]
                                corr_started = time.perf_counter()
                                corr_value = _pearson_from_vectors(
                                    x_values=x_values,
                                    y_values=score_arrays[score_metric],
                                    y_sum=score_info.sum_value,
                                    y_sum_sq=score_info.sum_sq_value,
                                )
                                elapsed = float(time.perf_counter() - corr_started)
                                phase_timings["pcc_sec"] += elapsed
                                _accumulate(emb_metric, score_metric, "pcc", float(corr_value), elapsed, group.n_pairs)

                        need_scc = "scc" in task.corr_metrics
                        need_kcc = "kcc" in task.corr_metrics
                        if need_scc or need_kcc:
                            sort_started = time.perf_counter()
                            if need_kcc:
                                if y_dense is None:
                                    raise RuntimeError("KCC requested but dense score ranks are missing")
                                order = np.lexsort((y_dense, x_values))
                            else:
                                order = np.argsort(x_values, kind="mergesort")
                            phase_timings["sort_sec"] += float(time.perf_counter() - sort_started)

                        if need_scc:
                            if y_avg is None:
                                raise RuntimeError("SCC requested but average score ranks are missing")
                            if group.rank_cache is None or group.rank_cache.rank_sum_sq is None:
                                raise RuntimeError("SCC requested but rank cache metadata is incomplete")
                            corr_started = time.perf_counter()
                            sum_x, sum_x2, sum_xy = _scan_groups_scc(order, x_values, y_avg)
                            corr_value = _pearson_from_sums(
                                n=group.n_pairs,
                                sum_x=float(sum_x),
                                sum_y=float(group.rank_cache.rank_sum),
                                sum_x2=float(sum_x2),
                                sum_y2=float(group.rank_cache.rank_sum_sq),
                                sum_xy=float(sum_xy),
                            )
                            elapsed = float(time.perf_counter() - corr_started)
                            phase_timings["scc_sec"] += elapsed
                            for score_metric in task.score_distances:
                                _accumulate(emb_metric, score_metric, "scc", float(corr_value), elapsed, group.n_pairs)

                        if need_kcc:
                            if y_dense is None:
                                raise RuntimeError("KCC requested but dense score ranks are missing")
                            if group.rank_cache is None:
                                raise RuntimeError("KCC requested but rank cache metadata is missing")
                            corr_started = time.perf_counter()
                            y_seq, ties_x, ties_xy = _build_y_sequence_and_ties(order, x_values, y_dense)
                            discordant = _count_inversions_int32(y_seq)
                            del y_seq
                            n0 = (group.n_pairs * (group.n_pairs - 1)) // 2
                            ties_y = int(group.rank_cache.tie_pairs)
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
                            for score_metric in task.score_distances:
                                _accumulate(emb_metric, score_metric, "kcc", float(corr_value), elapsed, group.n_pairs)
                    finally:
                        del x_values
                        if order is not None:
                            del order
            finally:
                for arr in score_arrays.values():
                    del arr
                if y_avg is not None:
                    del y_avg
                if y_dense is not None:
                    del y_dense
                if not task.keep_cache:
                    _cleanup_paths(vector_cleanup)
                    shutil.rmtree(group_tmp, ignore_errors=True)

        per_metric_rows: list[dict[str, Any]] = []
        n_groups_total = len(task.pair_groups)
        for emb_metric in task.embedding_distances:
            for score_metric in task.score_distances:
                for corr_metric in task.corr_metrics:
                    key = (emb_metric, score_metric, corr_metric)
                    used_groups = int(metric_count.get(key, 0))
                    value = float("nan") if used_groups == 0 else float(metric_sum[key] / used_groups)
                    per_metric_rows.append(
                        {
                            "dataset": task.dataset,
                            "model_key": task.model_key,
                            "layer_index": task.layer_index,
                            "layer_name": task.layer_name,
                            "layer_dim": task.layer_dim,
                            "embedding_distance": emb_metric,
                            "score_distance": score_metric,
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
    finally:
        if not task.keep_cache and layer_tmp.exists() and not any(layer_tmp.iterdir()):
            layer_tmp.rmdir()


def _resolve_layer_specs(
    outputs_root: Path,
    dataset: str,
    model_keys: Sequence[str],
    selectors: tuple[str, ...] | None,
) -> list[LayerSpec]:
    specs: list[LayerSpec] = []
    for model_key in model_keys:
        model_dir = outputs_root / dataset / model_key
        meta_path = model_dir / "meta.json"
        h5_path = model_dir / "layers.h5"
        if not meta_path.exists() or not h5_path.exists():
            raise FileNotFoundError(f"Missing meta/layers artifacts for {dataset}/{model_key}")

        layer_names = _load_layer_names(meta_path)
        selected = _resolve_layer_indices(layer_names, selectors)

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


def _cleanup_paths(paths: Sequence[str]) -> None:
    for path_text in paths:
        Path(path_text).unlink(missing_ok=True)


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
        current = best.get(key)
        value = row.get("value")
        if value is None:
            continue
        value_f = float(value)
        if math.isnan(value_f):
            continue
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
            item["model_key"],
            item["corr_metric"],
            item["embedding_distance"],
            item["score_distance"],
            -item["value"],
        ),
    )


def _prepare_layer_tasks(
    layer_specs: Sequence[LayerSpec],
    row_ids: Sequence[int],
    n_samples: int,
    n_pairs: int,
    pair_scope: str,
    group_field: str | None,
    config: EvaluationConfig,
    pair_groups: Sequence[PairGroupCache],
    dataset_tmp_dir: Path,
    task_logs_root: Path,
) -> list[LayerTask]:
    tasks: list[LayerTask] = []
    row_tuple = tuple(int(v) for v in row_ids)
    for spec in layer_specs:
        task_key = _task_key(spec.dataset, spec.model_key, spec.layer_index, spec.layer_name)
        task_log_dir = task_logs_root / _safe_token(spec.dataset) / _safe_token(spec.model_key)
        task_log_path = task_log_dir / f"layer_{spec.layer_index:03d}_{_safe_token(spec.layer_name)}.log"
        task = LayerTask(
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
            pair_scope=pair_scope,
            group_field=group_field,
            embedding_distances=config.embedding_distances,
            score_distances=config.score_distances,
            corr_metrics=config.corr_metrics,
            block_size=config.block_size,
            tmp_dir=str(dataset_tmp_dir),
            keep_cache=config.keep_cache,
            task_log_path=str(task_log_path),
            pair_groups=tuple(pair_groups),
        )
        tasks.append(task)
    return tasks


def _build_dataset_plan(config: EvaluationConfig, dataset: str) -> DatasetPlan:
    warnings: list[str] = []
    resolved_pair_scope = resolve_pair_scope(config.pair_scope, dataset)
    models = discover_models(config.outputs_root, dataset, config.models)
    if not models:
        warnings.append(f"dataset={dataset}: no completed model outputs found")
        return DatasetPlan(
            dataset=dataset,
            models=tuple(),
            row_ids=tuple(),
            scores_subset=np.empty((0,), dtype=np.float64),
            layer_specs=tuple(),
            n_samples=0,
            n_pairs=0,
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
        return DatasetPlan(
            dataset=dataset,
            models=tuple(models),
            row_ids=tuple(int(v) for v in row_ids.tolist()),
            scores_subset=scores_subset.astype(np.float64, copy=False),
            layer_specs=tuple(),
            n_samples=n_samples,
            n_pairs=_pair_count(n_samples),
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
    if not pair_groups:
        warnings.append(f"dataset={dataset}: no runnable pair groups for pair_scope={resolved_pair_scope}")
        return DatasetPlan(
            dataset=dataset,
            models=tuple(models),
            row_ids=tuple(int(v) for v in row_ids.tolist()),
            scores_subset=scores_subset.astype(np.float64, copy=False),
            layer_specs=tuple(),
            n_samples=n_samples,
            n_pairs=0,
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

    return DatasetPlan(
        dataset=dataset,
        models=tuple(models),
        row_ids=tuple(int(v) for v in row_ids.tolist()),
        scores_subset=scores_subset.astype(np.float64, copy=False),
        layer_specs=tuple(layer_specs),
        n_samples=n_samples,
        n_pairs=n_pairs,
        pair_scope=resolved_pair_scope,
        group_field=group_field,
        pair_groups=pair_groups,
        warnings=tuple(warnings),
    )


def _load_task_entries(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path) or [])


def _load_completed_task_keys(path: Path) -> set[str]:
    completed: set[str] = set()
    for entry in _iter_jsonl(path) or []:
        if str(entry.get("status")) == "completed":
            task_key = entry.get("task_key")
            if isinstance(task_key, str) and task_key:
                completed.add(task_key)
    return completed


def _flatten_result_entries(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if str(entry.get("status")) != "completed":
            continue
        task_rows = entry.get("rows")
        if isinstance(task_rows, list):
            rows.extend(task_rows)
    return rows


def _append_rows_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(TABLE_COLUMNS), delimiter="\t")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in TABLE_COLUMNS})


def _build_run_artifacts(run_dir: Path) -> RunArtifacts:
    return RunArtifacts(
        run_dir=run_dir,
        config_path=run_dir / "run_config.json",
        lock_path=run_dir / ".lock",
        results_jsonl_path=run_dir / "results.jsonl",
        errors_jsonl_path=run_dir / "errors.jsonl",
        progress_json_path=run_dir / "progress.json",
        report_json_path=run_dir / "report.json",
        table_tsv_path=run_dir / "results.tsv",
        run_log_path=run_dir / "run.log",
        cache_root=run_dir / "cache",
        task_logs_root=run_dir / "task_logs",
    )


def _acquire_run_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        stale = False
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = payload.get("pid")
            if isinstance(pid, int) and not psutil.pid_exists(pid):
                stale = True
        except Exception:
            stale = True
        if stale:
            lock_path.unlink(missing_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f"Run directory is locked by another process: {lock_path}") from exc

    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        fp.write(_json_dumps({"pid": os.getpid(), "started_at": _timestamp_now()}, indent=2))


def _release_run_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


class RunLogger:
    def __init__(self, artifacts: RunArtifacts, *, mode: str, heartbeat_sec: int, total_tasks: int, initial: int) -> None:
        resolved_mode = mode
        if resolved_mode == "auto":
            resolved_mode = "bar" if (tqdm is not None and sys.stdout.isatty()) else "log"
        if resolved_mode == "bar" and tqdm is None:
            resolved_mode = "log"
        if resolved_mode not in {"bar", "log", "off"}:
            raise ValueError(f"Unsupported progress mode: {mode}")

        self.artifacts = artifacts
        self.mode = resolved_mode
        self.heartbeat_sec = max(1, heartbeat_sec)
        self._last_heartbeat = time.monotonic()
        self._bar = None
        if self.mode == "bar" and tqdm is not None:
            self._bar = tqdm(total=total_tasks, initial=initial, desc="embedding-quality", unit="task")

    def log(self, message: str) -> None:
        line = f"[{_timestamp_now()}] {message}"
        with self.artifacts.run_log_path.open("a", encoding="utf-8") as fp:
            fp.write(line)
            fp.write("\n")

        if self.mode == "off":
            return
        if self._bar is not None:
            self._bar.write(line)
            return
        print(line, flush=True)

    def refresh(self, completed: int, total: int, *, dataset: str | None, dataset_completed: int | None, dataset_total: int | None) -> None:
        if self._bar is None:
            return
        delta = completed - int(self._bar.n)
        if delta > 0:
            self._bar.update(delta)
        parts: list[str] = []
        if dataset:
            parts.append(f"dataset={dataset}")
        if dataset_completed is not None and dataset_total is not None:
            parts.append(f"dataset_tasks={dataset_completed}/{dataset_total}")
        if parts:
            self._bar.set_postfix_str(" ".join(parts))

    def maybe_heartbeat(self, state: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat < self.heartbeat_sec:
            return
        self._last_heartbeat = now
        dataset = state.get("current_dataset")
        dataset_text = "-"
        if isinstance(dataset, str) and dataset:
            ds_state = state["datasets"].get(dataset, {})
            dataset_text = (
                f"{dataset} {ds_state.get('completed_tasks', 0)}/{ds_state.get('total_tasks', 0)}"
            )
        self.log(
            "heartbeat "
            f"completed={state['completed_tasks']}/{state['total_tasks']} "
            f"failed={state['failed_tasks']} "
            f"skipped={state['skipped_tasks']} "
            f"current={dataset_text}"
        )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _initial_progress_state(
    config: EvaluationConfig,
    artifacts: RunArtifacts,
    run_config_payload: dict[str, Any],
    plans: Sequence[DatasetPlan],
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


def _initialize_run_config(config: EvaluationConfig, artifacts: RunArtifacts) -> dict[str, Any]:
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


def _prepare_existing_outputs_for_resume(artifacts: RunArtifacts) -> set[str]:
    completed_keys = _load_completed_task_keys(artifacts.results_jsonl_path)
    if artifacts.results_jsonl_path.exists() and not artifacts.table_tsv_path.exists():
        rows = _flatten_result_entries(_load_task_entries(artifacts.results_jsonl_path))
        if rows:
            write_results_table(rows, artifacts.table_tsv_path)
    return completed_keys


def _mark_task_running(state: dict[str, Any], task: LayerTask) -> None:
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
    config: EvaluationConfig,
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


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
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

        plans: list[DatasetPlan] = []
        planning_errors: list[dict[str, Any]] = []
        for dataset in datasets:
            try:
                plans.append(_build_dataset_plan(config, dataset))
            except Exception as exc:
                error_entry = {
                    "status": "error",
                    "task_key": f"{dataset}::__dataset_plan__",
                    "dataset": dataset,
                    "model_key": "__dataset_plan__",
                    "layer_index": -1,
                    "layer_name": "__dataset_plan__",
                    "started_at": _timestamp_now(),
                    "finished_at": _timestamp_now(),
                    "elapsed_sec": 0.0,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "task_log_path": None,
                }
                planning_errors.append(error_entry)
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
                f"dataset start {dataset} models={len(plan.models)} layers={len(plan.layer_specs)} "
                f"samples={plan.n_samples} pairs={plan.n_pairs}"
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

            jobs = 0 if not pending_specs else _resolve_jobs(config.jobs, len(pending_specs), plan.n_pairs, config.corr_metrics)
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
            cache_dir = artifacts.cache_root / _safe_token(plan.dataset) / cache_hash
            ds_state["cache_dir"] = str(cache_dir)
            logger.log(f"dataset cache {dataset}: build/reuse -> {cache_dir}")
            pair_group_caches = _build_pair_group_caches(
                scores=plan.scores_subset,
                pair_groups=plan.pair_groups,
                score_distances=config.score_distances,
                corr_metrics=config.corr_metrics,
                cache_dir=cache_dir,
            )

            ds_state["status"] = "running"
            tasks = _prepare_layer_tasks(
                layer_specs=pending_specs,
                row_ids=plan.row_ids,
                n_samples=plan.n_samples,
                n_pairs=plan.n_pairs,
                pair_scope=plan.pair_scope,
                group_field=plan.group_field,
                config=config,
                pair_groups=pair_group_caches,
                dataset_tmp_dir=config.tmp_dir / _safe_token(dataset),
                task_logs_root=artifacts.task_logs_root,
            )

            logger.log(f"dataset execute {dataset}: pending_tasks={len(tasks)} jobs={jobs}")
            _write_progress_snapshot(artifacts, state)

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


def write_results_table(rows: Sequence[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    df = pd.DataFrame(row_list)
    if row_list and not df.empty:
        preferred_columns: list[str] = []
        row_keys = set().union(*(row.keys() for row in row_list))
        if {"model_key", "embedding_distance", "score_distance"}.issubset(row_keys):
            preferred_columns = [column for column in TABLE_COLUMNS if column in df.columns]
        elif {"reference_model_key", "candidate_model_key", "distance_metric"}.issubset(row_keys):
            from quality_backbones.alignment import ALIGNMENT_TABLE_COLUMNS

            preferred_columns = [column for column in ALIGNMENT_TABLE_COLUMNS if column in df.columns]
        if preferred_columns:
            remaining = [column for column in df.columns if column not in preferred_columns]
            df = df[preferred_columns + remaining]
    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out_path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(out_path, index=False)
        return
    # default for unknown suffix and .tsv
    df.to_csv(out_path, index=False, sep="\t")


def _validate_config(config: EvaluationConfig) -> None:
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

    bad_score = sorted(set(config.score_distances).difference(SUPPORTED_SCORE_DISTANCES))
    if bad_score:
        raise ValueError(f"Unsupported score distances: {bad_score}; supported={SUPPORTED_SCORE_DISTANCES}")

    bad_corr = sorted(set(config.corr_metrics).difference(SUPPORTED_CORR_METRICS))
    if bad_corr:
        raise ValueError(f"Unsupported corr metrics: {bad_corr}; supported={SUPPORTED_CORR_METRICS}")

    if not config.embedding_distances:
        raise ValueError("At least one embedding distance is required")
    if not config.score_distances:
        raise ValueError("At least one score distance is required")
    if not config.corr_metrics:
        raise ValueError("At least one correlation metric is required")


def parse_metric_list(raw: Sequence[str] | None, *, kind: str) -> tuple[str, ...] | None:
    tokens = _flatten_tokens(raw, lowercase=True)
    if tokens is None:
        return None
    if kind == "embedding":
        supported = SUPPORTED_EMBEDDING_DISTANCES
    elif kind == "score":
        supported = SUPPORTED_SCORE_DISTANCES
    elif kind == "corr":
        supported = SUPPORTED_CORR_METRICS
    else:
        raise ValueError(f"Unknown metric list kind: {kind}")

    unknown = sorted(set(tokens).difference(supported))
    if unknown:
        raise ValueError(f"Unknown {kind} metric(s): {unknown}. Supported: {supported}")

    ordered_unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            ordered_unique.append(token)
            seen.add(token)
    return tuple(ordered_unique)


def parse_layer_selectors(raw: Sequence[str] | None) -> tuple[str, ...] | None:
    return _flatten_tokens(raw, lowercase=False)


def parse_name_list(raw: Sequence[str] | None) -> tuple[str, ...] | None:
    return _flatten_tokens(raw, lowercase=False)
