from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from quality_backbones.cache import configure_cache_env
from quality_backbones.datasets import load_dataset_index
from quality_backbones.manifest import get_model_spec

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
except Exception as exc:  # pragma: no cover - import-time guard
    raise RuntimeError(
        "Missing `lightning` dependency. Install with: "
        "conda run -n encoders python -m pip install 'lightning>=2.4,<3'"
    ) from exc


_RANGE_RE = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    fraction: float


@dataclass(frozen=True)
class ResolvedLayer:
    model_key: str
    layer_index: int
    layer_name: str
    dim: int


@dataclass(frozen=True)
class DatasetSplit:
    train_positions: np.ndarray
    val_positions: np.ndarray


@dataclass(frozen=True)
class PreparedArrays:
    train_x: np.ndarray
    train_y: np.ndarray
    train_groups: np.ndarray | None
    train_dataset_ids: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_groups: np.ndarray | None
    val_dataset_ids: np.ndarray
    feature_plan: list[ResolvedLayer]
    split_stats: dict[str, dict[str, int]]


@dataclass(frozen=True)
class PreparedImageArrays:
    train_paths: np.ndarray
    train_y: np.ndarray
    train_groups: np.ndarray | None
    train_dataset_ids: np.ndarray
    val_paths: np.ndarray
    val_y: np.ndarray
    val_groups: np.ndarray | None
    val_dataset_ids: np.ndarray
    split_stats: dict[str, dict[str, int]]


def _parse_source_spec(raw: str) -> SourceSpec:
    text = raw.strip()
    if not text:
        raise ValueError("Empty source specification")

    if ":" not in text:
        return SourceSpec(dataset=text, fraction=1.0)

    dataset, fraction_str = text.split(":", 1)
    dataset = dataset.strip()
    fraction_str = fraction_str.strip()
    if not dataset:
        raise ValueError(f"Invalid source specification: {raw!r}")
    if not fraction_str:
        raise ValueError(f"Missing fraction in source specification: {raw!r}")

    fraction = float(fraction_str)
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"Source fraction must be in (0, 1], got {fraction} for {raw!r}")
    return SourceSpec(dataset=dataset, fraction=fraction)


def parse_source_specs(raw_specs: Iterable[str] | None) -> dict[str, SourceSpec]:
    result: dict[str, SourceSpec] = {}
    if raw_specs is None:
        return result
    for raw in raw_specs:
        spec = _parse_source_spec(raw)
        if spec.dataset in result:
            raise ValueError(f"Duplicate source dataset: {spec.dataset}")
        result[spec.dataset] = spec
    return result


def parse_feature_sources(raw_sources: Iterable[str]) -> OrderedDict[str, list[str]]:
    result: OrderedDict[str, list[str]] = OrderedDict()
    for raw in raw_sources:
        text = raw.strip()
        if not text:
            continue

        if ":" in text:
            model_key, selector = text.split(":", 1)
            model_key = model_key.strip()
            selector = selector.strip() or "all"
        else:
            model_key = text
            selector = "all"

        if not model_key:
            raise ValueError(f"Invalid feature source specification: {raw!r}")

        if model_key not in result:
            result[model_key] = []
        result[model_key].append(selector)

    if not result:
        raise ValueError("At least one --feature-source is required")
    return result


def _normalize_index(index: int, length: int) -> int:
    normalized = index if index >= 0 else length + index
    if normalized < 0 or normalized >= length:
        raise IndexError(f"Layer index {index} is out of bounds for {length} layers")
    return normalized


def _parse_layer_selector(selector: str, layer_names: list[str]) -> list[int]:
    text = selector.strip()
    if not text or text.lower() == "all" or text == "*":
        return list(range(len(layer_names)))

    selected: list[int] = []
    seen: set[int] = set()
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"Invalid empty layer selector: {selector!r}")

    for token in tokens:
        if token.lower() == "all" or token == "*":
            for idx in range(len(layer_names)):
                if idx not in seen:
                    selected.append(idx)
                    seen.add(idx)
            continue

        match = _RANGE_RE.match(token)
        if match is not None:
            start = _normalize_index(int(match.group(1)), len(layer_names))
            end = _normalize_index(int(match.group(2)), len(layer_names))
            step = 1 if end >= start else -1
            for idx in range(start, end + step, step):
                if idx not in seen:
                    selected.append(idx)
                    seen.add(idx)
            continue

        if token.lstrip("-").isdigit():
            idx = _normalize_index(int(token), len(layer_names))
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            continue

        if token not in layer_names:
            raise KeyError(
                f"Unknown layer token {token!r}. Available examples: {layer_names[:6]}"
            )
        idx = layer_names.index(token)
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)

    return selected


def _load_layer_names(meta_path: Path) -> list[str]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    layer_names = payload.get("layer_names")
    if not isinstance(layer_names, list) or not layer_names:
        raise ValueError(f"Invalid or empty layer_names in {meta_path}")
    return [str(name) for name in layer_names]


def _resolve_layers_for_dataset(
    outputs_root: Path,
    dataset: str,
    model_selectors: OrderedDict[str, list[str]],
) -> list[ResolvedLayer]:
    resolved: list[ResolvedLayer] = []
    for model_key, selectors in model_selectors.items():
        model_dir = outputs_root / dataset / model_key
        meta_path = model_dir / "meta.json"
        h5_path = model_dir / "layers.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {h5_path}")

        layer_names = _load_layer_names(meta_path)
        selected_indices: list[int] = []
        seen: set[int] = set()
        for selector in selectors:
            for idx in _parse_layer_selector(selector, layer_names):
                if idx not in seen:
                    selected_indices.append(idx)
                    seen.add(idx)

        if not selected_indices:
            raise ValueError(f"No layers selected for model {model_key}")

        with h5py.File(h5_path, "r") as fp:
            for idx in selected_indices:
                ds_name = f"layer_{idx:03d}"
                if ds_name not in fp:
                    raise KeyError(f"Missing dataset {ds_name} in {h5_path}")
                dim = int(fp[ds_name].shape[1])
                resolved.append(
                    ResolvedLayer(
                        model_key=model_key,
                        layer_index=idx,
                        layer_name=layer_names[idx],
                        dim=dim,
                    )
                )
    return resolved


def _load_dataset_table(
    datasets_root: Path,
    outputs_root: Path,
    dataset: str,
    model_keys: list[str],
) -> pd.DataFrame:
    if not model_keys:
        raise ValueError("model_keys must not be empty")

    anchor_key = model_keys[0]
    anchor_index_path = outputs_root / dataset / anchor_key / "index.parquet"
    if not anchor_index_path.exists():
        raise FileNotFoundError(f"Missing index parquet: {anchor_index_path}")

    anchor_df = pd.read_parquet(anchor_index_path)
    anchor_df = anchor_df.sort_values("row_id").reset_index(drop=True)
    if not anchor_df["row_id"].is_unique:
        raise ValueError(f"row_id is not unique in {anchor_index_path}")

    index_compare_cols = ["filename"]
    if "path" in anchor_df.columns:
        index_compare_cols.append("path")

    for model_key in model_keys[1:]:
        other_index_path = outputs_root / dataset / model_key / "index.parquet"
        if not other_index_path.exists():
            raise FileNotFoundError(f"Missing index parquet: {other_index_path}")
        other_df = pd.read_parquet(other_index_path)
        other_df = other_df.sort_values("row_id").reset_index(drop=True)
        if len(other_df) != len(anchor_df):
            raise ValueError(
                f"Row count mismatch for dataset={dataset} model={model_key}: "
                f"{len(other_df)} vs {len(anchor_df)}"
            )
        if not anchor_df["row_id"].equals(other_df["row_id"]):
            raise ValueError(f"row_id mismatch for dataset={dataset} model={model_key}")
        for col in index_compare_cols:
            if col not in other_df.columns:
                continue
            if not anchor_df[col].equals(other_df[col]):
                raise ValueError(f"{col} mismatch for dataset={dataset} model={model_key}")

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
        raise ValueError(
            f"Duplicate merge keys {merge_keys} in dataset CSV: {data_csv_path}"
        )

    rename_map = {
        col: f"{col}__csv"
        for col in csv_df.columns
        if col not in merge_keys and col in anchor_df.columns
    }
    csv_df = csv_df.rename(columns=rename_map)
    merged = anchor_df.merge(csv_df, on=merge_keys, how="left", sort=False, validate="m:1")
    return merged


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


def _build_metadata_payloads(df: pd.DataFrame) -> list[dict[str, Any] | None]:
    primary = df["metadata"] if "metadata" in df.columns else pd.Series([None] * len(df))
    fallback = df["metadata__csv"] if "metadata__csv" in df.columns else pd.Series([None] * len(df))

    payloads: list[dict[str, Any] | None] = []
    for p, f in zip(primary.tolist(), fallback.tolist(), strict=False):
        payload = _safe_json_loads(p)
        if payload is None:
            payload = _safe_json_loads(f)
        payloads.append(payload)
    return payloads


def extract_field_values(
    df: pd.DataFrame,
    field: str,
    *,
    numeric: bool,
) -> np.ndarray:
    field_name = field.strip()
    if not field_name:
        raise ValueError("Field name must not be empty")

    direct_candidates = [field_name, f"{field_name}__csv"]
    for candidate in direct_candidates:
        if candidate in df.columns:
            series = df[candidate]
            if numeric:
                return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
            return series.astype(str).to_numpy(dtype=object)

    payloads = _build_metadata_payloads(df)
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
        return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float32)

    normalized: list[str] = []
    for idx, value in enumerate(values):
        if value is None:
            normalized.append(f"__missing_row_{idx}")
        else:
            normalized.append(str(value))
    return np.asarray(normalized, dtype=object)


def _group_ids_for_positions(group_values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    ids: list[str] = []
    for pos in positions.tolist():
        value = group_values[pos]
        if value is None:
            ids.append(f"__missing_row_{pos}")
            continue
        text = str(value)
        if text == "" or text.lower() == "nan":
            ids.append(f"__missing_row_{pos}")
            continue
        ids.append(text)
    return np.asarray(ids, dtype=object)


def _sample_subset(
    positions: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
    group_values: np.ndarray | None = None,
) -> np.ndarray:
    if positions.size == 0:
        return np.empty(0, dtype=np.int64)
    if fraction >= 1.0:
        return np.sort(positions)

    target_count = max(1, int(round(positions.size * fraction)))
    target_count = min(target_count, positions.size)

    if group_values is None:
        selected = rng.choice(positions, size=target_count, replace=False)
        return np.sort(selected.astype(np.int64))

    group_ids = _group_ids_for_positions(group_values, positions)
    unique_groups = np.unique(group_ids)
    shuffled_groups = unique_groups[rng.permutation(len(unique_groups))]

    chosen: list[int] = []
    for group_id in shuffled_groups:
        members = positions[group_ids == group_id]
        chosen.extend(members.tolist())
        if len(chosen) >= target_count:
            break

    chosen_np = np.asarray(chosen, dtype=np.int64)
    chosen_np = np.unique(chosen_np)
    return np.sort(chosen_np)


def _split_random(
    positions: np.ndarray,
    val_ratio: float,
    rng: np.random.Generator,
) -> DatasetSplit:
    if positions.size < 2:
        raise ValueError("Need at least 2 samples to create train/val split")
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    perm = rng.permutation(positions)
    val_count = max(1, int(round(positions.size * val_ratio)))
    val_count = min(val_count, positions.size - 1)
    val_positions = np.sort(perm[:val_count].astype(np.int64))
    train_positions = np.sort(perm[val_count:].astype(np.int64))
    return DatasetSplit(train_positions=train_positions, val_positions=val_positions)


def _split_group_random(
    positions: np.ndarray,
    val_ratio: float,
    rng: np.random.Generator,
    group_values: np.ndarray,
) -> DatasetSplit:
    if positions.size < 2:
        raise ValueError("Need at least 2 samples to create group split")

    group_ids = _group_ids_for_positions(group_values, positions)
    unique_groups = np.unique(group_ids)
    if unique_groups.size < 2:
        raise ValueError("Need at least 2 groups to create a group-aware split")

    shuffled_groups = unique_groups[rng.permutation(len(unique_groups))]
    target_val = max(1, int(round(positions.size * val_ratio)))

    val_mask = np.zeros(positions.shape[0], dtype=bool)
    selected_count = 0
    for group_id in shuffled_groups:
        group_mask = group_ids == group_id
        val_mask |= group_mask
        selected_count += int(group_mask.sum())
        if selected_count >= target_val and np.any(~val_mask):
            break

    val_positions = np.sort(positions[val_mask].astype(np.int64))
    train_positions = np.sort(positions[~val_mask].astype(np.int64))

    if train_positions.size == 0 or val_positions.size == 0:
        return _split_random(positions=positions, val_ratio=val_ratio, rng=rng)

    return DatasetSplit(train_positions=train_positions, val_positions=val_positions)


def _split_predefined(
    df: pd.DataFrame,
    positions: np.ndarray,
    train_labels: set[str],
    val_labels: set[str],
) -> DatasetSplit:
    if "split" not in df.columns:
        raise ValueError("Column `split` is not available for predefined split policy")

    split_values = (
        df["split"].fillna("").astype(str).str.strip().str.lower().to_numpy(dtype=object)
    )
    subset_splits = split_values[positions]
    train_mask = np.isin(subset_splits, list(train_labels))
    val_mask = np.isin(subset_splits, list(val_labels))

    train_positions = positions[train_mask]
    val_positions = positions[val_mask]
    unknown_positions = positions[~train_mask & ~val_mask]
    if unknown_positions.size > 0:
        train_positions = np.concatenate([train_positions, unknown_positions])

    train_positions = np.unique(train_positions.astype(np.int64))
    val_positions = np.unique(val_positions.astype(np.int64))
    return DatasetSplit(
        train_positions=np.sort(train_positions),
        val_positions=np.sort(val_positions),
    )


def _split_explicit_two_way(
    positions: np.ndarray,
    train_fraction: float,
    val_fraction: float,
    rng: np.random.Generator,
    group_values: np.ndarray | None,
) -> DatasetSplit:
    if train_fraction < 0 or val_fraction < 0:
        raise ValueError("Explicit fractions must be non-negative")
    if train_fraction == 0 and val_fraction == 0:
        return DatasetSplit(
            train_positions=np.empty(0, dtype=np.int64),
            val_positions=np.empty(0, dtype=np.int64),
        )
    if train_fraction + val_fraction > 1.0 + 1e-8:
        raise ValueError(
            "For explicit train+val sources per dataset, fractions must satisfy train+val <= 1.0"
        )

    if group_values is None:
        perm = rng.permutation(positions)
        total = positions.size
        train_count = int(round(total * train_fraction))
        val_count = int(round(total * val_fraction))
        if train_fraction > 0:
            train_count = max(1, train_count)
        if val_fraction > 0:
            val_count = max(1, val_count)
        train_count = min(train_count, total)
        val_count = min(val_count, total - train_count)
        train_positions = np.sort(perm[:train_count].astype(np.int64))
        val_positions = np.sort(perm[train_count : train_count + val_count].astype(np.int64))
        return DatasetSplit(train_positions=train_positions, val_positions=val_positions)

    group_ids = _group_ids_for_positions(group_values, positions)
    unique_groups = np.unique(group_ids)
    shuffled_groups = unique_groups[rng.permutation(len(unique_groups))]

    target_train = int(round(positions.size * train_fraction))
    target_val = int(round(positions.size * val_fraction))
    if train_fraction > 0:
        target_train = max(1, target_train)
    if val_fraction > 0:
        target_val = max(1, target_val)

    train_chosen: list[int] = []
    val_chosen: list[int] = []
    train_count = 0
    val_count = 0
    state = "train"
    for group_id in shuffled_groups:
        members = positions[group_ids == group_id]
        if state == "train" and train_count < target_train:
            train_chosen.extend(members.tolist())
            train_count += members.size
            if train_count >= target_train:
                state = "val"
            continue
        if val_count < target_val:
            val_chosen.extend(members.tolist())
            val_count += members.size

    train_positions = np.sort(np.unique(np.asarray(train_chosen, dtype=np.int64)))
    val_positions = np.sort(np.unique(np.asarray(val_chosen, dtype=np.int64)))
    return DatasetSplit(train_positions=train_positions, val_positions=val_positions)


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


def _group_layers_by_model(layers: list[ResolvedLayer]) -> OrderedDict[str, list[ResolvedLayer]]:
    grouped: OrderedDict[str, list[ResolvedLayer]] = OrderedDict()
    for layer in layers:
        if layer.model_key not in grouped:
            grouped[layer.model_key] = []
        grouped[layer.model_key].append(layer)
    return grouped


def load_feature_matrix(
    outputs_root: Path,
    dataset: str,
    row_ids: np.ndarray,
    layers: list[ResolvedLayer],
    out_dtype: np.dtype,
) -> np.ndarray:
    grouped = _group_layers_by_model(layers)
    total_dim = sum(layer.dim for layer in layers)
    matrix = np.empty((row_ids.size, total_dim), dtype=out_dtype)

    cursor = 0
    for model_key, model_layers in grouped.items():
        h5_path = outputs_root / dataset / model_key / "layers.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing layers.h5 for dataset={dataset} model={model_key}: {h5_path}")
        with h5py.File(h5_path, "r") as fp:
            for layer in model_layers:
                ds_name = f"layer_{layer.layer_index:03d}"
                if ds_name not in fp:
                    raise KeyError(f"Missing dataset {ds_name} in {h5_path}")
                block = _read_rows(fp[ds_name], row_ids)
                if block.shape[1] != layer.dim:
                    raise ValueError(
                        f"Dimension mismatch for {dataset}/{model_key}/{ds_name}: "
                        f"expected {layer.dim}, got {block.shape[1]}"
                    )
                block = block.astype(out_dtype, copy=False)
                next_cursor = cursor + layer.dim
                matrix[:, cursor:next_cursor] = block
                cursor = next_cursor
    return matrix


class RegressionArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = features
        self.targets = targets.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


class RankingArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> None:
        self.features = features
        self.targets = targets.astype(np.float32, copy=False)
        self.groups = groups.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.groups.shape[0])

    def __getitem__(self, idx: int):
        indices = self.groups[idx]
        return self.features[indices], self.targets[indices]


def _build_group_buckets(group_values: np.ndarray) -> list[np.ndarray]:
    buckets_dict: dict[str, list[int]] = {}
    for idx, value in enumerate(group_values.tolist()):
        key = str(value)
        if key not in buckets_dict:
            buckets_dict[key] = []
        buckets_dict[key].append(idx)
    return [np.asarray(items, dtype=np.int64) for items in buckets_dict.values()]


def build_ranking_groups(
    num_items: int,
    num_choices: int,
    num_groups: int,
    rng: np.random.Generator,
    group_values: np.ndarray | None = None,
    within_group: bool = False,
) -> np.ndarray:
    if num_items <= 0:
        raise ValueError("num_items must be > 0")
    if num_choices < 2:
        raise ValueError("num_choices must be >= 2")
    if num_groups <= 0:
        raise ValueError("num_groups must be > 0")

    groups = np.empty((num_groups, num_choices), dtype=np.int64)
    if not within_group or group_values is None:
        replace = num_items < num_choices
        for i in range(num_groups):
            groups[i] = rng.choice(num_items, size=num_choices, replace=replace)
        return groups

    buckets = _build_group_buckets(group_values)
    valid_buckets = [bucket for bucket in buckets if bucket.size > 0]
    if not valid_buckets:
        raise ValueError("No groups available for within_group ranking sampling")

    for i in range(num_groups):
        bucket = valid_buckets[int(rng.integers(0, len(valid_buckets)))]
        replace = bucket.size < num_choices
        groups[i] = rng.choice(bucket, size=num_choices, replace=replace)
    return groups


class FusionScorer(nn.Module):
    def __init__(
        self,
        feature_dims: list[int],
        branch_dim: int,
        head_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("feature_dims must not be empty")
        if branch_dim <= 0:
            raise ValueError("branch_dim must be > 0")

        self.feature_dims = tuple(int(dim) for dim in feature_dims)
        self.total_input_dim = int(sum(self.feature_dims))

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, branch_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for dim in self.feature_dims
            ]
        )

        fused_dim = branch_dim * len(self.feature_dims)
        hidden_dim = head_hidden_dim if head_hidden_dim > 0 else max(64, min(1024, fused_dim // 2))
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.total_input_dim:
            raise ValueError(
                f"Expected last dim {self.total_input_dim}, got {x.shape[-1]}"
            )
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).float()
        chunks = torch.split(flat, self.feature_dims, dim=-1)
        projected = [branch(chunk) for branch, chunk in zip(self.branches, chunks, strict=False)]
        fused = torch.cat(projected, dim=-1)
        score = self.head(fused).squeeze(-1)
        return score.reshape(*original_shape)


class LinearScorer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        self.input_dim = int(input_dim)
        self.proj = nn.Linear(self.input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {x.shape[-1]}")
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).float()
        score = self.proj(flat).squeeze(-1)
        return score.reshape(*original_shape)


class MLPScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.input_dim
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = [
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {x.shape[-1]}")
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).float()
        score = self.net(flat).squeeze(-1)
        return score.reshape(*original_shape)


def _make_optimizer(
    params: list[dict[str, Any]] | Iterable[torch.nn.Parameter],
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    sgd_momentum: float,
) -> torch.optim.Optimizer:
    name = optimizer_name.strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps,
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=sgd_momentum,
            nesterov=(sgd_momentum > 0),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _resolve_warmup_steps(total_steps: int, warmup_steps: int, warmup_ratio: float) -> int:
    if total_steps <= 0:
        return 0
    if warmup_steps > 0:
        return min(int(warmup_steps), total_steps)
    if warmup_ratio <= 0:
        return 0
    return min(int(round(total_steps * warmup_ratio)), total_steps)


def _make_step_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    name = scheduler_name.strip().lower()
    if name == "none":
        return None
    if total_steps <= 0:
        return None
    if min_lr_ratio < 0 or min_lr_ratio > 1:
        raise ValueError("min_lr_ratio must be within [0, 1]")

    warmup = _resolve_warmup_steps(total_steps, warmup_steps, warmup_ratio)
    decay_steps = max(1, total_steps - warmup)

    def _lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-8, float(step + 1) / float(warmup))

        progress = float(step - warmup) / float(decay_steps)
        progress = max(0.0, min(1.0, progress))

        if name == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        if name == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


class EmbeddingTrainingModule(L.LightningModule):
    def __init__(
        self,
        scorer: nn.Module,
        task: str,
        lr: float,
        weight_decay: float,
        pairwise_target: str,
        listwise_target: str,
        target_temperature: float,
        learnable_logit_scale: bool,
        logit_temperature_init: float,
        logit_scale_max: float,
        run_dir: Path,
        save_epoch_metrics: bool,
        save_val_predictions: bool,
        val_predictions_max_rows: int,
        optimizer_name: str = "adamw",
        scheduler_name: str = "none",
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        min_lr_ratio: float = 0.0,
        optimizer_beta1: float = 0.9,
        optimizer_beta2: float = 0.999,
        optimizer_eps: float = 1e-8,
        sgd_momentum: float = 0.9,
        optim_param_groups: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.scorer = scorer
        self.task = task
        self.lr = lr
        self.weight_decay = weight_decay
        self.pairwise_target = pairwise_target
        self.listwise_target = listwise_target
        self.target_temperature = float(target_temperature)
        if self.target_temperature <= 0.0:
            raise ValueError("target_temperature must be > 0")

        self.logit_temperature_init = float(logit_temperature_init)
        if self.logit_temperature_init <= 0.0:
            raise ValueError("logit_temperature_init must be > 0")

        self.logit_scale_max = float(logit_scale_max)
        if self.logit_scale_max <= 0.0:
            raise ValueError("logit_scale_max must be > 0")

        ranking_task = task in {"pairwise", "listwise"}
        self.learnable_logit_scale = bool(learnable_logit_scale and ranking_task)
        self.logit_scale_param: nn.Parameter | None = None
        self.register_buffer(
            "_fixed_logit_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        if self.learnable_logit_scale:
            init_scale = 1.0 / self.logit_temperature_init
            self.logit_scale_param = nn.Parameter(
                torch.tensor(math.log(init_scale), dtype=torch.float32)
            )

        self.run_dir = Path(run_dir)
        self.save_epoch_metrics = save_epoch_metrics
        self.save_val_predictions = save_val_predictions
        self.val_predictions_max_rows = val_predictions_max_rows
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.warmup_steps = int(warmup_steps)
        self.warmup_ratio = float(warmup_ratio)
        self.min_lr_ratio = float(min_lr_ratio)
        self.optimizer_beta1 = float(optimizer_beta1)
        self.optimizer_beta2 = float(optimizer_beta2)
        self.optimizer_eps = float(optimizer_eps)
        self.sgd_momentum = float(sgd_momentum)
        self._optim_param_groups = optim_param_groups

        self._val_epoch_rows_local: list[dict[str, Any]] = []
        self._metrics_rows: list[dict[str, Any]] = []
        self._metrics_csv_path = self.run_dir / "epoch_metrics.csv"
        self._metrics_jsonl_path = self.run_dir / "epoch_metrics.jsonl"
        self._predictions_dir = self.run_dir / "epoch_predictions"
        if self.save_val_predictions:
            self._predictions_dir.mkdir(parents=True, exist_ok=True)

        self.save_hyperparameters(ignore=["scorer", "run_dir", "optim_param_groups"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x)

    def _current_logit_scale(self) -> torch.Tensor:
        if self.logit_scale_param is None:
            return self._fixed_logit_scale
        return torch.clamp(self.logit_scale_param.exp(), max=self.logit_scale_max)

    def _scale_ranking_scores(self, scores: torch.Tensor) -> torch.Tensor:
        return scores * self._current_logit_scale()

    def _log_ranking_temperature(self, prefix: str) -> None:
        if self.task not in {"pairwise", "listwise"}:
            return
        scale = self._current_logit_scale().detach()
        temperature = 1.0 / torch.clamp(scale, min=1e-8)
        self.log(
            f"{prefix}_logit_scale",
            scale,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_logit_temperature",
            temperature,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _step_regression(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        pred = self(x)
        y = y.float()
        loss = F.mse_loss(pred, y)
        mae = F.l1_loss(pred, y)
        return loss, mae, pred, y

    def _step_pairwise(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        scores = self._scale_ranking_scores(self(x))
        if scores.ndim != 2 or scores.shape[1] != 2:
            raise ValueError(f"Pairwise task expects scores of shape [B,2], got {tuple(scores.shape)}")

        delta = scores[:, 0] - scores[:, 1]
        y = y.float()
        if self.pairwise_target == "soft":
            target = torch.softmax(y / self.target_temperature, dim=1)[:, 0]
        else:
            target = (y[:, 0] > y[:, 1]).float()
        loss = F.binary_cross_entropy_with_logits(delta, target)

        gt_idx = torch.argmax(y, dim=1)
        pred_idx = (delta < 0).long()
        acc = (pred_idx == gt_idx).float().mean()
        return loss, acc, scores, y

    def _step_listwise(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        scores = self._scale_ranking_scores(self(x))
        if scores.ndim != 2 or scores.shape[1] < 2:
            raise ValueError(
                f"Listwise task expects scores of shape [B,N>=2], got {tuple(scores.shape)}"
            )

        y = y.float()
        if self.listwise_target == "soft":
            target_probs = torch.softmax(y / self.target_temperature, dim=1)
            log_probs = F.log_softmax(scores, dim=1)
            loss = -(target_probs * log_probs).sum(dim=1).mean()
        else:
            target_idx = torch.argmax(y, dim=1)
            loss = F.cross_entropy(scores, target_idx)

        gt_idx = torch.argmax(y, dim=1)
        pred_idx = torch.argmax(scores, dim=1)
        acc = (pred_idx == gt_idx).float().mean()
        return loss, acc, scores, y

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_rows_local = []

    def _append_regression_rows(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = pred.detach().float().cpu().numpy().reshape(-1)
        target_np = target.detach().float().cpu().numpy().reshape(-1)
        for pred_value, target_value in zip(pred_np, target_np, strict=False):
            row = {
                "pred": float(pred_value),
                "target": float(target_value),
                "abs_error": float(abs(pred_value - target_value)),
            }
            self._val_epoch_rows_local.append(row)

    def _append_ranking_rows(self, scores: torch.Tensor, target: torch.Tensor) -> None:
        scores_np = scores.detach().float().cpu().numpy()
        target_np = target.detach().float().cpu().numpy()
        num_choices = scores_np.shape[1]
        for score_row, target_row in zip(scores_np, target_np, strict=False):
            pred_choice = int(np.argmax(score_row))
            target_choice = int(np.argmax(target_row))
            row: dict[str, Any] = {
                "pred_choice": pred_choice,
                "target_choice": target_choice,
                "is_correct": int(pred_choice == target_choice),
            }
            for choice_idx in range(num_choices):
                row[f"pred_{choice_idx:02d}"] = float(score_row[choice_idx])
                row[f"target_{choice_idx:02d}"] = float(target_row[choice_idx])
            self._val_epoch_rows_local.append(row)

    def _gather_prediction_rows(self) -> list[dict[str, Any]]:
        if not self.save_val_predictions:
            return []

        rows = self._val_epoch_rows_local
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[list[dict[str, Any]] | None] | None
            if self.trainer.is_global_zero:
                gathered = [None] * torch.distributed.get_world_size()
            else:
                gathered = None
            torch.distributed.gather_object(rows, gathered, dst=0)
            if not self.trainer.is_global_zero:
                return []

            merged: list[dict[str, Any]] = []
            for part in gathered or []:
                if isinstance(part, list):
                    merged.extend(part)
            return merged

        return rows if self.trainer.is_global_zero else []

    def _collect_epoch_metrics(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in sorted(self.trainer.callback_metrics.items(), key=lambda x: str(x[0])):
            scalar = _to_scalar(value)
            if scalar is not None:
                row[str(key)] = scalar
        return row

    def _write_epoch_metrics(self) -> None:
        if not self.save_epoch_metrics:
            return
        pd.DataFrame(self._metrics_rows).to_csv(self._metrics_csv_path, index=False)
        _write_jsonl(self._metrics_jsonl_path, self._metrics_rows)

    def _write_epoch_predictions(self, rows: list[dict[str, Any]]) -> None:
        if not self.save_val_predictions:
            return

        if self.val_predictions_max_rows > 0:
            rows = rows[: self.val_predictions_max_rows]
        for idx, row in enumerate(rows):
            row["sample_idx"] = idx

        epoch = int(self.current_epoch)
        csv_path = self._predictions_dir / f"epoch_{epoch:03d}.csv"
        jsonl_path = self._predictions_dir / f"epoch_{epoch:03d}.jsonl"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        _write_jsonl(jsonl_path, rows)

    def training_step(self, batch, batch_idx: int):
        if self.task == "regression":
            loss, mae, _, _ = self._step_regression(batch)
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_mae", mae, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            return loss

        if self.task == "pairwise":
            loss, acc, _, _ = self._step_pairwise(batch)
        else:
            loss, acc, _, _ = self._step_listwise(batch)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self._log_ranking_temperature(prefix="train")
        return loss

    def validation_step(self, batch, batch_idx: int):
        if self.task == "regression":
            loss, mae, pred, target = self._step_regression(batch)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            if self.save_val_predictions and not self.trainer.sanity_checking:
                self._append_regression_rows(pred=pred, target=target)
            return

        if self.task == "pairwise":
            loss, acc, scores, target = self._step_pairwise(batch)
        else:
            loss, acc, scores, target = self._step_listwise(batch)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self._log_ranking_temperature(prefix="val")
        if self.save_val_predictions and not self.trainer.sanity_checking:
            self._append_ranking_rows(scores=scores, target=target)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._val_epoch_rows_local = []
            return

        gathered_rows = self._gather_prediction_rows()
        if self.trainer.is_global_zero:
            if self.save_epoch_metrics:
                self._metrics_rows.append(self._collect_epoch_metrics())
                self._write_epoch_metrics()
            if self.save_val_predictions:
                self._write_epoch_predictions(gathered_rows)
        self._val_epoch_rows_local = []

    def configure_optimizers(self):
        if self._optim_param_groups is None:
            trainable = [p for p in self.parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError("No trainable parameters found")
            optim_params: list[dict[str, Any]] | Iterable[torch.nn.Parameter] = trainable
        else:
            normalized_groups: list[dict[str, Any]] = []
            for group in self._optim_param_groups:
                params = [p for p in group.get("params", []) if isinstance(p, torch.nn.Parameter)]
                if not params:
                    continue
                merged = {k: v for k, v in group.items() if k != "params"}
                merged["params"] = params
                normalized_groups.append(merged)

            if self.logit_scale_param is not None and self.logit_scale_param.requires_grad:
                in_groups = any(
                    any(param is self.logit_scale_param for param in group.get("params", []))
                    for group in normalized_groups
                )
                if not in_groups:
                    normalized_groups.append(
                        {
                            "params": [self.logit_scale_param],
                            "lr": self.lr,
                            "weight_decay": 0.0,
                        }
                    )

            if not normalized_groups:
                raise RuntimeError("Optimizer param groups are empty")
            optim_params = normalized_groups

        optimizer = _make_optimizer(
            optim_params,
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
            beta1=self.optimizer_beta1,
            beta2=self.optimizer_beta2,
            eps=self.optimizer_eps,
            sgd_momentum=self.sgd_momentum,
        )
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        scheduler = _make_step_scheduler(
            optimizer,
            scheduler_name=self.scheduler_name,
            total_steps=total_steps,
            warmup_steps=self.warmup_steps,
            warmup_ratio=self.warmup_ratio,
            min_lr_ratio=self.min_lr_ratio,
        )
        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def _parse_labels(raw: str) -> set[str]:
    values = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return values


def _devices_arg(raw: str) -> int | list[int] | str:
    text = raw.strip()
    if text == "auto":
        return "auto"
    if "," in text:
        return [int(token.strip()) for token in text.split(",") if token.strip()]
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def _resolve_precision(precision: str, accelerator: str) -> str:
    text = precision.strip().lower()
    if text == "auto":
        return "bf16-mixed" if torch.cuda.is_available() else "32-true"

    if accelerator == "cpu" and text in {"16-mixed", "bf16-mixed"}:
        return "32-true"
    return precision


def _to_scalar(value: Any) -> float | int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, default=_json_default))
            fp.write("\n")


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


def _build_early_stopping_callback(patience: int) -> EarlyStopping | None:
    if patience <= 0:
        return None
    return EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
    )


def _early_stopping_config(patience: int) -> dict[str, Any]:
    enabled = patience > 0
    return {
        "enabled": enabled,
        "monitor": "val_loss" if enabled else None,
        "mode": "min" if enabled else None,
        "patience": int(patience),
    }


def _early_stopping_result(callback: EarlyStopping | None) -> dict[str, Any] | None:
    if callback is None:
        return None
    best_score = _to_scalar(callback.best_score)
    if isinstance(best_score, float) and not math.isfinite(best_score):
        best_score = None
    return {
        "monitor": str(callback.monitor),
        "mode": str(callback.mode),
        "patience": int(callback.patience),
        "best_score": best_score,
        "wait_count": int(callback.wait_count),
        "stopped_epoch": int(callback.stopped_epoch),
        "stopped_early": bool(callback.stopped_epoch > 0),
    }


def _print_available_fields(
    datasets_root: Path,
    outputs_root: Path,
    datasets: list[str],
    model_keys: list[str],
) -> None:
    print("Available fields summary:")
    for dataset in datasets:
        table = _load_dataset_table(
            datasets_root=datasets_root,
            outputs_root=outputs_root,
            dataset=dataset,
            model_keys=model_keys,
        )
        columns = sorted(table.columns.tolist())
        metadata_keys: set[str] = set()
        payloads = _build_metadata_payloads(table.iloc[: min(2000, len(table))])
        for payload in payloads:
            if isinstance(payload, dict):
                metadata_keys.update(str(key) for key in payload.keys())
        print(f"- {dataset}: rows={len(table)}")
        print(f"  columns: {columns}")
        print(f"  metadata keys (sampled): {sorted(metadata_keys)}")


def prepare_arrays(args: Any) -> PreparedArrays:
    train_sources = parse_source_specs(args.train_source)
    val_sources = parse_source_specs(args.val_source)
    if not train_sources:
        raise ValueError("At least one --train-source is required")

    feature_selectors = parse_feature_sources(args.feature_source)
    model_keys = list(feature_selectors.keys())
    dataset_names = sorted(set(train_sources.keys()) | set(val_sources.keys()))
    if not dataset_names:
        raise ValueError("No datasets selected")

    if args.list_target_fields:
        _print_available_fields(
            datasets_root=args.datasets_root,
            outputs_root=args.outputs_root,
            datasets=dataset_names,
            model_keys=model_keys,
        )
        raise SystemExit(0)

    rng = np.random.default_rng(args.seed)
    feature_plan_reference: list[ResolvedLayer] | None = None

    train_x_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    train_group_parts: list[np.ndarray] = []
    train_dataset_parts: list[np.ndarray] = []
    val_x_parts: list[np.ndarray] = []
    val_y_parts: list[np.ndarray] = []
    val_group_parts: list[np.ndarray] = []
    val_dataset_parts: list[np.ndarray] = []

    split_stats: dict[str, dict[str, int]] = {}
    train_labels = _parse_labels(args.train_split_labels)
    val_labels = _parse_labels(args.val_split_labels)

    for dataset in dataset_names:
        resolved_layers = _resolve_layers_for_dataset(
            outputs_root=args.outputs_root,
            dataset=dataset,
            model_selectors=feature_selectors,
        )
        signature = [(x.model_key, x.layer_name, x.dim) for x in resolved_layers]
        if feature_plan_reference is None:
            feature_plan_reference = resolved_layers
        else:
            ref_signature = [(x.model_key, x.layer_name, x.dim) for x in feature_plan_reference]
            if signature != ref_signature:
                raise ValueError(
                    "Selected layers are not compatible across datasets. "
                    f"Mismatch in dataset={dataset}."
                )

        table = _load_dataset_table(
            datasets_root=args.datasets_root,
            outputs_root=args.outputs_root,
            dataset=dataset,
            model_keys=model_keys,
        )

        target_values = extract_field_values(table, args.target_field, numeric=True)
        valid_positions = np.where(np.isfinite(target_values))[0].astype(np.int64)

        group_values: np.ndarray | None = None
        use_groups = (
            args.split_policy == "group"
            or args.group_field is not None
            or args.ranking_sampling == "within_group"
        )
        if use_groups:
            if args.group_field is None:
                raise ValueError(
                    "`--group-field` is required when split policy is `group` "
                    "or ranking sampling is `within_group`"
                )
            raw_group_values = extract_field_values(table, args.group_field, numeric=False)
            group_values = np.asarray(
                [f"{dataset}::{value}" for value in raw_group_values.tolist()],
                dtype=object,
            )

        train_frac = train_sources[dataset].fraction if dataset in train_sources else 0.0
        val_frac = val_sources[dataset].fraction if dataset in val_sources else 0.0

        if val_sources:
            if train_frac > 0 and val_frac > 0:
                split = _split_explicit_two_way(
                    positions=valid_positions,
                    train_fraction=train_frac,
                    val_fraction=val_frac,
                    rng=rng,
                    group_values=group_values,
                )
            elif train_frac > 0:
                split = DatasetSplit(
                    train_positions=_sample_subset(
                        positions=valid_positions,
                        fraction=train_frac,
                        rng=rng,
                        group_values=group_values if args.split_policy == "group" else None,
                    ),
                    val_positions=np.empty(0, dtype=np.int64),
                )
            elif val_frac > 0:
                split = DatasetSplit(
                    train_positions=np.empty(0, dtype=np.int64),
                    val_positions=_sample_subset(
                        positions=valid_positions,
                        fraction=val_frac,
                        rng=rng,
                        group_values=group_values if args.split_policy == "group" else None,
                    ),
                )
            else:
                split = DatasetSplit(
                    train_positions=np.empty(0, dtype=np.int64),
                    val_positions=np.empty(0, dtype=np.int64),
                )
        else:
            if train_frac <= 0:
                split = DatasetSplit(
                    train_positions=np.empty(0, dtype=np.int64),
                    val_positions=np.empty(0, dtype=np.int64),
                )
            else:
                pool = _sample_subset(
                    positions=valid_positions,
                    fraction=train_frac,
                    rng=rng,
                    group_values=None,
                )
                if pool.size == 0:
                    split = DatasetSplit(
                        train_positions=np.empty(0, dtype=np.int64),
                        val_positions=np.empty(0, dtype=np.int64),
                    )
                elif args.split_policy == "predefined":
                    split = _split_predefined(
                        df=table,
                        positions=pool,
                        train_labels=train_labels,
                        val_labels=val_labels,
                    )
                    if split.train_positions.size == 0 or split.val_positions.size == 0:
                        split = _split_random(
                            positions=pool,
                            val_ratio=args.val_ratio,
                            rng=rng,
                        )
                elif args.split_policy == "group":
                    if group_values is None:
                        raise ValueError("Group split policy requires group values")
                    split = _split_group_random(
                        positions=pool,
                        val_ratio=args.val_ratio,
                        rng=rng,
                        group_values=group_values,
                    )
                else:
                    split = _split_random(
                        positions=pool,
                        val_ratio=args.val_ratio,
                        rng=rng,
                    )

        train_positions = split.train_positions
        val_positions = split.val_positions
        split_stats[dataset] = {
            "total_rows": int(len(table)),
            "valid_target_rows": int(valid_positions.size),
            "train_rows": int(train_positions.size),
            "val_rows": int(val_positions.size),
        }

        if train_positions.size > 0:
            train_row_ids = table.iloc[train_positions]["row_id"].to_numpy(dtype=np.int64)
            train_x = load_feature_matrix(
                outputs_root=args.outputs_root,
                dataset=dataset,
                row_ids=train_row_ids,
                layers=resolved_layers,
                out_dtype=np.float16 if args.feature_dtype == "float16" else np.float32,
            )
            train_y = target_values[train_positions].astype(np.float32)
            train_x_parts.append(train_x)
            train_y_parts.append(train_y)
            train_dataset_parts.append(np.full(train_y.shape[0], dataset, dtype=object))
            if group_values is not None:
                train_group_parts.append(group_values[train_positions])

        if val_positions.size > 0:
            val_row_ids = table.iloc[val_positions]["row_id"].to_numpy(dtype=np.int64)
            val_x = load_feature_matrix(
                outputs_root=args.outputs_root,
                dataset=dataset,
                row_ids=val_row_ids,
                layers=resolved_layers,
                out_dtype=np.float16 if args.feature_dtype == "float16" else np.float32,
            )
            val_y = target_values[val_positions].astype(np.float32)
            val_x_parts.append(val_x)
            val_y_parts.append(val_y)
            val_dataset_parts.append(np.full(val_y.shape[0], dataset, dtype=object))
            if group_values is not None:
                val_group_parts.append(group_values[val_positions])

    if feature_plan_reference is None:
        raise RuntimeError("No feature plan was resolved")

    if not train_x_parts:
        raise ValueError("No training samples selected")
    if not val_x_parts:
        raise ValueError("No validation samples selected")

    train_x = np.concatenate(train_x_parts, axis=0)
    train_y = np.concatenate(train_y_parts, axis=0)
    val_x = np.concatenate(val_x_parts, axis=0)
    val_y = np.concatenate(val_y_parts, axis=0)

    train_groups = np.concatenate(train_group_parts, axis=0) if train_group_parts else None
    val_groups = np.concatenate(val_group_parts, axis=0) if val_group_parts else None
    train_dataset_ids = np.concatenate(train_dataset_parts, axis=0)
    val_dataset_ids = np.concatenate(val_dataset_parts, axis=0)

    if args.max_train_samples is not None and args.max_train_samples > 0 and train_x.shape[0] > args.max_train_samples:
        subset_rng = np.random.default_rng(args.seed + 101)
        subset = subset_rng.choice(train_x.shape[0], size=args.max_train_samples, replace=False)
        train_x = train_x[subset]
        train_y = train_y[subset]
        if train_groups is not None:
            train_groups = train_groups[subset]
        train_dataset_ids = train_dataset_ids[subset]

    if args.max_val_samples is not None and args.max_val_samples > 0 and val_x.shape[0] > args.max_val_samples:
        subset_rng = np.random.default_rng(args.seed + 202)
        subset = subset_rng.choice(val_x.shape[0], size=args.max_val_samples, replace=False)
        val_x = val_x[subset]
        val_y = val_y[subset]
        if val_groups is not None:
            val_groups = val_groups[subset]
        val_dataset_ids = val_dataset_ids[subset]

    return PreparedArrays(
        train_x=train_x,
        train_y=train_y,
        train_groups=train_groups,
        train_dataset_ids=train_dataset_ids,
        val_x=val_x,
        val_y=val_y,
        val_groups=val_groups,
        val_dataset_ids=val_dataset_ids,
        feature_plan=feature_plan_reference,
        split_stats=split_stats,
    )


def _build_dataloaders(args: Any, arrays: PreparedArrays) -> tuple[DataLoader, DataLoader]:
    pin_memory = args.accelerator != "cpu" and torch.cuda.is_available()

    if args.task == "regression":
        train_ds = RegressionArrayDataset(features=arrays.train_x, targets=arrays.train_y)
        val_ds = RegressionArrayDataset(features=arrays.val_x, targets=arrays.val_y)
    else:
        if args.task == "pairwise":
            num_choices = 2
        else:
            num_choices = args.num_choices
            if num_choices < 2:
                raise ValueError("--num-choices must be >= 2 for listwise task")

        train_group_count = args.train_ranking_groups
        if train_group_count <= 0:
            train_group_count = arrays.train_x.shape[0]
        val_group_count = args.val_ranking_groups
        if val_group_count <= 0:
            val_group_count = max(256, arrays.val_x.shape[0])

        train_rng = np.random.default_rng(args.seed + 303)
        val_rng = np.random.default_rng(args.seed + 404)
        ranking_sampling = str(getattr(args, "ranking_sampling", "within_dataset")).strip().lower()
        if ranking_sampling == "within_group":
            if arrays.train_groups is None or arrays.val_groups is None:
                raise ValueError(
                    "`--ranking-sampling within_group` requires valid group values in both splits"
                )
            train_group_values = arrays.train_groups
            val_group_values = arrays.val_groups
            within_group = True
        elif ranking_sampling == "within_dataset":
            train_group_values = arrays.train_dataset_ids
            val_group_values = arrays.val_dataset_ids
            within_group = True
        elif ranking_sampling == "global":
            train_group_values = None
            val_group_values = None
            within_group = False
        else:
            raise ValueError(f"Unsupported ranking sampling mode: {ranking_sampling}")

        train_groups = build_ranking_groups(
            num_items=arrays.train_x.shape[0],
            num_choices=num_choices,
            num_groups=train_group_count,
            rng=train_rng,
            group_values=train_group_values,
            within_group=within_group,
        )
        val_groups = build_ranking_groups(
            num_items=arrays.val_x.shape[0],
            num_choices=num_choices,
            num_groups=val_group_count,
            rng=val_rng,
            group_values=val_group_values,
            within_group=within_group,
        )

        train_ds = RankingArrayDataset(features=arrays.train_x, targets=arrays.train_y, groups=train_groups)
        val_ds = RankingArrayDataset(features=arrays.val_x, targets=arrays.val_y, groups=val_groups)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def _print_available_fields_from_csv(datasets_root: Path, datasets: list[str]) -> None:
    print("Available fields summary:")
    for dataset in datasets:
        table = load_dataset_index(datasets_root, dataset)
        columns = sorted(table.columns.tolist())
        metadata_keys: set[str] = set()
        payloads = _build_metadata_payloads(table.iloc[: min(2000, len(table))])
        for payload in payloads:
            if isinstance(payload, dict):
                metadata_keys.update(str(key) for key in payload.keys())
        print(f"- {dataset}: rows={len(table)}")
        print(f"  columns: {columns}")
        print(f"  metadata keys (sampled): {sorted(metadata_keys)}")


def prepare_image_arrays(args: Any) -> PreparedImageArrays:
    train_sources = parse_source_specs(args.train_source)
    val_sources = parse_source_specs(args.val_source)
    if not train_sources:
        raise ValueError("At least one --train-source is required")

    dataset_names = sorted(set(train_sources.keys()) | set(val_sources.keys()))
    if not dataset_names:
        raise ValueError("No datasets selected")

    if args.list_target_fields:
        _print_available_fields_from_csv(datasets_root=args.datasets_root, datasets=dataset_names)
        raise SystemExit(0)

    rng = np.random.default_rng(args.seed)
    train_path_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    train_group_parts: list[np.ndarray] = []
    train_dataset_parts: list[np.ndarray] = []
    val_path_parts: list[np.ndarray] = []
    val_y_parts: list[np.ndarray] = []
    val_group_parts: list[np.ndarray] = []
    val_dataset_parts: list[np.ndarray] = []

    split_stats: dict[str, dict[str, int]] = {}
    train_labels = _parse_labels(args.train_split_labels)
    val_labels = _parse_labels(args.val_split_labels)

    for dataset in dataset_names:
        table = load_dataset_index(args.datasets_root, dataset)
        target_values = extract_field_values(table, args.target_field, numeric=True)
        valid_positions = np.where(np.isfinite(target_values))[0].astype(np.int64)

        group_values: np.ndarray | None = None
        use_groups = (
            args.split_policy == "group"
            or args.group_field is not None
            or args.ranking_sampling == "within_group"
        )
        if use_groups:
            if args.group_field is None:
                raise ValueError(
                    "`--group-field` is required when split policy is `group` "
                    "or ranking sampling is `within_group`"
                )
            raw_group_values = extract_field_values(table, args.group_field, numeric=False)
            group_values = np.asarray(
                [f"{dataset}::{value}" for value in raw_group_values.tolist()],
                dtype=object,
            )

        train_frac = train_sources[dataset].fraction if dataset in train_sources else 0.0
        val_frac = val_sources[dataset].fraction if dataset in val_sources else 0.0

        if val_sources:
            if train_frac > 0 and val_frac > 0:
                split = _split_explicit_two_way(
                    positions=valid_positions,
                    train_fraction=train_frac,
                    val_fraction=val_frac,
                    rng=rng,
                    group_values=group_values,
                )
            elif train_frac > 0:
                split = DatasetSplit(
                    train_positions=_sample_subset(
                        positions=valid_positions,
                        fraction=train_frac,
                        rng=rng,
                        group_values=group_values if args.split_policy == "group" else None,
                    ),
                    val_positions=np.empty(0, dtype=np.int64),
                )
            elif val_frac > 0:
                split = DatasetSplit(
                    train_positions=np.empty(0, dtype=np.int64),
                    val_positions=_sample_subset(
                        positions=valid_positions,
                        fraction=val_frac,
                        rng=rng,
                        group_values=group_values if args.split_policy == "group" else None,
                    ),
                )
            else:
                split = DatasetSplit(
                    train_positions=np.empty(0, dtype=np.int64),
                    val_positions=np.empty(0, dtype=np.int64),
                )
        else:
            if train_frac <= 0:
                split = DatasetSplit(
                    train_positions=np.empty(0, dtype=np.int64),
                    val_positions=np.empty(0, dtype=np.int64),
                )
            else:
                pool = _sample_subset(
                    positions=valid_positions,
                    fraction=train_frac,
                    rng=rng,
                    group_values=None,
                )
                if pool.size == 0:
                    split = DatasetSplit(
                        train_positions=np.empty(0, dtype=np.int64),
                        val_positions=np.empty(0, dtype=np.int64),
                    )
                elif args.split_policy == "predefined":
                    split = _split_predefined(
                        df=table,
                        positions=pool,
                        train_labels=train_labels,
                        val_labels=val_labels,
                    )
                    if split.train_positions.size == 0 or split.val_positions.size == 0:
                        split = _split_random(
                            positions=pool,
                            val_ratio=args.val_ratio,
                            rng=rng,
                        )
                elif args.split_policy == "group":
                    if group_values is None:
                        raise ValueError("Group split policy requires group values")
                    split = _split_group_random(
                        positions=pool,
                        val_ratio=args.val_ratio,
                        rng=rng,
                        group_values=group_values,
                    )
                else:
                    split = _split_random(
                        positions=pool,
                        val_ratio=args.val_ratio,
                        rng=rng,
                    )

        train_positions = split.train_positions
        val_positions = split.val_positions
        split_stats[dataset] = {
            "total_rows": int(len(table)),
            "valid_target_rows": int(valid_positions.size),
            "train_rows": int(train_positions.size),
            "val_rows": int(val_positions.size),
        }

        if train_positions.size > 0:
            train_paths = (
                table.iloc[train_positions]["abs_image_path"].astype(str).to_numpy(dtype=object)
            )
            train_y = target_values[train_positions].astype(np.float32)
            train_path_parts.append(train_paths)
            train_y_parts.append(train_y)
            train_dataset_parts.append(np.full(train_y.shape[0], dataset, dtype=object))
            if group_values is not None:
                train_group_parts.append(group_values[train_positions])

        if val_positions.size > 0:
            val_paths = table.iloc[val_positions]["abs_image_path"].astype(str).to_numpy(dtype=object)
            val_y = target_values[val_positions].astype(np.float32)
            val_path_parts.append(val_paths)
            val_y_parts.append(val_y)
            val_dataset_parts.append(np.full(val_y.shape[0], dataset, dtype=object))
            if group_values is not None:
                val_group_parts.append(group_values[val_positions])

    if not train_path_parts:
        raise ValueError("No training samples selected")
    if not val_path_parts:
        raise ValueError("No validation samples selected")

    train_paths = np.concatenate(train_path_parts, axis=0)
    train_y = np.concatenate(train_y_parts, axis=0)
    val_paths = np.concatenate(val_path_parts, axis=0)
    val_y = np.concatenate(val_y_parts, axis=0)

    train_groups = np.concatenate(train_group_parts, axis=0) if train_group_parts else None
    val_groups = np.concatenate(val_group_parts, axis=0) if val_group_parts else None
    train_dataset_ids = np.concatenate(train_dataset_parts, axis=0)
    val_dataset_ids = np.concatenate(val_dataset_parts, axis=0)

    if (
        args.max_train_samples is not None
        and args.max_train_samples > 0
        and train_paths.shape[0] > args.max_train_samples
    ):
        subset_rng = np.random.default_rng(args.seed + 101)
        subset = subset_rng.choice(train_paths.shape[0], size=args.max_train_samples, replace=False)
        train_paths = train_paths[subset]
        train_y = train_y[subset]
        if train_groups is not None:
            train_groups = train_groups[subset]
        train_dataset_ids = train_dataset_ids[subset]

    if (
        args.max_val_samples is not None
        and args.max_val_samples > 0
        and val_paths.shape[0] > args.max_val_samples
    ):
        subset_rng = np.random.default_rng(args.seed + 202)
        subset = subset_rng.choice(val_paths.shape[0], size=args.max_val_samples, replace=False)
        val_paths = val_paths[subset]
        val_y = val_y[subset]
        if val_groups is not None:
            val_groups = val_groups[subset]
        val_dataset_ids = val_dataset_ids[subset]

    return PreparedImageArrays(
        train_paths=train_paths,
        train_y=train_y,
        train_groups=train_groups,
        train_dataset_ids=train_dataset_ids,
        val_paths=val_paths,
        val_y=val_y,
        val_groups=val_groups,
        val_dataset_ids=val_dataset_ids,
        split_stats=split_stats,
    )


class RegressionImagePathDataset(Dataset):
    def __init__(self, paths: np.ndarray, targets: np.ndarray) -> None:
        self.paths = paths.astype(object)
        self.targets = targets.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.paths.shape[0])

    def __getitem__(self, idx: int):
        return str(self.paths[idx]), float(self.targets[idx])


class RankingImagePathDataset(Dataset):
    def __init__(self, paths: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> None:
        self.paths = paths.astype(object)
        self.targets = targets.astype(np.float32, copy=False)
        self.groups = groups.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.groups.shape[0])

    def __getitem__(self, idx: int):
        indices = self.groups[idx]
        return self.paths[indices], self.targets[indices]


class VisionEncoderBackbone(nn.Module):
    _SUPPORTED_LOADERS = {"hf_auto_image", "timm_cnn_features", "timm_vit_blocks"}

    def __init__(self, model_key: str, weights_dir: Path) -> None:
        super().__init__()
        self.spec = get_model_spec(model_key)
        self.weights_dir = Path(weights_dir)
        self.loader = self.spec.loader
        if self.loader not in self._SUPPORTED_LOADERS:
            raise ValueError(
                "Encoder training currently supports loaders: "
                f"{sorted(self._SUPPORTED_LOADERS)}. Got {self.loader!r} for model {self.spec.key!r}."
            )

        self.image_processor: Any | None = None
        self.train_transform: Any | None = None
        self.eval_transform: Any | None = None

        if self.loader == "hf_auto_image":
            from transformers import AutoImageProcessor, AutoModel

            model_kwargs: dict[str, Any] = {"cache_dir": str(self.weights_dir / "hf")}
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.spec.pretrained_id,
                cache_dir=str(self.weights_dir / "hf"),
            )
            self.encoder = AutoModel.from_pretrained(self.spec.pretrained_id, **model_kwargs)
            self.backend = "hf"
            self.output_dim = self._infer_hf_output_dim()
        else:
            import timm

            if self.spec.timm_name is None:
                raise RuntimeError(f"Model {self.spec.key} has no timm_name")

            self.encoder = timm.create_model(self.spec.timm_name, pretrained=True, num_classes=0)
            cfg = timm.data.resolve_model_data_config(self.encoder)
            self.train_transform = timm.data.create_transform(**cfg, is_training=True)
            self.eval_transform = timm.data.create_transform(**cfg, is_training=False)
            self.backend = "timm"
            self.output_dim = int(getattr(self.encoder, "num_features", 0) or 0)
            if self.output_dim <= 0:
                self.output_dim = self._infer_output_dim_from_dummy()

    def _hf_image_size(self) -> int:
        processor = self.image_processor
        if processor is None:
            return 224
        size = getattr(processor, "size", None)
        if isinstance(size, int) and size > 0:
            return int(size)
        if isinstance(size, dict):
            for key in ("shortest_edge", "height", "width"):
                value = size.get(key)
                if isinstance(value, int) and value > 0:
                    return int(value)
        crop_size = getattr(processor, "crop_size", None)
        if isinstance(crop_size, dict):
            for key in ("height", "width"):
                value = crop_size.get(key)
                if isinstance(value, int) and value > 0:
                    return int(value)
        return 224

    def _infer_hf_output_dim(self) -> int:
        config = getattr(self.encoder, "config", None)
        for attr in ("hidden_size", "projection_dim", "embed_dim", "d_model"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)
        hidden_sizes = getattr(config, "hidden_sizes", None)
        if isinstance(hidden_sizes, (list, tuple)) and hidden_sizes:
            last = hidden_sizes[-1]
            if isinstance(last, int) and last > 0:
                return int(last)
        return self._infer_output_dim_from_dummy()

    def _infer_output_dim_from_dummy(self) -> int:
        image_size = self._hf_image_size() if self.backend == "hf" else 224
        dummy = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.inference_mode():
            emb = self(dummy)
        if was_training:
            self.encoder.train()
        if emb.ndim != 2:
            raise RuntimeError(f"Encoder dummy forward produced invalid shape: {tuple(emb.shape)}")
        return int(emb.shape[-1])

    @staticmethod
    def _pool_spatial_or_tokens(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            return t
        if t.ndim == 3:
            return t[:, 0]
        if t.ndim == 4:
            return t.mean(dim=(-1, -2))
        raise ValueError(f"Unsupported tensor shape for pooling: {tuple(t.shape)}")

    def _pool_timm_tokens(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 3:
            return self._pool_spatial_or_tokens(t)
        global_pool = str(getattr(self.encoder, "global_pool", "token") or "token")
        num_prefix_tokens = int(getattr(self.encoder, "num_prefix_tokens", 1) or 0)
        if global_pool in {"token", "tok", "cls", ""}:
            return t[:, 0]

        patch_tokens = t[:, num_prefix_tokens:] if num_prefix_tokens > 0 else t
        if patch_tokens.shape[1] == 0:
            patch_tokens = t
        if global_pool == "max":
            return patch_tokens.max(dim=1).values
        if global_pool == "avgmax":
            return 0.5 * (patch_tokens.mean(dim=1) + patch_tokens.max(dim=1).values)
        return patch_tokens.mean(dim=1)

    def preprocess_paths(self, paths: list[str], *, training: bool) -> torch.Tensor:
        if self.backend == "hf":
            if self.image_processor is None:
                raise RuntimeError("HF encoder is missing image_processor")
            images: list[Image.Image] = []
            for path in paths:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            batch = self.image_processor(images=images, return_tensors="pt")
            if "pixel_values" not in batch:
                raise RuntimeError("Image processor did not produce `pixel_values`")
            return batch["pixel_values"]

        transform = self.train_transform if training else self.eval_transform
        if transform is None:
            raise RuntimeError("timm encoder transform is not initialized")
        tensors: list[torch.Tensor] = []
        for path in paths:
            with Image.open(path) as img:
                tensors.append(transform(img.convert("RGB")))
        return torch.stack(tensors, dim=0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values [B,C,H,W], got {tuple(pixel_values.shape)}")

        if self.backend == "hf":
            outputs = self.encoder(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
            pooler_output = getattr(outputs, "pooler_output", None)
            if isinstance(pooler_output, torch.Tensor):
                return self._pool_spatial_or_tokens(pooler_output)

            last_hidden = getattr(outputs, "last_hidden_state", None)
            if not isinstance(last_hidden, torch.Tensor):
                raise RuntimeError("HF encoder returned neither pooler_output nor last_hidden_state")
            if self.spec.family == "I-JEPA" and last_hidden.ndim == 3:
                return last_hidden.mean(dim=1)
            return self._pool_spatial_or_tokens(last_hidden)

        encoded = self.encoder(pixel_values)
        if isinstance(encoded, (tuple, list)):
            tensor_output = next(
                (x for x in reversed(encoded) if isinstance(x, torch.Tensor)),
                None,
            )
            if tensor_output is None:
                raise RuntimeError("timm encoder returned no tensor output")
            encoded = tensor_output
        if not isinstance(encoded, torch.Tensor):
            raise RuntimeError(f"Unsupported timm encoder output type: {type(encoded)}")
        if encoded.ndim == 3:
            return self._pool_timm_tokens(encoded)
        return self._pool_spatial_or_tokens(encoded)


class EncoderWithHead(nn.Module):
    def __init__(self, backbone: VisionEncoderBackbone, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            embedding = self.backbone(x)
            return self.head(embedding)
        if x.ndim == 5:
            batch_size, num_choices = x.shape[:2]
            flat = x.reshape(batch_size * num_choices, *x.shape[2:])
            embedding = self.backbone(flat)
            scores = self.head(embedding)
            return scores.reshape(batch_size, num_choices)
        raise ValueError(f"Expected input rank 4 or 5, got {tuple(x.shape)}")


def _parse_lora_target_modules(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    return [token.strip() for token in text.split(",") if token.strip()]


def _infer_lora_target_modules(model: nn.Module) -> list[str]:
    blocked = {"classifier", "head", "score", "regressor", "lm_head"}
    names: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in blocked:
            continue
        names.add(leaf)
    return sorted(names)


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def _apply_tune_mode(backbone: VisionEncoderBackbone, args: Any) -> dict[str, Any]:
    tune_mode = str(args.tune_mode).strip().lower()
    info: dict[str, Any] = {"tune_mode": tune_mode}

    if tune_mode == "full":
        _set_requires_grad(backbone.encoder, True)
        backbone.encoder.train()
        return info
    if tune_mode == "frozen":
        _set_requires_grad(backbone.encoder, False)
        backbone.encoder.eval()
        return info
    if tune_mode != "lora":
        raise ValueError(f"Unsupported tune mode: {args.tune_mode}")

    if backbone.backend != "hf":
        raise ValueError("LoRA is currently supported only for HF encoder loaders")

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "LoRA mode requires `peft`. Install with: "
            "conda run -n encoders python -m pip install peft"
        ) from exc

    _set_requires_grad(backbone.encoder, False)
    target_modules = _parse_lora_target_modules(getattr(args, "lora_target_modules", ""))
    if not target_modules:
        target_modules = _infer_lora_target_modules(backbone.encoder)
    if not target_modules:
        raise RuntimeError("Unable to infer LoRA target modules")

    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(args.lora_dropout),
        bias=str(args.lora_bias),
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone.encoder = get_peft_model(backbone.encoder, lora_cfg)
    backbone.encoder.train()
    info["lora_target_modules"] = target_modules
    info["lora_r"] = int(args.lora_r)
    info["lora_alpha"] = int(args.lora_alpha)
    info["lora_dropout"] = float(args.lora_dropout)
    info["lora_bias"] = str(args.lora_bias)
    return info


def _load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        raw = payload["state_dict"]
    elif isinstance(payload, dict):
        raw = payload
    else:
        raise ValueError(f"Unsupported checkpoint format in {path}")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            state_dict[str(key)] = value
    if not state_dict:
        raise ValueError(f"No tensor state_dict values found in {path}")
    return state_dict


def _load_head_init_weights(head: nn.Module, init_path: Path) -> dict[str, Any]:
    source_state = _load_state_dict_file(init_path)
    target_state = head.state_dict()

    matched: dict[str, torch.Tensor] = {}
    considered = 0
    for key, value in source_state.items():
        candidates = [key]
        for prefix in (
            "module.scorer.head.",
            "module.scorer.",
            "module.head.",
            "scorer.head.",
            "scorer.",
            "head.",
        ):
            if key.startswith(prefix):
                candidates.append(key[len(prefix) :])
        for candidate in candidates:
            if candidate in target_state:
                considered += 1
                if tuple(target_state[candidate].shape) == tuple(value.shape):
                    matched[candidate] = value
                break

    missing, unexpected = head.load_state_dict(matched, strict=False)
    return {
        "init_path": str(init_path),
        "loaded_keys": len(matched),
        "considered_keys": int(considered),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _build_encoder_head(args: Any, input_dim: int) -> nn.Module:
    head_type = str(getattr(args, "head_type", "mlp")).strip().lower()
    if head_type == "linear":
        return LinearScorer(input_dim=input_dim)
    if head_type == "mlp":
        return MLPScorer(input_dim=input_dim, hidden_dim=input_dim, dropout=args.dropout)
    if head_type == "mlp_fusion":
        return FusionScorer(
            feature_dims=[input_dim],
            branch_dim=args.branch_dim,
            head_hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported head type: {head_type}")


def _collate_regression_images(
    batch: list[tuple[str, float]],
    *,
    backbone: VisionEncoderBackbone,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    paths, targets = zip(*batch, strict=False)
    x = backbone.preprocess_paths([str(path) for path in paths], training=training)
    y = torch.tensor(np.asarray(targets, dtype=np.float32), dtype=torch.float32)
    return x, y


def _collate_ranking_images(
    batch: list[tuple[np.ndarray, np.ndarray]],
    *,
    backbone: VisionEncoderBackbone,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    path_groups, target_groups = zip(*batch, strict=False)
    if not path_groups:
        raise RuntimeError("Empty ranking batch")

    num_choices = int(len(path_groups[0]))
    flat_paths: list[str] = []
    for group in path_groups:
        flat_paths.extend([str(item) for item in group.tolist()])

    x_flat = backbone.preprocess_paths(flat_paths, training=training)
    x = x_flat.reshape(len(path_groups), num_choices, *x_flat.shape[1:])
    y = torch.tensor(np.asarray(target_groups, dtype=np.float32), dtype=torch.float32)
    return x, y


def _build_encoder_dataloaders(
    args: Any,
    arrays: PreparedImageArrays,
    backbone: VisionEncoderBackbone,
) -> tuple[DataLoader, DataLoader]:
    num_workers = int(args.num_workers)
    if num_workers > 0:
        print("[encoder] forcing num_workers=0 for in-process image preprocessing")
        num_workers = 0

    pin_memory = args.accelerator != "cpu" and torch.cuda.is_available()

    if args.task == "regression":
        train_ds = RegressionImagePathDataset(paths=arrays.train_paths, targets=arrays.train_y)
        val_ds = RegressionImagePathDataset(paths=arrays.val_paths, targets=arrays.val_y)
        train_collate = lambda batch: _collate_regression_images(
            batch,
            backbone=backbone,
            training=True,
        )
        val_collate = lambda batch: _collate_regression_images(
            batch,
            backbone=backbone,
            training=False,
        )
    else:
        if args.task == "pairwise":
            num_choices = 2
        else:
            num_choices = args.num_choices
            if num_choices < 2:
                raise ValueError("--num-choices must be >= 2 for listwise task")

        train_group_count = args.train_ranking_groups
        if train_group_count <= 0:
            train_group_count = arrays.train_paths.shape[0]
        val_group_count = args.val_ranking_groups
        if val_group_count <= 0:
            val_group_count = max(64, arrays.val_paths.shape[0])

        train_rng = np.random.default_rng(args.seed + 303)
        val_rng = np.random.default_rng(args.seed + 404)
        ranking_sampling = str(getattr(args, "ranking_sampling", "within_dataset")).strip().lower()
        if ranking_sampling == "within_group":
            if arrays.train_groups is None or arrays.val_groups is None:
                raise ValueError(
                    "`--ranking-sampling within_group` requires valid group values in both splits"
                )
            train_group_values = arrays.train_groups
            val_group_values = arrays.val_groups
            within_group = True
        elif ranking_sampling == "within_dataset":
            train_group_values = arrays.train_dataset_ids
            val_group_values = arrays.val_dataset_ids
            within_group = True
        elif ranking_sampling == "global":
            train_group_values = None
            val_group_values = None
            within_group = False
        else:
            raise ValueError(f"Unsupported ranking sampling mode: {ranking_sampling}")

        train_groups = build_ranking_groups(
            num_items=arrays.train_paths.shape[0],
            num_choices=num_choices,
            num_groups=train_group_count,
            rng=train_rng,
            group_values=train_group_values,
            within_group=within_group,
        )
        val_groups = build_ranking_groups(
            num_items=arrays.val_paths.shape[0],
            num_choices=num_choices,
            num_groups=val_group_count,
            rng=val_rng,
            group_values=val_group_values,
            within_group=within_group,
        )

        train_ds = RankingImagePathDataset(paths=arrays.train_paths, targets=arrays.train_y, groups=train_groups)
        val_ds = RankingImagePathDataset(paths=arrays.val_paths, targets=arrays.val_y, groups=val_groups)
        train_collate = lambda batch: _collate_ranking_images(
            batch,
            backbone=backbone,
            training=True,
        )
        val_collate = lambda batch: _collate_ranking_images(
            batch,
            backbone=backbone,
            training=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=val_collate,
    )
    return train_loader, val_loader


def _param_count(module: nn.Module, *, trainable_only: bool) -> int:
    if trainable_only:
        return int(sum(param.numel() for param in module.parameters() if param.requires_grad))
    return int(sum(param.numel() for param in module.parameters()))


def run_encoder_training(args: Any) -> Path:
    if not getattr(args, "encoder_model", None):
        raise ValueError("--encoder-model is required when --train-mode=encoder")

    configure_cache_env(args.weights_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(args.output_dir, args.run_name, args.task)

    L.seed_everything(args.seed, workers=True)
    arrays = prepare_image_arrays(args)
    backbone = VisionEncoderBackbone(model_key=args.encoder_model, weights_dir=args.weights_dir)

    if getattr(args, "gradient_checkpointing", False) and hasattr(
        backbone.encoder, "gradient_checkpointing_enable"
    ):
        backbone.encoder.gradient_checkpointing_enable()

    tune_info = _apply_tune_mode(backbone, args)
    head = _build_encoder_head(args, input_dim=backbone.output_dim)
    head_init_info: dict[str, Any] | None = None
    if getattr(args, "init_head_from", None):
        head_init_info = _load_head_init_weights(head, Path(args.init_head_from))

    scorer = EncoderWithHead(backbone=backbone, head=head)
    train_loader, val_loader = _build_encoder_dataloaders(args, arrays, backbone)

    limit_train_batches = _normalize_limit_batches(args.limit_train_batches, len(train_loader))
    limit_val_batches = _normalize_limit_batches(args.limit_val_batches, len(val_loader))
    tensorboard_enabled = bool(getattr(args, "tensorboard", True))
    save_epoch_metrics = bool(getattr(args, "save_epoch_metrics", True))
    save_val_predictions = bool(getattr(args, "save_val_predictions", False))
    val_predictions_max_rows = int(getattr(args, "val_predictions_max_rows", 0) or 0)
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 0) or 0)
    learnable_logit_scale = bool(getattr(args, "learnable_logit_scale", False))
    logit_temperature_init = float(getattr(args, "logit_temperature_init", 0.07))
    logit_scale_max = float(getattr(args, "logit_scale_max", 100.0))

    encoder_params = [param for param in scorer.backbone.parameters() if param.requires_grad]
    head_params = [param for param in scorer.head.parameters() if param.requires_grad]
    if not head_params:
        raise RuntimeError("Head has no trainable parameters")

    optim_param_groups: list[dict[str, Any]] = []
    if encoder_params:
        optim_param_groups.append(
            {
                "params": encoder_params,
                "lr": float(args.encoder_lr if args.encoder_lr is not None else args.lr),
                "weight_decay": float(
                    args.encoder_weight_decay
                    if args.encoder_weight_decay is not None
                    else args.weight_decay
                ),
            }
        )
    optim_param_groups.append(
        {
            "params": head_params,
            "lr": float(args.head_lr if args.head_lr is not None else args.lr),
            "weight_decay": float(
                args.head_weight_decay if args.head_weight_decay is not None else args.weight_decay
            ),
        }
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_mode": "encoder",
        "task": args.task,
        "pairwise_target": args.pairwise_target,
        "listwise_target": args.listwise_target,
        "target_temperature": float(args.target_temperature),
        "num_choices": int(args.num_choices),
        "learnable_logit_scale": learnable_logit_scale,
        "logit_temperature_init": logit_temperature_init,
        "logit_scale_max": logit_scale_max,
        "target_field": args.target_field,
        "group_field": args.group_field,
        "split_policy": args.split_policy,
        "ranking_sampling": args.ranking_sampling,
        "val_ratio": args.val_ratio,
        "encoder_model": args.encoder_model,
        "encoder_family": backbone.spec.family,
        "encoder_loader": backbone.spec.loader,
        "encoder_source": backbone.spec.source,
        "encoder_pretrained_id": backbone.spec.pretrained_id,
        "encoder_timm_name": backbone.spec.timm_name,
        "encoder_output_dim": int(backbone.output_dim),
        "head_type": str(args.head_type),
        "tune": tune_info,
        "head_init": head_init_info,
        "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
        "split_stats": arrays.split_stats,
        "train_rows": int(arrays.train_paths.shape[0]),
        "val_rows": int(arrays.val_paths.shape[0]),
        "train_sources": list(args.train_source),
        "val_sources": list(args.val_source) if args.val_source else [],
        "train_target_min": float(np.min(arrays.train_y)),
        "train_target_max": float(np.max(arrays.train_y)),
        "val_target_min": float(np.min(arrays.val_y)),
        "val_target_max": float(np.max(arrays.val_y)),
        "parameters": {
            "encoder_total": _param_count(scorer.backbone, trainable_only=False),
            "encoder_trainable": _param_count(scorer.backbone, trainable_only=True),
            "head_total": _param_count(scorer.head, trainable_only=False),
            "head_trainable": _param_count(scorer.head, trainable_only=True),
            "model_total": _param_count(scorer, trainable_only=False),
            "model_trainable": _param_count(scorer, trainable_only=True),
        },
        "optimization": {
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "base_lr": args.lr,
            "weight_decay": args.weight_decay,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "encoder_weight_decay": args.encoder_weight_decay,
            "head_weight_decay": args.head_weight_decay,
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "max_grad_norm": args.max_grad_norm,
            "accumulate_grad_batches": args.accumulate_grad_batches,
        },
        "tensorboard_enabled": tensorboard_enabled,
        "save_epoch_metrics": save_epoch_metrics,
        "save_val_predictions": save_val_predictions,
        "val_predictions_max_rows": val_predictions_max_rows,
        "early_stopping": _early_stopping_config(early_stopping_patience),
        "trainer": {
            "accelerator": args.accelerator,
            "devices": _devices_arg(args.devices),
            "strategy": args.strategy,
            "precision": _resolve_precision(args.precision, args.accelerator),
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "limit_train_batches": limit_train_batches,
            "limit_val_batches": limit_val_batches,
            "log_every_n_steps": args.log_every_n_steps,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "run_summary": str(run_dir / "run_summary.json"),
            "result": str(run_dir / "result.json"),
            "checkpoints_dir": str(run_dir / "checkpoints"),
            "csv_logger_dir": str(run_dir / "logs"),
            "tensorboard_dir": str(run_dir / "tb") if tensorboard_enabled else None,
            "epoch_metrics_csv": str(run_dir / "epoch_metrics.csv") if save_epoch_metrics else None,
            "epoch_metrics_jsonl": str(run_dir / "epoch_metrics.jsonl") if save_epoch_metrics else None,
            "epoch_predictions_dir": str(run_dir / "epoch_predictions") if save_val_predictions else None,
        },
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"[dry-run] Prepared encoder data and config at: {run_dir}")
        return run_dir

    module = EmbeddingTrainingModule(
        scorer=scorer,
        task=args.task,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pairwise_target=args.pairwise_target,
        listwise_target=args.listwise_target,
        target_temperature=args.target_temperature,
        learnable_logit_scale=learnable_logit_scale,
        logit_temperature_init=logit_temperature_init,
        logit_scale_max=logit_scale_max,
        run_dir=run_dir,
        save_epoch_metrics=save_epoch_metrics,
        save_val_predictions=save_val_predictions,
        val_predictions_max_rows=val_predictions_max_rows,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        optimizer_eps=args.optimizer_eps,
        sgd_momentum=args.sgd_momentum,
        optim_param_groups=optim_param_groups,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_cb = _build_early_stopping_callback(early_stopping_patience)
    callbacks: list[Any] = [checkpoint_cb]
    if early_stopping_cb is not None:
        callbacks.append(early_stopping_cb)
    logger_instances: list[Any] = [CSVLogger(save_dir=str(run_dir), name="logs")]
    if tensorboard_enabled:
        try:
            logger_instances.append(TensorBoardLogger(save_dir=str(run_dir), name="tb"))
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard logging requires `tensorboard`. Install with: "
                "conda run -n encoders python -m pip install tensorboard"
            ) from exc
    logger: Any = logger_instances[0] if len(logger_instances) == 1 else logger_instances

    trainer = L.Trainer(
        default_root_dir=str(run_dir),
        accelerator=args.accelerator,
        devices=_devices_arg(args.devices),
        strategy=args.strategy,
        precision=_resolve_precision(args.precision, args.accelerator),
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=float(args.max_grad_norm) if float(args.max_grad_norm) > 0 else 0.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=max(1, int(args.accumulate_grad_batches)),
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    result = {
        "best_model_path": checkpoint_cb.best_model_path,
        "best_model_score": float(checkpoint_cb.best_model_score)
        if checkpoint_cb.best_model_score is not None
        else None,
        "early_stopping": _early_stopping_result(early_stopping_cb),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Training artifacts saved to: {run_dir}")
    return run_dir


def _make_run_dir(output_dir: Path, run_name: str | None, task: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = run_name.strip() if run_name else f"{task}_{timestamp}"
    candidate = output_dir / base_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    suffix = 1
    while True:
        alt = output_dir / f"{base_name}_{suffix:02d}"
        if not alt.exists():
            alt.mkdir(parents=True, exist_ok=False)
            return alt
        suffix += 1


def _normalize_limit_batches(raw_value: float, num_batches: int) -> float | int:
    value = float(raw_value)
    if value <= 0:
        raise ValueError("limit_*_batches must be > 0")
    if value > 1:
        return int(value)
    if num_batches <= 1:
        return 1
    min_fraction = 1.0 / float(num_batches)
    return max(value, min_fraction)


def run_training(args: Any) -> Path:
    train_mode = str(getattr(args, "train_mode", "embeddings")).strip().lower()
    if args.task not in {"regression", "pairwise", "listwise"}:
        raise ValueError(f"Unsupported task: {args.task}")
    if train_mode == "encoder":
        return run_encoder_training(args)
    if train_mode != "embeddings":
        raise ValueError(f"Unsupported train mode: {train_mode}")
    if not getattr(args, "feature_source", None):
        raise ValueError("At least one --feature-source is required in embeddings mode")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(args.output_dir, args.run_name, args.task)

    L.seed_everything(args.seed, workers=True)
    arrays = prepare_arrays(args)

    feature_dims = [layer.dim for layer in arrays.feature_plan]
    train_loader, val_loader = _build_dataloaders(args, arrays)

    limit_train_batches = _normalize_limit_batches(args.limit_train_batches, len(train_loader))
    limit_val_batches = _normalize_limit_batches(args.limit_val_batches, len(val_loader))
    tensorboard_enabled = bool(getattr(args, "tensorboard", True))
    save_epoch_metrics = bool(getattr(args, "save_epoch_metrics", True))
    save_val_predictions = bool(getattr(args, "save_val_predictions", False))
    val_predictions_max_rows = int(getattr(args, "val_predictions_max_rows", 0) or 0)
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 0) or 0)
    learnable_logit_scale = bool(getattr(args, "learnable_logit_scale", False))
    logit_temperature_init = float(getattr(args, "logit_temperature_init", 0.07))
    logit_scale_max = float(getattr(args, "logit_scale_max", 100.0))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_mode": "embeddings",
        "task": args.task,
        "pairwise_target": args.pairwise_target,
        "listwise_target": args.listwise_target,
        "target_temperature": float(args.target_temperature),
        "num_choices": int(args.num_choices),
        "learnable_logit_scale": learnable_logit_scale,
        "logit_temperature_init": logit_temperature_init,
        "logit_scale_max": logit_scale_max,
        "target_field": args.target_field,
        "group_field": args.group_field,
        "split_policy": args.split_policy,
        "ranking_sampling": args.ranking_sampling,
        "val_ratio": args.val_ratio,
        "feature_sources": list(args.feature_source),
        "feature_plan": [layer.__dict__ for layer in arrays.feature_plan],
        "feature_total_dim": int(sum(feature_dims)),
        "split_stats": arrays.split_stats,
        "train_rows": int(arrays.train_x.shape[0]),
        "val_rows": int(arrays.val_x.shape[0]),
        "train_sources": list(args.train_source),
        "val_sources": list(args.val_source) if args.val_source else [],
        "train_target_min": float(np.min(arrays.train_y)),
        "train_target_max": float(np.max(arrays.train_y)),
        "val_target_min": float(np.min(arrays.val_y)),
        "val_target_max": float(np.max(arrays.val_y)),
        "tensorboard_enabled": tensorboard_enabled,
        "save_epoch_metrics": save_epoch_metrics,
        "save_val_predictions": save_val_predictions,
        "val_predictions_max_rows": val_predictions_max_rows,
        "early_stopping": _early_stopping_config(early_stopping_patience),
        "head_type": str(getattr(args, "head_type", "mlp")),
        "head_init": None,
        "optimization": {
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "base_lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "max_grad_norm": args.max_grad_norm,
            "accumulate_grad_batches": args.accumulate_grad_batches,
        },
        "trainer": {
            "accelerator": args.accelerator,
            "devices": _devices_arg(args.devices),
            "strategy": args.strategy,
            "precision": _resolve_precision(args.precision, args.accelerator),
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "limit_train_batches": limit_train_batches,
            "limit_val_batches": limit_val_batches,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "run_summary": str(run_dir / "run_summary.json"),
            "result": str(run_dir / "result.json"),
            "checkpoints_dir": str(run_dir / "checkpoints"),
            "csv_logger_dir": str(run_dir / "logs"),
            "tensorboard_dir": str(run_dir / "tb") if tensorboard_enabled else None,
            "epoch_metrics_csv": str(run_dir / "epoch_metrics.csv") if save_epoch_metrics else None,
            "epoch_metrics_jsonl": str(run_dir / "epoch_metrics.jsonl") if save_epoch_metrics else None,
            "epoch_predictions_dir": str(run_dir / "epoch_predictions") if save_val_predictions else None,
        },
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"[dry-run] Prepared data and config at: {run_dir}")
        return run_dir

    head_type = str(getattr(args, "head_type", "mlp")).strip().lower()
    if head_type == "linear":
        scorer = LinearScorer(input_dim=int(sum(feature_dims)))
    elif head_type == "mlp":
        scorer = MLPScorer(
            input_dim=int(sum(feature_dims)),
            hidden_dim=int(sum(feature_dims)),
            dropout=args.dropout,
        )
    elif head_type == "mlp_fusion":
        scorer = FusionScorer(
            feature_dims=feature_dims,
            branch_dim=args.branch_dim,
            head_hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unsupported head type: {head_type}")

    if getattr(args, "init_head_from", None):
        head_init_info = _load_head_init_weights(scorer, Path(args.init_head_from))
        summary["head_init"] = head_init_info
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=_json_default),
            encoding="utf-8",
        )

    module = EmbeddingTrainingModule(
        scorer=scorer,
        task=args.task,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pairwise_target=args.pairwise_target,
        listwise_target=args.listwise_target,
        target_temperature=args.target_temperature,
        learnable_logit_scale=learnable_logit_scale,
        logit_temperature_init=logit_temperature_init,
        logit_scale_max=logit_scale_max,
        run_dir=run_dir,
        save_epoch_metrics=save_epoch_metrics,
        save_val_predictions=save_val_predictions,
        val_predictions_max_rows=val_predictions_max_rows,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        optimizer_eps=args.optimizer_eps,
        sgd_momentum=args.sgd_momentum,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_cb = _build_early_stopping_callback(early_stopping_patience)
    callbacks: list[Any] = [checkpoint_cb]
    if early_stopping_cb is not None:
        callbacks.append(early_stopping_cb)
    logger_instances: list[Any] = [CSVLogger(save_dir=str(run_dir), name="logs")]
    if tensorboard_enabled:
        try:
            logger_instances.append(TensorBoardLogger(save_dir=str(run_dir), name="tb"))
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard logging requires `tensorboard`. Install with: "
                "conda run -n encoders python -m pip install tensorboard"
            ) from exc
    logger: Any = logger_instances[0] if len(logger_instances) == 1 else logger_instances

    trainer = L.Trainer(
        default_root_dir=str(run_dir),
        accelerator=args.accelerator,
        devices=_devices_arg(args.devices),
        strategy=args.strategy,
        precision=_resolve_precision(args.precision, args.accelerator),
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=float(args.max_grad_norm) if float(args.max_grad_norm) > 0 else 0.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=max(1, int(args.accumulate_grad_batches)),
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    result = {
        "best_model_path": checkpoint_cb.best_model_path,
        "best_model_score": float(checkpoint_cb.best_model_score)
        if checkpoint_cb.best_model_score is not None
        else None,
        "early_stopping": _early_stopping_result(early_stopping_cb),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Training artifacts saved to: {run_dir}")
    return run_dir
