from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


def write_index_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def write_meta(meta: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


class H5LayerWriter:
    def __init__(self, h5_path: Path, num_rows: int, dtype: np.dtype = np.float16) -> None:
        self.h5_path = h5_path
        self.num_rows = num_rows
        self.dtype = dtype
        self._fp: h5py.File | None = None
        self._layers_created = False
        self._layer_names: list[str] = []

    def __enter__(self) -> "H5LayerWriter":
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = h5py.File(self.h5_path, "w")
        self._fp.attrs["schema_version"] = "v1"
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fp is not None:
            self._fp.close()

    @property
    def fp(self) -> h5py.File:
        if self._fp is None:
            raise RuntimeError("Writer is not open")
        return self._fp

    def create_layers(self, layer_names: Iterable[str], layer_dims: Iterable[int]) -> None:
        if self._layers_created:
            return
        self._layer_names = list(layer_names)
        dims = list(layer_dims)
        if len(self._layer_names) != len(dims):
            raise ValueError("layer_names and layer_dims length mismatch")
        for i, d in enumerate(dims):
            self.fp.create_dataset(
                f"layer_{i:03d}",
                shape=(self.num_rows, d),
                dtype=self.dtype,
                chunks=(min(2048, self.num_rows), d),
                compression="gzip",
                compression_opts=4,
            )
        self.fp.attrs["layer_names"] = np.array(self._layer_names, dtype=h5py.string_dtype(encoding="utf-8"))
        self.fp.attrs["row_count"] = int(self.num_rows)
        self._layers_created = True

    def write_batch(self, row_ids: np.ndarray, per_layer: list[np.ndarray]) -> None:
        if not self._layers_created:
            raise RuntimeError("Layers are not initialized")
        row_ids_np = np.asarray(row_ids).reshape(-1)
        write_order = np.argsort(row_ids_np)
        row_ids_sorted = row_ids_np[write_order]
        for i, arr in enumerate(per_layer):
            arr_np = np.asarray(arr)
            if arr_np.shape[0] != row_ids_sorted.shape[0]:
                raise ValueError(
                    f"Batch row count mismatch for layer_{i:03d}: "
                    f"rows={row_ids_sorted.shape[0]} arr={arr_np.shape[0]}"
                )
            self.fp[f"layer_{i:03d}"][row_ids_sorted] = arr_np[write_order].astype(self.dtype, copy=False)
