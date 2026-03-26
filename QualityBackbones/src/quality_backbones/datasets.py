from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def load_dataset_index(datasets_root: str | Path, dataset_name: str) -> pd.DataFrame:
    root = Path(datasets_root)
    dataset_dir = root / dataset_name
    csv_path = dataset_dir / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"filename", "subjective_score", "metadata"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df.copy().reset_index(drop=True)
    if "path" not in df.columns:
        df["path"] = df["filename"]

    path_values = df["path"].astype(str)
    fallback_values = df["filename"].astype(str)
    bad_path = path_values.str.strip().str.lower().isin({"", "nan", "none", "null"})
    path_values = path_values.where(~bad_path, fallback_values)

    df.insert(0, "row_id", df.index.astype(int))

    def _to_abs_path(rel_or_abs: str) -> str:
        candidate = Path(rel_or_abs)
        if candidate.is_absolute():
            return str(candidate)
        return str((dataset_dir / candidate).resolve())

    df["path"] = path_values
    df["abs_image_path"] = path_values.map(_to_abs_path)

    def _extract_split(md: str) -> str | None:
        try:
            payload = json.loads(md)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get("split")
        if value is None:
            value = payload.get("set")
        if value is None:
            return None
        return str(value)

    split_from_metadata = df["metadata"].map(_extract_split)
    if "split" in df.columns:
        raw_split = df["split"]
        split_missing = raw_split.isna() | raw_split.astype(str).str.strip().eq("")
        split_values = raw_split.where(~split_missing, split_from_metadata)
        df["split"] = split_values
    else:
        df["split"] = split_from_metadata

    return df


class ImageTableDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame) -> None:
        self.df = index_df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(row["abs_image_path"]).convert("RGB")
        return {
            "row_id": int(row["row_id"]),
            "image": image,
        }
