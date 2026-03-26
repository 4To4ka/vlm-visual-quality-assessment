from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.cache import configure_cache_env
from quality_backbones.datasets import ImageTableDataset, load_dataset_index
from quality_backbones.extractors import default_dtype_for_device, managed_extractor, select_device
from quality_backbones.manifest import get_model_spec, iter_enabled_image_model_specs
from quality_backbones.storage import H5LayerWriter, write_index_parquet, write_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compact embeddings for datasets/models")
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="*", default=None, help="Model keys. default: all enabled image models")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def collate_items(items: list[dict]) -> dict:
    return {
        "row_id": np.asarray([x["row_id"] for x in items], dtype=np.int64),
        "images": [x["image"] for x in items],
    }


def pick_specs(model_keys: list[str] | None):
    if model_keys:
        return [get_model_spec(k) for k in model_keys]
    return list(iter_enabled_image_model_specs())


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    args = parse_args()
    configure_cache_env(args.weights_dir)

    df = load_dataset_index(args.datasets_root, args.dataset)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy().reset_index(drop=True)
        df["row_id"] = np.arange(len(df), dtype=np.int64)

    dataset = ImageTableDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_items,
    )

    device = select_device(args.device)
    dtype = default_dtype_for_device(device)
    specs = pick_specs(args.models)

    for spec in specs:
        out_dir = args.output_dir / args.dataset / spec.key
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = out_dir / "index.parquet"
        meta_path = out_dir / "meta.json"
        h5_path = out_dir / "layers.h5"

        print(f"[extract] {spec.key}")
        try:
            with managed_extractor(spec, device=device, dtype=dtype, weights_dir=args.weights_dir) as extractor:
                write_index_parquet(df, index_path)
                with H5LayerWriter(h5_path, num_rows=len(df), dtype=np.float16) as writer:
                    layer_names: list[str] | None = None
                    for batch in tqdm(loader, total=len(loader), desc=spec.key):
                        row_ids = batch["row_id"]
                        images = batch["images"]
                        result = extractor.extract(images)
                        if layer_names is None:
                            layer_names = result.layer_names
                            writer.create_layers(layer_names, [arr.shape[1] for arr in result.per_layer_np])
                        writer.write_batch(row_ids=row_ids, per_layer=result.per_layer_np)

                meta = {
                    "schema_version": "v1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "dataset": args.dataset,
                    "num_samples": len(df),
                    "model_key": spec.key,
                    "family": spec.family,
                    "size": spec.size,
                    "source": spec.source,
                    "loader": spec.loader,
                    "pretrained_id": spec.pretrained_id,
                    "timm_name": spec.timm_name,
                    "regressor_dataset": spec.regressor_dataset,
                    "layer_names": layer_names,
                    "weights_dir": str(args.weights_dir.resolve()),
                    "index_path": str(index_path.resolve()),
                    "layers_path": str(h5_path.resolve()),
                }
                write_meta(meta, meta_path)

            error_path = out_dir / "error.json"
            error_path.unlink(missing_ok=True)
            print(f"  -> OK: {out_dir}")
        except Exception as exc:
            error_path = out_dir / "error.json"
            error_path.write_text(json.dumps({"error": str(exc)}, indent=2), encoding="utf-8")
            print(f"  -> FAIL: {exc}")


if __name__ == "__main__":
    main()
