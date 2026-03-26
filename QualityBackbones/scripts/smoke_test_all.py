from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.cache import configure_cache_env
from quality_backbones.datasets import load_dataset_index
from quality_backbones.extractors import default_dtype_for_device, managed_extractor, select_device
from quality_backbones.manifest import get_model_spec, iter_enabled_image_model_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test model loading and extraction")
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--dataset", type=str, default="koniq10k")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--max-models", type=int, default=20)
    parser.add_argument("--models", nargs="*", default=None, help="Explicit model keys")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("logs/smoke_test_results.json"))
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    args = parse_args()
    configure_cache_env(args.weights_dir)
    df = load_dataset_index(args.datasets_root, args.dataset).iloc[: args.max_samples].copy()
    images = []
    for _, row in df.iterrows():
        from PIL import Image

        images.append(Image.open(row["abs_image_path"]).convert("RGB"))

    device = select_device(args.device)
    dtype = default_dtype_for_device(device)
    results = []

    if args.models:
        specs = [get_model_spec(k) for k in args.models]
    else:
        specs = list(iter_enabled_image_model_specs())

    for i, spec in enumerate(specs):
        if i >= args.max_models:
            break
        item = {"model_key": spec.key, "ok": False}
        try:
            with managed_extractor(spec, device=device, dtype=dtype, weights_dir=args.weights_dir) as extractor:
                result = extractor.extract(images)
            item["ok"] = True
            item["num_layers"] = len(result.layer_names)
            item["layer_names_head"] = result.layer_names[:5]
            item["shapes_head"] = [list(arr.shape) for arr in result.per_layer_np[:5]]
            print(f"[OK] {spec.key} layers={item['num_layers']}")
        except Exception as exc:
            item["error"] = str(exc)
            print(f"[FAIL] {spec.key}: {exc}")
        results.append(item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
