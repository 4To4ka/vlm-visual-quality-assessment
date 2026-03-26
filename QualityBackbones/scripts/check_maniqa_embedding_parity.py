from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.cache import configure_cache_env
from quality_backbones.datasets import load_dataset_index
from quality_backbones.extractors import default_dtype_for_device, managed_extractor, select_device
from quality_backbones.manifest import get_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snapshot/compare MANIQA extractor embeddings for parity checks"
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--dataset", type=str, default="koniq10k")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--models",
        nargs="*",
        default=["maniqa_pipal22", "maniqa_kadid10k", "maniqa_koniq10k"],
        help="MANIQA model keys to check",
    )
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Write current embedding signature JSON to this path",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Compare current signature against this baseline JSON",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional report path for compare results",
    )
    return parser.parse_args()


def _load_images(datasets_root: Path, dataset: str, max_samples: int) -> list[Image.Image]:
    df = load_dataset_index(datasets_root, dataset).iloc[: max(1, int(max_samples))].copy()
    images: list[Image.Image] = []
    for _, row in df.iterrows():
        with Image.open(row["abs_image_path"]) as img:
            images.append(img.convert("RGB"))
    return images


def _capture_signature(
    *,
    datasets_root: Path,
    dataset: str,
    weights_dir: Path,
    model_keys: list[str],
    max_samples: int,
    device_raw: str,
    seed: int,
) -> dict[str, object]:
    configure_cache_env(weights_dir)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    images = _load_images(datasets_root=datasets_root, dataset=dataset, max_samples=max_samples)
    device = select_device(device_raw)
    dtype = default_dtype_for_device(device)

    payload: dict[str, object] = {
        "dataset": dataset,
        "num_samples": len(images),
        "device": str(device),
        "dtype": str(dtype),
        "models": {},
    }

    models_dict: dict[str, object] = {}
    for model_key in model_keys:
        spec = get_model_spec(model_key)
        with managed_extractor(spec, device=device, dtype=dtype, weights_dir=weights_dir) as extractor:
            result = extractor.extract(images)

        model_info = {
            "layer_names": list(result.layer_names),
            "layers": [],
        }
        for layer_name, arr in zip(result.layer_names, result.per_layer_np, strict=False):
            np_arr = np.asarray(arr)
            model_info["layers"].append(
                {
                    "name": str(layer_name),
                    "shape": list(np_arr.shape),
                    "dtype": str(np_arr.dtype),
                    "sha256": hashlib.sha256(np_arr.tobytes()).hexdigest(),
                }
            )
        models_dict[model_key] = model_info

    payload["models"] = models_dict
    return payload


def _compare_signatures(current: dict[str, object], baseline: dict[str, object]) -> list[str]:
    issues: list[str] = []

    current_models = current.get("models", current.get("model_hashes"))
    baseline_models = baseline.get("models", baseline.get("model_hashes"))
    if not isinstance(current_models, dict) or not isinstance(baseline_models, dict):
        return ["Invalid signature format: missing models dict"]

    baseline_keys = sorted(str(key) for key in baseline_models.keys())
    current_keys = sorted(str(key) for key in current_models.keys())
    if baseline_keys != current_keys:
        issues.append(f"Model key mismatch: baseline={baseline_keys}, current={current_keys}")

    for model_key in baseline_keys:
        if model_key not in current_models:
            issues.append(f"Missing model in current signature: {model_key}")
            continue

        baseline_model = baseline_models[model_key]
        current_model = current_models[model_key]
        if not isinstance(baseline_model, dict) or not isinstance(current_model, dict):
            issues.append(f"Invalid model payload type for {model_key}")
            continue

        baseline_layer_names = baseline_model.get("layer_names")
        current_layer_names = current_model.get("layer_names")
        if baseline_layer_names != current_layer_names:
            issues.append(f"Layer name/order mismatch for {model_key}")

        baseline_layers = baseline_model.get("layers")
        current_layers = current_model.get("layers")
        if not isinstance(baseline_layers, list) or not isinstance(current_layers, list):
            issues.append(f"Invalid layer payload for {model_key}")
            continue
        if len(baseline_layers) != len(current_layers):
            issues.append(
                f"Layer count mismatch for {model_key}: baseline={len(baseline_layers)} current={len(current_layers)}"
            )
            continue

        for index, (baseline_layer, current_layer) in enumerate(
            zip(baseline_layers, current_layers, strict=False)
        ):
            if not isinstance(baseline_layer, dict) or not isinstance(current_layer, dict):
                issues.append(f"Invalid layer entry type for {model_key} at index {index}")
                continue

            for key in ("name", "shape", "dtype", "sha256"):
                if baseline_layer.get(key) != current_layer.get(key):
                    issues.append(
                        f"Mismatch for {model_key} layer#{index} field={key}: "
                        f"baseline={baseline_layer.get(key)!r}, current={current_layer.get(key)!r}"
                    )

    return issues


def main() -> None:
    args = parse_args()
    if args.snapshot is None and args.compare is None:
        raise ValueError("At least one of --snapshot or --compare is required")

    current = _capture_signature(
        datasets_root=args.datasets_root,
        dataset=args.dataset,
        weights_dir=args.weights_dir,
        model_keys=[str(item) for item in args.models],
        max_samples=args.max_samples,
        device_raw=args.device,
        seed=args.seed,
    )

    if args.snapshot is not None:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"Saved snapshot: {args.snapshot}")

    if args.compare is None:
        return

    baseline = json.loads(args.compare.read_text(encoding="utf-8"))
    issues = _compare_signatures(current=current, baseline=baseline)
    report = {
        "ok": not issues,
        "issues": issues,
        "baseline": str(args.compare),
        "dataset": args.dataset,
        "models": list(args.models),
        "num_samples": int(args.max_samples),
    }

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved compare report: {args.report}")

    if issues:
        print("MANIQA parity check failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("MANIQA parity check passed: current embeddings match baseline.")


if __name__ == "__main__":
    main()
