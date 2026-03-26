from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.cache import configure_cache_env
from quality_backbones.extractors import default_dtype_for_device, managed_extractor, select_device
from quality_backbones.manifest import iter_enabled_image_model_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch model weights into local weights directory")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--only", nargs="*", default=None, help="Optional model keys")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    args = parse_args()
    env = configure_cache_env(args.weights_dir)
    print(f"Cache env: {env}")

    only = set(args.only or [])
    device = select_device("cpu")
    dtype = default_dtype_for_device(device)

    ok = 0
    fail = 0
    for spec in iter_enabled_image_model_specs():
        if only and spec.key not in only:
            continue
        print(f"[prefetch] {spec.key} ({spec.loader})")
        try:
            with managed_extractor(spec, device=device, dtype=dtype, weights_dir=args.weights_dir):
                pass
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"  -> FAIL: {exc}")

    print(f"Done. success={ok}, failed={fail}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
