from __future__ import annotations

import os
from pathlib import Path


def configure_cache_env(weights_dir: str | Path) -> dict[str, str]:
    root = Path(weights_dir).resolve()
    hf_root = root / "hf"
    hf_hub = hf_root / "hub"
    hf_tx = hf_root / "transformers"
    torch_home = root / "torch"
    for path in (root, hf_root, hf_hub, hf_tx, torch_home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        "HF_HOME": str(hf_root),
        "HUGGINGFACE_HUB_CACHE": str(hf_hub),
        "TRANSFORMERS_CACHE": str(hf_tx),
        "TORCH_HOME": str(torch_home),
    }
    os.environ.update(env)
    return env
