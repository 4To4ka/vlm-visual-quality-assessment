from __future__ import annotations

import inspect
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as TVT

from quality_backbones.cache import configure_cache_env
from quality_backbones.manifest import ModelSpec, get_model_spec, iter_enabled_image_model_specs
from quality_backbones.training import (
    EmbeddingTrainingModule,
    RegressionImagePathDataset,
    _devices_arg,
    _json_default,
    _make_run_dir,
    _normalize_limit_batches,
    _resolve_precision,
    prepare_image_arrays,
)

try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
except Exception as exc:  # pragma: no cover - import-time guard
    raise RuntimeError(
        "Missing `lightning` dependency. Install with: "
        "conda run -n encoders python -m pip install 'lightning>=2.4,<3'"
    ) from exc


_SUPPORTED_TRANSFORMER_LOADERS = {
    "timm_vit_blocks",
    "hf_auto_image",
    "hf_auto_image_remote",
    "hf_swin_vision",
    "hf_clip_vision",
    "hf_siglip_vision",
    "hf_siglip2_vision",
}

_NON_TRANSFORMER_FAMILIES = {
    "ResNet",
    "ResNeXt",
    "ConvNeXt",
    "FastViT",
    "VGG",
    "FastViTHD",
    "ARNIQA",
    "MANIQA",
}

_SWIN_TRANSFORMER_CLS: Any | None = None


def is_supported_transformer_encoder(spec: ModelSpec) -> bool:
    if spec.loader not in _SUPPORTED_TRANSFORMER_LOADERS:
        return False
    if spec.family in _NON_TRANSFORMER_FAMILIES:
        return False
    if spec.key.startswith("fastvithd_"):
        return False
    return True


def list_supported_transformer_encoder_keys() -> list[str]:
    keys: list[str] = []
    for spec in iter_enabled_image_model_specs():
        if is_supported_transformer_encoder(spec):
            keys.append(spec.key)
    return sorted(keys)


def _parse_int_csv(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    values: list[int] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        values.append(int(item))
    return values or None


def _as_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return int(value)
    return None


def _extract_patch_size(model: nn.Module) -> int | None:
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None:
        patch_size = getattr(patch_embed, "patch_size", None)
        if isinstance(patch_size, tuple) and patch_size:
            return _as_int(patch_size[0])
        if isinstance(patch_size, list) and patch_size:
            return _as_int(patch_size[0])
        out = _as_int(patch_size)
        if out is not None:
            return out

    config = getattr(model, "config", None)
    if config is None:
        return None

    for owner in (config, getattr(config, "vision_config", None)):
        if owner is None:
            continue
        value = getattr(owner, "patch_size", None)
        if isinstance(value, tuple) and value:
            out = _as_int(value[0])
        elif isinstance(value, list) and value:
            out = _as_int(value[0])
        else:
            out = _as_int(value)
        if out is not None:
            return out
    return None


def _extract_num_prefix_tokens(model: nn.Module) -> int | None:
    value = getattr(model, "num_prefix_tokens", None)
    if isinstance(value, int) and value >= 0:
        return int(value)

    config = getattr(model, "config", None)
    for owner in (config, getattr(config, "vision_config", None) if config is not None else None):
        if owner is None:
            continue
        for attr in ("num_prefix_tokens", "num_extra_tokens", "num_register_tokens"):
            maybe = getattr(owner, attr, None)
            if isinstance(maybe, int) and maybe >= 0:
                return int(maybe)
    return None


def _resolve_processor_image_size(processor: Any, default: int = 224) -> int:
    for attr in ("size", "crop_size"):
        value = getattr(processor, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)
        if isinstance(value, dict):
            for key in ("shortest_edge", "height", "width"):
                item = value.get(key)
                if isinstance(item, int) and item > 0:
                    return int(item)
    return int(default)


def _resolve_timm_image_size(cfg: dict[str, Any]) -> int:
    input_size = cfg.get("input_size")
    if isinstance(input_size, (tuple, list)) and len(input_size) >= 3:
        h = input_size[-2]
        if isinstance(h, int) and h > 0:
            return int(h)
    return 224


def _normalize_indices(indices: list[int], num_layers: int) -> list[int]:
    out: list[int] = []
    for index in indices:
        norm = index if index >= 0 else num_layers + index
        if norm < 0 or norm >= num_layers:
            raise IndexError(f"Layer index {index} is out of bounds for {num_layers} layers")
        out.append(norm)
    return out


def _factor_pair_closest_to_square(value: int) -> tuple[int, int]:
    if value <= 0:
        raise ValueError(f"value must be > 0, got {value}")
    root = int(math.sqrt(value))
    for h in range(root, 0, -1):
        if value % h == 0:
            return h, value // h
    return 1, value


def _is_square(value: int) -> bool:
    if value <= 0:
        return False
    root = int(math.sqrt(value))
    return root * root == value


def _resolve_window_size(preferred: int, h: int, w: int) -> int:
    start = max(1, min(preferred, h, w))
    for candidate in range(start, 0, -1):
        if h % candidate == 0 and w % candidate == 0:
            return candidate
    return 1


def _sanitize_heads(num_heads: list[int], channels: int) -> list[int]:
    out: list[int] = []
    for requested in num_heads:
        head = max(1, int(requested))
        head = min(head, channels)
        while head > 1 and channels % head != 0:
            head -= 1
        out.append(max(1, head))
    return out


def _ensure_maniqa_repo(weights_dir: Path) -> Path:
    repo_dir = weights_dir / "github" / "MANIQA"
    if repo_dir.exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        if repo_dir.exists():
            return repo_dir
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/IIGROUP/MANIQA", str(repo_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return repo_dir
        except Exception:
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    return repo_dir


def _patch_timm_layers_compat() -> None:
    import timm

    timm_models_layers = getattr(timm.models, "layers", None)
    timm_layers = getattr(timm, "layers", None)
    if timm_models_layers is None or timm_layers is None:
        return

    for name in ("DropPath", "to_2tuple", "trunc_normal_"):
        if hasattr(timm_models_layers, name):
            continue
        if hasattr(timm_layers, name):
            setattr(timm_models_layers, name, getattr(timm_layers, name))


def _load_swin_transformer_cls(weights_dir: Path):
    global _SWIN_TRANSFORMER_CLS
    if _SWIN_TRANSFORMER_CLS is not None:
        return _SWIN_TRANSFORMER_CLS

    repo_dir = _ensure_maniqa_repo(weights_dir)
    _patch_timm_layers_compat()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from models.swin import SwinTransformer  # type: ignore

    _SWIN_TRANSFORMER_CLS = SwinTransformer
    return _SWIN_TRANSFORMER_CLS


@dataclass(frozen=True)
class AdapterOutput:
    feature_maps: list[torch.Tensor]
    layer_indices: list[int]


class TABlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.1) -> None:
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape_as(x)
        x = self.proj_drop(x)
        return x + residual


class MANIQAHead(nn.Module):
    def __init__(
        self,
        *,
        swin_transformer_cls: Any,
        fused_channels: int,
        embed_dim: int,
        num_outputs: int,
        drop: float,
        depths: list[int],
        window_size: int,
        dim_mlp: int,
        num_heads: list[int],
        grid_size: tuple[int, int],
        num_tab: int,
        scale: float,
    ) -> None:
        super().__init__()
        h, w = grid_size
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid grid size: {grid_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")

        self.grid_size = (int(h), int(w))
        self.num_tokens = int(h * w)
        self.embed_dim = int(embed_dim)
        self.embed_dim_stage2 = max(1, int(embed_dim // 2))

        heads_stage1 = _sanitize_heads(num_heads, self.embed_dim)
        heads_stage2 = _sanitize_heads(num_heads, self.embed_dim_stage2)

        self.tablock1 = nn.ModuleList([TABlock(self.num_tokens, drop=drop) for _ in range(max(0, num_tab))])
        self.conv1 = nn.Conv2d(fused_channels, self.embed_dim, kernel_size=1, stride=1, padding=0)
        self.swintransformer1 = swin_transformer_cls(
            patches_resolution=self.grid_size,
            depths=depths,
            num_heads=heads_stage1,
            embed_dim=self.embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale,
        )

        self.tablock2 = nn.ModuleList([TABlock(self.num_tokens, drop=drop) for _ in range(max(0, num_tab))])
        self.conv2 = nn.Conv2d(self.embed_dim, self.embed_dim_stage2, kernel_size=1, stride=1, padding=0)
        self.swintransformer2 = swin_transformer_cls(
            patches_resolution=self.grid_size,
            depths=depths,
            num_heads=heads_stage2,
            embed_dim=self.embed_dim_stage2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale,
        )

        self.fc_score = nn.Sequential(
            nn.Linear(self.embed_dim_stage2, self.embed_dim_stage2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.embed_dim_stage2, num_outputs),
            nn.ReLU(),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(self.embed_dim_stage2, self.embed_dim_stage2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.embed_dim_stage2, num_outputs),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim != 4:
            raise ValueError(f"Expected fused map rank=4, got {tuple(fused.shape)}")

        b, c, h, w = fused.shape
        if (h, w) != self.grid_size:
            raise ValueError(f"Expected grid {self.grid_size}, got {(h, w)}")

        x = fused.reshape(b, c, h * w)
        for tab in self.tablock1:
            x = tab(x)
        x = x.reshape(b, c, h, w)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        b2, c2, h2, w2 = x.shape
        x = x.reshape(b2, c2, h2 * w2)
        for tab in self.tablock2:
            x = tab(x)
        x = x.reshape(b2, c2, h2, w2)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        tokens = x.flatten(2).transpose(1, 2)
        score_per_patch = self.fc_score(tokens).squeeze(-1)
        weight_per_patch = self.fc_weight(tokens).squeeze(-1)
        weighted = (score_per_patch * weight_per_patch).sum(dim=1)
        norm = weight_per_patch.sum(dim=1).clamp_min(1e-8)
        return weighted / norm


class MANIQATransformerEncoderAdapter(nn.Module):
    def __init__(
        self,
        *,
        model_key: str,
        weights_dir: Path,
        num_feature_layers: int,
        layer_indices: list[int] | None,
    ) -> None:
        super().__init__()
        spec = get_model_spec(model_key)
        if not is_supported_transformer_encoder(spec):
            supported = ", ".join(list_supported_transformer_encoder_keys()[:12])
            raise ValueError(
                f"Model {model_key!r} is not a supported transformer encoder for MANIQA training. "
                f"Examples: {supported}"
            )

        self.spec = spec
        self.loader = spec.loader
        self.weights_dir = Path(weights_dir)
        self.hf_cache_dir = self.weights_dir / "hf"
        self.num_feature_layers = int(num_feature_layers)
        if self.num_feature_layers < 1:
            raise ValueError("num_feature_layers must be >= 1")
        self.explicit_layer_indices = list(layer_indices) if layer_indices is not None else None

        self.model: nn.Module
        self.processor: Any | None = None
        self.train_transform: Any | None = None
        self.eval_transform: Any | None = None
        self._encoder_module: nn.Module

        self.patch_size_hint: int | None = None
        self.prefix_tokens_hint: int | None = None
        self.image_size: int = 224

        self._captured_blocks: list[torch.Tensor] = []
        self._hooks: list[Any] = []
        self._last_layer_indices: list[int] = []

        if self.loader == "timm_vit_blocks":
            self._init_timm_vit_blocks()
        elif self.loader in {"hf_auto_image", "hf_swin_vision"}:
            self._init_hf_auto_image(trust_remote_code=False)
        elif self.loader == "hf_auto_image_remote":
            self._init_hf_auto_image_remote()
        elif self.loader == "hf_clip_vision":
            self._init_hf_clip_vision()
        elif self.loader == "hf_siglip_vision":
            self._init_hf_siglip_vision()
        elif self.loader == "hf_siglip2_vision":
            self._init_hf_siglip2_vision()
        else:
            raise ValueError(f"Unsupported loader for MANIQA adapter: {self.loader}")

        if self.loader != "timm_vit_blocks" and self.processor is not None:
            self.train_transform = TVT.Compose(
                [
                    TVT.RandomResizedCrop(self.image_size, scale=(0.8, 1.0), interpolation=TVT.InterpolationMode.BICUBIC),
                    TVT.RandomHorizontalFlip(),
                ]
            )
            self.eval_transform = TVT.Compose(
                [
                    TVT.Resize(int(round(self.image_size * 1.15)), interpolation=TVT.InterpolationMode.BICUBIC),
                    TVT.CenterCrop(self.image_size),
                ]
            )

    @property
    def last_layer_indices(self) -> list[int]:
        return list(self._last_layer_indices)

    def _init_timm_vit_blocks(self) -> None:
        import timm

        if self.spec.timm_name is None:
            raise RuntimeError(f"Model {self.spec.key} has no timm_name")

        self.model = timm.create_model(self.spec.timm_name, pretrained=True)
        cfg = timm.data.resolve_model_data_config(self.model)
        self.train_transform = timm.data.create_transform(**cfg, is_training=True)
        self.eval_transform = timm.data.create_transform(**cfg, is_training=False)
        self.image_size = _resolve_timm_image_size(cfg)
        self.patch_size_hint = _extract_patch_size(self.model)
        self.prefix_tokens_hint = _extract_num_prefix_tokens(self.model)

        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError(f"Model {self.spec.timm_name} has no .blocks for ViT hidden states")
        for block in blocks:
            self._hooks.append(block.register_forward_hook(self._block_hook))
        self._encoder_module = self.model

    def _init_hf_auto_image(self, *, trust_remote_code: bool) -> None:
        from transformers import AutoImageProcessor, AutoModel

        if self.spec.pretrained_id is None:
            raise RuntimeError(f"Model {self.spec.key} has no pretrained_id")

        self.processor = AutoImageProcessor.from_pretrained(
            self.spec.pretrained_id,
            cache_dir=str(self.hf_cache_dir),
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            self.spec.pretrained_id,
            cache_dir=str(self.hf_cache_dir),
            trust_remote_code=trust_remote_code,
        )
        self.image_size = _resolve_processor_image_size(self.processor, default=224)
        self.patch_size_hint = _extract_patch_size(self.model)
        self.prefix_tokens_hint = _extract_num_prefix_tokens(self.model)
        self._encoder_module = self.model

    def _init_hf_auto_image_remote(self) -> None:
        if self.spec.pretrained_id is None:
            raise RuntimeError(f"Model {self.spec.key} has no pretrained_id")

        from quality_backbones.extractors import InternViTExtractor

        extractor = InternViTExtractor(
            self.spec,
            device=torch.device("cpu"),
            dtype=torch.float32,
            cache_dir=self.hf_cache_dir,
        )
        self.processor = extractor.processor
        self.model = extractor.model
        self.image_size = _resolve_processor_image_size(self.processor, default=448)
        self.patch_size_hint = _extract_patch_size(self.model)
        self.prefix_tokens_hint = _extract_num_prefix_tokens(self.model)
        self._encoder_module = self.model

    def _init_hf_clip_vision(self) -> None:
        from transformers import AutoProcessor, CLIPVisionModel

        if self.spec.pretrained_id is None:
            raise RuntimeError(f"Model {self.spec.key} has no pretrained_id")

        self.processor = AutoProcessor.from_pretrained(self.spec.pretrained_id, cache_dir=str(self.hf_cache_dir))
        self.model = CLIPVisionModel.from_pretrained(
            self.spec.pretrained_id,
            cache_dir=str(self.hf_cache_dir),
        )
        self.image_size = _resolve_processor_image_size(self.processor, default=224)
        self.patch_size_hint = _extract_patch_size(self.model)
        self.prefix_tokens_hint = _extract_num_prefix_tokens(self.model)
        self._encoder_module = self.model

    def _init_hf_siglip_vision(self) -> None:
        from transformers import AutoImageProcessor, SiglipVisionModel

        if self.spec.pretrained_id is None:
            raise RuntimeError(f"Model {self.spec.key} has no pretrained_id")

        self.processor = AutoImageProcessor.from_pretrained(self.spec.pretrained_id, cache_dir=str(self.hf_cache_dir))
        self.model = SiglipVisionModel.from_pretrained(
            self.spec.pretrained_id,
            cache_dir=str(self.hf_cache_dir),
        )
        self.image_size = _resolve_processor_image_size(self.processor, default=224)
        self.patch_size_hint = _extract_patch_size(self.model)
        self.prefix_tokens_hint = _extract_num_prefix_tokens(self.model)
        self._encoder_module = self.model

    def _init_hf_siglip2_vision(self) -> None:
        from transformers import AutoModel, AutoProcessor

        if self.spec.pretrained_id is None:
            raise RuntimeError(f"Model {self.spec.key} has no pretrained_id")

        self.processor = AutoProcessor.from_pretrained(self.spec.pretrained_id, cache_dir=str(self.hf_cache_dir))
        full_model = AutoModel.from_pretrained(
            self.spec.pretrained_id,
            cache_dir=str(self.hf_cache_dir),
        )
        vision_model = getattr(full_model, "vision_model", None)
        if vision_model is None:
            raise RuntimeError("SigLIP2 model has no vision_model")
        self.model = vision_model
        self.image_size = _resolve_processor_image_size(self.processor, default=224)
        self.patch_size_hint = _extract_patch_size(self.model)
        self.prefix_tokens_hint = _extract_num_prefix_tokens(self.model)
        self._encoder_module = self.model

    def encoder_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self._encoder_module.parameters()

    def _block_hook(self, _module: nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, torch.Tensor):
            self._captured_blocks.append(out)

    @staticmethod
    def _load_images(paths: list[str]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        return images

    def preprocess_paths(self, paths: list[str], *, training: bool) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.loader == "timm_vit_blocks":
            transform = self.train_transform if training else self.eval_transform
            if transform is None:
                raise RuntimeError("timm transform is not initialized")
            tensors: list[torch.Tensor] = []
            for image in self._load_images(paths):
                tensors.append(transform(image))
            return torch.stack(tensors, dim=0)

        if self.processor is None:
            raise RuntimeError("HF adapter is missing processor")

        pil_images = self._load_images(paths)
        transform = self.train_transform if training else self.eval_transform
        if transform is not None:
            pil_images = [transform(image) for image in pil_images]

        batch = self.processor(images=pil_images, return_tensors="pt")
        tensor_inputs: dict[str, torch.Tensor] = {
            key: value for key, value in batch.items() if isinstance(value, torch.Tensor)
        }

        if self.loader == "hf_siglip2_vision":
            if "spatial_shapes" not in tensor_inputs and "pixel_values" in tensor_inputs:
                pixel_values = tensor_inputs["pixel_values"]
                patch = self.patch_size_hint or 16
                h = int(pixel_values.shape[-2] // patch)
                w = int(pixel_values.shape[-1] // patch)
                tensor_inputs["spatial_shapes"] = torch.tensor(
                    [[h, w]] * pixel_values.shape[0],
                    dtype=torch.long,
                )
                if "pixel_attention_mask" not in tensor_inputs:
                    tensor_inputs["pixel_attention_mask"] = torch.ones(
                        (pixel_values.shape[0], h * w),
                        dtype=torch.bool,
                    )

        return tensor_inputs

    def _resolve_auto_indices(self, num_layers: int) -> list[int]:
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.explicit_layer_indices is not None:
            resolved = _normalize_indices(self.explicit_layer_indices, num_layers)
            while len(resolved) < self.num_feature_layers:
                resolved.append(resolved[-1])
            return resolved[: self.num_feature_layers]

        if num_layers >= self.num_feature_layers:
            start = max(0, num_layers // 2)
            sampled = np.linspace(start, num_layers - 1, num=self.num_feature_layers)
            resolved = [int(round(item)) for item in sampled.tolist()]
            resolved = _normalize_indices(resolved, num_layers)
            while len(resolved) < self.num_feature_layers:
                resolved.append(num_layers - 1)
            return resolved

        resolved = list(range(num_layers))
        while len(resolved) < self.num_feature_layers:
            resolved.append(num_layers - 1)
        return resolved

    @staticmethod
    def _coerce_map(hidden: torch.Tensor, *, input_h: int, input_w: int, patch_size_hint: int | None, prefix_tokens_hint: int | None) -> torch.Tensor:
        if hidden.ndim == 4:
            # Some backbones emit BHWC.
            if hidden.shape[1] <= 4 and hidden.shape[-1] > 4:
                return hidden.permute(0, 3, 1, 2).contiguous()
            return hidden
        if hidden.ndim != 3:
            raise ValueError(f"Unsupported hidden state shape: {tuple(hidden.shape)}")

        b, t, c = hidden.shape
        candidates: list[int] = []
        if prefix_tokens_hint is not None:
            candidates.append(int(prefix_tokens_hint))
        candidates.extend([0, 1, 2, 3])

        prefix = 0
        seen: set[int] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate >= t:
                continue
            patch_tokens = t - candidate
            if patch_tokens <= 0:
                continue
            if patch_size_hint is not None:
                expected = (input_h // patch_size_hint) * (input_w // patch_size_hint)
                if expected == patch_tokens:
                    prefix = candidate
                    break
            if _is_square(patch_tokens):
                prefix = candidate
                break

        if prefix > 0:
            hidden = hidden[:, prefix:]
            t = int(hidden.shape[1])

        if t <= 0:
            raise ValueError("No tokens left after prefix stripping")

        if patch_size_hint is not None:
            expected_h = max(1, input_h // patch_size_hint)
            expected_w = max(1, input_w // patch_size_hint)
            if expected_h * expected_w == t:
                h, w = expected_h, expected_w
            else:
                h, w = _factor_pair_closest_to_square(t)
        else:
            h, w = _factor_pair_closest_to_square(t)

        take = h * w
        if take != t:
            hidden = hidden[:, :take]
        return hidden.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _align_maps(feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        if not feature_maps:
            raise ValueError("feature_maps must not be empty")
        target = max(feature_maps, key=lambda item: int(item.shape[-2]) * int(item.shape[-1]))
        target_h, target_w = int(target.shape[-2]), int(target.shape[-1])

        aligned: list[torch.Tensor] = []
        for fmap in feature_maps:
            if int(fmap.shape[-2]) == target_h and int(fmap.shape[-1]) == target_w:
                aligned.append(fmap)
            else:
                aligned.append(
                    F.interpolate(
                        fmap,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
        return aligned

    def _forward_timm_hidden_states(self, pixel_values: torch.Tensor) -> tuple[list[torch.Tensor], int, int]:
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values rank=4, got {tuple(pixel_values.shape)}")
        self._captured_blocks = []
        _ = self.model.forward_features(pixel_values)
        if not self._captured_blocks:
            raise RuntimeError(f"Model {self.spec.key} produced no captured block outputs")
        return list(self._captured_blocks), int(pixel_values.shape[-2]), int(pixel_values.shape[-1])

    def _forward_hf_hidden_states(self, model_inputs: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], int, int]:
        pixel_values = model_inputs.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor):
            raise RuntimeError("model_inputs must include tensor `pixel_values`")

        if self.loader == "hf_siglip2_vision":
            inputs = dict(model_inputs)
            forward_params = set(inspect.signature(self.model.forward).parameters.keys())
            forward_kwargs: dict[str, Any] = {"output_hidden_states": True}
            if "return_dict" in forward_params:
                forward_kwargs["return_dict"] = True
            try:
                outputs = self.model(**inputs, **forward_kwargs)
            except TypeError:
                alt_inputs = dict(inputs)
                if "pixel_attention_mask" in alt_inputs and "attention_mask" not in alt_inputs:
                    alt_inputs["attention_mask"] = alt_inputs.pop("pixel_attention_mask")
                outputs = self.model(**alt_inputs, **forward_kwargs)
        else:
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if not isinstance(last_hidden, torch.Tensor):
                raise RuntimeError("Encoder returned no hidden_states and no last_hidden_state")
            hidden_states = (last_hidden,)

        return [item for item in hidden_states if isinstance(item, torch.Tensor)], int(pixel_values.shape[-2]), int(pixel_values.shape[-1])

    def forward(self, model_inputs: torch.Tensor | dict[str, torch.Tensor]) -> AdapterOutput:
        if self.loader == "timm_vit_blocks":
            if not isinstance(model_inputs, torch.Tensor):
                raise ValueError("timm adapter expects tensor model_inputs")
            hidden_states, input_h, input_w = self._forward_timm_hidden_states(model_inputs)
        else:
            if not isinstance(model_inputs, dict):
                raise ValueError("HF adapter expects dict model_inputs")
            hidden_states, input_h, input_w = self._forward_hf_hidden_states(model_inputs)

        if not hidden_states:
            raise RuntimeError("Encoder produced no hidden states")

        layer_indices = self._resolve_auto_indices(len(hidden_states))
        self._last_layer_indices = list(layer_indices)
        selected = [hidden_states[idx] for idx in layer_indices]

        maps = [
            self._coerce_map(
                hidden,
                input_h=input_h,
                input_w=input_w,
                patch_size_hint=self.patch_size_hint,
                prefix_tokens_hint=self.prefix_tokens_hint,
            )
            for hidden in selected
        ]
        maps = self._align_maps(maps)
        return AdapterOutput(feature_maps=maps, layer_indices=layer_indices)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class MANIQABasedScorer(nn.Module):
    def __init__(
        self,
        *,
        adapter: MANIQATransformerEncoderAdapter,
        weights_dir: Path,
        embed_dim: int,
        drop: float,
        depths: list[int],
        window_size: int,
        dim_mlp: int,
        num_heads: list[int],
        num_tab: int,
        scale: float,
        num_outputs: int,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.weights_dir = Path(weights_dir)
        self.embed_dim = int(embed_dim)
        self.drop = float(drop)
        self.depths = [int(item) for item in depths]
        self.window_size = int(window_size)
        self.dim_mlp = int(dim_mlp)
        self.num_heads = [int(item) for item in num_heads]
        self.num_tab = int(num_tab)
        self.scale = float(scale)
        self.num_outputs = int(num_outputs)

        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if not self.depths:
            raise ValueError("depths must not be empty")
        if not self.num_heads:
            raise ValueError("num_heads must not be empty")
        if len(self.num_heads) != len(self.depths):
            raise ValueError("num_heads must have same length as depths")

        self.projections: nn.ModuleList | None = None
        self.head: MANIQAHead | None = None
        self.feature_grid: tuple[int, int] | None = None
        self.feature_in_channels: list[int] = []
        self.selected_layer_indices: list[int] = []

    def encoder_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.adapter.encoder_parameters()

    def head_parameters(self) -> Iterable[torch.nn.Parameter]:
        if self.projections is not None:
            for module in self.projections:
                yield from module.parameters()
        if self.head is not None:
            yield from self.head.parameters()

    def _build_projectors(self, feature_maps: list[torch.Tensor]) -> None:
        channels = [int(item.shape[1]) for item in feature_maps]
        self.feature_in_channels = list(channels)
        self.projections = nn.ModuleList(
            [nn.Conv2d(ch, self.embed_dim, kernel_size=1, stride=1, padding=0) for ch in channels]
        )

    def _build_head(self, fused: torch.Tensor) -> None:
        if fused.ndim != 4:
            raise ValueError(f"Expected fused map rank=4, got {tuple(fused.shape)}")
        _, _, h, w = fused.shape
        self.feature_grid = (int(h), int(w))
        effective_window = _resolve_window_size(self.window_size, h=int(h), w=int(w))
        swin_cls = _load_swin_transformer_cls(self.weights_dir)
        self.head = MANIQAHead(
            swin_transformer_cls=swin_cls,
            fused_channels=int(fused.shape[1]),
            embed_dim=self.embed_dim,
            num_outputs=self.num_outputs,
            drop=self.drop,
            depths=self.depths,
            window_size=effective_window,
            dim_mlp=self.dim_mlp,
            num_heads=self.num_heads,
            grid_size=(int(h), int(w)),
            num_tab=self.num_tab,
            scale=self.scale,
        )

    def forward(self, model_inputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        adapter_out = self.adapter(model_inputs)
        feature_maps = adapter_out.feature_maps
        self.selected_layer_indices = list(adapter_out.layer_indices)

        if self.projections is None:
            self._build_projectors(feature_maps)
        if self.projections is None:
            raise RuntimeError("Failed to initialize projection layers")
        if len(self.projections) != len(feature_maps):
            raise RuntimeError(
                f"Projection count mismatch: {len(self.projections)} vs {len(feature_maps)} feature maps"
            )

        projected = [
            proj(feature_map)
            for proj, feature_map in zip(self.projections, feature_maps, strict=False)
        ]
        fused = torch.cat(projected, dim=1)

        if self.head is None:
            self._build_head(fused)
        if self.head is None:
            raise RuntimeError("Failed to initialize MANIQA head")
        return self.head(fused)


def _collate_regression_images(
    batch: list[tuple[str, float]],
    *,
    adapter: MANIQATransformerEncoderAdapter,
    training: bool,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor]:
    paths, targets = zip(*batch, strict=False)
    x = adapter.preprocess_paths([str(path) for path in paths], training=training)
    y = torch.tensor(np.asarray(targets, dtype=np.float32), dtype=torch.float32)
    return x, y


def _build_maniqa_dataloaders(
    args: Any,
    arrays: Any,
    adapter: MANIQATransformerEncoderAdapter,
) -> tuple[DataLoader, DataLoader]:
    num_workers = int(args.num_workers)
    if num_workers > 0:
        print("[maniqa] forcing num_workers=0 for in-process image preprocessing")
        num_workers = 0

    pin_memory = args.accelerator != "cpu" and torch.cuda.is_available()
    train_ds = RegressionImagePathDataset(paths=arrays.train_paths, targets=arrays.train_y)
    val_ds = RegressionImagePathDataset(paths=arrays.val_paths, targets=arrays.val_y)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=lambda batch: _collate_regression_images(batch, adapter=adapter, training=True),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=lambda batch: _collate_regression_images(batch, adapter=adapter, training=False),
    )
    return train_loader, val_loader


def _initialize_lazy_modules(
    scorer: MANIQABasedScorer,
    adapter: MANIQATransformerEncoderAdapter,
    arrays: Any,
) -> None:
    if arrays.train_paths.shape[0] == 0:
        raise ValueError("No training paths available for lazy module initialization")
    sample = adapter.preprocess_paths([str(arrays.train_paths[0])], training=False)
    was_training = scorer.training
    scorer.eval()
    with torch.no_grad():
        _ = scorer(sample)
    if was_training:
        scorer.train()


def _param_count(module: nn.Module, *, trainable_only: bool) -> int:
    if trainable_only:
        return int(sum(param.numel() for param in module.parameters() if param.requires_grad))
    return int(sum(param.numel() for param in module.parameters()))


def run_maniqa_training(args: Any) -> Path:
    if not getattr(args, "encoder_model", None):
        raise ValueError("--encoder-model is required")

    spec = get_model_spec(args.encoder_model)
    if not is_supported_transformer_encoder(spec):
        supported = list_supported_transformer_encoder_keys()
        raise ValueError(
            f"Encoder {args.encoder_model!r} is unsupported. "
            f"Supported transformer encoders (excluding fastvithd_*): {supported}"
        )

    configure_cache_env(args.weights_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(args.output_dir, args.run_name, "maniqa")

    L.seed_everything(args.seed, workers=True)
    arrays = prepare_image_arrays(args)

    adapter = MANIQATransformerEncoderAdapter(
        model_key=args.encoder_model,
        weights_dir=args.weights_dir,
        num_feature_layers=args.num_feature_layers,
        layer_indices=_parse_int_csv(args.encoder_layer_indices),
    )

    scorer = MANIQABasedScorer(
        adapter=adapter,
        weights_dir=args.weights_dir,
        embed_dim=args.maniqa_embed_dim,
        drop=args.maniqa_dropout,
        depths=[int(item) for item in str(args.maniqa_depths).split(",") if item.strip()],
        window_size=args.maniqa_window_size,
        dim_mlp=args.maniqa_dim_mlp,
        num_heads=[int(item) for item in str(args.maniqa_num_heads).split(",") if item.strip()],
        num_tab=args.maniqa_num_tab,
        scale=args.maniqa_scale,
        num_outputs=1,
    )

    if bool(getattr(args, "freeze_encoder", False)):
        for param in scorer.encoder_parameters():
            param.requires_grad = False

    train_loader, val_loader = _build_maniqa_dataloaders(args, arrays, adapter)
    _initialize_lazy_modules(scorer, adapter, arrays)

    encoder_params = [param for param in scorer.encoder_parameters() if param.requires_grad]
    head_params = [param for param in scorer.head_parameters() if param.requires_grad]
    if not head_params:
        raise RuntimeError("MANIQA head has no trainable parameters")

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

    limit_train_batches = _normalize_limit_batches(args.limit_train_batches, len(train_loader))
    limit_val_batches = _normalize_limit_batches(args.limit_val_batches, len(val_loader))
    tensorboard_enabled = bool(getattr(args, "tensorboard", True))
    save_epoch_metrics = bool(getattr(args, "save_epoch_metrics", True))
    save_val_predictions = bool(getattr(args, "save_val_predictions", True))
    val_predictions_max_rows = int(getattr(args, "val_predictions_max_rows", 0) or 0)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_mode": "maniqa",
        "task": "regression",
        "target_field": args.target_field,
        "group_field": args.group_field,
        "split_policy": args.split_policy,
        "val_ratio": args.val_ratio,
        "encoder_model": args.encoder_model,
        "encoder_family": spec.family,
        "encoder_loader": spec.loader,
        "encoder_source": spec.source,
        "encoder_pretrained_id": spec.pretrained_id,
        "encoder_timm_name": spec.timm_name,
        "num_feature_layers": int(args.num_feature_layers),
        "encoder_layer_indices_arg": str(args.encoder_layer_indices or ""),
        "resolved_layer_indices": list(scorer.selected_layer_indices),
        "feature_grid": list(scorer.feature_grid) if scorer.feature_grid is not None else None,
        "feature_in_channels": list(scorer.feature_in_channels),
        "maniqa": {
            "embed_dim": int(args.maniqa_embed_dim),
            "dropout": float(args.maniqa_dropout),
            "depths": [int(item) for item in str(args.maniqa_depths).split(",") if item.strip()],
            "num_heads": [int(item) for item in str(args.maniqa_num_heads).split(",") if item.strip()],
            "window_size": int(args.maniqa_window_size),
            "dim_mlp": int(args.maniqa_dim_mlp),
            "num_tab": int(args.maniqa_num_tab),
            "scale": float(args.maniqa_scale),
        },
        "freeze_encoder": bool(getattr(args, "freeze_encoder", False)),
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
            "encoder_total": int(sum(param.numel() for param in scorer.encoder_parameters())),
            "encoder_trainable": int(sum(param.numel() for param in scorer.encoder_parameters() if param.requires_grad)),
            "head_total": int(sum(param.numel() for param in scorer.head_parameters())),
            "head_trainable": int(sum(param.numel() for param in scorer.head_parameters() if param.requires_grad)),
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
        adapter.close()
        print(f"[dry-run] Prepared MANIQA data and config at: {run_dir}")
        return run_dir

    try:
        module = EmbeddingTrainingModule(
            scorer=scorer,
            task="regression",
            lr=args.lr,
            weight_decay=args.weight_decay,
            pairwise_target="hard",
            listwise_target="hard",
            target_temperature=1.0,
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
            callbacks=[checkpoint_cb],
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
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Training artifacts saved to: {run_dir}")
        return run_dir
    finally:
        adapter.close()
