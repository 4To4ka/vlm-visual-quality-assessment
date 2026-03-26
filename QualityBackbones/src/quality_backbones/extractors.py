from __future__ import annotations

import contextlib
import inspect
import json
import shutil
import time
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from quality_backbones.manifest import ModelSpec


_HF_VIT_DISABLE_POOLER_KEYS = {
    "vit_base",
    "vit_large",
    "vit_huge",
    "dino_vits8",
    "dino_vits16",
    "dino_vitb8",
    "dino_vitb16",
}


def select_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_dtype_for_device(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _pool_hidden_tensor(
    t: torch.Tensor,
    policy: str = "auto",
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    if t.ndim == 2:
        return t
    if t.ndim == 3:
        if policy in {"first_token", "cls_token"}:
            return t[:, 0]
        if policy == "mean_tokens":
            return t.mean(dim=1)
        if policy == "mean_patch_tokens":
            prefix = min(max(num_prefix_tokens, 0), t.shape[1] - 1)
            patch_tokens = t[:, prefix:]
            return patch_tokens.mean(dim=1)
        return t[:, 0]
    if t.ndim == 4:
        return t.mean(dim=(-1, -2))
    raise ValueError(f"Unsupported hidden tensor shape: {tuple(t.shape)}")


def _masked_mean_tokens(tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None:
        return tokens.mean(dim=1)
    mask = attention_mask
    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[0], -1)
    if mask.shape[1] != tokens.shape[1]:
        mask = mask[:, : tokens.shape[1]]
    mask = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (tokens * mask).sum(dim=1) / denom


def _l2_normalize(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return t / t.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class ExtractResult:
    layer_names: list[str]
    per_layer_np: list[np.ndarray]


class BaseExtractor:
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        raise NotImplementedError


class HFExtractor(BaseExtractor):
    def __init__(
        self,
        spec: ModelSpec,
        device: torch.device,
        dtype: torch.dtype,
        cache_dir: Path,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "cache_dir": str(cache_dir),
        }
        if spec.key in _HF_VIT_DISABLE_POOLER_KEYS:
            model_kwargs["add_pooling_layer"] = False
        if spec.loader == "hf_auto_image_remote":
            model_kwargs["trust_remote_code"] = True

        self.processor = AutoImageProcessor.from_pretrained(
            spec.pretrained_id,
            cache_dir=str(cache_dir),
            trust_remote_code=(spec.loader == "hf_auto_image_remote"),
        )
        self.model = AutoModel.from_pretrained(spec.pretrained_id, **model_kwargs).to(device).eval()
        self.spec = spec
        self.device = device
        self.model_dtype = dtype
        self.autocast_enabled = device.type == "cuda"

    def _hidden_state_pool_policy(self) -> str:
        if self.spec.family == "I-JEPA":
            return "mean_tokens"
        return "first_token"

    def _canonical_embedding(self, outputs: Any) -> torch.Tensor:
        pooler_output = getattr(outputs, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor) and self.spec.family in {"ResNet", "ConvNeXt", "DINOv2"}:
            return _pool_hidden_tensor(pooler_output, policy="auto")

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if not isinstance(last_hidden_state, torch.Tensor):
            raise RuntimeError("Model did not return last_hidden_state")

        if self.spec.family == "I-JEPA":
            return _pool_hidden_tensor(last_hidden_state, policy="mean_tokens")
        return _pool_hidden_tensor(last_hidden_state, policy="first_token")

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with _autocast_context(self.device, self.autocast_enabled, self.model_dtype):
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if not isinstance(last_hidden_state, torch.Tensor):
                raise RuntimeError("Model did not return hidden_states or last_hidden_state")
            hidden_states = (last_hidden_state,)

        pool_policy = self._hidden_state_pool_policy()
        layer_names = [f"hidden_state_{i:03d}" for i in range(len(hidden_states))]
        per_layer = [
            _pool_hidden_tensor(h, policy=pool_policy).detach().float().cpu().numpy() for h in hidden_states
        ]
        canonical = self._canonical_embedding(outputs)
        layer_names.append("canonical_embedding")
        per_layer.append(canonical.detach().float().cpu().numpy())
        return ExtractResult(layer_names=layer_names, per_layer_np=per_layer)


class HFCLIPVisionExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype, cache_dir: Path) -> None:
        from transformers import AutoProcessor, CLIPModel, CLIPVisionModel

        self.processor = AutoProcessor.from_pretrained(spec.pretrained_id, cache_dir=str(cache_dir))
        self.vision_model = CLIPVisionModel.from_pretrained(
            spec.pretrained_id,
            torch_dtype=dtype,
            cache_dir=str(cache_dir),
        ).to(device).eval()

        clip_model = CLIPModel.from_pretrained(
            spec.pretrained_id,
            torch_dtype=dtype,
            cache_dir=str(cache_dir),
        ).to(device).eval()
        self.visual_projection = clip_model.visual_projection
        self.visual_projection.eval()
        del clip_model

        self.device = device
        self.model_dtype = dtype
        self.autocast_enabled = device.type == "cuda"

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with _autocast_context(self.device, self.autocast_enabled, self.model_dtype):
            outputs = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            hidden_states = (outputs.last_hidden_state,)
        layer_names = [f"hidden_state_{i:03d}" for i in range(len(hidden_states))]
        per_layer = [_pool_hidden_tensor(h, policy="first_token").detach().float().cpu().numpy() for h in hidden_states]

        pooler_output = outputs.pooler_output
        pooler_output = pooler_output.to(dtype=self.visual_projection.weight.dtype)
        image_embeds = self.visual_projection(pooler_output)
        image_embeds_l2 = _l2_normalize(image_embeds)
        layer_names.extend(["pooler_output", "image_embeds", "image_embeds_l2"])
        per_layer.extend(
            [
                pooler_output.detach().float().cpu().numpy(),
                image_embeds.detach().float().cpu().numpy(),
                image_embeds_l2.detach().float().cpu().numpy(),
            ]
        )
        return ExtractResult(layer_names=layer_names, per_layer_np=per_layer)


class HFSwinVisionExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype, cache_dir: Path) -> None:
        from transformers import AutoImageProcessor, SwinModel, Swinv2Model

        if spec.pretrained_id is None:
            raise RuntimeError("Swin spec must include pretrained_id")

        self.processor = AutoImageProcessor.from_pretrained(spec.pretrained_id, cache_dir=str(cache_dir))
        if spec.family == "Swin":
            model = SwinModel.from_pretrained(
                spec.pretrained_id,
                torch_dtype=dtype,
                cache_dir=str(cache_dir),
            )
        elif spec.family == "SwinV2":
            model = Swinv2Model.from_pretrained(
                spec.pretrained_id,
                torch_dtype=dtype,
                cache_dir=str(cache_dir),
            )
        else:
            raise RuntimeError(f"Unsupported Swin family for extractor: {spec.family}")

        self.model = model.to(device).eval()
        self.device = device
        self.model_dtype = dtype
        self.autocast_enabled = device.type == "cuda"

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with _autocast_context(self.device, self.autocast_enabled, self.model_dtype):
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            hidden_states = (outputs.last_hidden_state,)
        layer_names = [f"hidden_state_{i:03d}" for i in range(len(hidden_states))]
        per_layer = [_pool_hidden_tensor(h, policy="mean_tokens").detach().float().cpu().numpy() for h in hidden_states]

        pooler_output = getattr(outputs, "pooler_output", None)
        if not isinstance(pooler_output, torch.Tensor):
            pooler_output = _pool_hidden_tensor(outputs.last_hidden_state, policy="mean_tokens")
        if not isinstance(pooler_output, torch.Tensor):
            raise RuntimeError("Swin model did not return a tensor pooler output")

        layer_names.append("canonical_embedding")
        per_layer.append(pooler_output.detach().float().cpu().numpy())
        return ExtractResult(layer_names=layer_names, per_layer_np=per_layer)


class HFSiglipVisionExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype, cache_dir: Path) -> None:
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.processor = AutoImageProcessor.from_pretrained(spec.pretrained_id, cache_dir=str(cache_dir))
        self.model = SiglipVisionModel.from_pretrained(
            spec.pretrained_id,
            torch_dtype=dtype,
            cache_dir=str(cache_dir),
        ).to(device).eval()
        self.device = device
        self.model_dtype = dtype
        self.autocast_enabled = device.type == "cuda"

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with _autocast_context(self.device, self.autocast_enabled, self.model_dtype):
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            hidden_states = (outputs.last_hidden_state,)

        siglip_vision = self.model.vision_model
        layer_names = [f"hidden_state_{i:03d}" for i in range(len(hidden_states))]
        per_layer: list[np.ndarray] = []
        for h in hidden_states:
            if h.ndim == 3 and getattr(siglip_vision, "head", None) is not None:
                h_for_head = h.to(dtype=siglip_vision.post_layernorm.weight.dtype)
                h_ln = siglip_vision.post_layernorm(h_for_head)
                if hasattr(siglip_vision.head, "probe"):
                    h_ln = h_ln.to(dtype=siglip_vision.head.probe.dtype)
                pooled = siglip_vision.head(h_ln)
            else:
                pooled = _pool_hidden_tensor(h, policy="mean_tokens")
            per_layer.append(pooled.detach().float().cpu().numpy())

        pooler_output = outputs.pooler_output
        if pooler_output is None:
            last_hidden = outputs.last_hidden_state
            last_hidden = last_hidden.to(dtype=siglip_vision.post_layernorm.weight.dtype)
            last_hidden = siglip_vision.post_layernorm(last_hidden)
            if hasattr(siglip_vision.head, "probe"):
                last_hidden = last_hidden.to(dtype=siglip_vision.head.probe.dtype)
            pooler_output = siglip_vision.head(last_hidden)
        image_embeds_l2 = _l2_normalize(pooler_output)
        layer_names.extend(["pooler_output", "image_embeds_l2"])
        per_layer.extend([pooler_output.detach().float().cpu().numpy(), image_embeds_l2.detach().float().cpu().numpy()])
        return ExtractResult(layer_names=layer_names, per_layer_np=per_layer)


class HFSiglip2VisionExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype, cache_dir: Path) -> None:
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(spec.pretrained_id, cache_dir=str(cache_dir))
        self.model = AutoModel.from_pretrained(
            spec.pretrained_id,
            torch_dtype=dtype,
            cache_dir=str(cache_dir),
        ).to(device).eval()
        self.model_type = str(getattr(self.model.config, "model_type", ""))
        self.device = device
        self.model_dtype = dtype
        self.autocast_enabled = device.type == "cuda"

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        inputs = self.processor(images=images, return_tensors="pt")
        vision_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        if "pixel_attention_mask" in vision_inputs and "attention_mask" not in vision_inputs:
            vision_inputs["attention_mask"] = vision_inputs["pixel_attention_mask"]

        vision_model = getattr(self.model, "vision_model", None)
        if vision_model is None:
            raise RuntimeError("Loaded SigLIP2 model has no vision_model")

        if self.model_type == "siglip2" and "spatial_shapes" not in vision_inputs and "pixel_values" in vision_inputs:
            pixel_values = vision_inputs["pixel_values"]
            if pixel_values.ndim == 4:
                patch_size = getattr(getattr(self.model.config, "vision_config", self.model.config), "patch_size", None)
                if isinstance(patch_size, int) and patch_size > 0:
                    h = pixel_values.shape[-2] // patch_size
                    w = pixel_values.shape[-1] // patch_size
                    vision_inputs["spatial_shapes"] = torch.tensor(
                        [[h, w]] * pixel_values.shape[0],
                        dtype=torch.long,
                        device=self.device,
                    )
                    if "attention_mask" not in vision_inputs:
                        vision_inputs["attention_mask"] = torch.ones(
                            (pixel_values.shape[0], h * w),
                            dtype=torch.bool,
                            device=self.device,
                        )

        vision_forward_params = set(inspect.signature(vision_model.forward).parameters.keys())
        forward_kwargs: dict[str, Any] = {"output_hidden_states": True}
        if "return_dict" in vision_forward_params:
            forward_kwargs["return_dict"] = True
        call_inputs = dict(vision_inputs)
        if "attention_mask" in call_inputs and "attention_mask" not in vision_forward_params:
            if "pixel_attention_mask" in vision_forward_params:
                call_inputs["pixel_attention_mask"] = call_inputs["attention_mask"]
            call_inputs.pop("attention_mask", None)
        if "pixel_attention_mask" in call_inputs and "pixel_attention_mask" not in vision_forward_params:
            if "attention_mask" in vision_forward_params:
                call_inputs["attention_mask"] = call_inputs["pixel_attention_mask"]
            call_inputs.pop("pixel_attention_mask", None)
        call_inputs = {k: v for k, v in call_inputs.items() if k in vision_forward_params}

        with _autocast_context(self.device, self.autocast_enabled, self.model_dtype):
            try:
                outputs = vision_model(**call_inputs, **forward_kwargs)
            except TypeError:
                alt_inputs = dict(call_inputs)
                if "attention_mask" in alt_inputs:
                    alt_inputs["pixel_attention_mask"] = alt_inputs.pop("attention_mask")
                if "pixel_attention_mask" in alt_inputs and "attention_mask" in vision_forward_params:
                    alt_inputs["attention_mask"] = alt_inputs.pop("pixel_attention_mask")
                alt_inputs = {k: v for k, v in alt_inputs.items() if k in vision_forward_params}
                outputs = vision_model(**alt_inputs, **forward_kwargs)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            hidden_states = (outputs.last_hidden_state,)

        attention_mask = call_inputs.get("attention_mask", call_inputs.get("pixel_attention_mask"))
        head_attention_mask: torch.Tensor | None
        if isinstance(attention_mask, torch.Tensor):
            mask_tensor: torch.Tensor = cast(torch.Tensor, attention_mask)
            if mask_tensor.dtype not in {
                torch.bool,
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            }:
                mask_tensor = mask_tensor.to(dtype=torch.bool)
            head_attention_mask = mask_tensor
        else:
            head_attention_mask = None
        layer_names = [f"hidden_state_{i:03d}" for i in range(len(hidden_states))]
        per_layer: list[np.ndarray] = []
        for h in hidden_states:
            if h.ndim == 3 and getattr(vision_model, "head", None) is not None:
                h_for_head = h.to(dtype=vision_model.post_layernorm.weight.dtype)
                h_ln = vision_model.post_layernorm(h_for_head)
                if hasattr(vision_model.head, "probe"):
                    h_ln = h_ln.to(dtype=vision_model.head.probe.dtype)
                try:
                    pooled = vision_model.head(h_ln, head_attention_mask)
                except TypeError:
                    pooled = vision_model.head(h_ln)
            elif h.ndim == 3:
                pooled = _masked_mean_tokens(h, attention_mask=attention_mask)
            else:
                pooled = _pool_hidden_tensor(h)
            per_layer.append(pooled.detach().float().cpu().numpy())

        pooler_output = getattr(outputs, "pooler_output", None)
        if not isinstance(pooler_output, torch.Tensor):
            last_hidden = outputs.last_hidden_state
            if getattr(vision_model, "head", None) is not None:
                last_hidden = last_hidden.to(dtype=vision_model.post_layernorm.weight.dtype)
                last_hidden = vision_model.post_layernorm(last_hidden)
                if hasattr(vision_model.head, "probe"):
                    last_hidden = last_hidden.to(dtype=vision_model.head.probe.dtype)
                try:
                    pooler_output = vision_model.head(last_hidden, head_attention_mask)
                except TypeError:
                    pooler_output = vision_model.head(last_hidden)
            else:
                pooler_output = _masked_mean_tokens(last_hidden, attention_mask=attention_mask)
        if not isinstance(pooler_output, torch.Tensor):
            raise RuntimeError("SigLIP2 vision model did not return a tensor pooler output")
        pooler_tensor = cast(torch.Tensor, pooler_output)
        image_embeds_l2 = _l2_normalize(pooler_tensor)
        layer_names.extend(["pooler_output", "image_embeds_l2"])
        per_layer.extend([pooler_tensor.detach().float().cpu().numpy(), image_embeds_l2.detach().float().cpu().numpy()])
        return ExtractResult(layer_names=layer_names, per_layer_np=per_layer)


def _internvit_local_dir(pretrained_id: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local = cache_dir / "internvit_local" / pretrained_id.replace("/", "__")
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=pretrained_id,
        local_dir=str(local),
    )
    return local


def _patch_internvit_flash_attention(local_model_dir: Path) -> None:
    flash_attention_file = local_model_dir / "flash_attention.py"
    if not flash_attention_file.exists():
        return
    patched = '''import torch\nimport torch.nn as nn\n\n\nclass FlashAttention(nn.Module):\n    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):\n        super().__init__()\n        self.softmax_scale = softmax_scale\n        self.dropout_p = attention_dropout\n\n    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None, max_s=None, need_weights=False):\n        raise RuntimeError(\n            "flash_attn is unavailable in this environment. "\n            "InternViT was loaded in no-flash mode; ensure use_flash_attn=False in config."\n        )\n'''
    flash_attention_file.write_text(patched, encoding="utf-8")


class InternViTExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype, cache_dir: Path) -> None:
        from transformers import AutoConfig, AutoModel, CLIPImageProcessor

        if spec.pretrained_id is None:
            raise RuntimeError("InternViT spec must include pretrained_id")
        local_model_dir = _internvit_local_dir(spec.pretrained_id, cache_dir)
        _patch_internvit_flash_attention(local_model_dir)

        config = AutoConfig.from_pretrained(str(local_model_dir), trust_remote_code=True)
        if hasattr(config, "use_flash_attn"):
            config.use_flash_attn = False

        self.processor = CLIPImageProcessor.from_pretrained(str(local_model_dir), trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        index_json = local_model_dir / "model.safetensors.index.json"
        if index_json.exists():
            from safetensors.torch import load_file

            index = json.loads(index_json.read_text(encoding="utf-8"))
            shard_files = sorted(set(index["weight_map"].values()))
            for shard_name in shard_files:
                shard_state = load_file(str(local_model_dir / shard_name), device="cpu")
                self.model.load_state_dict(shard_state, strict=False)
        else:
            safetensors_path = local_model_dir / "model.safetensors"
            if safetensors_path.exists():
                from safetensors.torch import load_file

                state = load_file(str(safetensors_path), device="cpu")
            else:
                state = torch.load(local_model_dir / "pytorch_model.bin", map_location="cpu")
            self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(device=device, dtype=dtype).eval()
        self.device = device
        self.model_dtype = dtype
        self.autocast_enabled = device.type == "cuda"

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with _autocast_context(self.device, self.autocast_enabled, self.model_dtype):
            outputs = self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            hidden_states = (outputs.last_hidden_state,)
        layer_names = [f"hidden_state_{i:03d}" for i in range(len(hidden_states))]
        per_layer = [_pool_hidden_tensor(h, policy="first_token").detach().float().cpu().numpy() for h in hidden_states]

        pooler_output = getattr(outputs, "pooler_output", None)
        if not isinstance(pooler_output, torch.Tensor):
            pooler_output = _pool_hidden_tensor(outputs.last_hidden_state, policy="first_token")
        if not isinstance(pooler_output, torch.Tensor):
            raise RuntimeError("InternViT model did not return a tensor pooler output")
        pooler_tensor = cast(torch.Tensor, pooler_output)
        layer_names.append("pooler_output")
        per_layer.append(pooler_tensor.detach().float().cpu().numpy())
        return ExtractResult(layer_names=layer_names, per_layer_np=per_layer)


class FastVLMFastViTHDExtractor(BaseExtractor):
    _MOBILECLIP_MODULE: Any = None

    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype, weights_dir: Path) -> None:
        from transformers import CLIPImageProcessor

        if spec.pretrained_id is None:
            raise RuntimeError("FastViTHD spec must include pretrained_id")

        self.device = device
        self.model_dtype = dtype if device.type == "cuda" else torch.float32
        self.autocast_enabled = device.type == "cuda" and self.model_dtype in {
            torch.float16,
            torch.bfloat16,
        }

        repo_dir = self._ensure_repo(weights_dir)
        mobileclip_module = self._load_mobileclip_module(repo_dir)
        vision_tower_name = self._resolve_vision_tower_name(
            spec.pretrained_id,
            cache_dir=weights_dir / "hf",
        )
        model_cfg = mobileclip_module.load_model_config(vision_tower_name)
        image_cfg = cast(dict[str, Any], model_cfg.get("image_cfg", {}))
        default_image_size = int(image_cfg.get("image_size", 1024))
        image_size = self._parse_image_size(vision_tower_name, default_image_size)
        image_cfg["image_size"] = image_size

        self.processor = CLIPImageProcessor(
            crop_size={"height": image_size, "width": image_size},
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            size={"shortest_edge": image_size},
        )

        self.model = mobileclip_module.MCi(
            model_name=str(image_cfg["model_name"]),
            projection_dim=int(model_cfg["embed_dim"]),
        )
        state_dict = self._load_vision_state_dict(spec.pretrained_id, cache_dir=weights_dir / "hf")
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device=self.device, dtype=self.model_dtype).eval()

        self._layer_names: list[str] = []
        self._captured: dict[str, np.ndarray] = {}
        self._hooks = []
        self._register_block_hooks()

    @classmethod
    def _load_mobileclip_module(cls, repo_dir: Path):
        if cls._MOBILECLIP_MODULE is not None:
            return cls._MOBILECLIP_MODULE

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        for mod_name in list(sys.modules.keys()):
            if mod_name == "llava" or mod_name.startswith("llava."):
                del sys.modules[mod_name]

        from llava.model.multimodal_encoder import mobileclip as mobileclip_module  # type: ignore

        cls._MOBILECLIP_MODULE = mobileclip_module
        return mobileclip_module

    def _ensure_repo(self, weights_dir: Path) -> Path:
        repo_dir = weights_dir / "github" / "ml-fastvlm"
        if repo_dir.exists():
            return repo_dir
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            if repo_dir.exists():
                return repo_dir
            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/apple/ml-fastvlm",
                        str(repo_dir),
                    ],
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

    def _resolve_vision_tower_name(self, pretrained_id: str, cache_dir: Path) -> str:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=pretrained_id,
            filename="config.json",
            cache_dir=str(cache_dir),
        )
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        value = payload.get("mm_vision_tower")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "mobileclip_l_1024"

    @staticmethod
    def _parse_image_size(vision_tower_name: str, default: int) -> int:
        tail = vision_tower_name.split("_")[-1]
        if tail.isdigit():
            return int(tail)
        return default

    def _load_vision_state_dict(self, pretrained_id: str, cache_dir: Path) -> dict[str, torch.Tensor]:
        from huggingface_hub import hf_hub_download

        prefixes = (
            "model.vision_tower.vision_tower.",
            "vision_tower.vision_tower.",
            "model.vision_tower.",
            "vision_tower.",
        )
        try:
            safetensors_path = Path(
                hf_hub_download(
                    repo_id=pretrained_id,
                    filename="model.safetensors",
                    cache_dir=str(cache_dir),
                )
            )
            return self._load_prefixed_safetensors([safetensors_path], prefixes)
        except Exception:
            index_path = Path(
                hf_hub_download(
                    repo_id=pretrained_id,
                    filename="model.safetensors.index.json",
                    cache_dir=str(cache_dir),
                )
            )
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = cast(dict[str, str], index_payload.get("weight_map", {}))
            shard_names = sorted(set(weight_map.values()))
            if not shard_names:
                raise RuntimeError(f"No shard files listed in {index_path}")
            shard_paths = [
                Path(
                    hf_hub_download(
                        repo_id=pretrained_id,
                        filename=shard_name,
                        cache_dir=str(cache_dir),
                    )
                )
                for shard_name in shard_names
            ]
            return self._load_prefixed_safetensors(shard_paths, prefixes)

    @staticmethod
    def _load_prefixed_safetensors(
        file_paths: list[Path],
        prefixes: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        state: dict[str, torch.Tensor] = {}
        selected_prefix: str | None = None
        for file_path in file_paths:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if selected_prefix is None:
                    for prefix in prefixes:
                        if any(key.startswith(prefix) for key in keys):
                            selected_prefix = prefix
                            break
                if selected_prefix is None:
                    continue
                for key in keys:
                    if key.startswith(selected_prefix):
                        state[key[len(selected_prefix) :]] = handle.get_tensor(key)
        if not state:
            raise RuntimeError(
                "Unable to find FastViTHD vision keys in safetensors files. "
                f"Expected one of prefixes: {prefixes}"
            )
        return state

    def _register_block_hooks(self) -> None:
        network = getattr(getattr(self.model, "model", None), "network", None)
        if not isinstance(network, torch.nn.ModuleList):
            raise RuntimeError("FastViTHD model has no expected network ModuleList")
        network_modules = cast(torch.nn.ModuleList, network)

        block_index = 0
        for module in network_modules:
            if not isinstance(module, torch.nn.Sequential):
                continue
            for block in module:
                layer_name = f"block_{block_index:03d}"
                self._layer_names.append(layer_name)
                self._hooks.append(block.register_forward_hook(self._make_block_hook(layer_name)))
                block_index += 1

        if not self._layer_names:
            raise RuntimeError("FastViTHD extractor did not discover any intermediate blocks")

    def _make_block_hook(self, layer_name: str):
        def _hook_fn(_module, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                return
            pooled = _pool_hidden_tensor(out)
            self._captured[layer_name] = pooled.detach().float().cpu().numpy()

        return _hook_fn

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.model_dtype)

        self._captured = {}
        with _autocast_context(self.device, self.autocast_enabled, self.model_dtype):
            outputs = self.model(pixel_values, return_image_embeddings=True)

        missing = [name for name in self._layer_names if name not in self._captured]
        if missing:
            raise RuntimeError(f"FastViTHD extractor missed hooked layers: {missing[:4]}")

        layer_names = list(self._layer_names)
        per_layer = [self._captured[name] for name in self._layer_names]

        image_embeddings = outputs.get("image_embeddings") if isinstance(outputs, dict) else None
        if isinstance(image_embeddings, torch.Tensor):
            layer_names.append("image_embeddings")
            per_layer.append(_pool_hidden_tensor(image_embeddings).detach().float().cpu().numpy())

        logits = outputs.get("logits") if isinstance(outputs, dict) else None
        canonical = logits if isinstance(logits, torch.Tensor) else image_embeddings
        if not isinstance(canonical, torch.Tensor):
            raise RuntimeError("FastViTHD model did not return a tensor canonical embedding")
        layer_names.append("canonical_embedding")
        per_layer.append(_pool_hidden_tensor(canonical).detach().float().cpu().numpy())
        return ExtractResult(layer_names=layer_names, per_layer_np=per_layer)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class TimmCNNFeaturesExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype) -> None:
        import timm

        self.device = device
        self.model_dtype = dtype if device.type == "cuda" else torch.float32
        self.autocast_enabled = device.type == "cuda" and self.model_dtype in {torch.float16, torch.bfloat16}

        self.model = timm.create_model(spec.timm_name, pretrained=True, features_only=True).to(
            device=device,
            dtype=self.model_dtype,
        ).eval()
        cfg = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**cfg, is_training=False)

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        x = torch.stack([self.transform(im) for im in images], dim=0).to(
            device=self.device,
            dtype=self.model_dtype,
        )
        amp_context = (
            torch.autocast(device_type="cuda", dtype=self.model_dtype)
            if self.autocast_enabled
            else contextlib.nullcontext()
        )
        with amp_context:
            feats = self.model(x)
        layer_names = [f"feature_map_{i:03d}" for i in range(len(feats))]
        per_layer = [_pool_hidden_tensor(t).detach().float().cpu().numpy() for t in feats]
        return ExtractResult(layer_names=layer_names, per_layer_np=per_layer)


class TimmViTBlocksExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype) -> None:
        import timm

        self.device = device
        self.model_dtype = dtype if device.type == "cuda" else torch.float32
        self.autocast_enabled = device.type == "cuda" and self.model_dtype in {torch.float16, torch.bfloat16}

        self.model = timm.create_model(spec.timm_name, pretrained=True).to(
            device=device,
            dtype=self.model_dtype,
        ).eval()
        self._global_pool = str(getattr(self.model, "global_pool", "token"))
        self._num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1) or 0)
        cfg = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**cfg, is_training=False)
        self._captured: list[torch.Tensor] = []
        self._hooks = []

        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError(f"Model {spec.timm_name} has no .blocks for ViT extraction")
        for block in blocks:
            self._hooks.append(block.register_forward_hook(self._hook_fn))

    def _hook_fn(self, _module, _inp, out):
        if isinstance(out, tuple):
            out = out[0]
        self._captured.append(out)

    def _pool_block_tokens(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 3:
            return _pool_hidden_tensor(t)

        if self._global_pool in {"token", "tok", "cls", ""}:
            return _pool_hidden_tensor(t, policy="first_token")

        patch_tokens = t[:, self._num_prefix_tokens :] if self._num_prefix_tokens > 0 else t
        if patch_tokens.shape[1] == 0:
            patch_tokens = t

        if self._global_pool == "max":
            return patch_tokens.max(dim=1).values
        if self._global_pool == "avgmax":
            return 0.5 * (patch_tokens.mean(dim=1) + patch_tokens.max(dim=1).values)
        return patch_tokens.mean(dim=1)

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        self._captured = []
        x = torch.stack([self.transform(im) for im in images], dim=0).to(
            device=self.device,
            dtype=self.model_dtype,
        )
        amp_context = (
            torch.autocast(device_type="cuda", dtype=self.model_dtype)
            if self.autocast_enabled
            else contextlib.nullcontext()
        )
        with amp_context:
            features = self.model.forward_features(x)
        layer_names = [f"block_{i:03d}" for i in range(len(self._captured))]
        per_layer = [self._pool_block_tokens(t).detach().float().cpu().numpy() for t in self._captured]

        with amp_context:
            canonical = self.model.forward_head(features, pre_logits=True)
        if isinstance(canonical, tuple):
            canonical = canonical[0]
        if not isinstance(canonical, torch.Tensor):
            raise RuntimeError("timm ViT model returned unsupported canonical embedding output")
        if canonical.ndim != 2:
            canonical = self._pool_block_tokens(canonical)
        layer_names.append("canonical_embedding")
        per_layer.append(canonical.detach().float().cpu().numpy())
        return ExtractResult(layer_names=layer_names, per_layer_np=per_layer)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


class ARNIQAExtractor(BaseExtractor):
    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype) -> None:
        self.model = torch.hub.load(
            repo_or_dir="miccunifi/ARNIQA",
            source="github",
            model="ARNIQA",
            regressor_dataset=spec.regressor_dataset,
            trust_repo=True,
        ).to(device).eval()
        self.device = device
        self.model_dtype = dtype
        self.preprocess = T.Compose(
            [
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        full = torch.stack([self.preprocess(im) for im in images], dim=0).to(self.device)
        ds_imgs = [T.Resize((im.size[1] // 2, im.size[0] // 2))(im) for im in images]
        down = torch.stack([self.preprocess(im) for im in ds_imgs], dim=0).to(self.device)
        with _autocast_context(self.device, self.device.type == "cuda", self.model_dtype):
            out = self.model(full, down, return_embedding=True, scale_score=True)

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            emb = out[1]
        else:
            emb = out
        if not isinstance(emb, torch.Tensor):
            raise RuntimeError("ARNIQA did not return tensor embeddings")
        emb_tensor = cast(torch.Tensor, emb)
        if emb_tensor.ndim == 1:
            emb_tensor = emb_tensor.unsqueeze(0)
        return ExtractResult(layer_names=["arniqa_embedding"], per_layer_np=[emb_tensor.detach().float().cpu().numpy()])


class MANIQAExtractor(BaseExtractor):
    _CKPT_INFO = {
        "PIPAL22": ("ckpt_valid", "https://github.com/IIGROUP/MANIQA/releases/download/PIPAL22-VALID-CKPT/ckpt_valid"),
        "KADID10K": ("ckpt_kadid10k.pt", "https://github.com/IIGROUP/MANIQA/releases/download/Kadid10k/ckpt_kadid10k.pt"),
        "KONIQ10K": ("ckpt_koniq10k.pt", "https://github.com/IIGROUP/MANIQA/releases/download/Koniq10k/ckpt_koniq10k.pt"),
    }
    _MANIQA_MODULE: Any = None

    def __init__(self, spec: ModelSpec, device: torch.device, dtype: torch.dtype, weights_dir: Path) -> None:
        from einops import rearrange

        self._rearrange = rearrange
        self.device = device
        self.model_dtype = dtype if device.type == "cuda" else torch.float32
        self.autocast_enabled = device.type == "cuda"

        repo_dir = self._ensure_repo(weights_dir)
        ckpt_path = self._ensure_checkpoint(weights_dir, spec.size)

        maniqa_module = self._load_maniqa_module(repo_dir)
        MANIQA = maniqa_module.MANIQA

        self.model = MANIQA(
            embed_dim=768,
            num_outputs=1,
            dim_mlp=768,
            patch_size=8,
            img_size=224,
            window_size=4,
            depths=[2, 2],
            num_heads=[4, 4],
            num_tab=2,
            scale=0.8,
        )
        try:
            import importlib

            sys.modules["timm.models.layers"] = importlib.import_module("timm.layers")
            for submodule in ("patch_embed", "mlp", "drop"):
                alias = f"timm.models.layers.{submodule}"
                target = f"timm.layers.{submodule}"
                sys.modules[alias] = importlib.import_module(target)
        except Exception:
            pass
        state_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
            state = state_obj["state_dict"]
        elif isinstance(state_obj, dict):
            state = state_obj
        elif hasattr(state_obj, "state_dict"):
            state = state_obj.state_dict()
        else:
            raise RuntimeError(f"Unsupported MANIQA checkpoint format: {type(state_obj)}")
        self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(device=self.device, dtype=self.model_dtype).eval()

        self.preprocess = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @classmethod
    def _load_maniqa_module(cls, repo_dir: Path):
        if cls._MANIQA_MODULE is not None:
            return cls._MANIQA_MODULE

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        for mod_name in list(sys.modules.keys()):
            if mod_name == "models" or mod_name.startswith("models."):
                del sys.modules[mod_name]

        from models import maniqa as maniqa_module  # type: ignore

        original_create_model = maniqa_module.timm.create_model

        def _create_model_no_pretrained(*args, **kwargs):
            kwargs["pretrained"] = False
            return original_create_model(*args, **kwargs)

        maniqa_module.timm.create_model = _create_model_no_pretrained
        cls._MANIQA_MODULE = maniqa_module
        return maniqa_module

    def _ensure_repo(self, weights_dir: Path) -> Path:
        repo_dir = weights_dir / "github" / "MANIQA"
        vendored_timm = repo_dir / "timm"
        disabled_timm = repo_dir / "timm_vendor_disabled"
        if repo_dir.exists():
            if vendored_timm.exists() and not disabled_timm.exists():
                vendored_timm.rename(disabled_timm)
            return repo_dir
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            if repo_dir.exists():
                return repo_dir
            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/IIGROUP/MANIQA",
                        str(repo_dir),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if vendored_timm.exists() and not disabled_timm.exists():
                    vendored_timm.rename(disabled_timm)
                return repo_dir
            except Exception:
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)
                if attempt == 3:
                    raise
                time.sleep(2.0 * (attempt + 1))
        return repo_dir

    def _ensure_checkpoint(self, weights_dir: Path, size: str) -> Path:
        if size not in self._CKPT_INFO:
            raise RuntimeError(f"Unsupported MANIQA size: {size}")
        filename, url = self._CKPT_INFO[size]
        ckpt_dir = weights_dir / "maniqa"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / filename
        if ckpt_path.exists():
            return ckpt_path
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as response, ckpt_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                return ckpt_path
            except Exception:
                if not ckpt_path.exists():
                    try:
                        subprocess.run(
                            ["curl", "-L", "-A", "Mozilla/5.0", url, "-o", str(ckpt_path)],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        return ckpt_path
                    except Exception:
                        pass
                if ckpt_path.exists() and ckpt_path.stat().st_size == 0:
                    ckpt_path.unlink(missing_ok=True)
                if attempt == 4:
                    raise
                time.sleep(3.0 * (attempt + 1))
        return ckpt_path

    def _embedding_forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.model.vit(x)
        h = self.model.extract_feature(self.model.save_output)
        self.model.save_output.outputs.clear()

        h = self._rearrange(h, "b (h w) c -> b c (h w)", h=self.model.input_size, w=self.model.input_size)
        for tab in self.model.tablock1:
            h = tab(h)
        h = self._rearrange(h, "b c (h w) -> b c h w", h=self.model.input_size, w=self.model.input_size)
        h = self.model.conv1(h)
        h = self.model.swintransformer1(h)

        h = self._rearrange(h, "b c h w -> b c (h w)", h=self.model.input_size, w=self.model.input_size)
        for tab in self.model.tablock2:
            h = tab(h)
        h = self._rearrange(h, "b c (h w) -> b c h w", h=self.model.input_size, w=self.model.input_size)
        h = self.model.conv2(h)
        h = self.model.swintransformer2(h)
        h = self._rearrange(h, "b c h w -> b (h w) c", h=self.model.input_size, w=self.model.input_size)
        return h.mean(dim=1)

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> ExtractResult:
        x = torch.stack([self.preprocess(im) for im in images], dim=0).to(self.device, dtype=self.model_dtype)
        with _autocast_context(self.device, self.autocast_enabled, self.model_dtype):
            emb = self._embedding_forward(x)
        return ExtractResult(layer_names=["maniqa_embedding"], per_layer_np=[emb.detach().float().cpu().numpy()])


def build_extractor(
    spec: ModelSpec,
    device: torch.device,
    dtype: torch.dtype,
    weights_dir: Path,
) -> BaseExtractor:
    if spec.loader == "hf_auto_image":
        return HFExtractor(spec, device=device, dtype=dtype, cache_dir=weights_dir / "hf")
    if spec.loader == "hf_auto_image_remote" and spec.family == "InternViT":
        return InternViTExtractor(spec, device=device, dtype=dtype, cache_dir=weights_dir / "hf")
    if spec.loader == "hf_auto_image_remote":
        return HFExtractor(spec, device=device, dtype=dtype, cache_dir=weights_dir / "hf")
    if spec.loader == "hf_swin_vision":
        return HFSwinVisionExtractor(spec, device=device, dtype=dtype, cache_dir=weights_dir / "hf")
    if spec.loader == "hf_clip_vision":
        return HFCLIPVisionExtractor(spec, device=device, dtype=dtype, cache_dir=weights_dir / "hf")
    if spec.loader == "hf_siglip_vision":
        return HFSiglipVisionExtractor(spec, device=device, dtype=dtype, cache_dir=weights_dir / "hf")
    if spec.loader == "hf_siglip2_vision":
        return HFSiglip2VisionExtractor(spec, device=device, dtype=dtype, cache_dir=weights_dir / "hf")
    if spec.loader == "timm_cnn_features":
        return TimmCNNFeaturesExtractor(spec, device=device, dtype=dtype)
    if spec.loader == "timm_vit_blocks":
        return TimmViTBlocksExtractor(spec, device=device, dtype=dtype)
    if spec.loader == "arniqa_torchhub":
        return ARNIQAExtractor(spec, device=device, dtype=dtype)
    if spec.loader == "fastvlm_fastvithd":
        return FastVLMFastViTHDExtractor(spec, device=device, dtype=dtype, weights_dir=weights_dir)
    if spec.loader == "maniqa_repo":
        return MANIQAExtractor(spec, device=device, dtype=dtype, weights_dir=weights_dir)
    raise ValueError(f"Unsupported loader: {spec.loader}")


@contextlib.contextmanager
def managed_extractor(spec: ModelSpec, device: torch.device, dtype: torch.dtype, weights_dir: Path):
    extractor = build_extractor(spec=spec, device=device, dtype=dtype, weights_dir=weights_dir)
    try:
        yield extractor
    finally:
        close = getattr(extractor, "close", None)
        if callable(close):
            close()
