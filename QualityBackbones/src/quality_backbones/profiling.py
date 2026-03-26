from __future__ import annotations

import inspect
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from quality_backbones.extractors import (
    ARNIQAExtractor,
    BaseExtractor,
    FastVLMFastViTHDExtractor,
    HFCLIPVisionExtractor,
    HFExtractor,
    HFSiglip2VisionExtractor,
    HFSiglipVisionExtractor,
    HFSwinVisionExtractor,
    InternViTExtractor,
    MANIQAExtractor,
    TimmCNNFeaturesExtractor,
    TimmViTBlocksExtractor,
    _autocast_context,
    _l2_normalize,
    _masked_mean_tokens,
    _pool_hidden_tensor,
    build_extractor,
    default_dtype_for_device,
    managed_extractor,
    select_device,
)
from quality_backbones.manifest import ModelSpec, get_model_spec, iter_enabled_image_model_specs


PreparedBatch = torch.Tensor | dict[str, torch.Tensor]


class _EarlyStop(RuntimeError):
    pass


def _extract_first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unable to extract tensor from value of type {type(value)!r}")


def _module_list(module: nn.Module, *attr_names: str) -> nn.ModuleList:
    for name in attr_names:
        value = getattr(module, name, None)
        if isinstance(value, nn.ModuleList):
            return value
    raise RuntimeError(f"Unable to find ModuleList among attributes {attr_names}")


def _select_primary_tensor(batch: PreparedBatch) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    preferred = (
        "pixel_values",
        "input_pixels",
        "images",
        "full",
        "x",
    )
    for key in preferred:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return value
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError("Prepared batch does not contain a tensor input")


def _stringify_dtype(dtype: torch.dtype) -> str:
    text = str(dtype)
    return text.replace("torch.", "")


def _normalize_index(value: int, size: int) -> int:
    if size <= 0:
        raise ValueError("size must be positive")
    if value < 0:
        value += size
    if value < 0 or value >= size:
        raise IndexError(f"Layer index {value} is out of range for size {size}")
    return value


def parse_profile_layer_selectors(raw_values: Sequence[str] | None) -> tuple[str, ...] | None:
    if raw_values is None:
        return None
    tokens: list[str] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            token = part.strip()
            if token:
                tokens.append(token)
    return tuple(tokens) if tokens else None


def resolve_profile_target_indices(target_names: Sequence[str], selectors: Sequence[str] | None) -> list[int]:
    if not target_names:
        raise ValueError("target_names must not be empty")
    if selectors is None:
        return list(range(len(target_names)))

    selected: list[int] = []
    for raw_selector in selectors:
        selector = str(raw_selector).strip()
        if not selector:
            continue
        lowered = selector.lower()
        if lowered == "all":
            selected.extend(range(len(target_names)))
            continue
        if lowered == "last":
            selected.append(len(target_names) - 1)
            continue
        if "-" in selector and selector.count("-") == 1:
            left_text, right_text = selector.split("-", 1)
            if left_text.strip().lstrip("+-").isdigit() and right_text.strip().lstrip("+-").isdigit():
                left = _normalize_index(int(left_text), len(target_names))
                right = _normalize_index(int(right_text), len(target_names))
                step = 1 if right >= left else -1
                selected.extend(range(left, right + step, step))
                continue
        if selector.lstrip("+-").isdigit():
            selected.append(_normalize_index(int(selector), len(target_names)))
            continue
        if selector not in target_names:
            raise KeyError(f"Unknown layer token {selector!r}. Example available layers: {list(target_names)[:6]}")
        selected.append(target_names.index(selector))

    ordered: list[int] = []
    seen: set[int] = set()
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def write_profile_results_table(rows: Sequence[dict[str, Any]], out_path: Path) -> None:
    frame = pd.DataFrame(list(rows))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".tsv":
        frame.to_csv(out_path, sep="\t", index=False)
        return
    if suffix == ".csv":
        frame.to_csv(out_path, index=False)
        return
    if suffix == ".parquet":
        frame.to_parquet(out_path, index=False)
        return
    raise ValueError(f"Unsupported table output suffix: {out_path.suffix}")


@dataclass(frozen=True)
class ProfileTarget:
    layer_index: int
    layer_name: str


@dataclass
class ProfileRecord:
    model_key: str
    family: str
    loader: str
    layer_index: int
    layer_name: str
    batch_size: int
    input_height: int | None
    input_width: int | None
    device: str
    dtype: str
    output_dim: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    throughput_img_s: float
    flops_total: int
    macs_total: int
    flops_per_image: float
    macs_per_image: float
    delta_latency_ms: float | None = None
    delta_flops_total: int | None = None
    parity_ok: bool | None = None
    parity_max_abs_error: float | None = None
    parity_mean_abs_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForwardRunner:
    def run(self) -> torch.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _CallableForwardRunner(ForwardRunner):
    def __init__(self, fn: Callable[[], torch.Tensor]) -> None:
        self._fn = fn

    def run(self) -> torch.Tensor:
        return self._fn()


class _HookedForwardRunner(ForwardRunner):
    def __init__(
        self,
        *,
        target_module: nn.Module,
        output_transform: Callable[[Any], torch.Tensor],
        forward_fn: Callable[[], Any],
    ) -> None:
        self._target_module = target_module
        self._output_transform = output_transform
        self._forward_fn = forward_fn
        self._captured: torch.Tensor | None = None
        self._handle = target_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        self._captured = self._output_transform(output)
        raise _EarlyStop()

    def run(self) -> torch.Tensor:
        self._captured = None
        try:
            self._forward_fn()
        except _EarlyStop:
            pass
        if not isinstance(self._captured, torch.Tensor):
            raise RuntimeError("Target hook did not capture an output tensor")
        return self._captured

    def close(self) -> None:
        self._handle.remove()


@dataclass
class TargetProfileResult:
    record: ProfileRecord
    output: torch.Tensor


class BaseProfileAdapter:
    def __init__(self, spec: ModelSpec, extractor: BaseExtractor) -> None:
        self.spec = spec
        self.extractor = extractor

    @property
    def device(self) -> torch.device:
        return cast(torch.device, getattr(self.extractor, "device"))

    @property
    def model_dtype(self) -> torch.dtype:
        return cast(torch.dtype, getattr(self.extractor, "model_dtype"))

    @property
    def autocast_enabled(self) -> bool:
        return bool(getattr(self.extractor, "autocast_enabled", self.device.type == "cuda"))

    def autocast_context(self):
        return _autocast_context(self.device, self.autocast_enabled, self.model_dtype)

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        raise NotImplementedError

    def list_targets(self) -> list[ProfileTarget]:
        raise NotImplementedError

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        raise NotImplementedError

    def primary_input_shape(self, batch: PreparedBatch) -> tuple[int | None, int | None]:
        tensor = _select_primary_tensor(batch)
        if tensor.ndim < 4:
            return None, None
        return int(tensor.shape[-2]), int(tensor.shape[-1])

    def verify_against_extract_result(
        self,
        target: ProfileTarget,
        output: torch.Tensor,
        extract_result_layer_map: dict[str, np.ndarray],
        *,
        atol: float,
        rtol: float,
    ) -> tuple[bool, float, float]:
        expected = extract_result_layer_map[target.layer_name]
        actual = output.detach().float().cpu().numpy()
        delta = np.abs(actual - expected)
        max_abs = float(delta.max()) if delta.size else 0.0
        mean_abs = float(delta.mean()) if delta.size else 0.0
        ok = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
        return ok, max_abs, mean_abs


class HFProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: HFExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.hf_extractor = extractor
        self.model = extractor.model
        self.embedding_module = cast(nn.Module, getattr(self.model, "embeddings", getattr(self.model, "embedder", None)))
        if self.embedding_module is None:
            raise RuntimeError(f"Model {spec.key} has no HF embedding module")
        self.encoder_layers = _module_list(self.model.encoder, "layer", "layers", "stages")

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        inputs = self.hf_extractor.processor(images=images, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    def list_targets(self) -> list[ProfileTarget]:
        names = [f"hidden_state_{i:03d}" for i in range(len(self.encoder_layers) + 1)]
        names.append("canonical_embedding")
        return [ProfileTarget(layer_index=i, layer_name=name) for i, name in enumerate(names)]

    def _pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return _pool_hidden_tensor(hidden, policy=self.hf_extractor._hidden_state_pool_policy())

    def _call_model(self, batch: dict[str, torch.Tensor], **extra_kwargs: Any) -> Any:
        call_kwargs = dict(batch)
        call_kwargs.update(extra_kwargs)
        with self.autocast_context():
            return self.model(**call_kwargs)

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        inputs = cast(dict[str, torch.Tensor], batch)
        if target.layer_name == "canonical_embedding":
            def _run_canonical() -> torch.Tensor:
                outputs = self._call_model(inputs, return_dict=True)
                return self.hf_extractor._canonical_embedding(outputs)

            return _CallableForwardRunner(_run_canonical)

        if target.layer_index == 0:
            target_module = self.embedding_module
        else:
            target_module = self.encoder_layers[target.layer_index - 1]

        def _forward() -> Any:
            return self._call_model(inputs)

        return _HookedForwardRunner(
            target_module=target_module,
            output_transform=lambda raw: self._pool_hidden(_extract_first_tensor(raw)),
            forward_fn=_forward,
        )


class HFSwinProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: HFSwinVisionExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.swin_extractor = extractor
        self.model = extractor.model
        self.encoder_layers = _module_list(self.model.encoder, "layers")

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        inputs = self.swin_extractor.processor(images=images, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].to(self.device)}

    def list_targets(self) -> list[ProfileTarget]:
        names = [f"hidden_state_{i:03d}" for i in range(len(self.encoder_layers) + 1)]
        names.append("canonical_embedding")
        return [ProfileTarget(layer_index=i, layer_name=name) for i, name in enumerate(names)]

    def _pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return _pool_hidden_tensor(hidden, policy="mean_tokens")

    def _call_model(self, batch: dict[str, torch.Tensor], **extra_kwargs: Any) -> Any:
        with self.autocast_context():
            return self.model(pixel_values=batch["pixel_values"], **extra_kwargs)

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        inputs = cast(dict[str, torch.Tensor], batch)
        if target.layer_name == "canonical_embedding":
            def _run_canonical() -> torch.Tensor:
                outputs = self._call_model(inputs, return_dict=True)
                pooler_output = getattr(outputs, "pooler_output", None)
                if not isinstance(pooler_output, torch.Tensor):
                    pooler_output = _pool_hidden_tensor(outputs.last_hidden_state, policy="mean_tokens")
                return cast(torch.Tensor, pooler_output)

            return _CallableForwardRunner(_run_canonical)

        target_module = self.model.embeddings if target.layer_index == 0 else self.encoder_layers[target.layer_index - 1]
        output_transform = (
            lambda raw: self._pool_hidden(_extract_first_tensor(raw))
            if target.layer_index == 0
            else self._pool_hidden(_extract_first_tensor(raw))
        )
        return _HookedForwardRunner(
            target_module=target_module,
            output_transform=output_transform,
            forward_fn=lambda: self._call_model(inputs),
        )


class HFCLIPProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: HFCLIPVisionExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.clip_extractor = extractor
        self.model = extractor.vision_model
        self.vision_model = extractor.vision_model.vision_model
        self.encoder_layers = _module_list(self.vision_model.encoder, "layers")
        self.visual_projection = extractor.visual_projection

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        inputs = self.clip_extractor.processor(images=images, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].to(self.device)}

    def list_targets(self) -> list[ProfileTarget]:
        names = [f"hidden_state_{i:03d}" for i in range(len(self.encoder_layers) + 1)]
        names.extend(["pooler_output", "image_embeds", "image_embeds_l2"])
        return [ProfileTarget(layer_index=i, layer_name=name) for i, name in enumerate(names)]

    def _call_model(self, batch: dict[str, torch.Tensor], **extra_kwargs: Any) -> Any:
        with self.autocast_context():
            return self.model(pixel_values=batch["pixel_values"], **extra_kwargs)

    def _run_head(self, batch: dict[str, torch.Tensor], layer_name: str) -> torch.Tensor:
        outputs = self._call_model(batch, return_dict=True)
        pooler_output = outputs.pooler_output.to(dtype=self.visual_projection.weight.dtype)
        if layer_name == "pooler_output":
            return pooler_output
        image_embeds = self.visual_projection(pooler_output)
        if layer_name == "image_embeds":
            return image_embeds
        if layer_name == "image_embeds_l2":
            return _l2_normalize(image_embeds)
        raise KeyError(layer_name)

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        inputs = cast(dict[str, torch.Tensor], batch)
        if target.layer_name in {"pooler_output", "image_embeds", "image_embeds_l2"}:
            return _CallableForwardRunner(lambda: self._run_head(inputs, target.layer_name))

        target_module = self.vision_model.pre_layrnorm if target.layer_index == 0 else self.encoder_layers[target.layer_index - 1]
        return _HookedForwardRunner(
            target_module=target_module,
            output_transform=lambda raw: _pool_hidden_tensor(_extract_first_tensor(raw), policy="first_token"),
            forward_fn=lambda: self._call_model(inputs),
        )


class HFSiglipProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: HFSiglipVisionExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.siglip_extractor = extractor
        self.model = extractor.model
        self.vision_model = extractor.model.vision_model
        self.encoder_layers = _module_list(self.vision_model.encoder, "layers")

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        inputs = self.siglip_extractor.processor(images=images, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].to(self.device)}

    def list_targets(self) -> list[ProfileTarget]:
        names = [f"hidden_state_{i:03d}" for i in range(len(self.encoder_layers) + 1)]
        names.extend(["pooler_output", "image_embeds_l2"])
        return [ProfileTarget(layer_index=i, layer_name=name) for i, name in enumerate(names)]

    def _call_model(self, batch: dict[str, torch.Tensor], **extra_kwargs: Any) -> Any:
        with self.autocast_context():
            return self.model(pixel_values=batch["pixel_values"], **extra_kwargs)

    def _pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3 and getattr(self.vision_model, "head", None) is not None:
            hidden_for_head = hidden.to(dtype=self.vision_model.post_layernorm.weight.dtype)
            hidden_ln = self.vision_model.post_layernorm(hidden_for_head)
            if hasattr(self.vision_model.head, "probe"):
                hidden_ln = hidden_ln.to(dtype=self.vision_model.head.probe.dtype)
            return cast(torch.Tensor, self.vision_model.head(hidden_ln))
        return _pool_hidden_tensor(hidden, policy="mean_tokens")

    def _run_head(self, batch: dict[str, torch.Tensor], layer_name: str) -> torch.Tensor:
        outputs = self._call_model(batch, return_dict=True)
        pooler_output = outputs.pooler_output
        if pooler_output is None:
            last_hidden = outputs.last_hidden_state
            last_hidden = last_hidden.to(dtype=self.vision_model.post_layernorm.weight.dtype)
            last_hidden = self.vision_model.post_layernorm(last_hidden)
            if hasattr(self.vision_model.head, "probe"):
                last_hidden = last_hidden.to(dtype=self.vision_model.head.probe.dtype)
            pooler_output = self.vision_model.head(last_hidden)
        if layer_name == "pooler_output":
            return cast(torch.Tensor, pooler_output)
        if layer_name == "image_embeds_l2":
            return _l2_normalize(cast(torch.Tensor, pooler_output))
        raise KeyError(layer_name)

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        inputs = cast(dict[str, torch.Tensor], batch)
        if target.layer_name in {"pooler_output", "image_embeds_l2"}:
            return _CallableForwardRunner(lambda: self._run_head(inputs, target.layer_name))

        target_module = self.vision_model.embeddings if target.layer_index == 0 else self.encoder_layers[target.layer_index - 1]
        return _HookedForwardRunner(
            target_module=target_module,
            output_transform=lambda raw: self._pool_hidden(_extract_first_tensor(raw)),
            forward_fn=lambda: self._call_model(inputs),
        )


class HFSiglip2ProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: HFSiglip2VisionExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.siglip2_extractor = extractor
        self.model = extractor.model
        self.vision_model = getattr(self.model, "vision_model", None)
        if self.vision_model is None:
            raise RuntimeError("Loaded SigLIP2 model has no vision_model")
        self.encoder_layers = _module_list(self.vision_model.encoder, "layers")
        self.vision_forward_params = set(inspect.signature(self.vision_model.forward).parameters.keys())

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        inputs = self.siglip2_extractor.processor(images=images, return_tensors="pt")
        vision_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        if "pixel_attention_mask" in vision_inputs and "attention_mask" not in vision_inputs:
            vision_inputs["attention_mask"] = vision_inputs["pixel_attention_mask"]

        if self.siglip2_extractor.model_type == "siglip2" and "spatial_shapes" not in vision_inputs and "pixel_values" in vision_inputs:
            pixel_values = vision_inputs["pixel_values"]
            if pixel_values.ndim == 4:
                patch_size = getattr(getattr(self.model.config, "vision_config", self.model.config), "patch_size", None)
                if isinstance(patch_size, int) and patch_size > 0:
                    height = pixel_values.shape[-2] // patch_size
                    width = pixel_values.shape[-1] // patch_size
                    vision_inputs["spatial_shapes"] = torch.tensor(
                        [[height, width]] * pixel_values.shape[0],
                        dtype=torch.long,
                        device=self.device,
                    )
                    if "attention_mask" not in vision_inputs:
                        vision_inputs["attention_mask"] = torch.ones(
                            (pixel_values.shape[0], height * width),
                            dtype=torch.bool,
                            device=self.device,
                        )

        return self._filter_vision_inputs(vision_inputs)

    def list_targets(self) -> list[ProfileTarget]:
        names = ["hidden_state_000", "pooler_output", "image_embeds_l2"]
        return [ProfileTarget(layer_index=i, layer_name=name) for i, name in enumerate(names)]

    def _filter_vision_inputs(self, vision_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        call_inputs = dict(vision_inputs)
        if "attention_mask" in call_inputs and "attention_mask" not in self.vision_forward_params:
            if "pixel_attention_mask" in self.vision_forward_params:
                call_inputs["pixel_attention_mask"] = call_inputs["attention_mask"]
            call_inputs.pop("attention_mask", None)
        if "pixel_attention_mask" in call_inputs and "pixel_attention_mask" not in self.vision_forward_params:
            if "attention_mask" in self.vision_forward_params:
                call_inputs["attention_mask"] = call_inputs["pixel_attention_mask"]
            call_inputs.pop("pixel_attention_mask", None)
        return {k: v for k, v in call_inputs.items() if k in self.vision_forward_params}

    def _head_attention_mask(self, call_inputs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        attention_mask = call_inputs.get("attention_mask", call_inputs.get("pixel_attention_mask"))
        if not isinstance(attention_mask, torch.Tensor):
            return None
        mask_tensor = cast(torch.Tensor, attention_mask)
        if mask_tensor.dtype not in {torch.bool, torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            mask_tensor = mask_tensor.to(dtype=torch.bool)
        return mask_tensor

    def _call_model(self, batch: dict[str, torch.Tensor], **extra_kwargs: Any) -> Any:
        filtered_extra = {k: v for k, v in extra_kwargs.items() if k in self.vision_forward_params}
        with self.autocast_context():
            try:
                return self.vision_model(**batch, **filtered_extra)
            except TypeError:
                alt_inputs = dict(batch)
                if "attention_mask" in alt_inputs:
                    alt_inputs["pixel_attention_mask"] = alt_inputs.pop("attention_mask")
                if "pixel_attention_mask" in alt_inputs and "attention_mask" in self.vision_forward_params:
                    alt_inputs["attention_mask"] = alt_inputs.pop("pixel_attention_mask")
                return self.vision_model(**alt_inputs, **filtered_extra)

    def _pool_hidden(self, hidden: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        attention_mask = batch.get("attention_mask", batch.get("pixel_attention_mask"))
        head_attention_mask = self._head_attention_mask(batch)
        if hidden.ndim == 3 and getattr(self.vision_model, "head", None) is not None:
            hidden_for_head = hidden.to(dtype=self.vision_model.post_layernorm.weight.dtype)
            hidden_ln = self.vision_model.post_layernorm(hidden_for_head)
            if hasattr(self.vision_model.head, "probe"):
                hidden_ln = hidden_ln.to(dtype=self.vision_model.head.probe.dtype)
            try:
                return cast(torch.Tensor, self.vision_model.head(hidden_ln, head_attention_mask))
            except TypeError:
                return cast(torch.Tensor, self.vision_model.head(hidden_ln))
        if hidden.ndim == 3:
            return _masked_mean_tokens(hidden, attention_mask=attention_mask)
        return _pool_hidden_tensor(hidden)

    def _run_head(self, batch: dict[str, torch.Tensor], layer_name: str) -> torch.Tensor:
        outputs = self._call_model(batch, return_dict=True)
        pooler_output = getattr(outputs, "pooler_output", None)
        if not isinstance(pooler_output, torch.Tensor):
            last_hidden = outputs.last_hidden_state
            if getattr(self.vision_model, "head", None) is not None:
                last_hidden = last_hidden.to(dtype=self.vision_model.post_layernorm.weight.dtype)
                last_hidden = self.vision_model.post_layernorm(last_hidden)
                if hasattr(self.vision_model.head, "probe"):
                    last_hidden = last_hidden.to(dtype=self.vision_model.head.probe.dtype)
                try:
                    pooler_output = self.vision_model.head(last_hidden, self._head_attention_mask(batch))
                except TypeError:
                    pooler_output = self.vision_model.head(last_hidden)
            else:
                attention_mask = batch.get("attention_mask", batch.get("pixel_attention_mask"))
                pooler_output = _masked_mean_tokens(last_hidden, attention_mask=attention_mask)
        pooler_tensor = cast(torch.Tensor, pooler_output)
        if layer_name == "pooler_output":
            return pooler_tensor
        if layer_name == "image_embeds_l2":
            return _l2_normalize(pooler_tensor)
        raise KeyError(layer_name)

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        inputs = cast(dict[str, torch.Tensor], batch)
        if target.layer_name == "hidden_state_000":
            return _CallableForwardRunner(
                lambda: self._pool_hidden(cast(torch.Tensor, self._call_model(inputs, return_dict=True).last_hidden_state), inputs)
            )
        if target.layer_name in {"pooler_output", "image_embeds_l2"}:
            return _CallableForwardRunner(lambda: self._run_head(inputs, target.layer_name))
        raise KeyError(target.layer_name)


class InternViTProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: InternViTExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.intern_extractor = extractor
        self.model = extractor.model
        self.encoder_layers = _module_list(self.model.encoder, "layers")

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        inputs = self.intern_extractor.processor(images=images, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].to(self.device)}

    def list_targets(self) -> list[ProfileTarget]:
        names = [f"hidden_state_{i:03d}" for i in range(len(self.encoder_layers) + 1)]
        names.append("pooler_output")
        return [ProfileTarget(layer_index=i, layer_name=name) for i, name in enumerate(names)]

    def _call_model(self, batch: dict[str, torch.Tensor], **extra_kwargs: Any) -> Any:
        with self.autocast_context():
            return self.model(pixel_values=batch["pixel_values"], **extra_kwargs)

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        inputs = cast(dict[str, torch.Tensor], batch)
        if target.layer_name == "pooler_output":
            def _run_pooler() -> torch.Tensor:
                outputs = self._call_model(inputs, return_dict=True)
                pooler_output = getattr(outputs, "pooler_output", None)
                if not isinstance(pooler_output, torch.Tensor):
                    pooler_output = _pool_hidden_tensor(outputs.last_hidden_state, policy="first_token")
                return cast(torch.Tensor, pooler_output)

            return _CallableForwardRunner(_run_pooler)

        target_module = self.model.embeddings if target.layer_index == 0 else self.encoder_layers[target.layer_index - 1]
        return _HookedForwardRunner(
            target_module=target_module,
            output_transform=lambda raw: _pool_hidden_tensor(_extract_first_tensor(raw), policy="first_token"),
            forward_fn=lambda: self._call_model(inputs),
        )


class TimmCNNProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: TimmCNNFeaturesExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.timm_extractor = extractor
        self.model = extractor.model
        feature_info = getattr(self.model, "feature_info", None)
        if feature_info is None:
            raise RuntimeError(f"Model {spec.key} has no feature_info")
        module_names = feature_info.module_name()
        self.target_modules = [self._resolve_target_module(name) for name in module_names]

    def _resolve_target_module(self, name: str) -> nn.Module:
        candidates = [name]
        alt = name.replace(".", "_")
        if alt != name:
            candidates.append(alt)
        for candidate in candidates:
            try:
                return self.model.get_submodule(candidate)
            except AttributeError:
                continue
        raise RuntimeError(f"Unable to resolve feature module {name!r} on {type(self.model).__name__}")

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        return torch.stack([self.timm_extractor.transform(im) for im in images], dim=0).to(
            device=self.device,
            dtype=self.model_dtype,
        )

    def list_targets(self) -> list[ProfileTarget]:
        names = [f"feature_map_{i:03d}" for i in range(len(self.target_modules))]
        return [ProfileTarget(layer_index=i, layer_name=name) for i, name in enumerate(names)]

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        x = cast(torch.Tensor, batch)
        target_module = self.target_modules[target.layer_index]
        return _HookedForwardRunner(
            target_module=target_module,
            output_transform=lambda raw: _pool_hidden_tensor(_extract_first_tensor(raw)),
            forward_fn=lambda: self._forward(x),
        )

    def _forward(self, x: torch.Tensor) -> Any:
        with self.autocast_context():
            return self.model(x)


class TimmViTProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: TimmViTBlocksExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.timm_extractor = extractor
        self.model = extractor.model
        blocks = getattr(self.model, "blocks", None)
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            raise RuntimeError(f"Model {spec.key} has no sequential .blocks for ViT profiling")
        self.blocks = blocks

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        return torch.stack([self.timm_extractor.transform(im) for im in images], dim=0).to(
            device=self.device,
            dtype=self.model_dtype,
        )

    def list_targets(self) -> list[ProfileTarget]:
        names = [f"block_{i:03d}" for i in range(len(self.blocks))]
        names.append("canonical_embedding")
        return [ProfileTarget(layer_index=i, layer_name=name) for i, name in enumerate(names)]

    def _forward_features(self, x: torch.Tensor) -> Any:
        with self.autocast_context():
            return self.model.forward_features(x)

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        x = cast(torch.Tensor, batch)
        if target.layer_name == "canonical_embedding":
            def _run_canonical() -> torch.Tensor:
                features = self._forward_features(x)
                with self.autocast_context():
                    canonical = self.model.forward_head(features, pre_logits=True)
                if isinstance(canonical, tuple):
                    canonical = canonical[0]
                if not isinstance(canonical, torch.Tensor):
                    raise RuntimeError("timm ViT model returned unsupported canonical embedding output")
                if canonical.ndim != 2:
                    canonical = self.timm_extractor._pool_block_tokens(canonical)
                return canonical

            return _CallableForwardRunner(_run_canonical)

        target_module = self.blocks[target.layer_index]
        return _HookedForwardRunner(
            target_module=target_module,
            output_transform=lambda raw: self.timm_extractor._pool_block_tokens(_extract_first_tensor(raw)),
            forward_fn=lambda: self._forward_features(x),
        )


class FastViTHDProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: FastVLMFastViTHDExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.fastvithd_extractor = extractor
        self.model = extractor.model
        network = getattr(getattr(self.model, "model", None), "network", None)
        if not isinstance(network, nn.ModuleList):
            raise RuntimeError("FastViTHD model has no expected network ModuleList")
        self.blocks: list[nn.Module] = []
        for module in cast(nn.ModuleList, network):
            if not isinstance(module, nn.Sequential):
                continue
            for block in module:
                self.blocks.append(block)

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        inputs = self.fastvithd_extractor.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(device=self.device, dtype=self.model_dtype)

    def list_targets(self) -> list[ProfileTarget]:
        names = [f"block_{i:03d}" for i in range(len(self.blocks))]
        names.extend(["image_embeddings", "canonical_embedding"])
        return [ProfileTarget(layer_index=i, layer_name=name) for i, name in enumerate(names)]

    def _forward(self, x: torch.Tensor) -> Any:
        with self.autocast_context():
            return self.model(x, return_image_embeddings=True)

    def _run_head(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        outputs = self._forward(x)
        image_embeddings = outputs.get("image_embeddings") if isinstance(outputs, dict) else None
        logits = outputs.get("logits") if isinstance(outputs, dict) else None
        if layer_name == "image_embeddings":
            if not isinstance(image_embeddings, torch.Tensor):
                raise RuntimeError("FastViTHD model did not return image_embeddings")
            return _pool_hidden_tensor(image_embeddings)
        canonical = logits if isinstance(logits, torch.Tensor) else image_embeddings
        if not isinstance(canonical, torch.Tensor):
            raise RuntimeError("FastViTHD model did not return a tensor canonical embedding")
        return _pool_hidden_tensor(canonical)

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        x = cast(torch.Tensor, batch)
        if target.layer_name in {"image_embeddings", "canonical_embedding"}:
            return _CallableForwardRunner(lambda: self._run_head(x, target.layer_name))

        target_module = self.blocks[target.layer_index]
        return _HookedForwardRunner(
            target_module=target_module,
            output_transform=lambda raw: _pool_hidden_tensor(_extract_first_tensor(raw)),
            forward_fn=lambda: self._forward(x),
        )


class ARNIQAProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: ARNIQAExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.arniqa_extractor = extractor

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        full = torch.stack([self.arniqa_extractor.preprocess(im) for im in images], dim=0).to(self.device)
        ds_images = [T.Resize((im.size[1] // 2, im.size[0] // 2))(im) for im in images]
        down = torch.stack([self.arniqa_extractor.preprocess(im) for im in ds_images], dim=0).to(self.device)
        return {"full": full, "down": down}

    def list_targets(self) -> list[ProfileTarget]:
        return [ProfileTarget(layer_index=0, layer_name="arniqa_embedding")]

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        inputs = cast(dict[str, torch.Tensor], batch)

        def _run() -> torch.Tensor:
            with self.autocast_context():
                out = self.arniqa_extractor.model(inputs["full"], inputs["down"], return_embedding=True, scale_score=True)
            emb = out[1] if isinstance(out, (tuple, list)) and len(out) >= 2 else out
            if not isinstance(emb, torch.Tensor):
                raise RuntimeError("ARNIQA did not return tensor embeddings")
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            return cast(torch.Tensor, emb)

        return _CallableForwardRunner(_run)


class MANIQAProfileAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: MANIQAExtractor) -> None:
        super().__init__(spec=spec, extractor=extractor)
        self.maniqa_extractor = extractor

    def prepare_batch(self, images: list[Image.Image]) -> PreparedBatch:
        return torch.stack([self.maniqa_extractor.preprocess(im) for im in images], dim=0).to(
            self.device,
            dtype=self.model_dtype,
        )

    def list_targets(self) -> list[ProfileTarget]:
        return [ProfileTarget(layer_index=0, layer_name="maniqa_embedding")]

    def make_runner(self, target: ProfileTarget, batch: PreparedBatch) -> ForwardRunner:
        x = cast(torch.Tensor, batch)
        return _CallableForwardRunner(lambda: self._run(x))

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        with self.autocast_context():
            return self.maniqa_extractor._embedding_forward(x)


def build_profile_adapter(spec: ModelSpec, extractor: BaseExtractor) -> BaseProfileAdapter:
    if isinstance(extractor, InternViTExtractor):
        return InternViTProfileAdapter(spec, extractor)
    if isinstance(extractor, HFExtractor):
        return HFProfileAdapter(spec, extractor)
    if isinstance(extractor, HFSwinVisionExtractor):
        return HFSwinProfileAdapter(spec, extractor)
    if isinstance(extractor, HFCLIPVisionExtractor):
        return HFCLIPProfileAdapter(spec, extractor)
    if isinstance(extractor, HFSiglipVisionExtractor):
        return HFSiglipProfileAdapter(spec, extractor)
    if isinstance(extractor, HFSiglip2VisionExtractor):
        return HFSiglip2ProfileAdapter(spec, extractor)
    if isinstance(extractor, TimmCNNFeaturesExtractor):
        return TimmCNNProfileAdapter(spec, extractor)
    if isinstance(extractor, TimmViTBlocksExtractor):
        return TimmViTProfileAdapter(spec, extractor)
    if isinstance(extractor, FastVLMFastViTHDExtractor):
        return FastViTHDProfileAdapter(spec, extractor)
    if isinstance(extractor, ARNIQAExtractor):
        return ARNIQAProfileAdapter(spec, extractor)
    if isinstance(extractor, MANIQAExtractor):
        return MANIQAProfileAdapter(spec, extractor)
    raise TypeError(f"Unsupported extractor type for profiling: {type(extractor)!r}")


def _measure_flops(fn: Callable[[], torch.Tensor]) -> tuple[int, torch.Tensor]:
    with FlopCounterMode(display=False) as counter:
        output = fn()
    return int(counter.get_total_flops()), output


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_latency_ms(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(max(0, warmup)):
        fn()
    _synchronize_if_needed(device)

    values: list[float] = []
    for _ in range(max(1, iters)):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            values.append(float(start.elapsed_time(end)))
        else:
            started = time.perf_counter()
            fn()
            values.append(float((time.perf_counter() - started) * 1000.0))
    return values


def profile_adapter_target(
    adapter: BaseProfileAdapter,
    batch: PreparedBatch,
    target: ProfileTarget,
    *,
    batch_size: int,
    warmup: int,
    iters: int,
    extract_result_layer_map: dict[str, np.ndarray] | None = None,
    parity_atol: float = 1e-4,
    parity_rtol: float = 1e-4,
) -> TargetProfileResult:
    runner = adapter.make_runner(target, batch)
    try:
        flops_total, output = _measure_flops(runner.run)
        timings_ms = _measure_latency_ms(runner.run, device=adapter.device, warmup=warmup, iters=iters)
    finally:
        runner.close()

    input_height, input_width = adapter.primary_input_shape(batch)
    latency_ms_mean = float(np.mean(timings_ms))
    latency_ms_p50 = float(np.percentile(timings_ms, 50))
    latency_ms_p95 = float(np.percentile(timings_ms, 95))
    macs_total = flops_total // 2
    record = ProfileRecord(
        model_key=adapter.spec.key,
        family=adapter.spec.family,
        loader=adapter.spec.loader,
        layer_index=target.layer_index,
        layer_name=target.layer_name,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        device=str(adapter.device),
        dtype=_stringify_dtype(adapter.model_dtype),
        output_dim=int(output.shape[-1]),
        latency_ms_mean=latency_ms_mean,
        latency_ms_p50=latency_ms_p50,
        latency_ms_p95=latency_ms_p95,
        throughput_img_s=float(batch_size / max(latency_ms_mean / 1000.0, 1e-12)),
        flops_total=flops_total,
        macs_total=macs_total,
        flops_per_image=float(flops_total / max(batch_size, 1)),
        macs_per_image=float(macs_total / max(batch_size, 1)),
    )
    if extract_result_layer_map is not None:
        ok, max_abs, mean_abs = adapter.verify_against_extract_result(
            target,
            output,
            extract_result_layer_map,
            atol=parity_atol,
            rtol=parity_rtol,
        )
        record.parity_ok = ok
        record.parity_max_abs_error = max_abs
        record.parity_mean_abs_error = mean_abs
    return TargetProfileResult(record=record, output=output)


def profile_adapter_targets(
    adapter: BaseProfileAdapter,
    images: list[Image.Image],
    *,
    selectors: Sequence[str] | None = None,
    warmup: int = 10,
    iters: int = 30,
    verify_parity: bool = False,
    parity_atol: float = 1e-4,
    parity_rtol: float = 1e-4,
) -> list[ProfileRecord]:
    batch = adapter.prepare_batch(images)
    targets = adapter.list_targets()
    target_names = [item.layer_name for item in targets]
    selected_indices = resolve_profile_target_indices(target_names, selectors)
    selected_targets = [targets[idx] for idx in selected_indices]

    extract_result_layer_map: dict[str, np.ndarray] | None = None
    if verify_parity:
        extract_result = adapter.extractor.extract(images)
        extract_result_layer_map = dict(zip(extract_result.layer_names, extract_result.per_layer_np))

    records: list[ProfileRecord] = []
    previous: ProfileRecord | None = None
    for target in selected_targets:
        result = profile_adapter_target(
            adapter,
            batch,
            target,
            batch_size=len(images),
            warmup=warmup,
            iters=iters,
            extract_result_layer_map=extract_result_layer_map,
            parity_atol=parity_atol,
            parity_rtol=parity_rtol,
        )
        record = result.record
        if previous is not None:
            record.delta_latency_ms = float(record.latency_ms_mean - previous.latency_ms_mean)
            record.delta_flops_total = int(record.flops_total - previous.flops_total)
        records.append(record)
        previous = record
    return records


def profile_model_layers(
    model_key: str,
    images: list[Image.Image],
    *,
    device: str | None = None,
    weights_dir: Path = Path("weights"),
    selectors: Sequence[str] | None = None,
    warmup: int = 10,
    iters: int = 30,
    verify_parity: bool = False,
    parity_atol: float = 1e-4,
    parity_rtol: float = 1e-4,
) -> list[ProfileRecord]:
    spec = get_model_spec(model_key)
    device_obj = select_device(device)
    dtype = default_dtype_for_device(device_obj)
    with managed_extractor(spec, device=device_obj, dtype=dtype, weights_dir=weights_dir) as extractor:
        adapter = build_profile_adapter(spec, extractor)
        return profile_adapter_targets(
            adapter,
            images,
            selectors=selectors,
            warmup=warmup,
            iters=iters,
            verify_parity=verify_parity,
            parity_atol=parity_atol,
            parity_rtol=parity_rtol,
        )


def iter_selected_model_specs(model_keys: Sequence[str] | None = None) -> list[ModelSpec]:
    if model_keys:
        return [get_model_spec(key) for key in model_keys]
    return list(iter_enabled_image_model_specs())


__all__ = [
    "BaseProfileAdapter",
    "ProfileRecord",
    "ProfileTarget",
    "build_extractor",
    "build_profile_adapter",
    "default_dtype_for_device",
    "iter_selected_model_specs",
    "managed_extractor",
    "parse_profile_layer_selectors",
    "profile_adapter_target",
    "profile_adapter_targets",
    "profile_model_layers",
    "resolve_profile_target_indices",
    "select_device",
    "write_profile_results_table",
]
