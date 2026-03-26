from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.manifest import ModelSpec  # noqa: E402
from quality_backbones.profiling import (  # noqa: E402
    BaseProfileAdapter,
    ProfileTarget,
    _CallableForwardRunner,
    _HookedForwardRunner,
    _extract_first_tensor,
    _measure_flops,
    profile_adapter_targets,
    resolve_profile_target_indices,
)


class _ToyExtractor:
    def __init__(self, model: nn.Module, batch: torch.Tensor) -> None:
        self.device = torch.device("cpu")
        self.model_dtype = torch.float32
        self.autocast_enabled = False
        self.model = model
        self.batch = batch

    def extract(self, _images: list[Image.Image]):
        with torch.inference_mode():
            first = self.model.l1(self.batch)
            second = self.model.l2(first)
        return type(
            "ExtractResult",
            (),
            {
                "layer_names": ["layer_0", "layer_1"],
                "per_layer_np": [first.detach().cpu().numpy(), second.detach().cpu().numpy()],
            },
        )()


class _ToyAdapter(BaseProfileAdapter):
    def __init__(self, spec: ModelSpec, extractor: _ToyExtractor) -> None:
        super().__init__(spec, extractor)
        self.model = extractor.model

    def prepare_batch(self, _images: list[Image.Image]):
        return self.extractor.batch

    def list_targets(self) -> list[ProfileTarget]:
        return [ProfileTarget(0, "layer_0"), ProfileTarget(1, "layer_1")]

    def make_runner(self, target: ProfileTarget, batch):
        x = batch
        if target.layer_name == "layer_0":
            return _HookedForwardRunner(
                target_module=self.model.l1,
                output_transform=lambda raw: _extract_first_tensor(raw),
                forward_fn=lambda: self.model(x),
            )
        return _CallableForwardRunner(lambda: self.model(x))


class ProfilingTests(unittest.TestCase):
    def test_resolve_profile_target_indices_supports_mixed_selectors(self) -> None:
        target_names = ["hidden_state_000", "hidden_state_001", "hidden_state_002", "canonical_embedding"]

        selected = resolve_profile_target_indices(target_names, ("0-1", "last", "hidden_state_001"))

        self.assertEqual(selected, [0, 1, 3])

    def test_hooked_forward_runner_counts_prefix_flops_only(self) -> None:
        class ToyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = nn.Linear(8, 16, bias=True)
                self.l2 = nn.Linear(16, 4, bias=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.l1(x)
                return self.l2(x)

        model = ToyModel().eval()
        batch = torch.randn(2, 8)
        runner = _HookedForwardRunner(
            target_module=model.l1,
            output_transform=lambda raw: _extract_first_tensor(raw),
            forward_fn=lambda: model(batch),
        )
        try:
            flops_total, output = _measure_flops(runner.run)
        finally:
            runner.close()

        self.assertEqual(flops_total, 512)
        self.assertEqual(tuple(output.shape), (2, 16))

    def test_profile_adapter_targets_computes_deltas_and_parity(self) -> None:
        class ToyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = nn.Linear(4, 4, bias=False)
                self.l2 = nn.Linear(4, 4, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.l1(x)
                return self.l2(x)

        torch.manual_seed(0)
        model = ToyModel().eval()
        batch = torch.randn(3, 4)
        extractor = _ToyExtractor(model, batch)
        spec = ModelSpec("toy", "Toy", "tiny", "unit", "unit")
        adapter = _ToyAdapter(spec, extractor)

        records = profile_adapter_targets(
            adapter,
            images=[Image.new("RGB", (4, 4))],
            warmup=0,
            iters=1,
            verify_parity=True,
        )

        self.assertEqual(len(records), 2)
        self.assertIsNone(records[0].delta_flops_total)
        self.assertIsNone(records[0].delta_latency_ms)
        self.assertTrue(records[0].parity_ok)
        self.assertTrue(records[1].parity_ok)
        self.assertGreater(records[1].flops_total, records[0].flops_total)
        self.assertGreater(records[1].delta_flops_total or 0, 0)
        self.assertAlmostEqual(records[0].parity_max_abs_error or 0.0, 0.0)
        self.assertAlmostEqual(records[1].parity_mean_abs_error or 0.0, 0.0)
        self.assertEqual(records[0].output_dim, 4)
        self.assertEqual(records[1].output_dim, 4)


if __name__ == "__main__":
    unittest.main()
