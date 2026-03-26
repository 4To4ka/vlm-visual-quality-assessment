from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.triplet_evaluation import (  # noqa: E402
    TripletEvaluationConfig,
    _count_anchor_triplets,
    run_triplet_evaluation,
)


def _anchor_triplet_bruteforce(x_values: np.ndarray, y_values: np.ndarray) -> tuple[int, int]:
    total = 0
    correct = 0
    for left in range(len(x_values) - 1):
        for right in range(left + 1, len(x_values)):
            total += 2
            if (x_values[left] < x_values[right]) == (y_values[left] < y_values[right]):
                correct += 1
            if (x_values[right] < x_values[left]) == (y_values[right] < y_values[left]):
                correct += 1
    return total, correct


def _group_triplet_bruteforce(scores: np.ndarray, embeddings: np.ndarray) -> tuple[float, int, int]:
    total = 0
    correct = 0
    for anchor in range(len(scores)):
        others = [idx for idx in range(len(scores)) if idx != anchor]
        x_values = np.asarray(
            [abs(float(embeddings[anchor]) - float(embeddings[idx])) for idx in others],
            dtype=np.float64,
        )
        y_values = np.asarray(
            [abs(float(scores[anchor]) - float(scores[idx])) for idx in others],
            dtype=np.float64,
        )
        anchor_total, anchor_correct = _anchor_triplet_bruteforce(x_values, y_values)
        total += anchor_total
        correct += anchor_correct
    value = float("nan") if total == 0 else float(correct / total)
    return value, total, correct


def _write_dataset(
    datasets_root: Path,
    outputs_root: Path,
    *,
    dataset: str,
    model_key: str,
    scores: list[float],
    embeddings: list[float],
    ref_filenames: list[str] | None,
) -> None:
    dataset_dir = datasets_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    row_ids = np.arange(len(scores), dtype=np.int64)
    paths = [f"{dataset.lower()}_{idx:03d}.png" for idx in range(len(scores))]

    data = {
        "path": paths,
        "normalized_score": scores,
    }
    if ref_filenames is not None:
        data["ref_filename"] = ref_filenames
    pd.DataFrame(data).to_csv(dataset_dir / "data.csv", index=False)

    model_dir = outputs_root / dataset / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": row_ids, "path": paths}).to_parquet(model_dir / "index.parquet", index=False)
    (model_dir / "meta.json").write_text(json.dumps({"layer_names": ["embedding"]}), encoding="utf-8")

    with h5py.File(model_dir / "layers.h5", "w") as fp:
        fp.create_dataset("layer_000", data=np.asarray(embeddings, dtype=np.float32).reshape(-1, 1))


class TripletEvaluationTests(unittest.TestCase):
    def test_count_anchor_triplets_respects_strict_ties(self) -> None:
        x_values = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        y_values = np.asarray([0.0, 1.0, 1.0], dtype=np.float64)

        summary = _count_anchor_triplets(x_values, y_values)

        self.assertEqual(summary.total, 6)
        self.assertEqual(summary.correct, 4)
        self.assertEqual(summary.ties_x, 1)
        self.assertEqual(summary.ties_y, 1)
        self.assertEqual(summary.ties_xy, 0)
        self.assertEqual(summary.discordant, 0)

    def test_run_triplet_evaluation_within_ref_uses_group_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_path = Path(tmp_dir_str)
            datasets_root = tmp_path / "datasets"
            outputs_root = tmp_path / "outputs"
            run_dir = tmp_path / "run"
            tmp_dir = tmp_path / "tmp"
            model_key = "demo_model"

            scores = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
            embeddings = np.asarray([0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            refs = ["ref_a.png", "ref_a.png", "ref_a.png", "ref_b.png", "ref_b.png", "ref_b.png", "ref_b.png"]

            _write_dataset(
                datasets_root,
                outputs_root,
                dataset="CSIQ",
                model_key=model_key,
                scores=scores.tolist(),
                embeddings=embeddings.tolist(),
                ref_filenames=refs,
            )

            config = TripletEvaluationConfig(
                datasets_root=datasets_root,
                outputs_root=outputs_root,
                datasets=("CSIQ",),
                models=(model_key,),
                layer_selectors=("0",),
                target_field="normalized_score",
                sample_limit=None,
                seed=42,
                embedding_distances=("l2",),
                pair_scope="auto",
                block_size=2,
                jobs=1,
                tmp_dir=tmp_dir,
                run_dir=run_dir,
                keep_cache=False,
                resume=False,
                progress_mode="off",
                heartbeat_sec=30,
                fail_fast=True,
            )

            report = run_triplet_evaluation(config)
            row = report["results"][0]
            summary = report["dataset_summaries"][0]

            group_a = _group_triplet_bruteforce(scores[:3], embeddings[:3])
            group_b = _group_triplet_bruteforce(scores[3:], embeddings[3:])
            expected_group_mean = (group_a[0] + group_b[0]) / 2.0
            expected_pooled = float((group_a[2] + group_b[2]) / (group_a[1] + group_b[1]))

            self.assertEqual(summary["pair_scope"], "within_ref")
            self.assertEqual(summary["group_field"], "ref_filename")
            self.assertEqual(summary["n_groups"], 2)
            self.assertEqual(row["pair_scope"], "within_ref")
            self.assertEqual(row["group_field"], "ref_filename")
            self.assertEqual(row["n_groups_total"], 2)
            self.assertEqual(row["n_groups_used"], 2)
            self.assertEqual(row["n_pairs"], 9)
            self.assertEqual(row["n_triplets_total"], 30)
            self.assertEqual(row["n_triplets_correct"], group_a[2] + group_b[2])
            self.assertTrue(np.isclose(row["value"], expected_group_mean))
            self.assertTrue(np.isclose(row["pooled_value"], expected_pooled))
            self.assertFalse(np.isclose(row["value"], row["pooled_value"]))

    def test_run_triplet_evaluation_auto_uses_global_for_nr_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_path = Path(tmp_dir_str)
            datasets_root = tmp_path / "datasets"
            outputs_root = tmp_path / "outputs"
            run_dir = tmp_path / "run"
            tmp_dir = tmp_path / "tmp"
            model_key = "demo_model"

            scores = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
            embeddings = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

            _write_dataset(
                datasets_root,
                outputs_root,
                dataset="NRDATA",
                model_key=model_key,
                scores=scores.tolist(),
                embeddings=embeddings.tolist(),
                ref_filenames=None,
            )

            config = TripletEvaluationConfig(
                datasets_root=datasets_root,
                outputs_root=outputs_root,
                datasets=("NRDATA",),
                models=(model_key,),
                layer_selectors=("0",),
                target_field="normalized_score",
                sample_limit=None,
                seed=42,
                embedding_distances=("l2",),
                pair_scope="auto",
                block_size=2,
                jobs=1,
                tmp_dir=tmp_dir,
                run_dir=run_dir,
                keep_cache=False,
                resume=False,
                progress_mode="off",
                heartbeat_sec=30,
                fail_fast=True,
            )

            report = run_triplet_evaluation(config)
            row = report["results"][0]
            summary = report["dataset_summaries"][0]

            expected_value, expected_total, expected_correct = _group_triplet_bruteforce(scores, embeddings)

            self.assertEqual(summary["pair_scope"], "global")
            self.assertIsNone(summary["group_field"])
            self.assertEqual(summary["n_groups"], 1)
            self.assertEqual(summary["n_triplets"], expected_total)
            self.assertEqual(row["pair_scope"], "global")
            self.assertIsNone(row["group_field"])
            self.assertEqual(row["n_groups_total"], 1)
            self.assertEqual(row["n_groups_used"], 1)
            self.assertEqual(row["n_pairs"], 6)
            self.assertEqual(row["n_triplets_total"], expected_total)
            self.assertEqual(row["n_triplets_correct"], expected_correct)
            self.assertTrue(np.isclose(row["value"], expected_value))
            self.assertTrue(np.isclose(row["pooled_value"], expected_value))


if __name__ == "__main__":
    unittest.main()
