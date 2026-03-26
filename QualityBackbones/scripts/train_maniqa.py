from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.maniqa_training import (
    list_supported_transformer_encoder_keys,
    run_maniqa_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train MANIQA-based IQA metric with a swappable transformer vision encoder."
        )
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Cache/weights root for encoder + MANIQA repo assets",
    )

    parser.add_argument(
        "--train-source",
        action="append",
        default=None,
        help="Training source spec: dataset[:fraction]. Repeatable.",
    )
    parser.add_argument(
        "--val-source",
        action="append",
        default=None,
        help=(
            "Optional explicit validation source spec: dataset[:fraction]. Repeatable. "
            "If omitted, validation is split from train sources via --split-policy."
        ),
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="normalized_score",
        help="Target field from data CSV columns or metadata path like metadata.score",
    )
    parser.add_argument(
        "--split-policy",
        choices=["predefined", "random", "group"],
        default="predefined",
        help="Split policy when --val-source is omitted",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--group-field",
        type=str,
        default=None,
        help=(
            "Optional grouping field for group-aware split. "
            "Supports dataframe columns or metadata path (metadata.<key>)."
        ),
    )
    parser.add_argument(
        "--train-split-labels",
        type=str,
        default="train,training",
        help="Comma-separated split labels used as train for predefined policy",
    )
    parser.add_argument(
        "--val-split-labels",
        type=str,
        default="val,validation,valid,test",
        help="Comma-separated split labels used as val for predefined policy",
    )
    parser.add_argument(
        "--ranking-sampling",
        choices=["global", "within_group"],
        default="global",
        help="Compatibility flag for shared split prep; keep default for MANIQA regression",
    )

    parser.add_argument(
        "--encoder-model",
        type=str,
        default=None,
        help="Transformer model key from manifest to use as MANIQA vision encoder",
    )
    parser.add_argument(
        "--list-encoders",
        action="store_true",
        help="List supported transformer encoder keys and exit",
    )
    parser.add_argument(
        "--encoder-layer-indices",
        type=str,
        default="",
        help="Optional comma-separated layer indices (supports negative), e.g. -4,-3,-2,-1",
    )
    parser.add_argument(
        "--num-feature-layers",
        type=int,
        default=4,
        help="How many encoder hidden layers to feed into MANIQA pipeline",
    )
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze encoder weights and train only MANIQA head/projectors",
    )

    parser.add_argument("--maniqa-embed-dim", type=int, default=768)
    parser.add_argument("--maniqa-dropout", type=float, default=0.1)
    parser.add_argument("--maniqa-depths", type=str, default="2,2")
    parser.add_argument("--maniqa-num-heads", type=str, default="4,4")
    parser.add_argument("--maniqa-window-size", type=int, default=4)
    parser.add_argument("--maniqa-dim-mlp", type=int, default=768)
    parser.add_argument("--maniqa-num-tab", type=int, default=2)
    parser.add_argument("--maniqa-scale", type=float, default=0.8)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["none", "cosine", "linear"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    parser.add_argument("--optimizer-beta2", type=float, default=0.999)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--sgd-momentum", type=float, default=0.9)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--encoder-weight-decay", type=float, default=None)
    parser.add_argument("--head-weight-decay", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="auto")
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TensorBoard logging (use --no-tensorboard to disable)",
    )
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Lightning limit_train_batches value (fraction or absolute count)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Lightning limit_val_batches value (fraction or absolute count)",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maniqa_training_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--save-epoch-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save epoch-level metrics to CSV and JSONL",
    )
    parser.add_argument(
        "--save-val-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save validation predictions per epoch to CSV and JSONL",
    )
    parser.add_argument(
        "--val-predictions-max-rows",
        type=int,
        default=0,
        help="Optional cap for saved validation prediction rows per epoch (0 = all)",
    )

    parser.add_argument(
        "--list-target-fields",
        action="store_true",
        help="Print available fields for selected datasets and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare splits/config and model graph init; skip training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_encoders:
        for key in list_supported_transformer_encoder_keys():
            print(key)
        return

    run_maniqa_training(args)


if __name__ == "__main__":
    main()
