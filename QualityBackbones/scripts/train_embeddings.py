from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quality_backbones.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train torch/lightning models either on extracted embeddings or by fine-tuning encoders."
        )
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--train-mode",
        choices=["embeddings", "encoder"],
        default="embeddings",
        help="`embeddings`: train on extracted features, `encoder`: fine-tune encoder + head on images",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default=None,
        help="Model key from manifest for --train-mode=encoder",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Cache/weights root used by encoder training",
    )

    parser.add_argument(
        "--train-source",
        action="append",
        required=True,
        help="Training source spec: dataset[:fraction]. Repeatable.",
    )
    parser.add_argument(
        "--val-source",
        action="append",
        default=None,
        help=(
            "Optional explicit validation source spec: dataset[:fraction]. Repeatable. "
            "If omitted, validation is split from train sources using --split-policy."
        ),
    )

    parser.add_argument(
        "--feature-source",
        action="append",
        default=[],
        help=(
            "Feature source for embeddings mode: model_key:selector. Repeatable. "
            "selector supports all/*, indices (0,2,-1), ranges (0-5), names, or comma combinations."
        ),
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="normalized_score",
        help="Target field from index/data CSV columns or metadata path like metadata.MOS",
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
            "Optional grouping field for group-aware split/ranking sampling. "
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
        "--task",
        choices=["regression", "pairwise", "listwise"],
        default="regression",
        help="Training mode",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=4,
        help="N choices for listwise mode (ignored in pairwise mode)",
    )
    parser.add_argument("--pairwise-target", choices=["hard", "soft"], default="hard")
    parser.add_argument("--listwise-target", choices=["hard", "soft"], default="hard")
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=1.0,
        help="Temperature used to build soft GT target distributions",
    )
    parser.add_argument(
        "--learnable-logit-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable CLIP-style learnable logit scale for ranking losses "
            "(pairwise/listwise tasks)."
        ),
    )
    parser.add_argument(
        "--logit-temperature-init",
        type=float,
        default=0.07,
        help="Initial model logit temperature used when --learnable-logit-scale is enabled",
    )
    parser.add_argument(
        "--logit-scale-max",
        type=float,
        default=100.0,
        help="Maximum CLIP-style logit scale value (after exp) when learnable",
    )
    parser.add_argument(
        "--ranking-sampling",
        choices=["within_dataset", "global", "within_group"],
        default="within_dataset",
        help="How to build ranking candidate groups",
    )
    parser.add_argument(
        "--train-ranking-groups",
        type=int,
        default=0,
        help="Number of ranking groups per train epoch (0 -> auto)",
    )
    parser.add_argument(
        "--val-ranking-groups",
        type=int,
        default=0,
        help="Number of ranking groups for validation (0 -> auto)",
    )

    parser.add_argument("--branch-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--head-type",
        choices=["linear", "mlp", "mlp_fusion"],
        default="mlp",
    )
    parser.add_argument(
        "--init-head-from",
        type=Path,
        default=None,
        help="Optional checkpoint/state_dict path to initialize head weights",
    )
    parser.add_argument(
        "--tune-mode",
        choices=["frozen", "full", "lora"],
        default="frozen",
        help="Encoder tuning mode for --train-mode=encoder",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing when supported by encoder",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", choices=["none", "all", "lora_only"], default="none")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="",
        help="Comma-separated LoRA target module names (auto-infer when empty)",
    )
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Validation patience in epochs for early stopping (0 disables)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
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

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training_runs"))
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
        help="Prepare splits/features/config and exit without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
