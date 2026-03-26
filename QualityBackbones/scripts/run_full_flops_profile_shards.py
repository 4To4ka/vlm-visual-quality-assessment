from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "scripts" / "profile_layer_performance.py"

SHARDS: dict[str, list[str]] = {
    "cnn_core": [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
        "fastvit_t8",
        "fastvit_t12",
        "fastvit_s12",
        "fastvit_sa12",
        "fastvit_sa24",
        "fastvit_sa36",
        "fastvit_ma36",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "swin_tiny",
        "swin_small",
        "swin_base",
        "swin_large",
        "swinv2_tiny",
        "swinv2_small",
        "swinv2_base",
        "swinv2_large",
    ],
    "vit_clip_mae": [
        "vit_tiny",
        "vit_small",
        "vit_base",
        "vit_large",
        "vit_huge",
        "vit_mae_base",
        "vit_mae_large",
        "vit_mae_huge",
        "clip_vit_b32",
        "clip_vit_b16",
        "clip_vit_l14",
        "clip_vit_l14_336",
    ],
    "ssl_core": [
        "dino_vits8",
        "dino_vits16",
        "dino_vitb8",
        "dino_vitb16",
        "dinov2_small",
        "dinov2_base",
        "dinov2_large",
        "dinov2_giant",
        "ijepa_vith14",
    ],
    "dinov3": [
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
        "dinov3_vith16plus",
        "dinov3_vit7b16",
    ],
    "siglip": [
        "siglip_base",
        "siglip_large",
        "siglip_so400m",
        "siglip2_base",
        "siglip2_large",
        "siglip2_so400m",
        "siglip2_base_naflex",
        "siglip2_so400m_naflex",
    ],
    "remote_heavy": [
        "internvit_300m",
        "internvit_300m_v25",
        "internvit_6b_v25",
        "fastvithd_05b",
        "fastvithd_15b",
        "fastvithd_7b",
    ],
    "iqa_heads": [
        "arniqa_live",
        "arniqa_csiq",
        "arniqa_tid2013",
        "arniqa_kadid10k",
        "arniqa_flive",
        "arniqa_spaq",
        "arniqa_clive",
        "arniqa_koniq10k",
        "maniqa_pipal22",
        "maniqa_kadid10k",
        "maniqa_koniq10k",
    ],
}


def _resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _default_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    stamp = _default_stamp()
    parser = argparse.ArgumentParser(description="Run full FLOPs profiling shards sequentially")
    parser.add_argument("--datasets-root", type=Path, default=Path("/mnt/l/IQA"))
    parser.add_argument("--dataset", type=str, default="koniq10k")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs") / "profile_runs" / f"full_flops_{stamp}",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs") / f"full_flops_{stamp}",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--shards", nargs="*", default=list(SHARDS.keys()))
    parser.add_argument("--verify-parity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _build_command(args: argparse.Namespace, shard: str, run_root: Path, log_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--datasets-root",
        str(args.datasets_root),
        "--dataset",
        args.dataset,
        "--weights-dir",
        str(args.weights_dir),
        "--models",
        *SHARDS[shard],
        "--max-samples",
        str(args.max_samples),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--run-dir",
        str(run_root / shard),
        "--table-out",
        str(log_root / f"{shard}.tsv"),
        "--json-out",
        str(log_root / f"{shard}.json"),
    ]
    if args.verify_parity:
        command.append("--verify-parity")
    if args.resume or (run_root / shard).exists():
        command.append("--resume")
    return command


def _read_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_launcher_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = _resolve_from_root(args.run_root)
    log_root = _resolve_from_root(args.log_root)
    args.datasets_root = _resolve_from_root(args.datasets_root)
    args.weights_dir = _resolve_from_root(args.weights_dir)

    unknown = [name for name in args.shards if name not in SHARDS]
    if unknown:
        raise SystemExit(f"Unknown shard names: {', '.join(unknown)}")

    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    launcher_state_path = run_root / "launcher_state.json"
    config = {
        "datasets_root": str(args.datasets_root),
        "dataset": args.dataset,
        "weights_dir": str(args.weights_dir),
        "run_root": str(run_root),
        "log_root": str(log_root),
        "device": args.device,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "verify_parity": bool(args.verify_parity),
        "resume": bool(args.resume),
        "shards": [{"name": name, "models": SHARDS[name]} for name in args.shards],
    }
    launch_started_at = datetime.now(timezone.utc).isoformat()
    _write_launcher_state(
        launcher_state_path,
        {
            "config": config,
            "started_at": launch_started_at,
            "shards": [],
        },
    )

    shard_statuses: list[dict[str, Any]] = []
    any_failure = False
    for shard in args.shards:
        shard_run_dir = run_root / shard
        shard_stdout_log = log_root / f"{shard}.stdout.log"
        command = _build_command(args, shard, run_root, log_root)
        started_at = datetime.now(timezone.utc).isoformat()
        print(f"=== START {shard} ===")
        print(shlex.join(command))
        with shard_stdout_log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# START {started_at}\n")
            handle.write(f"# COMMAND {shlex.join(command)}\n")
            handle.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            finished_at = datetime.now(timezone.utc).isoformat()
            handle.write(f"# END {finished_at} returncode={result.returncode}\n")

        report = _read_report(shard_run_dir / "report.json")
        status = {
            "name": shard,
            "started_at": started_at,
            "finished_at": finished_at,
            "returncode": result.returncode,
            "stdout_log": str(shard_stdout_log),
            "run_dir": str(shard_run_dir),
            "table_out": str(log_root / f"{shard}.tsv"),
            "json_out": str(log_root / f"{shard}.json"),
            "report_rows": report.get("rows") if report else None,
            "report_errors": report.get("errors") if report else None,
            "report_models": report.get("models") if report else None,
        }
        shard_statuses.append(status)
        _write_launcher_state(
            launcher_state_path,
            {
                "config": config,
                "started_at": launch_started_at,
                "updated_at": finished_at,
                "shards": shard_statuses,
            },
        )
        print(
            f"=== END {shard} returncode={result.returncode} "
            f"rows={status['report_rows']} errors={status['report_errors']} ==="
        )
        if result.returncode != 0:
            any_failure = True
            if args.stop_on_error:
                break

    _write_launcher_state(
        launcher_state_path,
        {
            "config": config,
            "started_at": launch_started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "shards": shard_statuses,
            "any_failure": any_failure,
        },
    )
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
