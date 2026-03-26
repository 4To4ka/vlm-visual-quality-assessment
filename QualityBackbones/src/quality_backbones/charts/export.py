from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from quality_backbones.charts.registry import FigureSpec, REPO_ROOT
from quality_backbones.charts.style import EXPORT_SEED, get_style_profile


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _default_title(slug: str) -> str:
    return slug.replace("__", " ").replace("_", " ").strip().title()


def build_figure_record(
    *,
    experiment_slug: str,
    spec: FigureSpec,
    slug: str,
    output_path: Path,
    inputs: dict[str, Any],
    title: str | None = None,
) -> dict[str, Any]:
    return {
        "figure_id": f"{experiment_slug}/{slug}",
        "title": title or _default_title(slug),
        "inputs": inputs,
        "style_profile": get_style_profile(),
        "seed": EXPORT_SEED,
        "output_paths": {output_path.suffix.lstrip("."): str(output_path)},
        "timestamp": _utc_timestamp(),
        "git_commit": _git_commit(),
        "renderer": spec.renderer,
        "tier": spec.tier,
        "preset": spec.preset,
        "layout": spec.layout,
        "variant": spec.variant,
        "annotation_policy": spec.annotation_policy,
        "review_priority": spec.review_priority,
    }


def write_figure_manifest(
    path: Path,
    *,
    experiment_slug: str,
    records: list[dict[str, Any]],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment_slug,
        "style_profile": get_style_profile(),
        "seed": EXPORT_SEED,
        "generated_at": _utc_timestamp(),
        "git_commit": _git_commit(),
        "figures": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)
