from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
CHARTS_ROOT = REPO_ROOT / "charts"
REGISTRY_ROOT = CHARTS_ROOT / "registry"


@dataclass(frozen=True)
class ExperimentSpec:
    slug: str
    title: str
    kind: str
    slice_name: str | None
    quality_report: Path | None
    quality_report_alt: Path | None
    alignment_report: Path | None
    triplet_report: Path | None
    top_k: int
    exclude_families: tuple[str, ...]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class FigureSpec:
    slug: str
    renderer: str
    tier: str
    preset: str
    family: str | None = None
    layout: str | None = None
    variant: str | None = None
    annotation_policy: str | None = None
    export_formats: tuple[str, ...] = ("pdf",)
    review_priority: str = "medium"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_optional_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    return (REPO_ROOT / raw).resolve()


def load_experiments(path: Path | None = None) -> list[ExperimentSpec]:
    registry_path = path or (REGISTRY_ROOT / "experiments.json")
    payload = _read_json(registry_path)
    experiments: list[ExperimentSpec] = []
    for item in payload.get("experiments", []):
        experiments.append(
            ExperimentSpec(
                slug=str(item["slug"]),
                title=str(item.get("title", item["slug"])),
                kind=str(item.get("kind", "report")),
                slice_name=item.get("slice"),
                quality_report=_resolve_optional_path(item.get("quality_report")),
                quality_report_alt=_resolve_optional_path(item.get("quality_report_alt")),
                alignment_report=_resolve_optional_path(item.get("alignment_report")),
                triplet_report=_resolve_optional_path(item.get("triplet_report")),
                top_k=int(item.get("top_k", 10)),
                exclude_families=tuple(str(value) for value in item.get("exclude_families", [])),
                depends_on=tuple(str(value) for value in item.get("depends_on", [])),
            )
        )
    return experiments


def load_figure_registry(path: Path | None = None) -> dict[str, list[FigureSpec]]:
    registry_path = path or (REGISTRY_ROOT / "figures.json")
    payload = _read_json(registry_path)
    parsed: dict[str, list[FigureSpec]] = {}
    for key, items in payload.items():
        parsed[key] = [
            FigureSpec(
                slug=str(item["slug"]),
                renderer=str(item["renderer"]),
                tier=str(item.get("tier", "paper")),
                preset=str(item.get("preset", "double_column")),
                family=item.get("family"),
                layout=item.get("layout"),
                variant=item.get("variant"),
                annotation_policy=item.get("annotation_policy"),
                export_formats=tuple(str(value) for value in item.get("export_formats", ["pdf"])),
                review_priority=str(item.get("review_priority", "medium")),
            )
            for item in items
        ]
    return parsed


def load_style_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = path or (REGISTRY_ROOT / "style_presets.json")
    return _read_json(registry_path)


def get_experiment_map(experiments: list[ExperimentSpec]) -> dict[str, ExperimentSpec]:
    return {experiment.slug: experiment for experiment in experiments}
