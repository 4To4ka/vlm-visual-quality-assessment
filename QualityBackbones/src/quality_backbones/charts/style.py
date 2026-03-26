from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from quality_backbones.charts.registry import load_style_registry

matplotlib.use("Agg")

BACKGROUND = "#ffffff"
PANEL = "#ffffff"
INK = "#111111"
MUTED = "#5f6368"
GRID = "#d9d9d9"
TEAL = "#1f77b4"
GOLD = "#c69214"
CORAL = "#c75d3b"
NAVY = "#1f3b73"
SAND = "#f4f4f4"
MISSING = "#efefef"
HIGHLIGHT = "#eab308"
EXPORT_SEED = 0

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "quality_backbones_charts_heatmap",
    ["#f7fbff", "#d6e6f4", "#9ecae1", "#4f93c5", "#0b4f8a"],
)

LAYER_DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "quality_backbones_layer_depth",
    ["#b8d9f0", "#7fb6dc", "#3f89bf", "#0b4f8a"],
)

FAMILY_PALETTE = [
    "#264653",
    "#2a9d8f",
    "#d9a441",
    "#d06b4d",
    "#7f5539",
    "#669bbc",
    "#588157",
    "#8d99ae",
    "#bc6c25",
    "#3d405b",
    "#6a994e",
    "#b56576",
    "#457b9d",
    "#9c6644",
    "#4d908e",
    "#ffb703",
]

FALLBACK_STYLE_REGISTRY: dict[str, Any] = {
    "conference": "ACM Multimedia 2026",
    "style_profile": "research_charts_acmmm_2026",
    "palette_policy": "preserve_existing_palette_where_possible",
    "font_family": "DejaVu Serif",
    "export": {
        "creator": "research-charts",
        "subject": "QualityBackbones publication figures",
        "pad_inches": 0.04,
    },
    "typography": {
        "figure_title_size": 11.0,
        "panel_title_size": 10.0,
        "panel_subtitle_size": 8.0,
        "axis_label_size": 9.2,
        "tick_label_size": 8.2,
        "legend_size": 7.8,
        "annotation_size": 7.5,
        "footnote_size": 7.4,
    },
    "lines": {
        "thin": 0.9,
        "normal": 1.2,
        "thick": 1.8,
        "grid_alpha": 0.22,
    },
    "markers": {
        "small": 18,
        "medium": 34,
        "large": 58,
        "highlight": 92,
    },
    "presets": {
        "paper_1col": {"width": 3.35, "height": 2.45, "dpi": 320},
        "paper_1p5col": {"width": 5.3, "height": 3.2, "dpi": 320},
        "paper_2col": {"width": 7.1, "height": 3.9, "dpi": 320},
        "paper_2col_tall": {"width": 7.1, "height": 4.75, "dpi": 320},
        "supp_fullwidth": {"width": 10.4, "height": 4.9, "dpi": 320},
        "supp_tall": {"width": 10.4, "height": 6.6, "dpi": 320},
        "layerwise_sheet": {"width": 11.6, "height": 10.6, "dpi": 320},
        "single_column": {"width": 3.35, "height": 2.45, "dpi": 320},
        "one_and_half_column": {"width": 5.3, "height": 3.2, "dpi": 320},
        "double_column": {"width": 7.1, "height": 3.9, "dpi": 320},
        "paper_landscape": {"width": 7.1, "height": 3.9, "dpi": 320},
        "supplementary_page": {"width": 10.4, "height": 6.6, "dpi": 320},
        "layerwise_page": {"width": 11.6, "height": 9.8, "dpi": 320},
    },
}


@dataclass(frozen=True)
class FigurePreset:
    name: str
    width: float
    height: float
    dpi: int = 300


@lru_cache(maxsize=1)
def get_style_registry() -> dict[str, Any]:
    try:
        return load_style_registry()
    except Exception:
        return FALLBACK_STYLE_REGISTRY


def get_style_value(section: str, key: str, default: float | int | str) -> float | int | str:
    registry = get_style_registry()
    section_payload = registry.get(section, {})
    if not isinstance(section_payload, dict):
        return default
    return section_payload.get(key, default)


def get_style_profile() -> str:
    registry = get_style_registry()
    return str(registry.get("style_profile", FALLBACK_STYLE_REGISTRY["style_profile"]))


def get_export_value(key: str, default: float | int | str) -> float | int | str:
    registry = get_style_registry()
    export_payload = registry.get("export", {})
    if not isinstance(export_payload, dict):
        return default
    return export_payload.get(key, default)


def get_line_width(name: str) -> float:
    return float(get_style_value("lines", name, FALLBACK_STYLE_REGISTRY["lines"][name]))


def get_marker_size(name: str) -> float:
    return float(get_style_value("markers", name, FALLBACK_STYLE_REGISTRY["markers"][name]))


def get_font_size(name: str) -> float:
    return float(get_style_value("typography", name, FALLBACK_STYLE_REGISTRY["typography"][name]))


def configure_style() -> None:
    registry = get_style_registry()
    font_family = str(registry.get("font_family", FALLBACK_STYLE_REGISTRY["font_family"]))
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "grid.color": GRID,
            "font.family": font_family,
            "axes.titlesize": get_font_size("panel_title_size"),
            "axes.titleweight": "bold",
            "axes.labelsize": get_font_size("axis_label_size"),
            "xtick.labelsize": get_font_size("tick_label_size"),
            "ytick.labelsize": get_font_size("tick_label_size"),
            "figure.titlesize": get_font_size("figure_title_size"),
            "legend.fontsize": get_font_size("legend_size"),
            "legend.frameon": False,
            "legend.title_fontsize": get_font_size("legend_size"),
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_preset(name: str) -> FigurePreset:
    registry = get_style_registry()
    presets = registry.get("presets", FALLBACK_STYLE_REGISTRY["presets"])
    raw = presets.get(name)
    if raw is None:
        raise KeyError(f"Unknown figure preset: {name}")
    return FigurePreset(
        name=name,
        width=float(raw["width"]),
        height=float(raw["height"]),
        dpi=int(raw.get("dpi", 320)),
    )


def create_family_color_map(families: list[str]) -> dict[str, str]:
    unique_families: list[str] = []
    for family in families:
        if family not in unique_families:
            unique_families.append(family)
    return {
        family: FAMILY_PALETTE[index % len(FAMILY_PALETTE)]
        for index, family in enumerate(unique_families)
    }


def soften_color(color: str, amount: float = 0.78) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(color), dtype=np.float64)
    return tuple((1.0 - amount) * rgb + amount * np.ones(3, dtype=np.float64))


def format_score(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    value_f = float(value)
    if np.isnan(value_f):
        return "--"
    return f"{value_f:.{digits}f}"


def short_layer_name(name: str) -> str:
    text = str(name)
    replacements = {
        "canonical_embedding": "canon",
        "pooler_output": "pool",
        "image_embeds_l2": "img_l2",
        "image_embeds": "img",
        "image_embeddings": "img",
        "arniqa_embedding": "arniqa",
        "maniqa_embedding": "maniqa",
    }
    if text in replacements:
        return replacements[text]
    if text.startswith("hidden_state_"):
        return f"hs{text.rsplit('_', maxsplit=1)[-1]}"
    if text.startswith("block_"):
        return f"b{text.rsplit('_', maxsplit=1)[-1]}"
    if text.startswith("feature_map_"):
        return f"fm{text.rsplit('_', maxsplit=1)[-1]}"
    compact = text.replace("_", " ")
    if len(compact) <= 14:
        return compact
    return f"{compact[:11]}..."


def save_figure(fig: plt.Figure, path: Path, *, dpi: int | None = None, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {
        "bbox_inches": "tight",
        "pad_inches": float(get_export_value("pad_inches", 0.04)),
        "metadata": {
            "Creator": str(get_export_value("creator", "research-charts")),
            "Title": title or path.stem,
            "Subject": str(get_export_value("subject", get_style_profile())),
        },
    }
    if path.suffix.lower() == ".png":
        save_kwargs["dpi"] = dpi or 320
    fig.savefig(path, **save_kwargs)
