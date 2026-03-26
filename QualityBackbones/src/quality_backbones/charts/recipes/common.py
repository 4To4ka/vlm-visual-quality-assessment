from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from quality_backbones.charts.style import (
    BACKGROUND,
    GOLD,
    GRID,
    HEATMAP_CMAP,
    INK,
    MISSING,
    MUTED,
    PANEL,
    TEAL,
    format_score,
    get_font_size,
    get_line_width,
    get_marker_size,
    get_style_value,
    short_layer_name,
    soften_color,
)


def apply_card(ax: plt.Axes, *, title: str | None = None, subtitle: str | None = None) -> None:
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_linewidth(get_line_width("thin"))
    ax.spines["bottom"].set_linewidth(get_line_width("thin"))
    if title:
        ax.text(
            0.0,
            1.105,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=get_font_size("panel_title_size"),
            fontweight="bold",
            color=INK,
            clip_on=False,
        )
    if subtitle:
        ax.text(
            0.0,
            1.03,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=get_font_size("panel_subtitle_size"),
            color=MUTED,
            wrap=True,
            clip_on=False,
        )


def _grid_alpha() -> float:
    return float(get_style_value("lines", "grid_alpha", 0.22))


def _candidate_offsets(x_span: float, y_span: float) -> list[tuple[float, float]]:
    return [
        (0.015 * x_span, 0.018 * y_span),
        (0.016 * x_span, -0.018 * y_span),
        (-0.065 * x_span, 0.018 * y_span),
        (-0.065 * x_span, -0.02 * y_span),
        (0.006 * x_span, 0.038 * y_span),
        (0.006 * x_span, -0.04 * y_span),
        (0.05 * x_span, 0.0),
        (-0.09 * x_span, 0.0),
        (0.03 * x_span, 0.055 * y_span),
        (0.03 * x_span, -0.055 * y_span),
        (-0.1 * x_span, 0.05 * y_span),
        (-0.1 * x_span, -0.05 * y_span),
    ]


def _annotate_points_greedy(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    label_col: str,
    color_col: str | None = None,
    default_color: str = INK,
) -> None:
    if data.empty:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_span = max(1e-6, float(data[x_col].max()) - float(data[x_col].min()))
    y_span = max(1e-6, float(data[y_col].max()) - float(data[y_col].min()))
    axis_bbox = ax.get_window_extent(renderer=renderer)
    placed_bboxes = []
    offsets = _candidate_offsets(x_span, y_span)
    for _, row in data.iterrows():
        x_value = float(row[x_col])
        y_value = float(row[y_col])
        color = str(row[color_col]) if color_col is not None else default_color
        chosen_text = None
        for dx, dy in offsets:
            trial = ax.text(
                x_value + dx,
                y_value + dy,
                str(row[label_col]),
                fontsize=get_font_size("annotation_size"),
                color=color,
                bbox={"facecolor": BACKGROUND, "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
                zorder=5,
                visible=False,
            )
            bbox = trial.get_window_extent(renderer=renderer).expanded(1.05, 1.18)
            inside_axes = (
                bbox.x0 >= axis_bbox.x0 + 3.0
                and bbox.x1 <= axis_bbox.x1 - 3.0
                and bbox.y0 >= axis_bbox.y0 + 3.0
                and bbox.y1 <= axis_bbox.y1 - 3.0
            )
            if inside_axes and not any(bbox.overlaps(existing) for existing in placed_bboxes):
                trial.set_visible(True)
                chosen_text = trial
                break
            trial.remove()
        if chosen_text is None:
            chosen_text = ax.text(
                x_value + offsets[0][0],
                y_value + offsets[0][1],
                str(row[label_col]),
                fontsize=get_font_size("annotation_size"),
                color=color,
                bbox={"facecolor": BACKGROUND, "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
                zorder=5,
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed_bboxes.append(chosen_text.get_window_extent(renderer=renderer).expanded(1.05, 1.18))


def _build_family_legend_handles(families: list[str], family_color_map: dict[str, str]) -> list[Line2D]:
    ordered = list(dict.fromkeys(families))
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=family_color_map.get(family, TEAL),
            markeredgecolor="white",
            markersize=7.5,
            label=family,
        )
        for family in ordered
    ]


def _prepare_top_models(df: pd.DataFrame, *, score_col: str, top_n: int = 12) -> pd.DataFrame:
    plot_df = df.copy()
    if "headline_eligible" in plot_df.columns:
        eligible = plot_df[plot_df["headline_eligible"]].copy()
        if not eligible.empty:
            plot_df = eligible
    plot_df = plot_df.sort_values(score_col, ascending=False).head(top_n).reset_index(drop=True)
    return plot_df


def _ranking_baseline(values: pd.Series) -> float:
    return max(0.0, float(values.min()) - 0.045)


def _draw_rank_lollipop(
    ax: plt.Axes,
    *,
    plot_df: pd.DataFrame,
    score_col: str,
    family_color_map: dict[str, str],
    x_label: str,
) -> None:
    apply_card(ax)
    baseline = _ranking_baseline(plot_df[score_col])
    y_positions = np.arange(len(plot_df))
    top_k = min(3, len(plot_df))
    for idx, row in plot_df.iterrows():
        color = family_color_map.get(str(row["family"]), TEAL)
        y_pos = y_positions[idx]
        value = float(row[score_col])
        ax.hlines(
            y=y_pos,
            xmin=baseline,
            xmax=value,
            color=soften_color(color, 0.6),
            linewidth=get_line_width("normal"),
            zorder=1,
        )
        marker_size = get_marker_size("highlight") if idx < top_k else get_marker_size("large")
        edge_color = GOLD if idx < top_k else "white"
        edge_width = get_line_width("normal") if idx < top_k else get_line_width("thin")
        ax.scatter(
            value,
            y_pos,
            s=marker_size,
            color=color,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=3,
        )
        ax.text(
            value + 0.005,
            y_pos,
            format_score(value),
            va="center",
            ha="left",
            fontsize=get_font_size("annotation_size"),
            color=INK,
        )
    ax.set_xlim(baseline, min(1.01, float(plot_df[score_col].max()) + 0.06))
    ax.set_xlabel(x_label)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["model_key"].tolist())
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=_grid_alpha())
    ax.tick_params(axis="y", length=0)


def _draw_layer_index_strip(
    ax: plt.Axes,
    *,
    plot_df: pd.DataFrame,
    layer_index_col: str,
    layer_name_col: str,
    family_color_map: dict[str, str],
    title: str,
) -> None:
    apply_card(ax, title=title, subtitle="Best global layer per ranked model.")
    y_positions = np.arange(len(plot_df))
    max_layer = max(1, int(plot_df[layer_index_col].max()))
    for idx, row in plot_df.iterrows():
        color = family_color_map.get(str(row["family"]), TEAL)
        y_pos = y_positions[idx]
        layer_index = int(row[layer_index_col])
        ax.scatter(
            layer_index,
            y_pos,
            s=get_marker_size("medium"),
            color=color,
            edgecolor="white",
            linewidth=get_line_width("thin"),
            zorder=3,
        )
        ax.text(
            layer_index + 0.35,
            y_pos,
            short_layer_name(str(row[layer_name_col])),
            va="center",
            ha="left",
            fontsize=get_font_size("annotation_size"),
            color=MUTED,
        )
    ax.set_xlim(-0.5, max_layer + 2.5)
    ax.set_xlabel("Best layer index")
    ax.set_yticks(y_positions)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(axis="x", alpha=_grid_alpha())
    ax.tick_params(axis="y", length=0, labelleft=False, left=False)


def _pareto_frontier(df: pd.DataFrame, *, x_col: str, y_col: str) -> pd.DataFrame:
    ordered = df.sort_values([x_col, y_col], ascending=[False, False]).copy()
    best_y = -np.inf
    keep: list[int] = []
    for idx, row in ordered.iterrows():
        y_value = float(row[y_col])
        if y_value > best_y:
            keep.append(idx)
            best_y = y_value
    return ordered.loc[keep].sort_values(x_col).reset_index(drop=True)


def _select_joint_highlights(merged: pd.DataFrame, policy: str | None) -> pd.DataFrame:
    selection: list[str] = []
    selection.extend(merged.nlargest(1, "joint_score")["model_key"].tolist())
    if policy == "frontier_plus_extremes":
        specialization = merged.assign(
            quality_specialization=lambda x: x["global_quality_score"] - x["global_alignment_score"],
            alignment_specialization=lambda x: x["global_alignment_score"] - x["global_quality_score"],
        )
        selection.extend(specialization.nlargest(1, "quality_specialization")["model_key"].tolist())
        selection.extend(specialization.nlargest(1, "alignment_specialization")["model_key"].tolist())
    selected = merged[merged["model_key"].isin(dict.fromkeys(selection))].copy()
    return selected.drop_duplicates("model_key")


def _draw_tradeoff_scatter(
    ax: plt.Axes,
    *,
    merged: pd.DataFrame,
    family_color_map: dict[str, str],
    highlights: pd.DataFrame,
) -> None:
    apply_card(ax, title="Quality-alignment trade-off", subtitle="Pareto frontier and representative highlighted models.")
    quality_med = float(merged["global_quality_score"].median())
    alignment_med = float(merged["global_alignment_score"].median())
    ax.axvspan(quality_med, 1.02, ymin=alignment_med / 1.02, ymax=1.0, color=soften_color(GOLD, 0.88), zorder=0)
    ax.scatter(
        merged["global_quality_score"],
        merged["global_alignment_score"],
        s=get_marker_size("medium"),
        color="#d0d4d8",
        edgecolor="white",
        linewidth=get_line_width("thin"),
        alpha=0.95,
        zorder=1,
    )
    frontier = _pareto_frontier(merged, x_col="global_quality_score", y_col="global_alignment_score")
    ax.plot(
        frontier["global_quality_score"],
        frontier["global_alignment_score"],
        linestyle="--",
        color=GOLD,
        linewidth=get_line_width("thick"),
        zorder=2,
    )
    ax.scatter(
        frontier["global_quality_score"],
        frontier["global_alignment_score"],
        s=get_marker_size("large"),
        facecolor=BACKGROUND,
        edgecolor=GOLD,
        linewidth=get_line_width("normal"),
        zorder=3,
    )
    if not highlights.empty:
        highlights = highlights.copy()
        highlights["label_color"] = highlights["family"].map(
            lambda family: family_color_map.get(str(family), TEAL)
        )
        ax.scatter(
            highlights["global_quality_score"],
            highlights["global_alignment_score"],
            s=get_marker_size("highlight"),
            c=highlights["label_color"],
            edgecolor="white",
            linewidth=get_line_width("normal"),
            zorder=4,
        )
        _annotate_points_greedy(
            ax,
            highlights,
            x_col="global_quality_score",
            y_col="global_alignment_score",
            label_col="model_key",
            color_col="label_color",
        )
    ax.axvline(quality_med, color=MUTED, linewidth=get_line_width("thin"), linestyle=":")
    ax.axhline(alignment_med, color=MUTED, linewidth=get_line_width("thin"), linestyle=":")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Global quality score")
    ax.set_ylabel("Global alignment score")
    ax.grid(alpha=_grid_alpha())


def _draw_family_tradeoff_panel(
    ax: plt.Axes,
    *,
    merged: pd.DataFrame,
    family_color_map: dict[str, str],
    label_limit: int = 6,
) -> None:
    apply_card(ax, title="Family medians", subtitle="Median family position with interquartile range.")
    family_stats = (
        merged.groupby("family", as_index=False)
        .agg(
            quality_median=("global_quality_score", "median"),
            quality_q1=("global_quality_score", lambda values: float(np.quantile(values, 0.25))),
            quality_q3=("global_quality_score", lambda values: float(np.quantile(values, 0.75))),
            alignment_median=("global_alignment_score", "median"),
            alignment_q1=("global_alignment_score", lambda values: float(np.quantile(values, 0.25))),
            alignment_q3=("global_alignment_score", lambda values: float(np.quantile(values, 0.75))),
            model_count=("model_key", "nunique"),
        )
        .sort_values(["quality_median", "alignment_median"], ascending=[False, False])
        .reset_index(drop=True)
    )
    for _, row in family_stats.iterrows():
        color = family_color_map.get(str(row["family"]), TEAL)
        ax.hlines(
            float(row["alignment_median"]),
            float(row["quality_q1"]),
            float(row["quality_q3"]),
            color=soften_color(color, 0.35),
            linewidth=get_line_width("normal"),
            zorder=1,
        )
        ax.vlines(
            float(row["quality_median"]),
            float(row["alignment_q1"]),
            float(row["alignment_q3"]),
            color=soften_color(color, 0.35),
            linewidth=get_line_width("normal"),
            zorder=1,
        )
        ax.scatter(
            float(row["quality_median"]),
            float(row["alignment_median"]),
            s=get_marker_size("large") + 6.0 * float(row["model_count"]),
            color=color,
            edgecolor="white",
            linewidth=get_line_width("thin"),
            zorder=3,
        )
    top_joint = family_stats.head(label_limit).copy()
    alignment_tail = family_stats.nsmallest(1, "alignment_median")
    quality_tail = family_stats.nsmallest(1, "quality_median")
    annotated = pd.concat([top_joint, alignment_tail, quality_tail], ignore_index=True).drop_duplicates("family")
    annotated["label_color"] = annotated["family"].map(
        lambda family: family_color_map.get(str(family), TEAL)
    )
    annotated = annotated.rename(columns={"quality_median": "x", "alignment_median": "y"})
    _annotate_points_greedy(ax, annotated, x_col="x", y_col="y", label_col="family", color_col="label_color")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Median quality score")
    ax.set_ylabel("Median alignment score")
    ax.grid(alpha=_grid_alpha())


def build_heatmap_payload(
    dataset_best: pd.DataFrame,
    *,
    value_col: str,
    model_order: list[str],
    dataset_order: list[str],
    model_col: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    pivot = dataset_best.pivot_table(index=model_col, columns="dataset", values=value_col, aggfunc="first")
    matrix = np.full((len(model_order), len(dataset_order)), np.nan, dtype=np.float64)
    for i, model in enumerate(model_order):
        for j, dataset in enumerate(dataset_order):
            if model in pivot.index and dataset in pivot.columns:
                value = pivot.loc[model, dataset]
                if pd.notna(value):
                    matrix[i, j] = float(value)
    return matrix, model_order, dataset_order


def _build_annotation_mask(matrix: np.ndarray, policy: str | None) -> np.ndarray:
    mask = np.zeros_like(matrix, dtype=bool)
    if policy == "column_maxima":
        for col_idx in range(matrix.shape[1]):
            column = matrix[:, col_idx]
            finite = np.isfinite(column)
            if finite.any():
                row_idx = int(np.nanargmax(column))
                mask[row_idx, col_idx] = True
    elif policy == "all":
        mask = np.isfinite(matrix)
    return mask


def _draw_heatmap_panel(
    ax: plt.Axes,
    *,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    subtitle: str,
    colorbar_label: str,
    annotation_policy: str | None,
) -> None:
    apply_card(ax, title=title, subtitle=subtitle)
    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad(MISSING)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=BACKGROUND, linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)

    mask = _build_annotation_mask(matrix, annotation_policy)
    norm = Normalize(vmin=0.0, vmax=1.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if mask[i, j] and np.isfinite(value):
                text_color = "white" if norm(value) > 0.62 else INK
                ax.text(
                    j,
                    i,
                    format_score(value, 2),
                    ha="center",
                    va="center",
                    fontsize=get_font_size("annotation_size"),
                    color=text_color,
                )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.ax.tick_params(labelsize=get_font_size("tick_label_size"))
    cbar.set_label(colorbar_label, fontsize=get_font_size("axis_label_size"))


def _draw_row_summary_bars(
    ax: plt.Axes,
    *,
    values: np.ndarray,
    families: list[str],
    family_color_map: dict[str, str],
    title: str,
    xlabel: str,
    show_y_labels: bool = False,
) -> None:
    apply_card(ax, title=title, subtitle="Row-wise mean across datasets.")
    y_positions = np.arange(len(values))
    colors = [family_color_map.get(family, TEAL) for family in families]
    ax.barh(y_positions, values, color=colors, edgecolor="white", linewidth=get_line_width("thin"))
    for ypos, value in enumerate(values):
        ax.text(float(value) + 0.01, ypos, format_score(value), va="center", fontsize=get_font_size("annotation_size"))
    left_bound = 0.0 if not show_y_labels else max(0.0, float(np.nanmin(values)) - 0.04)
    right_bound = min(1.02, float(np.nanmax(values)) + 0.08)
    ax.set_xlim(left_bound, right_bound)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y_positions if show_y_labels else [])
    ax.tick_params(axis="y", labelleft=show_y_labels, left=show_y_labels)
    if not show_y_labels:
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=_grid_alpha())
    if show_y_labels:
        ax.set_yticklabels([str(index + 1) for index in y_positions])


def _draw_family_distribution(
    ax: plt.Axes,
    *,
    ranking_df: pd.DataFrame,
    score_col: str,
    family_order: list[str],
    family_color_map: dict[str, str],
    title: str,
    subtitle: str,
    xlabel: str,
    show_y_labels: bool = True,
) -> None:
    apply_card(ax, title=title, subtitle=subtitle)
    y_positions = np.arange(len(family_order))
    for ypos, family in enumerate(family_order):
        family_rows = ranking_df[ranking_df["family"] == family].copy()
        if family_rows.empty:
            continue
        values = family_rows[score_col].to_numpy(dtype=np.float64)
        jitter = (
            np.linspace(-0.16, 0.16, num=len(values))
            if len(values) > 1
            else np.zeros(1, dtype=np.float64)
        )
        color = family_color_map.get(family, TEAL)
        ax.scatter(
            values,
            np.full(len(values), ypos, dtype=np.float64) + jitter,
            s=get_marker_size("small"),
            color=soften_color(color, 0.35),
            edgecolor="white",
            linewidth=get_line_width("thin"),
            alpha=0.95,
            zorder=2,
        )
        q1 = float(np.quantile(values, 0.25))
        median = float(np.median(values))
        q3 = float(np.quantile(values, 0.75))
        ax.hlines(ypos, q1, q3, color=color, linewidth=get_line_width("thick"), zorder=3)
        ax.scatter(median, ypos, s=get_marker_size("large"), color=color, edgecolor="white", linewidth=get_line_width("thin"), zorder=4)
    ax.set_yticks(y_positions)
    if show_y_labels:
        ax.set_yticklabels(family_order)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=_grid_alpha())
    ax.tick_params(axis="y", length=0, labelleft=show_y_labels)


def _draw_dumbbell_gap(
    ax: plt.Axes,
    *,
    plot_df: pd.DataFrame,
    fr_col: str,
    nr_col: str,
    family_color_map: dict[str, str],
    title: str,
    score_label: str,
    subtitle: str = "Largest paired shifts shown as score intervals.",
    left_label: str = "FR",
    right_label: str = "NR",
    label_mode: str = "endpoint",
) -> None:
    apply_card(ax, title=title, subtitle=subtitle)
    y_positions = np.arange(len(plot_df))
    for idx, row in plot_df.iterrows():
        family = str(row["family"])
        color = family_color_map.get(family, TEAL)
        nr_value = float(row[nr_col])
        fr_value = float(row[fr_col])
        y_pos = y_positions[idx]
        ax.hlines(
            y_pos,
            min(nr_value, fr_value),
            max(nr_value, fr_value),
            color=soften_color(color, 0.55),
            linewidth=get_line_width("thick"),
            zorder=1,
        )
        ax.scatter(
            nr_value,
            y_pos,
            s=get_marker_size("medium"),
            facecolor=BACKGROUND,
            edgecolor=soften_color(color, 0.2),
            linewidth=get_line_width("normal"),
            zorder=3,
        )
        ax.scatter(
            fr_value,
            y_pos,
            s=get_marker_size("medium"),
            facecolor=color,
            edgecolor="white",
            linewidth=get_line_width("thin"),
            zorder=4,
        )
        if label_mode == "endpoint":
            ax.text(
                max(nr_value, fr_value) + 0.012,
                y_pos,
                row["model_key"],
                va="center",
                fontsize=get_font_size("annotation_size"),
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["model_key"].tolist() if label_mode == "ytick" else [])
    ax.invert_yaxis()
    ax.set_xlim(max(0.0, float(np.nanmin(plot_df[[fr_col, nr_col]].to_numpy())) - 0.05), 1.02)
    ax.set_xlabel(score_label)
    ax.grid(axis="x", alpha=_grid_alpha())
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BACKGROUND, markeredgecolor=MUTED, markersize=7.5, label=right_label),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markeredgecolor="white", markersize=7.5, label=left_label),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.tick_params(axis="y", length=0)


def _draw_signed_delta_lollipop(
    ax: plt.Axes,
    *,
    plot_df: pd.DataFrame,
    delta_col: str,
    family_color_map: dict[str, str],
    title: str,
    xlabel: str,
    show_y_labels: bool = True,
) -> None:
    apply_card(ax, title=title, subtitle="Signed effect size; positive means FR is stronger.")
    ordered = plot_df.sort_values(delta_col).reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    for idx, row in ordered.iterrows():
        family = str(row["family"])
        color = family_color_map.get(family, TEAL)
        value = float(row[delta_col])
        y_pos = y_positions[idx]
        ax.hlines(y_pos, 0.0, value, color=soften_color(color, 0.55), linewidth=get_line_width("thick"), zorder=1)
        ax.scatter(value, y_pos, s=get_marker_size("medium"), color=color, edgecolor="white", linewidth=get_line_width("thin"), zorder=3)
        ax.text(
            value + (0.007 if value >= 0 else -0.007),
            y_pos,
            format_score(value),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=get_font_size("annotation_size"),
        )
    max_abs = max(0.08, float(np.nanmax(np.abs(ordered[delta_col].to_numpy()))))
    ax.axvline(0.0, color=MUTED, linewidth=get_line_width("thin"), linestyle="--")
    ax.set_xlim(-max_abs * 1.18, max_abs * 1.18)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["model_key"].tolist() if show_y_labels else [])
    ax.grid(axis="x", alpha=_grid_alpha())
    ax.tick_params(axis="y", length=0, labelleft=show_y_labels)


__all__ = [
    "_annotate_points_greedy",
    "_build_family_legend_handles",
    "_draw_dumbbell_gap",
    "_draw_family_distribution",
    "_draw_family_tradeoff_panel",
    "_draw_heatmap_panel",
    "_draw_layer_index_strip",
    "_draw_rank_lollipop",
    "_draw_row_summary_bars",
    "_draw_signed_delta_lollipop",
    "_draw_tradeoff_scatter",
    "_grid_alpha",
    "_prepare_top_models",
    "_select_joint_highlights",
    "apply_card",
    "build_heatmap_payload",
]
