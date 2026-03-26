from __future__ import annotations

from pathlib import Path
from typing import Callable

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

from quality_backbones.charts.data import (
    DeltaExperimentArtifacts,
    QualityScopeComparisonArtifacts,
    ReportExperimentArtifacts,
    TripletDeltaExperimentArtifacts,
    TripletExperimentArtifacts,
    build_alignment_flops_comparison,
    build_quality_flops_comparison,
    build_triplet_flops_comparison,
    rank_to_percentile,
)
from quality_backbones.charts.export import build_figure_record, write_figure_manifest
from quality_backbones.charts.recipes.common import (
    _annotate_points_greedy,
    _build_family_legend_handles,
    _draw_dumbbell_gap,
    _draw_family_distribution,
    _draw_family_tradeoff_panel,
    _draw_heatmap_panel,
    _draw_layer_index_strip,
    _draw_rank_lollipop,
    _draw_row_summary_bars,
    _draw_signed_delta_lollipop,
    _draw_tradeoff_scatter,
    _grid_alpha,
    _prepare_top_models,
    _select_joint_highlights,
    apply_card,
    build_heatmap_payload,
)
from quality_backbones.charts.registry import FigureSpec
from quality_backbones.charts.style import (
    BACKGROUND,
    GOLD,
    HEATMAP_CMAP,
    INK,
    LAYER_DEPTH_CMAP,
    MISSING,
    MUTED,
    TEAL,
    create_family_color_map,
    format_score,
    get_font_size,
    get_line_width,
    get_marker_size,
    get_preset,
    save_figure,
    short_layer_name,
    soften_color,
)


def _apply_figure_layout(
    fig: plt.Figure,
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> None:
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)


def _apply_paper_layout(fig: plt.Figure, *, left: float = 0.09, right: float = 0.98) -> None:
    _apply_figure_layout(fig, left=left, right=right, top=0.83, bottom=0.16)


def _apply_supplementary_layout(
    fig: plt.Figure,
    *,
    left: float = 0.12,
    right: float = 0.985,
    top: float = 0.87,
    bottom: float = 0.12,
) -> None:
    _apply_figure_layout(fig, left=left, right=right, top=top, bottom=bottom)


def _apply_layerwise_layout(fig: plt.Figure) -> None:
    _apply_figure_layout(fig, left=0.07, right=0.95, top=0.90, bottom=0.08)


def _format_gflops(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    numeric = float(value)
    if numeric >= 100.0:
        return f"{numeric:.0f}"
    if numeric >= 10.0:
        return f"{numeric:.1f}"
    return f"{numeric:.2f}"


def build_quality_ranking_figure(artifacts: ReportExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    plot_df = _prepare_top_models(artifacts.quality.model_ranking, score_col="global_quality_score", top_n=12)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.12)
    _apply_paper_layout(fig)

    ax_rank = fig.add_subplot(gs[0, 0])
    apply_card(
        ax_rank,
        title=f"Embedding quality ({artifacts.slice_name.upper()})",
        subtitle="Top non-IQA models by global quality.",
    )
    _draw_rank_lollipop(
        ax_rank,
        plot_df=plot_df,
        score_col="global_quality_score",
        family_color_map=artifacts.family_color_map,
        x_label="Global quality score",
    )
    ax_layer = fig.add_subplot(gs[0, 1], sharey=ax_rank)
    _draw_layer_index_strip(
        ax_layer,
        plot_df=plot_df,
        layer_index_col="global_best_layer_index",
        layer_name_col="global_best_layer",
        family_color_map=artifacts.family_color_map,
        title="Where does the best layer live?",
    )
    return fig


def build_alignment_ranking_figure(artifacts: ReportExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    plot_df = _prepare_top_models(artifacts.alignment.model_ranking, score_col="global_alignment_score", top_n=12)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.12)
    _apply_paper_layout(fig)

    ax_rank = fig.add_subplot(gs[0, 0])
    apply_card(
        ax_rank,
        title=f"Embedding alignment ({artifacts.slice_name.upper()})",
        subtitle="Top non-IQA models by global alignment.",
    )
    _draw_rank_lollipop(
        ax_rank,
        plot_df=plot_df,
        score_col="global_alignment_score",
        family_color_map=artifacts.family_color_map,
        x_label="Global alignment score",
    )
    ax_layer = fig.add_subplot(gs[0, 1], sharey=ax_rank)
    _draw_layer_index_strip(
        ax_layer,
        plot_df=plot_df,
        layer_index_col="global_best_layer_index",
        layer_name_col="global_best_layer",
        family_color_map=artifacts.family_color_map,
        title="Best alignment layer index",
    )
    return fig


def build_joint_scatter_figure(artifacts: ReportExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.18)
    _apply_paper_layout(fig)
    merged = artifacts.joint_ranking.copy()
    highlights = _select_joint_highlights(merged, spec.annotation_policy)

    ax_left = fig.add_subplot(gs[0, 0])
    _draw_tradeoff_scatter(
        ax_left,
        merged=merged,
        family_color_map=artifacts.family_color_map,
        highlights=highlights,
    )

    ax_right = fig.add_subplot(gs[0, 1])
    _draw_family_tradeoff_panel(ax_right, merged=merged, family_color_map=artifacts.family_color_map)
    return fig


def build_quality_heatmap_figure(artifacts: ReportExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    plot_df = _prepare_top_models(artifacts.quality.model_ranking, score_col="global_quality_score", top_n=12)
    model_order = plot_df["model_key"].tolist()
    matrix, rows, cols = build_heatmap_payload(
        artifacts.quality.dataset_best,
        value_col="quality_score",
        model_order=model_order,
        dataset_order=artifacts.quality.dataset_order,
        model_col="model_key",
    )
    row_means = np.nanmean(matrix, axis=1)
    families = plot_df["family"].tolist()

    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.2], wspace=0.12)
    _apply_supplementary_layout(fig, left=0.18, right=0.985, top=0.86, bottom=0.14)
    ax_heatmap = fig.add_subplot(gs[0, 0])
    _draw_heatmap_panel(
        ax_heatmap,
        matrix=matrix,
        row_labels=rows,
        col_labels=cols,
        title=f"Best-layer quality heatmap ({artifacts.slice_name.upper()})",
        subtitle="Top-ranked models; column maxima annotated.",
        colorbar_label="Quality score",
        annotation_policy=spec.annotation_policy,
    )
    ax_summary = fig.add_subplot(gs[0, 1])
    _draw_row_summary_bars(
        ax_summary,
        values=row_means,
        families=families,
        family_color_map=artifacts.family_color_map,
        title="Row means",
        xlabel="Mean quality",
    )
    return fig


def build_alignment_reference_figures(
    artifacts: ReportExperimentArtifacts,
    spec: FigureSpec,
) -> list[tuple[str, plt.Figure]]:
    preset = get_preset(spec.preset)
    figures: list[tuple[str, plt.Figure]] = []
    for reference_key in artifacts.alignment.reference_order:
        reference_rank = artifacts.alignment.reference_ranking[
            artifacts.alignment.reference_ranking["reference_model_key"] == reference_key
        ].copy()
        model_order = reference_rank.head(10)["model_key"].tolist()
        if not model_order:
            continue
        families = reference_rank.head(10)["family"].tolist()
        ref_rows = artifacts.alignment.rows[
            artifacts.alignment.rows["reference_model_key"] == reference_key
        ].copy()
        ref_rows["reference_alignment_percentile"] = ref_rows.groupby(
            ["dataset", "corr_metric", "distance_metric"]
        )["value"].transform(rank_to_percentile)
        ref_layer_scores = (
            ref_rows.groupby(
                ["dataset", "candidate_model_key", "candidate_layer_index", "candidate_layer_name"],
                as_index=False,
            )
            .agg(alignment_score=("reference_alignment_percentile", "mean"))
            .sort_values(
                ["dataset", "candidate_model_key", "alignment_score", "candidate_layer_index"],
                ascending=[True, True, False, True],
            )
            .drop_duplicates(["dataset", "candidate_model_key"])
            .rename(columns={"candidate_model_key": "model_key", "candidate_layer_name": "layer_name"})
        )
        matrix, rows, cols = build_heatmap_payload(
            ref_layer_scores,
            value_col="alignment_score",
            model_order=model_order,
            dataset_order=artifacts.alignment.dataset_order,
            model_col="model_key",
        )
        row_means = np.nanmean(matrix, axis=1)
        fig = plt.figure(figsize=(preset.width, preset.height))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.2], wspace=0.12)
        _apply_supplementary_layout(fig, left=0.18, right=0.985, top=0.86, bottom=0.14)
        ax_heatmap = fig.add_subplot(gs[0, 0])
        _draw_heatmap_panel(
            ax_heatmap,
            matrix=matrix,
            row_labels=rows,
            col_labels=cols,
            title=f"Alignment to {reference_key}",
            subtitle="Top candidate models under a fixed reference.",
            colorbar_label="Alignment score",
            annotation_policy=spec.annotation_policy,
        )
        ax_summary = fig.add_subplot(gs[0, 1])
        _draw_row_summary_bars(
            ax_summary,
            values=row_means,
            families=families,
            family_color_map=artifacts.family_color_map,
            title="Row means",
            xlabel="Mean align.",
        )
        figures.append((f"alignment_heatmap__{reference_key}", fig))
    return figures


def build_family_summary_figure(artifacts: ReportExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    _apply_supplementary_layout(fig, left=0.16, right=0.985, top=0.87, bottom=0.11)
    quality_df = artifacts.quality.model_ranking[["family", "global_quality_score", "model_key"]].copy()
    alignment_df = artifacts.alignment.model_ranking[["family", "global_alignment_score", "model_key"]].copy()
    family_order = (
        quality_df.groupby("family", as_index=False)["global_quality_score"]
        .median()
        .merge(
            alignment_df.groupby("family", as_index=False)["global_alignment_score"].median(),
            on="family",
            how="outer",
        )
        .fillna(0.0)
        .assign(joint=lambda x: x["global_quality_score"] + x["global_alignment_score"])
        .sort_values("joint", ascending=False)["family"]
        .tolist()
    )
    ax_quality = fig.add_subplot(gs[0, 0])
    _draw_family_distribution(
        ax_quality,
        ranking_df=quality_df.rename(columns={"global_quality_score": "score"}),
        score_col="score",
        family_order=family_order,
        family_color_map=artifacts.family_color_map,
        title="Family quality distribution",
        subtitle="Dots = models, thick segment = IQR, filled marker = median.",
        xlabel="Global quality score",
    )
    ax_alignment = fig.add_subplot(gs[0, 1], sharey=ax_quality)
    _draw_family_distribution(
        ax_alignment,
        ranking_df=alignment_df.rename(columns={"global_alignment_score": "score"}),
        score_col="score",
        family_order=family_order,
        family_color_map=artifacts.family_color_map,
        title="Family alignment distribution",
        subtitle="This makes family breadth visible instead of reporting medians alone.",
        xlabel="Global alignment score",
        show_y_labels=False,
    )
    return fig


def _build_selected_layer_flops_figure(
    plot_df: pd.DataFrame,
    *,
    family_color_map: dict[str, str],
    spec: FigureSpec,
    title: str,
    subtitle: str,
) -> plt.Figure:
    preset = get_preset(spec.preset)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 1)
    _apply_supplementary_layout(fig, left=0.24, right=0.83, top=0.84, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    apply_card(ax, title=title, subtitle=subtitle)

    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No FLOPs-profiled top models available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=get_font_size("panel_subtitle_size"),
            color=MUTED,
        )
        ax.set_axis_off()
        return fig

    y_positions = np.arange(len(plot_df))
    bar_height = 0.66
    highlight_count = min(3, len(plot_df))
    colors = [family_color_map.get(str(family), TEAL) for family in plot_df["family"]]
    ax.barh(y_positions, np.ones(len(plot_df)), height=bar_height, color=MISSING, edgecolor="none", zorder=1)
    ax.barh(
        y_positions,
        plot_df["selected_to_full_ratio"],
        height=bar_height,
        color=colors,
        edgecolor="white",
        linewidth=get_line_width("thin"),
        zorder=2,
    )

    annotation_transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        1.01,
        1.025,
        "Selected / full GFLOPs | layer",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=get_font_size("annotation_size"),
        color=MUTED,
        clip_on=False,
    )

    for index, row in enumerate(plot_df.itertuples(index=False)):
        ratio = float(row.selected_to_full_ratio)
        color = colors[index]
        marker_size = get_marker_size("highlight") if index < highlight_count else get_marker_size("medium")
        edge_color = GOLD if index < highlight_count else "white"
        edge_width = get_line_width("normal") if index < highlight_count else get_line_width("thin")
        ax.scatter(
            ratio,
            index,
            s=marker_size,
            color=color,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=3,
        )

        if ratio >= 0.88:
            share_x = max(0.04, ratio - 0.018)
            share_ha = "right"
            share_color = BACKGROUND
        else:
            share_x = min(0.99, ratio + 0.018)
            share_ha = "left"
            share_color = INK
        ax.text(
            share_x,
            index,
            f"{ratio:.0%}",
            va="center",
            ha=share_ha,
            fontsize=get_font_size("annotation_size"),
            color=share_color,
            zorder=4,
        )

        ax.text(
            1.01,
            index,
            f"{_format_gflops(row.selected_flops_g)}/{_format_gflops(row.full_flops_g)} G | {short_layer_name(str(row.global_best_layer))}",
            transform=annotation_transform,
            ha="left",
            va="center",
            fontsize=get_font_size("annotation_size"),
            color=MUTED,
            clip_on=False,
        )

    ax.axvline(0.5, color=MUTED, linewidth=get_line_width("thin"), linestyle=":", zorder=0)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("FLOPs up to selected layer / full model FLOPs")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["model_key"].tolist())
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="x", alpha=_grid_alpha())
    ax.tick_params(axis="y", length=0)

    legend_handles = _build_family_legend_handles(plot_df["family"].tolist(), family_color_map)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.34),
            ncol=min(5, len(legend_handles)),
            handletextpad=0.5,
            columnspacing=0.9,
        )
    return fig


def build_quality_flops_figure(artifacts: ReportExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    plot_df = build_quality_flops_comparison(artifacts.quality)
    plot_df = plot_df[plot_df["selected_to_full_ratio"].notna()].copy()
    plot_df = _prepare_top_models(plot_df, score_col="global_quality_score", top_n=12)
    return _build_selected_layer_flops_figure(
        plot_df,
        family_color_map=artifacts.family_color_map,
        spec=spec,
        title=f"Quality-selected compute share ({artifacts.slice_name.upper()})",
        subtitle="Top non-IQA models by global quality. Color shows FLOPs up to the best quality layer; pale remainder is compute after that layer.",
    )


def build_alignment_flops_figure(artifacts: ReportExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    plot_df = build_alignment_flops_comparison(artifacts.alignment)
    plot_df = plot_df[plot_df["selected_to_full_ratio"].notna()].copy()
    plot_df = _prepare_top_models(plot_df, score_col="global_alignment_score", top_n=12)
    return _build_selected_layer_flops_figure(
        plot_df,
        family_color_map=artifacts.family_color_map,
        spec=spec,
        title=f"Alignment-selected compute share ({artifacts.slice_name.upper()})",
        subtitle="Top non-IQA models by global alignment. This shows how early the best alignment layer arrives relative to full-model compute.",
    )


def build_triplet_ranking_figure(artifacts: TripletExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    plot_df = _prepare_top_models(artifacts.triplet.model_ranking, score_col="global_triplet_score", top_n=12)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.12)
    _apply_paper_layout(fig)

    ax_rank = fig.add_subplot(gs[0, 0])
    apply_card(
        ax_rank,
        title=f"Embedding triplet agreement ({artifacts.slice_name.upper()})",
        subtitle="Top non-IQA models by global triplet agreement.",
    )
    _draw_rank_lollipop(
        ax_rank,
        plot_df=plot_df,
        score_col="global_triplet_score",
        family_color_map=artifacts.family_color_map,
        x_label="Global triplet score",
    )

    ax_layer = fig.add_subplot(gs[0, 1], sharey=ax_rank)
    _draw_layer_index_strip(
        ax_layer,
        plot_df=plot_df,
        layer_index_col="global_best_layer_index",
        layer_name_col="global_best_layer",
        family_color_map=artifacts.family_color_map,
        title="Where does triplet peak?",
    )
    return fig


def build_triplet_heatmap_figure(artifacts: TripletExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    plot_df = _prepare_top_models(artifacts.triplet.model_ranking, score_col="global_triplet_score", top_n=12)
    model_order = plot_df["model_key"].tolist()
    matrix, rows, cols = build_heatmap_payload(
        artifacts.triplet.dataset_best,
        value_col="triplet_score",
        model_order=model_order,
        dataset_order=artifacts.triplet.dataset_order,
        model_col="model_key",
    )
    row_means = np.nanmean(matrix, axis=1)
    families = plot_df["family"].tolist()

    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.2], wspace=0.12)
    _apply_supplementary_layout(fig, left=0.18, right=0.985, top=0.86, bottom=0.14)
    ax_heatmap = fig.add_subplot(gs[0, 0])
    _draw_heatmap_panel(
        ax_heatmap,
        matrix=matrix,
        row_labels=rows,
        col_labels=cols,
        title=f"Best-layer triplet heatmap ({artifacts.slice_name.upper()})",
        subtitle="Top-ranked models; column maxima annotated.",
        colorbar_label="Triplet score",
        annotation_policy=spec.annotation_policy,
    )
    ax_summary = fig.add_subplot(gs[0, 1])
    _draw_row_summary_bars(
        ax_summary,
        values=row_means,
        families=families,
        family_color_map=artifacts.family_color_map,
        title="Row means",
        xlabel="Mean triplet",
    )
    return fig


def build_triplet_family_summary_figure(artifacts: TripletExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    _apply_supplementary_layout(fig, left=0.16, right=0.985, top=0.87, bottom=0.11)

    ranking_df = artifacts.triplet.model_ranking[
        ["family", "global_triplet_score", "global_best_layer_fraction", "model_key"]
    ].copy()
    family_order = (
        ranking_df.groupby("family", as_index=False)["global_triplet_score"]
        .median()
        .sort_values(["global_triplet_score", "family"], ascending=[False, True])["family"]
        .tolist()
    )

    ax_triplet = fig.add_subplot(gs[0, 0])
    _draw_family_distribution(
        ax_triplet,
        ranking_df=ranking_df.rename(columns={"global_triplet_score": "score"}),
        score_col="score",
        family_order=family_order,
        family_color_map=artifacts.family_color_map,
        title="Family triplet distribution",
        subtitle="Dots = models, thick segment = IQR, filled marker = median.",
        xlabel="Global triplet score",
    )

    ax_depth = fig.add_subplot(gs[0, 1], sharey=ax_triplet)
    _draw_family_distribution(
        ax_depth,
        ranking_df=ranking_df.rename(columns={"global_best_layer_fraction": "score"}),
        score_col="score",
        family_order=family_order,
        family_color_map=artifacts.family_color_map,
        title="Where triplet peaks",
        subtitle="Best-layer position normalized by model depth.",
        xlabel="Relative best-layer position",
        show_y_labels=False,
    )
    return fig


def build_triplet_flops_figure(artifacts: TripletExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    plot_df = build_triplet_flops_comparison(artifacts.triplet)
    plot_df = plot_df[plot_df["selected_to_full_ratio"].notna()].copy()
    plot_df = _prepare_top_models(plot_df, score_col="global_triplet_score", top_n=12)
    return _build_selected_layer_flops_figure(
        plot_df,
        family_color_map=artifacts.family_color_map,
        spec=spec,
        title=f"Triplet-selected compute share ({artifacts.slice_name.upper()})",
        subtitle="Top non-IQA models by global triplet agreement. Right-hand labels report selected/full GFLOPs and the chosen layer.",
    )


def build_gap_figure(
    gap_df: pd.DataFrame,
    *,
    score_col: str,
    left_label: str,
    right_label: str,
    title: str,
    family_color_map: dict[str, str],
    spec: FigureSpec,
) -> plt.Figure:
    preset = get_preset(spec.preset)
    top_shift = gap_df.head(14).copy().reset_index(drop=True)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    _apply_paper_layout(fig, left=0.24, right=0.985)

    ax_left = fig.add_subplot(gs[0, 0])
    _draw_dumbbell_gap(
        ax_left,
        plot_df=top_shift,
        fr_col=f"{score_col}_{left_label}",
        nr_col=f"{score_col}_{right_label}",
        family_color_map=family_color_map,
        title=title,
        score_label=score_col.replace("_", " "),
        label_mode="ytick",
    )

    ax_right = fig.add_subplot(gs[0, 1])
    _draw_signed_delta_lollipop(
        ax_right,
        plot_df=top_shift,
        delta_col=f"delta_{score_col}",
        family_color_map=family_color_map,
        title="Signed FR-NR effect",
        xlabel=f"FR - NR {score_col.replace('_', ' ')}",
        show_y_labels=False,
    )
    return fig


def build_joint_gap_figure(delta: DeltaExperimentArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    gap_df = delta.joint_gap.copy()
    family_color_map = delta.fr.family_color_map
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.38, 0.92], wspace=0.18)
    _apply_paper_layout(fig, left=0.11, right=0.985)

    ax_left = fig.add_subplot(gs[0, 0])
    apply_card(
        ax_left,
        title="FR-NR joint shift",
        subtitle="Quadrants separate FR-favoring and NR-favoring behavior in quality and alignment.",
    )
    ax_left.axvspan(
        0.0,
        max(0.01, float(gap_df["delta_quality_score"].max()) * 1.1),
        ymin=0.5,
        ymax=1.0,
        color=soften_color(GOLD, 0.88),
        zorder=0,
    )
    ax_left.axhline(0.0, color=MUTED, linewidth=get_line_width("thin"), linestyle="--")
    ax_left.axvline(0.0, color=MUTED, linewidth=get_line_width("thin"), linestyle="--")
    ax_left.scatter(
        gap_df["delta_quality_score"],
        gap_df["delta_alignment_score"],
        s=get_marker_size("medium"),
        color="#d0d4d8",
        edgecolor="white",
        linewidth=get_line_width("thin"),
        alpha=0.95,
        zorder=1,
    )
    highlighted = pd.concat(
        [gap_df[gap_df["delta_joint_score"] > 0.0].head(1), gap_df[gap_df["delta_joint_score"] < 0.0].head(2)],
        ignore_index=True,
    )
    highlighted["label_color"] = highlighted["family"].map(
        lambda family: family_color_map.get(str(family), TEAL)
    )
    highlighted["annotation_label"] = highlighted["model_key"].replace(
        {"siglip2_base_naflex": "s2_base_nf"}
    )
    highlighted["annotation_label"] = highlighted["annotation_label"].str.replace("_naflex", "_nf", regex=False)
    ax_left.scatter(
        highlighted["delta_quality_score"],
        highlighted["delta_alignment_score"],
        s=get_marker_size("highlight"),
        c=highlighted["label_color"],
        edgecolor="white",
        linewidth=get_line_width("normal"),
        zorder=4,
    )
    q_max = max(0.08, float(np.nanmax(np.abs(gap_df["delta_quality_score"].to_numpy()))))
    a_max = max(0.08, float(np.nanmax(np.abs(gap_df["delta_alignment_score"].to_numpy()))))
    for row in highlighted.itertuples(index=False):
        x_value = float(row.delta_quality_score)
        y_value = float(row.delta_alignment_score)
        if x_value > 0.0 and y_value > 0.0:
            x_text = x_value - 0.16 * q_max
            y_text = y_value + 0.03 * a_max
            ha = "left"
        else:
            x_text = x_value + 0.02 * q_max
            y_text = y_value - 0.02 * a_max
            ha = "left"
        ax_left.text(
            x_text,
            y_text,
            str(row.annotation_label),
            fontsize=get_font_size("annotation_size"),
            color=str(row.label_color),
            bbox={"facecolor": BACKGROUND, "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
            ha=ha,
            va="center",
            zorder=5,
        )
    ax_left.set_xlim(-q_max * 1.18, q_max * 1.34)
    ax_left.set_ylim(-a_max * 1.16, a_max * 1.22)
    ax_left.set_xlabel("FR - NR quality score")
    ax_left.set_ylabel("FR - NR alignment score")
    ax_left.grid(alpha=_grid_alpha())

    ax_right = fig.add_subplot(gs[0, 1])
    joint_top = gap_df.head(10).copy()
    _draw_signed_delta_lollipop(
        ax_right,
        plot_df=joint_top,
        delta_col="delta_joint_score",
        family_color_map=family_color_map,
        title="Signed joint effect",
        xlabel="FR - NR joint score",
        show_y_labels=False,
    )
    return fig


def _select_quality_scope_highlights(model_comparison: pd.DataFrame) -> pd.DataFrame:
    selection: list[str] = []
    selection.extend(model_comparison.head(3)["model_key"].tolist())
    selection.extend(
        model_comparison.sort_values("abs_quality_delta", ascending=False)
        .head(3)["model_key"]
        .tolist()
    )
    selection.extend(
        model_comparison.sort_values("global_quality_score_correct", ascending=False)
        .head(1)["model_key"]
        .tolist()
    )
    selection.extend(
        model_comparison.sort_values("global_quality_score_wrong", ascending=False)
        .head(1)["model_key"]
        .tolist()
    )
    selected = model_comparison[model_comparison["model_key"].isin(dict.fromkeys(selection))].copy()
    return selected.drop_duplicates("model_key")


def build_quality_scope_gap_figure(artifacts: QualityScopeComparisonArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    top_shift = artifacts.model_comparison.head(14).copy().reset_index(drop=True)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    _apply_paper_layout(fig, left=0.24, right=0.985)

    ax_left = fig.add_subplot(gs[0, 0])
    _draw_dumbbell_gap(
        ax_left,
        plot_df=top_shift,
        fr_col="global_quality_score_correct",
        nr_col="global_quality_score_wrong",
        family_color_map=artifacts.family_color_map,
        title="Within-group vs global quality",
        score_label="Global FR quality score",
        subtitle="Largest FR-model shifts under the two aggregation protocols.",
        left_label="Within-group",
        right_label="Global",
        label_mode="ytick",
    )

    ax_right = fig.add_subplot(gs[0, 1])
    _draw_signed_delta_lollipop(
        ax_right,
        plot_df=top_shift,
        delta_col="delta_quality_score",
        family_color_map=artifacts.family_color_map,
        title="Signed aggregation effect",
        xlabel="Within-group - global quality score",
        show_y_labels=False,
    )
    return fig


def build_quality_scope_scatter_figure(artifacts: QualityScopeComparisonArtifacts, spec: FigureSpec) -> plt.Figure:
    preset = get_preset(spec.preset)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 0.88], wspace=0.34)
    _apply_paper_layout(fig, left=0.11, right=0.985)
    model_comparison = artifacts.model_comparison.copy()
    highlights = _select_quality_scope_highlights(model_comparison)
    highlights["label_color"] = highlights["family"].map(
        lambda family: artifacts.family_color_map.get(str(family), TEAL)
    )

    ax_left = fig.add_subplot(gs[0, 0])
    apply_card(
        ax_left,
        title="Score preservation under wrong aggregation",
        subtitle="Diagonal agreement means the aggregation choice barely changes the score.",
    )
    x_line = np.linspace(0.0, 1.02, 128)
    ax_left.fill_between(x_line, x_line, 1.02, color=soften_color(GOLD, 0.9), zorder=0)
    ax_left.scatter(
        model_comparison["global_quality_score_wrong"],
        model_comparison["global_quality_score_correct"],
        s=get_marker_size("medium"),
        color="#d0d4d8",
        edgecolor="white",
        linewidth=get_line_width("thin"),
        alpha=0.95,
        zorder=1,
    )
    ax_left.plot([0.0, 1.02], [0.0, 1.02], linestyle="--", color=MUTED, linewidth=get_line_width("thin"), zorder=2)
    ax_left.scatter(
        highlights["global_quality_score_wrong"],
        highlights["global_quality_score_correct"],
        s=get_marker_size("highlight"),
        c=highlights["label_color"],
        edgecolor="white",
        linewidth=get_line_width("normal"),
        zorder=3,
    )
    _annotate_points_greedy(
        ax_left,
        highlights.rename(columns={"global_quality_score_wrong": "x", "global_quality_score_correct": "y"}),
        x_col="x",
        y_col="y",
        label_col="model_key",
        color_col="label_color",
    )
    ax_left.set_xlim(0.0, 1.02)
    ax_left.set_ylim(0.0, 1.02)
    ax_left.set_xlabel("Wrong global FR score")
    ax_left.set_ylabel("Correct within-group FR score")
    ax_left.grid(alpha=_grid_alpha())

    ax_right = fig.add_subplot(gs[0, 1])
    family_shift = artifacts.family_shift.copy().sort_values(
        ["mean_abs_delta_quality", "family"],
        ascending=[False, True],
    ).head(5)
    apply_card(ax_right, title="Most sensitive families")
    y_positions = np.arange(len(family_shift))
    colors = [artifacts.family_color_map.get(str(family), TEAL) for family in family_shift["family"]]
    ax_right.barh(y_positions, family_shift["mean_delta_quality"], color=colors, edgecolor="white", linewidth=get_line_width("thin"))
    ax_right.axvline(0.0, color=MUTED, linestyle="--", linewidth=get_line_width("thin"))
    ax_right.set_yticks(y_positions)
    ax_right.set_yticklabels(family_shift["family"].tolist())
    ax_right.invert_yaxis()
    ax_right.tick_params(axis="y", labelsize=get_font_size("annotation_size"))
    limit = max(0.08, float(np.nanmax(np.abs(family_shift["mean_delta_quality"].to_numpy()))))
    ax_right.set_xlim(-limit * 1.2, limit * 1.2)
    ax_right.set_xlabel("Mean within-group - global score")
    ax_right.grid(axis="x", alpha=_grid_alpha())
    ax_right.tick_params(axis="y", length=0)

    return fig


def build_quality_scope_dataset_heatmap_figure(
    artifacts: QualityScopeComparisonArtifacts,
    spec: FigureSpec,
) -> plt.Figure:
    preset = get_preset(spec.preset)
    plot_df = (
        artifacts.dataset_model_comparison.groupby(["model_key", "family"], as_index=False)
        .agg(mean_abs_delta_quality=("abs_quality_delta", "mean"), mean_delta_quality=("delta_quality_score", "mean"))
        .sort_values(["mean_abs_delta_quality", "mean_delta_quality", "model_key"], ascending=[False, False, True])
        .head(14)
    )
    model_order = plot_df["model_key"].tolist()
    dataset_order = artifacts.correct.dataset_order
    pivot = artifacts.dataset_model_comparison.pivot_table(
        index="model_key",
        columns="dataset",
        values="delta_quality_score",
        aggfunc="first",
    )
    matrix = np.full((len(model_order), len(dataset_order)), np.nan, dtype=np.float64)
    for row_idx, model_key in enumerate(model_order):
        for col_idx, dataset in enumerate(dataset_order):
            if model_key in pivot.index and dataset in pivot.columns:
                value = pivot.loc[model_key, dataset]
                if pd.notna(value):
                    matrix[row_idx, col_idx] = float(value)
    row_means = np.nanmean(np.abs(matrix), axis=1)
    families = plot_df["family"].tolist()
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.1], wspace=0.12)
    _apply_supplementary_layout(fig, left=0.19, right=0.985, top=0.86, bottom=0.14)
    ax_heatmap = fig.add_subplot(gs[0, 0])
    apply_card(
        ax_heatmap,
        title="Per-dataset FR score deltas",
        subtitle="Positive cells mean within-group aggregation raises the best-layer score.",
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(MISSING)
    max_abs = max(0.08, float(np.nanmax(np.abs(matrix))))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    im = ax_heatmap.imshow(matrix, cmap=cmap, aspect="auto", norm=norm)
    ax_heatmap.set_xticks(np.arange(len(dataset_order)))
    ax_heatmap.set_xticklabels(dataset_order, rotation=24, ha="right")
    ax_heatmap.set_yticks(np.arange(len(model_order)))
    ax_heatmap.set_yticklabels(model_order)
    ax_heatmap.set_xticks(np.arange(-0.5, len(dataset_order), 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, len(model_order), 1), minor=True)
    ax_heatmap.grid(which="minor", color=BACKGROUND, linewidth=1.0)
    ax_heatmap.tick_params(which="minor", bottom=False, left=False)
    ax_heatmap.tick_params(length=0)
    for col_idx in range(matrix.shape[1]):
        column = np.abs(matrix[:, col_idx])
        if np.isfinite(column).any():
            row_idx = int(np.nanargmax(column))
            value = matrix[row_idx, col_idx]
            text_color = "white" if abs(value) > 0.55 * max_abs else "black"
            ax_heatmap.text(
                col_idx,
                row_idx,
                format_score(value, 2),
                ha="center",
                va="center",
                fontsize=get_font_size("annotation_size"),
                color=text_color,
            )
    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.028, pad=0.02)
    cbar.ax.tick_params(labelsize=get_font_size("tick_label_size"))
    cbar.set_label("Within-group - global score", fontsize=get_font_size("axis_label_size"))

    ax_summary = fig.add_subplot(gs[0, 1])
    _draw_row_summary_bars(
        ax_summary,
        values=row_means,
        families=families,
        family_color_map=artifacts.family_color_map,
        title="Mean |delta|",
        xlabel="Mean abs. delta",
    )
    return fig


def build_quality_scope_family_shift_figure(
    artifacts: QualityScopeComparisonArtifacts,
    spec: FigureSpec,
) -> plt.Figure:
    preset = get_preset(spec.preset)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    _apply_supplementary_layout(fig, left=0.12, right=0.985, top=0.87, bottom=0.11)
    family_shift = artifacts.family_shift.copy().sort_values(
        ["mean_abs_delta_quality", "family"],
        ascending=[False, True],
    )

    ax_left = fig.add_subplot(gs[0, 0])
    apply_card(
        ax_left,
        title="Family shift magnitude",
        subtitle="Signed means reveal which families benefit from the proper within-group protocol.",
    )
    y_positions = np.arange(len(family_shift))
    colors = [artifacts.family_color_map.get(str(family), TEAL) for family in family_shift["family"]]
    ax_left.barh(y_positions, family_shift["mean_delta_quality"], color=colors, edgecolor="white", linewidth=get_line_width("thin"))
    ax_left.axvline(0.0, color=MUTED, linestyle="--", linewidth=get_line_width("thin"))
    for ypos, value in enumerate(family_shift["mean_delta_quality"]):
        ax_left.text(
            float(value) + (0.006 if value >= 0 else -0.006),
            ypos,
            format_score(value),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=get_font_size("annotation_size"),
        )
    ax_left.set_yticks(y_positions)
    ax_left.set_yticklabels(family_shift["family"].tolist())
    ax_left.invert_yaxis()
    left_limit = max(0.08, float(np.nanmax(np.abs(family_shift["mean_delta_quality"].to_numpy()))))
    ax_left.set_xlim(-left_limit * 1.18, left_limit * 1.18)
    ax_left.set_xlabel("Mean within-group - global score")
    ax_left.grid(axis="x", alpha=_grid_alpha())
    ax_left.tick_params(axis="y", length=0)

    ax_right = fig.add_subplot(gs[0, 1])
    apply_card(
        ax_right,
        title="Layer instability",
        subtitle="Fraction of models whose best global FR layer changes under wrong aggregation.",
    )
    ax_right.barh(y_positions, family_shift["layer_changed_rate"], color=colors, edgecolor="white", linewidth=get_line_width("thin"))
    for ypos, value in enumerate(family_shift["layer_changed_rate"]):
        ax_right.text(float(value) + 0.02, ypos, f"{100.0 * float(value):.0f}%", va="center", fontsize=get_font_size("annotation_size"))
    ax_right.set_xlim(0.0, 1.02)
    ax_right.set_xlabel("Layer-changed rate")
    ax_right.set_yticks(y_positions)
    ax_right.set_yticklabels([])
    ax_right.invert_yaxis()
    ax_right.grid(axis="x", alpha=_grid_alpha())
    ax_right.tick_params(axis="y", length=0)
    return fig


def build_model_layer_bundle_figure(
    model_key: str,
    *,
    slices: dict[str, ReportExperimentArtifacts],
    spec: FigureSpec,
) -> plt.Figure | None:
    preset = get_preset(spec.preset)
    slice_order = ["overall", "fr", "nr"]
    available_profiles: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    family: str | None = None
    max_layer = 1
    for slice_name in slice_order:
        artifacts = slices.get(slice_name)
        if artifacts is None:
            continue
        quality_profile = artifacts.quality.layer_profiles[
            artifacts.quality.layer_profiles["model_key"] == model_key
        ].copy()
        alignment_profile = artifacts.alignment.layer_profiles[
            artifacts.alignment.layer_profiles["model_key"] == model_key
        ].copy()
        joint_profile = artifacts.joint_layer_profiles[
            artifacts.joint_layer_profiles["model_key"] == model_key
        ].copy()
        if quality_profile.empty or alignment_profile.empty or joint_profile.empty:
            continue
        available_profiles[slice_name] = (quality_profile, alignment_profile, joint_profile)
        if family is None:
            family = str(quality_profile.iloc[0]["family"])
        max_layer = max(
            max_layer,
            int(
                max(
                    quality_profile["layer_index"].max(),
                    alignment_profile["layer_index"].max(),
                    joint_profile["layer_index"].max(),
                )
            ),
        )
    if not available_profiles:
        return None

    norm = Normalize(vmin=0, vmax=max_layer)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.0, 1.0, 0.05], hspace=0.4, wspace=0.26)
    _apply_layerwise_layout(fig)
    scatter_handle = None
    dense_line_alpha = 0.28 if max_layer > 20 else 0.34
    depth_point_size = max(12.0, get_marker_size("medium") - 0.35 * max_layer)
    trend_point_size = max(9.0, get_marker_size("small") - 0.12 * max_layer)
    x_margin = max(0.8, 0.025 * max_layer)

    for row_index, slice_name in enumerate(slice_order):
        quality_profile, alignment_profile, joint_profile = available_profiles.get(
            slice_name,
            (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
        )
        if quality_profile.empty:
            for col_index in range(3):
                ax = fig.add_subplot(gs[row_index, col_index])
                ax.axis("off")
            continue
        color = slices[slice_name].family_color_map.get(family or "", TEAL)

        ax_quality = fig.add_subplot(gs[row_index, 0])
        apply_card(
            ax_quality,
            title="Quality over depth" if row_index == 0 else None,
            subtitle="Mean with min-max envelope." if row_index == 0 else None,
        )
        best_quality = quality_profile.sort_values(["mean_quality", "layer_index"], ascending=[False, True]).iloc[0]
        ax_quality.fill_between(
            quality_profile["layer_index"],
            quality_profile["min_quality"],
            quality_profile["max_quality"],
            color=soften_color(color, 0.74),
            alpha=dense_line_alpha,
        )
        ax_quality.plot(quality_profile["layer_index"], quality_profile["mean_quality"], color=color, linewidth=get_line_width("thick"))
        ax_quality.scatter(quality_profile["layer_index"], quality_profile["mean_quality"], color=color, s=trend_point_size, zorder=3)
        ax_quality.scatter(best_quality["layer_index"], best_quality["mean_quality"], color=GOLD, edgecolor="white", linewidth=get_line_width("thin"), s=get_marker_size("highlight"), zorder=4)
        ax_quality.axvline(best_quality["layer_index"], color=GOLD, linewidth=get_line_width("thin"), linestyle="--")
        ax_quality.text(
            0.10,
            0.08,
            f"best {int(best_quality['layer_index'])}: {short_layer_name(str(best_quality['layer_name']))}",
            transform=ax_quality.transAxes,
            fontsize=get_font_size("annotation_size"),
            color=MUTED,
            clip_on=False,
        )
        ax_quality.set_xlim(-x_margin, max_layer + x_margin)
        ax_quality.set_ylim(0.0, 1.02)
        ax_quality.set_xlabel("Layer index")
        ax_quality.set_ylabel(f"{slice_name.upper()}\nQuality")
        ax_quality.grid(alpha=_grid_alpha())
        ax_quality.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        ax_alignment = fig.add_subplot(gs[row_index, 1])
        apply_card(
            ax_alignment,
            title="Alignment over depth" if row_index == 0 else None,
            subtitle="Mean with min-max envelope." if row_index == 0 else None,
        )
        best_alignment = alignment_profile.sort_values(["mean_alignment", "layer_index"], ascending=[False, True]).iloc[0]
        ax_alignment.fill_between(
            alignment_profile["layer_index"],
            alignment_profile["min_alignment"],
            alignment_profile["max_alignment"],
            color=soften_color(color, 0.74),
            alpha=dense_line_alpha,
        )
        ax_alignment.plot(alignment_profile["layer_index"], alignment_profile["mean_alignment"], color=color, linewidth=get_line_width("thick"))
        ax_alignment.scatter(alignment_profile["layer_index"], alignment_profile["mean_alignment"], color=color, s=trend_point_size, zorder=3)
        ax_alignment.scatter(best_alignment["layer_index"], best_alignment["mean_alignment"], color=GOLD, edgecolor="white", linewidth=get_line_width("thin"), s=get_marker_size("highlight"), zorder=4)
        ax_alignment.axvline(best_alignment["layer_index"], color=GOLD, linewidth=get_line_width("thin"), linestyle="--")
        ax_alignment.text(
            0.10,
            0.08,
            f"best {int(best_alignment['layer_index'])}: {short_layer_name(str(best_alignment['layer_name']))}",
            transform=ax_alignment.transAxes,
            fontsize=get_font_size("annotation_size"),
            color=MUTED,
            clip_on=False,
        )
        ax_alignment.set_xlim(-x_margin, max_layer + x_margin)
        ax_alignment.set_ylim(0.0, 1.02)
        ax_alignment.set_xlabel("Layer index")
        ax_alignment.set_ylabel(f"{slice_name.upper()}\nAlignment")
        ax_alignment.grid(alpha=_grid_alpha())
        ax_alignment.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        ax_joint = fig.add_subplot(gs[row_index, 2])
        apply_card(
            ax_joint,
            title="Quality vs alignment trajectory" if row_index == 0 else None,
            subtitle="Color encodes layer depth." if row_index == 0 else None,
        )
        best_joint = joint_profile.sort_values(["joint_score", "layer_index"], ascending=[False, True]).iloc[0]
        ax_joint.plot(
            joint_profile["mean_quality"],
            joint_profile["mean_alignment"],
            color=soften_color(color, 0.45),
            linewidth=get_line_width("thin"),
            alpha=0.9,
            zorder=1,
        )
        scatter_handle = ax_joint.scatter(
            joint_profile["mean_quality"],
            joint_profile["mean_alignment"],
            c=joint_profile["layer_index"],
            cmap=LAYER_DEPTH_CMAP,
            norm=norm,
            s=depth_point_size,
            edgecolor="white",
            linewidth=get_line_width("thin"),
            zorder=3,
        )
        start_joint = joint_profile.sort_values("layer_index", ascending=True).iloc[0]
        end_joint = joint_profile.sort_values("layer_index", ascending=True).iloc[-1]
        ax_joint.scatter(
            start_joint["mean_quality"],
            start_joint["mean_alignment"],
            marker="s",
            s=depth_point_size * 1.15,
            facecolor=BACKGROUND,
            edgecolor=color,
            linewidth=get_line_width("normal"),
            zorder=4,
        )
        ax_joint.scatter(
            end_joint["mean_quality"],
            end_joint["mean_alignment"],
            marker="^",
            s=depth_point_size * 1.25,
            facecolor=color,
            edgecolor="white",
            linewidth=get_line_width("thin"),
            zorder=4,
        )
        ax_joint.scatter(best_joint["mean_quality"], best_joint["mean_alignment"], color=GOLD, edgecolor="white", linewidth=get_line_width("thin"), s=get_marker_size("highlight"), zorder=4)
        ax_joint.text(
            0.10,
            0.08,
            f"best joint layer {int(best_joint['layer_index'])}",
            transform=ax_joint.transAxes,
            fontsize=get_font_size("annotation_size"),
            color=MUTED,
            clip_on=False,
        )
        ax_joint.set_xlim(0.0, 1.02)
        ax_joint.set_ylim(0.0, 1.02)
        ax_joint.set_xlabel("Quality")
        ax_joint.set_ylabel(f"{slice_name.upper()}\nAlignment")
        ax_joint.grid(alpha=_grid_alpha())

    cax = fig.add_subplot(gs[:, 3])
    if scatter_handle is not None:
        cbar = fig.colorbar(scatter_handle, cax=cax)
        cbar.ax.tick_params(labelsize=get_font_size("tick_label_size"))
        cbar.set_label("Layer index", fontsize=get_font_size("axis_label_size"))
    else:
        cax.axis("off")
    fig.text(
        0.01,
        0.99,
        f"Layerwise bundle: {model_key} ({family or '-'})",
        ha="left",
        va="top",
        fontsize=get_font_size("figure_title_size"),
        fontweight="bold",
    )
    return fig


def build_model_triplet_layer_bundle_figure(
    model_key: str,
    *,
    slices: dict[str, TripletExperimentArtifacts],
    spec: FigureSpec,
) -> plt.Figure | None:
    preset = get_preset(spec.preset)
    slice_order = ["overall", "fr", "nr"]
    available_profiles: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]] = {}
    family: str | None = None
    max_layer = 1
    curve_floor = 1.0
    curve_ceiling = 0.0
    score_floor = 1.0
    score_ceiling = 0.0

    for slice_name in slice_order:
        artifacts = slices.get(slice_name)
        if artifacts is None:
            continue
        triplet_profile = artifacts.triplet.layer_profiles[
            artifacts.triplet.layer_profiles["model_key"] == model_key
        ].copy()
        dataset_best = artifacts.triplet.dataset_best[
            artifacts.triplet.dataset_best["model_key"] == model_key
        ].copy()
        if triplet_profile.empty:
            continue
        available_profiles[slice_name] = (triplet_profile, dataset_best, artifacts.triplet.dataset_order)
        if family is None:
            family = str(triplet_profile.iloc[0]["family"])
        max_layer = max(max_layer, int(triplet_profile["layer_index"].max()))
        curve_floor = min(curve_floor, float(triplet_profile["min_triplet"].min()))
        curve_ceiling = max(curve_ceiling, float(triplet_profile["max_triplet"].max()))
        if not dataset_best.empty:
            score_floor = min(score_floor, float(dataset_best["triplet_score"].min()))
            score_ceiling = max(score_ceiling, float(dataset_best["triplet_score"].max()))
    if not available_profiles:
        return None

    norm = Normalize(vmin=0, vmax=max_layer)
    curve_ymin = max(0.0, curve_floor - 0.04)
    curve_ymax = min(1.0, curve_ceiling + 0.03)
    score_xmin = max(0.45, score_floor - 0.045)
    score_xmax = min(1.0, score_ceiling + 0.08)
    fig = plt.figure(figsize=(preset.width, preset.height))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1.0, 0.05], hspace=0.42, wspace=0.28)
    _apply_layerwise_layout(fig)
    scatter_handle = None
    dense_line_alpha = 0.28 if max_layer > 20 else 0.34
    trend_point_size = max(9.0, get_marker_size("small") - 0.12 * max_layer)
    dataset_point_size = max(18.0, get_marker_size("medium") - 0.10 * max_layer)
    x_margin = max(0.8, 0.025 * max_layer)

    for row_index, slice_name in enumerate(slice_order):
        triplet_profile, dataset_best, dataset_order = available_profiles.get(
            slice_name,
            (pd.DataFrame(), pd.DataFrame(), []),
        )
        if triplet_profile.empty:
            for col_index in range(2):
                ax = fig.add_subplot(gs[row_index, col_index])
                ax.axis("off")
            continue
        color = slices[slice_name].family_color_map.get(family or "", TEAL)

        ax_curve = fig.add_subplot(gs[row_index, 0])
        apply_card(
            ax_curve,
            title="Triplet over depth" if row_index == 0 else None,
            subtitle="Mean with min-max envelope." if row_index == 0 else None,
        )
        best_triplet = triplet_profile.sort_values(["mean_triplet", "layer_index"], ascending=[False, True]).iloc[0]
        ax_curve.fill_between(
            triplet_profile["layer_index"],
            triplet_profile["min_triplet"],
            triplet_profile["max_triplet"],
            color=soften_color(color, 0.74),
            alpha=dense_line_alpha,
        )
        ax_curve.plot(
            triplet_profile["layer_index"],
            triplet_profile["mean_triplet"],
            color=color,
            linewidth=get_line_width("thick"),
        )
        ax_curve.scatter(
            triplet_profile["layer_index"],
            triplet_profile["mean_triplet"],
            color=color,
            s=trend_point_size,
            zorder=3,
        )
        ax_curve.scatter(
            best_triplet["layer_index"],
            best_triplet["mean_triplet"],
            color=GOLD,
            edgecolor="white",
            linewidth=get_line_width("thin"),
            s=get_marker_size("highlight"),
            zorder=4,
        )
        ax_curve.axvline(best_triplet["layer_index"], color=GOLD, linewidth=get_line_width("thin"), linestyle="--")
        ax_curve.text(
            0.10,
            0.08,
            f"best {int(best_triplet['layer_index'])}: {short_layer_name(str(best_triplet['layer_name']))}",
            transform=ax_curve.transAxes,
            fontsize=get_font_size("annotation_size"),
            color=MUTED,
            clip_on=False,
        )
        ax_curve.set_xlim(-x_margin, max_layer + x_margin)
        ax_curve.set_ylim(curve_ymin, curve_ymax)
        ax_curve.set_xlabel("Layer index")
        ax_curve.set_ylabel(f"{slice_name.upper()}\nTriplet")
        ax_curve.grid(alpha=_grid_alpha())
        ax_curve.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        ax_dataset = fig.add_subplot(gs[row_index, 1])
        apply_card(
            ax_dataset,
            title="Dataset-specific peaks" if row_index == 0 else None,
            subtitle="Color encodes best layer index." if row_index == 0 else None,
        )
        ordered_datasets = [dataset for dataset in dataset_order if dataset in set(dataset_best["dataset"].tolist())]
        if not ordered_datasets:
            ordered_datasets = sorted(dataset_best["dataset"].unique().tolist())
        dataset_best = dataset_best.copy()
        dataset_best["dataset"] = pd.Categorical(dataset_best["dataset"], categories=ordered_datasets, ordered=True)
        dataset_best = dataset_best.sort_values(["dataset", "triplet_score"], ascending=[True, False]).reset_index(drop=True)
        if not dataset_best.empty:
            y_positions = np.arange(len(dataset_best))
            scatter_handle = ax_dataset.scatter(
                dataset_best["triplet_score"],
                y_positions,
                c=dataset_best["layer_index"],
                cmap=LAYER_DEPTH_CMAP,
                norm=norm,
                s=dataset_point_size,
                edgecolor="white",
                linewidth=get_line_width("thin"),
                zorder=3,
            )
            for ypos, row in enumerate(dataset_best.itertuples(index=False)):
                ax_dataset.text(
                    float(row.triplet_score) + 0.008,
                    ypos,
                    short_layer_name(str(row.layer_name)),
                    va="center",
                    ha="left",
                    fontsize=get_font_size("annotation_size"),
                    color=MUTED,
                )
            ax_dataset.set_yticks(y_positions)
            ax_dataset.set_yticklabels([str(value) for value in dataset_best["dataset"].tolist()])
            ax_dataset.invert_yaxis()
        ax_dataset.set_xlim(score_xmin, score_xmax)
        ax_dataset.set_xlabel("Best dataset triplet score")
        ax_dataset.set_ylabel(f"{slice_name.upper()}\nDataset")
        ax_dataset.grid(axis="x", alpha=_grid_alpha())
        ax_dataset.tick_params(axis="y", length=0)

    cax = fig.add_subplot(gs[:, 2])
    if scatter_handle is not None:
        cbar = fig.colorbar(scatter_handle, cax=cax)
        cbar.ax.tick_params(labelsize=get_font_size("tick_label_size"))
        cbar.set_label("Best layer index", fontsize=get_font_size("axis_label_size"))
    else:
        cax.axis("off")
    fig.text(
        0.01,
        0.99,
        f"Triplet layerwise bundle: {model_key} ({family or '-'})",
        ha="left",
        va="top",
        fontsize=get_font_size("figure_title_size"),
        fontweight="bold",
    )
    return fig


def _build_report_renderer_map() -> dict[str, Callable[[ReportExperimentArtifacts, FigureSpec], plt.Figure]]:
    return {
        "quality_ranking": build_quality_ranking_figure,
        "alignment_ranking": build_alignment_ranking_figure,
        "quality_vs_alignment": build_joint_scatter_figure,
        "quality_flops": build_quality_flops_figure,
        "alignment_flops": build_alignment_flops_figure,
        "quality_heatmap": build_quality_heatmap_figure,
        "family_summary": build_family_summary_figure,
    }


def _build_triplet_renderer_map() -> dict[str, Callable[[TripletExperimentArtifacts, FigureSpec], plt.Figure]]:
    return {
        "triplet_ranking": build_triplet_ranking_figure,
        "triplet_flops": build_triplet_flops_figure,
        "triplet_heatmap": build_triplet_heatmap_figure,
        "triplet_family_summary": build_triplet_family_summary_figure,
    }


def _record_title(slug: str) -> str | None:
    if slug.startswith("alignment_heatmap__"):
        return f"Alignment to {slug.split('__', maxsplit=1)[1]}"
    return None


def render_report_figures(
    experiment_dir: Path,
    artifacts: ReportExperimentArtifacts,
    figure_specs: list[FigureSpec],
) -> tuple[dict[str, str], str]:
    renderer_map = _build_report_renderer_map()
    paper_dir = experiment_dir / "figures" / "paper"
    supplementary_dir = experiment_dir / "figures" / "supplementary"
    exports_dir = experiment_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    paper_bundle_path = exports_dir / "paper_bundle.pdf"
    supplementary_bundle_path = exports_dir / "supplementary_bundle.pdf"
    saved: dict[str, str] = {}
    records: list[dict[str, object]] = []
    inputs = {
        "quality_source": artifacts.quality_source,
        "alignment_source": artifacts.alignment_source,
        "slice": artifacts.slice_name,
    }

    with PdfPages(paper_bundle_path) as paper_pdf, PdfPages(supplementary_bundle_path) as supplementary_pdf:
        for spec in figure_specs:
            target_dir = paper_dir if spec.tier == "paper" else supplementary_dir
            if spec.renderer == "alignment_reference_heatmaps":
                built = build_alignment_reference_figures(artifacts, spec)
            else:
                built = [(spec.slug, renderer_map[spec.renderer](artifacts, spec))]
            for slug, fig in built:
                output_path = target_dir / f"{slug}.pdf"
                save_figure(fig, output_path, title=_record_title(slug))
                saved[slug] = str(output_path)
                records.append(
                    build_figure_record(
                        experiment_slug=artifacts.slug,
                        spec=spec,
                        slug=slug,
                        output_path=output_path,
                        inputs=inputs,
                        title=_record_title(slug),
                    )
                )
                if spec.tier == "paper":
                    paper_pdf.savefig(fig, bbox_inches="tight")
                else:
                    supplementary_pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    figure_manifest_path = write_figure_manifest(
        exports_dir / "figures_manifest.json",
        experiment_slug=artifacts.slug,
        records=records,
    )
    saved["paper_bundle"] = str(paper_bundle_path)
    saved["supplementary_bundle"] = str(supplementary_bundle_path)
    return saved, figure_manifest_path


def render_triplet_figures(
    experiment_dir: Path,
    artifacts: TripletExperimentArtifacts,
    figure_specs: list[FigureSpec],
) -> tuple[dict[str, str], str]:
    renderer_map = _build_triplet_renderer_map()
    paper_dir = experiment_dir / "figures" / "paper"
    supplementary_dir = experiment_dir / "figures" / "supplementary"
    exports_dir = experiment_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    paper_bundle_path = exports_dir / "paper_bundle.pdf"
    supplementary_bundle_path = exports_dir / "supplementary_bundle.pdf"
    saved: dict[str, str] = {}
    records: list[dict[str, object]] = []
    inputs = {
        "triplet_source": artifacts.triplet_source,
        "slice": artifacts.slice_name,
    }

    with PdfPages(paper_bundle_path) as paper_pdf, PdfPages(supplementary_bundle_path) as supplementary_pdf:
        for spec in figure_specs:
            fig = renderer_map[spec.renderer](artifacts, spec)
            target_dir = paper_dir if spec.tier == "paper" else supplementary_dir
            output_path = target_dir / f"{spec.slug}.pdf"
            save_figure(fig, output_path)
            saved[spec.slug] = str(output_path)
            records.append(
                build_figure_record(
                    experiment_slug=artifacts.slug,
                    spec=spec,
                    slug=spec.slug,
                    output_path=output_path,
                    inputs=inputs,
                )
            )
            if spec.tier == "paper":
                paper_pdf.savefig(fig, bbox_inches="tight")
            else:
                supplementary_pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    figure_manifest_path = write_figure_manifest(
        exports_dir / "figures_manifest.json",
        experiment_slug=artifacts.slug,
        records=records,
    )
    saved["paper_bundle"] = str(paper_bundle_path)
    saved["supplementary_bundle"] = str(supplementary_bundle_path)
    return saved, figure_manifest_path


def render_delta_figures(
    experiment_dir: Path,
    artifacts: DeltaExperimentArtifacts,
    figure_specs: list[FigureSpec],
) -> tuple[dict[str, str], str]:
    paper_dir = experiment_dir / "figures" / "paper"
    exports_dir = experiment_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = exports_dir / "paper_bundle.pdf"
    saved: dict[str, str] = {}
    records: list[dict[str, object]] = []
    inputs = {
        "fr_manifest": str(experiment_dir.parent / "fr" / "manifest.json"),
        "nr_manifest": str(experiment_dir.parent / "nr" / "manifest.json"),
    }

    with PdfPages(bundle_path) as pdf:
        for spec in figure_specs:
            if spec.renderer == "quality_fr_vs_nr_delta":
                fig = build_gap_figure(
                    artifacts.quality_gap,
                    score_col="global_quality_score",
                    left_label="fr",
                    right_label="nr",
                    title="Quality FR-NR gap",
                    family_color_map=artifacts.fr.family_color_map,
                    spec=spec,
                )
            elif spec.renderer == "alignment_fr_vs_nr_delta":
                fig = build_gap_figure(
                    artifacts.alignment_gap,
                    score_col="global_alignment_score",
                    left_label="fr",
                    right_label="nr",
                    title="Alignment FR-NR gap",
                    family_color_map=artifacts.fr.family_color_map,
                    spec=spec,
                )
            elif spec.renderer == "joint_fr_vs_nr_shift":
                fig = build_joint_gap_figure(artifacts, spec)
            else:
                raise KeyError(f"Unsupported delta renderer: {spec.renderer}")
            output_path = paper_dir / f"{spec.slug}.pdf"
            save_figure(fig, output_path)
            saved[spec.slug] = str(output_path)
            records.append(
                build_figure_record(
                    experiment_slug=artifacts.slug,
                    spec=spec,
                    slug=spec.slug,
                    output_path=output_path,
                    inputs=inputs,
                )
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    figure_manifest_path = write_figure_manifest(
        exports_dir / "figures_manifest.json",
        experiment_slug=artifacts.slug,
        records=records,
    )
    saved["paper_bundle"] = str(bundle_path)
    return saved, figure_manifest_path


def render_triplet_delta_figures(
    experiment_dir: Path,
    artifacts: TripletDeltaExperimentArtifacts,
    figure_specs: list[FigureSpec],
) -> tuple[dict[str, str], str]:
    paper_dir = experiment_dir / "figures" / "paper"
    exports_dir = experiment_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = exports_dir / "paper_bundle.pdf"
    saved: dict[str, str] = {}
    records: list[dict[str, object]] = []
    inputs = {
        "fr_manifest": str(experiment_dir.parent / "triplet_fr" / "manifest.json"),
        "nr_manifest": str(experiment_dir.parent / "triplet_nr" / "manifest.json"),
    }

    with PdfPages(bundle_path) as pdf:
        for spec in figure_specs:
            if spec.renderer != "triplet_fr_vs_nr_delta":
                raise KeyError(f"Unsupported triplet delta renderer: {spec.renderer}")
            fig = build_gap_figure(
                artifacts.triplet_gap,
                score_col="global_triplet_score",
                left_label="fr",
                right_label="nr",
                title="Triplet FR-NR gap",
                family_color_map=artifacts.fr.family_color_map,
                spec=spec,
            )
            output_path = paper_dir / f"{spec.slug}.pdf"
            save_figure(fig, output_path)
            saved[spec.slug] = str(output_path)
            records.append(
                build_figure_record(
                    experiment_slug=artifacts.slug,
                    spec=spec,
                    slug=spec.slug,
                    output_path=output_path,
                    inputs=inputs,
                )
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    figure_manifest_path = write_figure_manifest(
        exports_dir / "figures_manifest.json",
        experiment_slug=artifacts.slug,
        records=records,
    )
    saved["paper_bundle"] = str(bundle_path)
    return saved, figure_manifest_path


def render_quality_scope_comparison_figures(
    experiment_dir: Path,
    artifacts: QualityScopeComparisonArtifacts,
    figure_specs: list[FigureSpec],
) -> tuple[dict[str, str], str]:
    paper_dir = experiment_dir / "figures" / "paper"
    supplementary_dir = experiment_dir / "figures" / "supplementary"
    exports_dir = experiment_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    renderer_map: dict[str, Callable[[QualityScopeComparisonArtifacts, FigureSpec], plt.Figure]] = {
        "quality_scope_gap": build_quality_scope_gap_figure,
        "quality_scope_scatter": build_quality_scope_scatter_figure,
        "quality_scope_dataset_heatmap": build_quality_scope_dataset_heatmap_figure,
        "quality_scope_family_shift": build_quality_scope_family_shift_figure,
    }
    paper_bundle_path = exports_dir / "paper_bundle.pdf"
    supplementary_bundle_path = exports_dir / "supplementary_bundle.pdf"
    saved: dict[str, str] = {}
    records: list[dict[str, object]] = []
    inputs = {
        "correct_quality_source": artifacts.correct_source,
        "wrong_quality_source": artifacts.wrong_source,
        "slice": artifacts.slice_name,
    }

    with PdfPages(paper_bundle_path) as paper_pdf, PdfPages(supplementary_bundle_path) as supplementary_pdf:
        for spec in figure_specs:
            fig = renderer_map[spec.renderer](artifacts, spec)
            target_dir = paper_dir if spec.tier == "paper" else supplementary_dir
            output_path = target_dir / f"{spec.slug}.pdf"
            save_figure(fig, output_path)
            saved[spec.slug] = str(output_path)
            records.append(
                build_figure_record(
                    experiment_slug=artifacts.slug,
                    spec=spec,
                    slug=spec.slug,
                    output_path=output_path,
                    inputs=inputs,
                )
            )
            if spec.tier == "paper":
                paper_pdf.savefig(fig, bbox_inches="tight")
            else:
                supplementary_pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    figure_manifest_path = write_figure_manifest(
        exports_dir / "figures_manifest.json",
        experiment_slug=artifacts.slug,
        records=records,
    )
    saved["paper_bundle"] = str(paper_bundle_path)
    saved["supplementary_bundle"] = str(supplementary_bundle_path)
    return saved, figure_manifest_path


def render_layerwise_figures(
    experiment_dir: Path,
    *,
    slices: dict[str, ReportExperimentArtifacts],
    model_keys: list[str],
    figure_specs: list[FigureSpec],
) -> tuple[pd.DataFrame, str]:
    spec = (
        figure_specs[0]
        if figure_specs
        else FigureSpec(
            slug="model_layer_bundle",
            renderer="model_layer_bundle",
            tier="supplementary",
            preset="layerwise_sheet",
        )
    )
    exports_dir = experiment_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    output_records: list[dict[str, str]] = []
    for model_key in model_keys:
        model_dir = experiment_dir / "models" / model_key
        figure_dir = model_dir / "figures" / "supplementary"
        fig = build_model_layer_bundle_figure(model_key, slices=slices, spec=spec)
        if fig is None:
            continue
        output_path = figure_dir / "layerwise_bundle.pdf"
        save_figure(fig, output_path, title=f"{model_key} layerwise bundle")
        plt.close(fig)
        output_records.append({"model_key": model_key, "bundle_pdf": str(output_path)})
        records.append(
            build_figure_record(
                experiment_slug="layerwise",
                spec=spec,
                slug=f"{model_key}_layerwise_bundle",
                output_path=output_path,
                inputs={"model_key": model_key, "slices": ["overall", "fr", "nr"]},
                title=f"{model_key} layerwise bundle",
            )
        )
    index_df = pd.DataFrame(output_records)
    if not index_df.empty:
        index_df = index_df.sort_values("model_key").reset_index(drop=True)
    figure_manifest_path = write_figure_manifest(
        exports_dir / "figures_manifest.json",
        experiment_slug="layerwise",
        records=records,
    )
    return index_df, figure_manifest_path


def render_triplet_layerwise_figures(
    experiment_dir: Path,
    *,
    slices: dict[str, TripletExperimentArtifacts],
    model_keys: list[str],
    figure_specs: list[FigureSpec],
) -> tuple[pd.DataFrame, str]:
    spec = (
        figure_specs[0]
        if figure_specs
        else FigureSpec(
            slug="model_triplet_layer_bundle",
            renderer="model_triplet_layer_bundle",
            tier="supplementary",
            preset="layerwise_sheet",
        )
    )
    exports_dir = experiment_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    output_records: list[dict[str, str]] = []
    experiment_slug = experiment_dir.name
    for model_key in model_keys:
        model_dir = experiment_dir / "models" / model_key
        figure_dir = model_dir / "figures" / "supplementary"
        fig = build_model_triplet_layer_bundle_figure(model_key, slices=slices, spec=spec)
        if fig is None:
            continue
        output_path = figure_dir / "layerwise_bundle.pdf"
        save_figure(fig, output_path, title=f"{model_key} triplet layerwise bundle")
        plt.close(fig)
        output_records.append({"model_key": model_key, "bundle_pdf": str(output_path)})
        records.append(
            build_figure_record(
                experiment_slug=experiment_slug,
                spec=spec,
                slug=f"{model_key}_triplet_layerwise_bundle",
                output_path=output_path,
                inputs={"model_key": model_key, "slices": ["overall", "fr", "nr"]},
                title=f"{model_key} triplet layerwise bundle",
            )
        )
    index_df = pd.DataFrame(output_records)
    if not index_df.empty:
        index_df = index_df.sort_values("model_key").reset_index(drop=True)
    figure_manifest_path = write_figure_manifest(
        exports_dir / "figures_manifest.json",
        experiment_slug=experiment_slug,
        records=records,
    )
    return index_df, figure_manifest_path


def build_family_map_from_slices(slices: dict[str, ReportExperimentArtifacts]) -> dict[str, str]:
    families: list[str] = []
    for artifacts in slices.values():
        families.extend(artifacts.quality.model_ranking["family"].tolist())
        families.extend(artifacts.alignment.model_ranking["family"].tolist())
    return create_family_color_map(families)
