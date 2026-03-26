from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from quality_backbones.charts.data import (
    DeltaExperimentArtifacts,
    QualityScopeComparisonArtifacts,
    ReportExperimentArtifacts,
    TripletDeltaExperimentArtifacts,
    TripletExperimentArtifacts,
    build_alignment_flops_comparison,
    build_alignment_dataset_metric_family_scores,
    build_alignment_metric_family_scores,
    build_gap_table,
    build_joint_gap_table,
    build_joint_layer_profiles,
    build_joint_ranking,
    build_model_metadata,
    build_quality_flops_comparison,
    build_quality_scope_dataset_model_comparison,
    build_quality_scope_dataset_summary,
    build_quality_scope_family_shift,
    build_quality_scope_layer_changes,
    build_quality_scope_model_comparison,
    build_quality_scope_summary_metrics,
    build_quality_dataset_metric_family_scores,
    build_quality_metric_family_scores,
    build_triplet_flops_comparison,
    build_triplet_dataset_family_scores,
    build_triplet_family_scores,
    filter_rows_by_slice,
    load_report_table,
    prepare_alignment_artifacts,
    prepare_quality_artifacts,
    prepare_triplet_artifacts,
)
from quality_backbones.charts.registry import CHARTS_ROOT, ExperimentSpec, get_experiment_map, load_experiments, load_figure_registry
from quality_backbones.charts.renderers import (
    render_delta_figures,
    render_layerwise_figures,
    render_quality_scope_comparison_figures,
    render_report_figures,
    render_triplet_delta_figures,
    render_triplet_figures,
    render_triplet_layerwise_figures,
)
from quality_backbones.charts.style import configure_style, create_family_color_map


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _experiment_dir(slug: str) -> Path:
    return CHARTS_ROOT / "experiments" / slug


def _spec_to_dict(spec: ExperimentSpec) -> dict[str, Any]:
    payload = asdict(spec)
    if spec.quality_report is not None:
        payload["quality_report"] = str(spec.quality_report)
    if spec.quality_report_alt is not None:
        payload["quality_report_alt"] = str(spec.quality_report_alt)
    if spec.alignment_report is not None:
        payload["alignment_report"] = str(spec.alignment_report)
    if spec.triplet_report is not None:
        payload["triplet_report"] = str(spec.triplet_report)
    return payload


def build_report_experiment(spec: ExperimentSpec, figure_registry: dict[str, list]) -> ReportExperimentArtifacts:
    if spec.slice_name is None or spec.quality_report is None or spec.alignment_report is None:
        raise ValueError(f"Report experiment is incomplete: {spec.slug}")

    configure_style()
    model_meta = build_model_metadata()
    excluded_families = set(spec.exclude_families)

    quality_rows, quality_meta = load_report_table(
        spec.quality_report,
        preferred_names=("report.json", "results.tsv", "results.csv"),
        results_key="results",
    )
    alignment_rows, alignment_meta = load_report_table(
        spec.alignment_report,
        preferred_names=("report.json", "results.tsv", "results.csv"),
        results_key="results",
    )

    quality_rows = filter_rows_by_slice(quality_rows, "dataset", spec.slice_name)
    alignment_rows = filter_rows_by_slice(alignment_rows, "dataset", spec.slice_name)

    quality = prepare_quality_artifacts(
        quality_rows,
        quality_meta,
        model_meta=model_meta,
        excluded_families=excluded_families,
        top_k=spec.top_k,
    )
    alignment = prepare_alignment_artifacts(
        alignment_rows,
        alignment_meta,
        model_meta=model_meta,
        excluded_families=excluded_families,
        top_k=spec.top_k,
    )

    family_color_map = create_family_color_map(
        quality.model_ranking["family"].tolist() + alignment.model_ranking["family"].tolist()
    )
    artifacts = ReportExperimentArtifacts(
        slug=spec.slug,
        title=spec.title,
        slice_name=spec.slice_name,
        quality_source=str(quality_meta.get("source_path", spec.quality_report)),
        alignment_source=str(alignment_meta.get("source_path", spec.alignment_report)),
        excluded_families=tuple(sorted(excluded_families)),
        quality=quality,
        alignment=alignment,
        joint_ranking=build_joint_ranking(quality, alignment),
        joint_layer_profiles=build_joint_layer_profiles(quality, alignment),
        quality_family_metric_scores=build_quality_metric_family_scores(quality),
        alignment_family_metric_scores=build_alignment_metric_family_scores(alignment),
        quality_dataset_family_metric_scores=build_quality_dataset_metric_family_scores(quality),
        alignment_dataset_family_metric_scores=build_alignment_dataset_metric_family_scores(alignment),
        family_color_map=family_color_map,
    )

    experiment_dir = _experiment_dir(spec.slug)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / "config.json", _spec_to_dict(spec))
    _write_json(
        experiment_dir / "sources" / "inputs.json",
        {
            "quality_source": artifacts.quality_source,
            "alignment_source": artifacts.alignment_source,
            "slice": spec.slice_name,
            "excluded_families": list(artifacts.excluded_families),
            "top_k": spec.top_k,
        },
    )

    aggregates = write_report_aggregates(experiment_dir, artifacts)
    figures, figure_manifest = render_report_figures(
        experiment_dir,
        artifacts,
        figure_registry["report_experiment"],
    )
    _write_json(
        experiment_dir / "manifest.json",
        {
            "slug": spec.slug,
            "title": spec.title,
            "kind": spec.kind,
            "slice": spec.slice_name,
            "sources": {
                "quality": artifacts.quality_source,
                "alignment": artifacts.alignment_source,
            },
            "excluded_families": list(artifacts.excluded_families),
            "aggregates": aggregates,
            "figure_manifest": figure_manifest,
            "figures": figures,
        },
    )
    return artifacts


def write_report_aggregates(experiment_dir: Path, artifacts: ReportExperimentArtifacts) -> dict[str, str]:
    paper_dir = experiment_dir / "aggregates" / "paper"
    supplementary_dir = experiment_dir / "aggregates" / "supplementary"
    cache_dir = experiment_dir / "aggregates" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "quality_model_ranking": paper_dir / "quality_model_ranking.tsv",
        "alignment_model_ranking": paper_dir / "alignment_model_ranking.tsv",
        "quality_alignment_joint": paper_dir / "quality_alignment_joint.tsv",
        "quality_best_layers_by_dataset": supplementary_dir / "quality_best_layers_by_dataset.tsv",
        "quality_dataset_winners": supplementary_dir / "quality_dataset_winners.tsv",
        "quality_selected_layer_flops": supplementary_dir / "quality_selected_layer_flops.tsv",
        "alignment_best_layers_by_dataset": supplementary_dir / "alignment_best_layers_by_dataset.tsv",
        "alignment_reference_ranking": supplementary_dir / "alignment_reference_ranking.tsv",
        "alignment_selected_layer_flops": supplementary_dir / "alignment_selected_layer_flops.tsv",
        "quality_family_scc_pcc": supplementary_dir / "quality_family_scc_pcc.tsv",
        "alignment_family_scc_pcc": supplementary_dir / "alignment_family_scc_pcc.tsv",
        "quality_family_scc_pcc_by_dataset": supplementary_dir / "quality_family_scc_pcc_by_dataset.tsv",
        "alignment_family_scc_pcc_by_dataset": supplementary_dir / "alignment_family_scc_pcc_by_dataset.tsv",
        "quality_layer_profiles": supplementary_dir / "quality_layer_profiles.tsv",
        "alignment_layer_profiles": supplementary_dir / "alignment_layer_profiles.tsv",
        "joint_layer_profiles": supplementary_dir / "joint_layer_profiles.tsv",
    }

    _write_dataframe(paths["quality_model_ranking"], artifacts.quality.model_ranking)
    _write_dataframe(paths["alignment_model_ranking"], artifacts.alignment.model_ranking)
    _write_dataframe(paths["quality_alignment_joint"], artifacts.joint_ranking)
    _write_dataframe(paths["quality_best_layers_by_dataset"], artifacts.quality.dataset_best)
    _write_dataframe(paths["quality_dataset_winners"], artifacts.quality.dataset_winners)
    _write_dataframe(paths["quality_selected_layer_flops"], build_quality_flops_comparison(artifacts.quality))
    _write_dataframe(paths["alignment_best_layers_by_dataset"], artifacts.alignment.dataset_best)
    _write_dataframe(paths["alignment_reference_ranking"], artifacts.alignment.reference_ranking)
    _write_dataframe(paths["alignment_selected_layer_flops"], build_alignment_flops_comparison(artifacts.alignment))
    _write_dataframe(paths["quality_family_scc_pcc"], artifacts.quality_family_metric_scores)
    _write_dataframe(paths["alignment_family_scc_pcc"], artifacts.alignment_family_metric_scores)
    _write_dataframe(paths["quality_family_scc_pcc_by_dataset"], artifacts.quality_dataset_family_metric_scores)
    _write_dataframe(paths["alignment_family_scc_pcc_by_dataset"], artifacts.alignment_dataset_family_metric_scores)
    _write_dataframe(paths["quality_layer_profiles"], artifacts.quality.layer_profiles)
    _write_dataframe(paths["alignment_layer_profiles"], artifacts.alignment.layer_profiles)
    _write_dataframe(paths["joint_layer_profiles"], artifacts.joint_layer_profiles)
    return {key: str(path) for key, path in paths.items()}


def build_triplet_experiment(spec: ExperimentSpec, figure_registry: dict[str, list]) -> TripletExperimentArtifacts:
    if spec.slice_name is None or spec.triplet_report is None:
        raise ValueError(f"Triplet experiment is incomplete: {spec.slug}")

    configure_style()
    model_meta = build_model_metadata()
    excluded_families = set(spec.exclude_families)

    triplet_rows, triplet_meta = load_report_table(
        spec.triplet_report,
        preferred_names=("report.json", "results.tsv", "results.csv"),
        results_key="results",
    )
    triplet_rows = filter_rows_by_slice(triplet_rows, "dataset", spec.slice_name)
    triplet = prepare_triplet_artifacts(
        triplet_rows,
        triplet_meta,
        model_meta=model_meta,
        excluded_families=excluded_families,
        top_k=spec.top_k,
    )

    family_color_map = create_family_color_map(triplet.model_ranking["family"].tolist())
    artifacts = TripletExperimentArtifacts(
        slug=spec.slug,
        title=spec.title,
        slice_name=spec.slice_name,
        triplet_source=str(triplet_meta.get("source_path", spec.triplet_report)),
        excluded_families=tuple(sorted(excluded_families)),
        triplet=triplet,
        triplet_family_scores=build_triplet_family_scores(triplet),
        triplet_dataset_family_scores=build_triplet_dataset_family_scores(triplet),
        family_color_map=family_color_map,
    )

    experiment_dir = _experiment_dir(spec.slug)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / "config.json", _spec_to_dict(spec))
    _write_json(
        experiment_dir / "sources" / "inputs.json",
        {
            "triplet_source": artifacts.triplet_source,
            "slice": spec.slice_name,
            "excluded_families": list(artifacts.excluded_families),
            "top_k": spec.top_k,
        },
    )

    aggregates = write_triplet_aggregates(experiment_dir, artifacts)
    figures, figure_manifest = render_triplet_figures(
        experiment_dir,
        artifacts,
        figure_registry["triplet_experiment"],
    )
    _write_json(
        experiment_dir / "manifest.json",
        {
            "slug": spec.slug,
            "title": spec.title,
            "kind": spec.kind,
            "slice": spec.slice_name,
            "sources": {"triplet": artifacts.triplet_source},
            "excluded_families": list(artifacts.excluded_families),
            "aggregates": aggregates,
            "figure_manifest": figure_manifest,
            "figures": figures,
        },
    )
    return artifacts


def write_triplet_aggregates(experiment_dir: Path, artifacts: TripletExperimentArtifacts) -> dict[str, str]:
    paper_dir = experiment_dir / "aggregates" / "paper"
    supplementary_dir = experiment_dir / "aggregates" / "supplementary"

    paths = {
        "triplet_model_ranking": paper_dir / "triplet_model_ranking.tsv",
        "triplet_best_layers_by_dataset": supplementary_dir / "triplet_best_layers_by_dataset.tsv",
        "triplet_dataset_winners": supplementary_dir / "triplet_dataset_winners.tsv",
        "triplet_selected_layer_flops": supplementary_dir / "triplet_selected_layer_flops.tsv",
        "triplet_family_scores": supplementary_dir / "triplet_family_scores.tsv",
        "triplet_family_scores_by_dataset": supplementary_dir / "triplet_family_scores_by_dataset.tsv",
        "triplet_layer_profiles": supplementary_dir / "triplet_layer_profiles.tsv",
    }

    _write_dataframe(paths["triplet_model_ranking"], artifacts.triplet.model_ranking)
    _write_dataframe(paths["triplet_best_layers_by_dataset"], artifacts.triplet.dataset_best)
    _write_dataframe(paths["triplet_dataset_winners"], artifacts.triplet.dataset_winners)
    _write_dataframe(paths["triplet_selected_layer_flops"], build_triplet_flops_comparison(artifacts.triplet))
    _write_dataframe(paths["triplet_family_scores"], artifacts.triplet_family_scores)
    _write_dataframe(paths["triplet_family_scores_by_dataset"], artifacts.triplet_dataset_family_scores)
    _write_dataframe(paths["triplet_layer_profiles"], artifacts.triplet.layer_profiles)
    return {key: str(path) for key, path in paths.items()}


def build_delta_experiment(
    spec: ExperimentSpec,
    *,
    fr: ReportExperimentArtifacts,
    nr: ReportExperimentArtifacts,
    figure_registry: dict[str, list],
) -> DeltaExperimentArtifacts:
    configure_style()
    experiment_dir = _experiment_dir(spec.slug)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / "config.json", _spec_to_dict(spec))
    _write_json(
        experiment_dir / "sources" / "inputs.json",
        {
            "depends_on": [fr.slug, nr.slug],
            "fr_manifest": str(_experiment_dir(fr.slug) / "manifest.json"),
            "nr_manifest": str(_experiment_dir(nr.slug) / "manifest.json"),
        },
    )

    artifacts = DeltaExperimentArtifacts(
        slug=spec.slug,
        title=spec.title,
        quality_gap=build_gap_table(
            fr.quality.model_ranking,
            nr.quality.model_ranking,
            score_col="global_quality_score",
            rank_col="quality_rank",
            left_label="fr",
            right_label="nr",
        ),
        alignment_gap=build_gap_table(
            fr.alignment.model_ranking,
            nr.alignment.model_ranking,
            score_col="global_alignment_score",
            rank_col="alignment_rank",
            left_label="fr",
            right_label="nr",
        ),
        joint_gap=build_joint_gap_table(fr.joint_ranking, nr.joint_ranking),
        fr=fr,
        nr=nr,
    )

    paper_dir = experiment_dir / "aggregates" / "paper"
    supplementary_dir = experiment_dir / "aggregates" / "supplementary"
    aggregates = {
        "quality_fr_nr_gap": str(paper_dir / "quality_fr_nr_gap.tsv"),
        "alignment_fr_nr_gap": str(paper_dir / "alignment_fr_nr_gap.tsv"),
        "joint_fr_nr_gap": str(paper_dir / "joint_fr_nr_gap.tsv"),
        "quality_full_table": str(supplementary_dir / "quality_fr_nr_gap_full.tsv"),
        "alignment_full_table": str(supplementary_dir / "alignment_fr_nr_gap_full.tsv"),
        "joint_full_table": str(supplementary_dir / "joint_fr_nr_gap_full.tsv"),
    }
    _write_dataframe(Path(aggregates["quality_fr_nr_gap"]), artifacts.quality_gap.head(25))
    _write_dataframe(Path(aggregates["alignment_fr_nr_gap"]), artifacts.alignment_gap.head(25))
    _write_dataframe(Path(aggregates["joint_fr_nr_gap"]), artifacts.joint_gap.head(25))
    _write_dataframe(Path(aggregates["quality_full_table"]), artifacts.quality_gap)
    _write_dataframe(Path(aggregates["alignment_full_table"]), artifacts.alignment_gap)
    _write_dataframe(Path(aggregates["joint_full_table"]), artifacts.joint_gap)

    figures, figure_manifest = render_delta_figures(
        experiment_dir,
        artifacts,
        figure_registry["delta_experiment"],
    )
    _write_json(
        experiment_dir / "manifest.json",
        {
            "slug": spec.slug,
            "title": spec.title,
            "kind": spec.kind,
            "depends_on": [fr.slug, nr.slug],
            "aggregates": aggregates,
            "figure_manifest": figure_manifest,
            "figures": figures,
        },
    )
    return artifacts


def build_triplet_delta_experiment(
    spec: ExperimentSpec,
    *,
    fr: TripletExperimentArtifacts,
    nr: TripletExperimentArtifacts,
    figure_registry: dict[str, list],
) -> TripletDeltaExperimentArtifacts:
    configure_style()
    experiment_dir = _experiment_dir(spec.slug)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / "config.json", _spec_to_dict(spec))
    _write_json(
        experiment_dir / "sources" / "inputs.json",
        {
            "depends_on": [fr.slug, nr.slug],
            "fr_manifest": str(_experiment_dir(fr.slug) / "manifest.json"),
            "nr_manifest": str(_experiment_dir(nr.slug) / "manifest.json"),
        },
    )

    artifacts = TripletDeltaExperimentArtifacts(
        slug=spec.slug,
        title=spec.title,
        triplet_gap=build_gap_table(
            fr.triplet.model_ranking,
            nr.triplet.model_ranking,
            score_col="global_triplet_score",
            rank_col="triplet_rank",
            left_label="fr",
            right_label="nr",
        ),
        fr=fr,
        nr=nr,
    )

    paper_dir = experiment_dir / "aggregates" / "paper"
    supplementary_dir = experiment_dir / "aggregates" / "supplementary"
    aggregates = {
        "triplet_fr_nr_gap": str(paper_dir / "triplet_fr_nr_gap.tsv"),
        "triplet_full_table": str(supplementary_dir / "triplet_fr_nr_gap_full.tsv"),
    }
    _write_dataframe(Path(aggregates["triplet_fr_nr_gap"]), artifacts.triplet_gap.head(25))
    _write_dataframe(Path(aggregates["triplet_full_table"]), artifacts.triplet_gap)

    figures, figure_manifest = render_triplet_delta_figures(
        experiment_dir,
        artifacts,
        figure_registry["triplet_delta_experiment"],
    )
    _write_json(
        experiment_dir / "manifest.json",
        {
            "slug": spec.slug,
            "title": spec.title,
            "kind": spec.kind,
            "depends_on": [fr.slug, nr.slug],
            "aggregates": aggregates,
            "figure_manifest": figure_manifest,
            "figures": figures,
        },
    )
    return artifacts


def build_quality_scope_comparison_experiment(
    spec: ExperimentSpec,
    *,
    figure_registry: dict[str, list],
) -> QualityScopeComparisonArtifacts:
    if spec.slice_name is None or spec.quality_report is None or spec.quality_report_alt is None:
        raise ValueError(f"Quality scope comparison experiment is incomplete: {spec.slug}")

    configure_style()
    model_meta = build_model_metadata()
    excluded_families = set(spec.exclude_families)

    correct_rows, correct_meta = load_report_table(
        spec.quality_report,
        preferred_names=("report.json", "results.tsv", "results.csv"),
        results_key="results",
    )
    wrong_rows, wrong_meta = load_report_table(
        spec.quality_report_alt,
        preferred_names=("report.json", "results.tsv", "results.csv"),
        results_key="results",
    )

    correct_rows = filter_rows_by_slice(correct_rows, "dataset", spec.slice_name)
    wrong_rows = filter_rows_by_slice(wrong_rows, "dataset", spec.slice_name)

    correct = prepare_quality_artifacts(
        correct_rows,
        correct_meta,
        model_meta=model_meta,
        excluded_families=excluded_families,
        top_k=spec.top_k,
    )
    wrong = prepare_quality_artifacts(
        wrong_rows,
        wrong_meta,
        model_meta=model_meta,
        excluded_families=excluded_families,
        top_k=spec.top_k,
    )

    family_color_map = create_family_color_map(
        correct.model_ranking["family"].tolist() + wrong.model_ranking["family"].tolist()
    )
    model_comparison = build_quality_scope_model_comparison(correct, wrong)
    dataset_model_comparison = build_quality_scope_dataset_model_comparison(correct, wrong)
    dataset_summary = build_quality_scope_dataset_summary(correct_meta, wrong_meta, dataset_model_comparison)
    family_shift = build_quality_scope_family_shift(model_comparison)
    layer_changes = build_quality_scope_layer_changes(model_comparison, dataset_model_comparison)
    summary_metrics = build_quality_scope_summary_metrics(model_comparison, dataset_summary, top_k=min(10, spec.top_k))

    artifacts = QualityScopeComparisonArtifacts(
        slug=spec.slug,
        title=spec.title,
        slice_name=spec.slice_name,
        correct_source=str(correct_meta.get("source_path", spec.quality_report)),
        wrong_source=str(wrong_meta.get("source_path", spec.quality_report_alt)),
        excluded_families=tuple(sorted(excluded_families)),
        correct=correct,
        wrong=wrong,
        model_comparison=model_comparison,
        dataset_model_comparison=dataset_model_comparison,
        dataset_summary=dataset_summary,
        family_shift=family_shift,
        layer_changes=layer_changes,
        summary_metrics=summary_metrics,
        family_color_map=family_color_map,
    )

    experiment_dir = _experiment_dir(spec.slug)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / "config.json", _spec_to_dict(spec))
    _write_json(
        experiment_dir / "sources" / "inputs.json",
        {
            "correct_quality_source": artifacts.correct_source,
            "wrong_quality_source": artifacts.wrong_source,
            "slice": spec.slice_name,
            "excluded_families": list(artifacts.excluded_families),
            "comparison": {
                "correct_label": "within_group",
                "wrong_label": "global",
            },
        },
    )

    paper_dir = experiment_dir / "aggregates" / "paper"
    supplementary_dir = experiment_dir / "aggregates" / "supplementary"
    aggregates = {
        "model_comparison": str(paper_dir / "quality_scope_model_comparison.tsv"),
        "dataset_summary": str(paper_dir / "quality_scope_dataset_summary.tsv"),
        "dataset_model_comparison": str(supplementary_dir / "quality_scope_dataset_model_comparison.tsv"),
        "family_shift": str(supplementary_dir / "quality_scope_family_shift.tsv"),
        "layer_changes": str(supplementary_dir / "quality_scope_layer_changes.tsv"),
        "summary_metrics": str(experiment_dir / "aggregates" / "paper" / "quality_scope_summary.json"),
    }
    _write_dataframe(Path(aggregates["model_comparison"]), artifacts.model_comparison)
    _write_dataframe(Path(aggregates["dataset_summary"]), artifacts.dataset_summary)
    _write_dataframe(Path(aggregates["dataset_model_comparison"]), artifacts.dataset_model_comparison)
    _write_dataframe(Path(aggregates["family_shift"]), artifacts.family_shift)
    _write_dataframe(Path(aggregates["layer_changes"]), artifacts.layer_changes)
    _write_json(Path(aggregates["summary_metrics"]), artifacts.summary_metrics)

    figures, figure_manifest = render_quality_scope_comparison_figures(
        experiment_dir,
        artifacts,
        figure_registry["quality_scope_comparison_experiment"],
    )
    _write_json(
        experiment_dir / "manifest.json",
        {
            "slug": spec.slug,
            "title": spec.title,
            "kind": spec.kind,
            "slice": spec.slice_name,
            "sources": {
                "correct_quality": artifacts.correct_source,
                "wrong_quality": artifacts.wrong_source,
            },
            "comparison": {
                "correct_label": "within_group",
                "wrong_label": "global",
            },
            "excluded_families": list(artifacts.excluded_families),
            "summary_metrics": artifacts.summary_metrics,
            "aggregates": aggregates,
            "figure_manifest": figure_manifest,
            "figures": figures,
        },
    )
    return artifacts


def build_layerwise_experiment(
    spec: ExperimentSpec,
    *,
    overall: ReportExperimentArtifacts,
    fr: ReportExperimentArtifacts,
    nr: ReportExperimentArtifacts,
    figure_registry: dict[str, list],
) -> pd.DataFrame:
    configure_style()
    experiment_dir = _experiment_dir(spec.slug)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    slices = {"overall": overall, "fr": fr, "nr": nr}
    _write_json(experiment_dir / "config.json", _spec_to_dict(spec))
    _write_json(
        experiment_dir / "sources" / "inputs.json",
        {
            "depends_on": [overall.slug, fr.slug, nr.slug],
            "slice_manifests": {
                key: str(_experiment_dir(value.slug) / "manifest.json")
                for key, value in slices.items()
            },
        },
    )

    model_sets = [set(artifacts.joint_ranking["model_key"].tolist()) for artifacts in slices.values()]
    model_keys = sorted(set.intersection(*model_sets)) if model_sets else []

    supplementary_dir = experiment_dir / "aggregates" / "supplementary"
    aggregate_paths: dict[str, str] = {}
    for slice_name, artifacts in slices.items():
        for key, df in {
            f"{slice_name}_quality_layer_profiles": artifacts.quality.layer_profiles,
            f"{slice_name}_alignment_layer_profiles": artifacts.alignment.layer_profiles,
            f"{slice_name}_joint_layer_profiles": artifacts.joint_layer_profiles,
        }.items():
            path = supplementary_dir / f"{key}.tsv"
            _write_dataframe(path, df)
            aggregate_paths[key] = str(path)

    layerwise_index, figure_manifest = render_layerwise_figures(
        experiment_dir,
        slices=slices,
        model_keys=model_keys,
        figure_specs=figure_registry["layerwise_experiment"],
    )
    index_path = experiment_dir / "exports" / "model_index.tsv"
    _write_dataframe(index_path, layerwise_index)
    aggregate_paths["model_index"] = str(index_path)

    for row in layerwise_index.itertuples(index=False):
        model_dir = experiment_dir / "models" / str(row.model_key)
        _write_json(
            model_dir / "manifest.json",
            {
                "model_key": str(row.model_key),
                "bundle_pdf": str(row.bundle_pdf),
                "slices": ["overall", "fr", "nr"],
            },
        )

    _write_json(
        experiment_dir / "manifest.json",
        {
            "slug": spec.slug,
            "title": spec.title,
            "kind": spec.kind,
            "depends_on": [overall.slug, fr.slug, nr.slug],
            "model_count": int(layerwise_index.shape[0]),
            "aggregates": aggregate_paths,
            "figure_manifest": figure_manifest,
        },
    )
    return layerwise_index


def build_triplet_layerwise_experiment(
    spec: ExperimentSpec,
    *,
    overall: TripletExperimentArtifacts,
    fr: TripletExperimentArtifacts,
    nr: TripletExperimentArtifacts,
    figure_registry: dict[str, list],
) -> pd.DataFrame:
    configure_style()
    experiment_dir = _experiment_dir(spec.slug)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    slices = {"overall": overall, "fr": fr, "nr": nr}
    _write_json(experiment_dir / "config.json", _spec_to_dict(spec))
    _write_json(
        experiment_dir / "sources" / "inputs.json",
        {
            "depends_on": [overall.slug, fr.slug, nr.slug],
            "slice_manifests": {
                key: str(_experiment_dir(value.slug) / "manifest.json")
                for key, value in slices.items()
            },
        },
    )

    model_sets = [set(artifacts.triplet.model_ranking["model_key"].tolist()) for artifacts in slices.values()]
    model_keys = sorted(set.intersection(*model_sets)) if model_sets else []

    supplementary_dir = experiment_dir / "aggregates" / "supplementary"
    aggregate_paths: dict[str, str] = {}
    for slice_name, artifacts in slices.items():
        for key, df in {
            f"{slice_name}_triplet_layer_profiles": artifacts.triplet.layer_profiles,
            f"{slice_name}_triplet_best_layers": artifacts.triplet.dataset_best,
        }.items():
            path = supplementary_dir / f"{key}.tsv"
            _write_dataframe(path, df)
            aggregate_paths[key] = str(path)

    layerwise_index, figure_manifest = render_triplet_layerwise_figures(
        experiment_dir,
        slices=slices,
        model_keys=model_keys,
        figure_specs=figure_registry["triplet_layerwise_experiment"],
    )
    index_path = experiment_dir / "exports" / "model_index.tsv"
    _write_dataframe(index_path, layerwise_index)
    aggregate_paths["model_index"] = str(index_path)

    for row in layerwise_index.itertuples(index=False):
        model_dir = experiment_dir / "models" / str(row.model_key)
        _write_json(
            model_dir / "manifest.json",
            {
                "model_key": str(row.model_key),
                "bundle_pdf": str(row.bundle_pdf),
                "slices": ["overall", "fr", "nr"],
            },
        )

    _write_json(
        experiment_dir / "manifest.json",
        {
            "slug": spec.slug,
            "title": spec.title,
            "kind": spec.kind,
            "depends_on": [overall.slug, fr.slug, nr.slug],
            "model_count": int(layerwise_index.shape[0]),
            "aggregates": aggregate_paths,
            "figure_manifest": figure_manifest,
        },
    )
    return layerwise_index


def build_all(selected_slugs: set[str] | None = None) -> dict[str, Any]:
    experiments = load_experiments()
    experiment_map = get_experiment_map(experiments)
    figure_registry = load_figure_registry()
    built_reports: dict[str, ReportExperimentArtifacts] = {}
    built_triplet_reports: dict[str, TripletExperimentArtifacts] = {}
    built_any: dict[str, Any] = {}

    report_slugs = [slug for slug, spec in experiment_map.items() if spec.kind == "report"]
    for slug in report_slugs:
        if selected_slugs is not None and slug not in selected_slugs and "layerwise" not in selected_slugs and "fr_vs_nr" not in selected_slugs:
            continue
        built_reports[slug] = build_report_experiment(experiment_map[slug], figure_registry)
        built_any[slug] = str(_experiment_dir(slug) / "manifest.json")

    if (selected_slugs is None or "fr_vs_nr" in selected_slugs) and "fr" in built_reports and "nr" in built_reports:
        delta_spec = experiment_map["fr_vs_nr"]
        build_delta_experiment(delta_spec, fr=built_reports["fr"], nr=built_reports["nr"], figure_registry=figure_registry)
        built_any[delta_spec.slug] = str(_experiment_dir(delta_spec.slug) / "manifest.json")

    if selected_slugs is None or "fr_quality_scope_comparison" in selected_slugs:
        comparison_spec = experiment_map.get("fr_quality_scope_comparison")
        if comparison_spec is not None:
            build_quality_scope_comparison_experiment(comparison_spec, figure_registry=figure_registry)
            built_any[comparison_spec.slug] = str(_experiment_dir(comparison_spec.slug) / "manifest.json")

    if (selected_slugs is None or "layerwise" in selected_slugs) and {"overall", "fr", "nr"}.issubset(built_reports):
        layer_spec = experiment_map["layerwise"]
        build_layerwise_experiment(
            layer_spec,
            overall=built_reports["overall"],
            fr=built_reports["fr"],
            nr=built_reports["nr"],
            figure_registry=figure_registry,
        )
        built_any[layer_spec.slug] = str(_experiment_dir(layer_spec.slug) / "manifest.json")

    triplet_report_slugs = [slug for slug, spec in experiment_map.items() if spec.kind == "triplet_report"]
    for slug in triplet_report_slugs:
        if selected_slugs is not None and slug not in selected_slugs and "triplet_layerwise" not in selected_slugs and "triplet_fr_vs_nr" not in selected_slugs:
            continue
        built_triplet_reports[slug] = build_triplet_experiment(experiment_map[slug], figure_registry)
        built_any[slug] = str(_experiment_dir(slug) / "manifest.json")

    if (
        (selected_slugs is None or "triplet_fr_vs_nr" in selected_slugs)
        and {"triplet_fr", "triplet_nr"}.issubset(built_triplet_reports)
    ):
        delta_spec = experiment_map["triplet_fr_vs_nr"]
        build_triplet_delta_experiment(
            delta_spec,
            fr=built_triplet_reports["triplet_fr"],
            nr=built_triplet_reports["triplet_nr"],
            figure_registry=figure_registry,
        )
        built_any[delta_spec.slug] = str(_experiment_dir(delta_spec.slug) / "manifest.json")

    if (
        (selected_slugs is None or "triplet_layerwise" in selected_slugs)
        and {"triplet_overall", "triplet_fr", "triplet_nr"}.issubset(built_triplet_reports)
    ):
        layer_spec = experiment_map["triplet_layerwise"]
        build_triplet_layerwise_experiment(
            layer_spec,
            overall=built_triplet_reports["triplet_overall"],
            fr=built_triplet_reports["triplet_fr"],
            nr=built_triplet_reports["triplet_nr"],
            figure_registry=figure_registry,
        )
        built_any[layer_spec.slug] = str(_experiment_dir(layer_spec.slug) / "manifest.json")

    manifest_payload = {"experiments": built_any}
    manifest_path = CHARTS_ROOT / "manifest.json"
    if selected_slugs is not None and manifest_path.exists():
        try:
            existing_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_payload = {}
        existing_experiments = existing_payload.get("experiments", {})
        if isinstance(existing_experiments, dict):
            merged_experiments = dict(existing_experiments)
            merged_experiments.update(built_any)
            manifest_payload = {"experiments": merged_experiments}
    _write_json(manifest_path, manifest_payload)
    return built_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication-grade chart experiments under charts/")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment slugs, for example: overall fr nr fr_vs_nr layerwise",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.experiments) if args.experiments else None
    built = build_all(selected)
    print("Built chart experiments:")
    for slug, manifest_path in built.items():
        print(f"- {slug}: {manifest_path}")
