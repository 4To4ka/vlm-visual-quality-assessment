# Research README

This document is the research-facing index for `QualityBackbones`.

It is written for a human reader or a research agent that needs to answer questions such as:

- What does this repository study?
- Which code paths implement the main pipeline?
- Which result bundles are canonical?
- Which findings are safe to cite, and which are only exploratory?
- How are datasets, metrics, and artifacts connected?

It complements `README.md`, which is more command-oriented.

## 1. Research Agent Start Here

If you only inspect a small part of the repository, start with these paths:

### Canonical result bundles

- Primary headline bundle: `outputs/embedding_supplementary_report_refgrouped_complete_intersection/`
- Bundle manifest: `outputs/embedding_supplementary_report_refgrouped_complete_intersection/manifest.json`
- Text summary of best models: `outputs/embedding_supplementary_report_refgrouped_complete_intersection/best_models_report.txt`
- FR-only bundle: `outputs/embedding_supplementary_report_refgrouped_fr/`
- NR-only bundle: `outputs/embedding_supplementary_report_refgrouped_nr/`
- Overall triplet ranking: `charts/experiments/triplet_overall/aggregates/paper/triplet_model_ranking.tsv`

### Methodology-critical artifact

- FR aggregation caveat / ablation: `charts/experiments/fr_quality_scope_comparison/aggregates/paper/quality_scope_summary.json`

This artifact is important because it shows that full-reference (FR) quality rankings change dramatically if image pairs are aggregated globally instead of within reference groups.

### Core code entry points

- Model registry: `src/quality_backbones/manifest.py`
- Extraction script: `scripts/extract_embeddings.py`
- Quality evaluation: `scripts/report_embedding_quality.py`
- Alignment evaluation: `scripts/report_embedding_alignment.py`
- Triplet evaluation: `scripts/report_embedding_triplet_accuracy.py`
- Training: `scripts/train_embeddings.py`
- MANIQA-style training: `scripts/train_maniqa.py`
- Profiling: `scripts/profile_layer_performance.py`
- Chart builder: `scripts/build_charts.py`

### Supporting maps

- Chart experiment registry: `charts/manifest.json`
- Chart workspace guide: `charts/README.md`
- Existing operational README: `README.md`

## 2. What This Repository Does

`QualityBackbones` studies vision backbones as embedding spaces for image quality assessment (IQA) and related ranking/alignment tasks.

At a high level, the repository does five things:

1. Registers many vision encoders from different families.
2. Extracts per-layer embeddings for IQA datasets.
3. Evaluates embedding geometry with quality, alignment, and triplet metrics.
4. Profiles latency and approximate FLOPs for truncated encoder prefixes.
5. Trains lightweight heads, direct encoder fine-tunes, and MANIQA-style models with swappable encoders.

The current model registry exposes `86` enabled image models across `20` families in `src/quality_backbones/manifest.py`.

Representative families include:

- ResNet, ResNeXt, ConvNeXt, VGG
- ViT, MAE, CLIP, DINO, DINOv2, DINOv3, I-JEPA
- Swin, SwinV2
- SigLIP, SigLIP2
- InternViT, FastViT, FastViTHD
- ARNIQA and MANIQA reference/IQA-oriented models

## 3. Repository Map

| Path | Role | Notes |
| --- | --- | --- |
| `src/quality_backbones/` | Core Python package | Main implementation lives here. |
| `scripts/` | User-facing and batch entry points | Extraction, reports, profiling, training, chart building. |
| `outputs/` | Generated embeddings and result bundles | Main experiment artifacts live here. |
| `charts/` | Paper-facing chart workspace | Registries, aggregates, figures, exported PDF bundles. |
| `logs/` | Smoke, repro, parity, profiling logs | Mostly diagnostics and validation, not headline results. |
| `weights/` | Model weights, HF cache, vendored upstream repos | Dependency cache, not a canonical result location. |
| `tests/` | Minimal unit tests | Only a small subset of logic is covered. |
| `train_all.slurm`, `multinode_train_all.slurm`, `train_all_array.slurm` | Cluster launchers | Useful for large sweeps, but cluster-specific. |
| `models.txt` | Model inventory / audit-style artifact | Helpful context, not a primary source of results. |

## 4. Core Code Architecture

### Package modules

| File | Responsibility |
| --- | --- |
| `src/quality_backbones/manifest.py` | Central model registry via `ModelSpec`. |
| `src/quality_backbones/datasets.py` | Normalizes `data.csv` tables and image loading. |
| `src/quality_backbones/extractors.py` | Unified extractor layer across HF, timm, remote-code, IQA, and FastVLM-style models. |
| `src/quality_backbones/storage.py` | Writes extraction artifacts: `meta.json`, `index.parquet`, `layers.h5`. |
| `src/quality_backbones/cache.py` | Local cache and model-storage configuration. |
| `src/quality_backbones/evaluation.py` | Pairwise embedding-quality evaluation against target score distances. |
| `src/quality_backbones/alignment.py` | Geometry alignment evaluation against reference embedding spaces. |
| `src/quality_backbones/triplet_evaluation.py` | Exact triplet-order agreement evaluation. |
| `src/quality_backbones/profiling.py` | Prefix latency/FLOPs profiling and parity checks. |
| `src/quality_backbones/training.py` | Embedding-head training and encoder fine-tuning with Lightning. |
| `src/quality_backbones/maniqa_training.py` | MANIQA-style training with a swappable encoder backbone. |

### Chart subpackage

| File | Responsibility |
| --- | --- |
| `src/quality_backbones/charts/pipeline.py` | Orchestrates chart experiments end-to-end. |
| `src/quality_backbones/charts/data.py` | Converts raw reports into ranked/aggregated chart tables. |
| `src/quality_backbones/charts/renderers.py` | Matplotlib figure generation. |
| `src/quality_backbones/charts/registry.py` | Loads experiment and style registries. |
| `src/quality_backbones/charts/style.py` | Figure styles and plot presets. |
| `src/quality_backbones/charts/export.py` | Figure manifest and export metadata. |

## 5. End-to-End Workflow

The repository has one main research pipeline and several secondary branches.

### A. Embedding extraction pipeline

1. Read dataset CSV from `datasets_root/<dataset>/data.csv`.
2. Normalize rows and paths with `src/quality_backbones/datasets.py`.
3. Instantiate a backbone-specific extractor from `src/quality_backbones/extractors.py`.
4. Export per-layer embeddings into compact storage.

Main script:

- `scripts/extract_embeddings.py`

Output schema per dataset/model:

- `outputs/{dataset}/{model_key}/meta.json`
- `outputs/{dataset}/{model_key}/index.parquet`
- `outputs/{dataset}/{model_key}/layers.h5`

### B. Embedding evaluation pipeline

Three downstream metric families are built on top of extracted embeddings:

| Metric family | Script | Main output location |
| --- | --- | --- |
| Quality correlation | `scripts/report_embedding_quality.py` | `outputs/embedding_quality_runs/` |
| Cross-model alignment | `scripts/report_embedding_alignment.py` | `outputs/embedding_alignment_runs/` |
| Triplet agreement | `scripts/report_embedding_triplet_accuracy.py` | `outputs/embedding_triplet_runs/` |

These scripts read `index.parquet`, `meta.json`, and `layers.h5`, then emit resumable JSON/TSV reports.

### C. Charting and paper-facing reporting

Two artifact layers summarize the metric runs:

- Supplementary report bundles under `outputs/embedding_supplementary_report*`
- Paper-facing chart workspaces under `charts/experiments/*`

Main scripts:

- `scripts/visualize_embedding_reports.py`
- `scripts/build_charts.py`

### D. Profiling branch

- Script: `scripts/profile_layer_performance.py`
- Batch wrapper: `scripts/run_full_flops_profile_shards.py`
- Output locations: `outputs/profile_runs/` and `logs/full_flops_*`

### E. Training branch

- Main training: `scripts/train_embeddings.py`
- Large LOO sweep orchestration: `scripts/train_all_loo.py`
- Cluster launchers: `multinode_train_all.slurm`, `train_all_array.slurm`, `train_all.slurm`
- Output location: `outputs/training_runs/` and `outputs/training_runs_loo_all/`

### F. MANIQA-specific branch

- Script: `scripts/train_maniqa.py`
- Regression/parity helper: `scripts/check_maniqa_embedding_parity.py`
- Output location: `outputs/maniqa_training_runs/` plus parity logs under `logs/maniqa_parity/`

## 6. Datasets, Scope, and Coverage

The repository expects external IQA datasets under a root such as `/mnt/l/IQA`.

Datasets visible in stored results include:

- `AGIQA-3K`
- `CSIQ`
- `FLIVE_Database`
- `LIVEC`
- `PieAPP`
- `PIPAL`
- `SPAQ`
- `TID2013`
- `kadid10k`
- `koniq10k`

The most defensible headline bundle in the repository is the ref-grouped, complete-intersection report:

- `outputs/embedding_supplementary_report_refgrouped_complete_intersection/`

Why this bundle is the safest headline source:

- It uses reference-grouped aggregation for FR datasets.
- It applies a complete-intersection filter for fair model coverage.
- It excludes `ARNIQA` and `MANIQA` from the non-IQA backbone ranking.

According to `outputs/embedding_supplementary_report_refgrouped_complete_intersection/manifest.json`, this canonical slice retains:

- `8` datasets
- `73` models
- datasets: `AGIQA-3K`, `CSIQ`, `LIVEC`, `PieAPP`, `SPAQ`, `TID2013`, `kadid10k`, `koniq10k`

## 7. Canonical Result Bundles

### Headline quality/alignment bundles

These directories contain text summaries, TSV rankings, and PDF bundles.

- `outputs/embedding_supplementary_report/`
- `outputs/embedding_supplementary_report_complete_intersection/`
- `outputs/embedding_supplementary_report_refgrouped/`
- `outputs/embedding_supplementary_report_refgrouped_complete_intersection/`
- `outputs/embedding_supplementary_report_refgrouped_fr/`
- `outputs/embedding_supplementary_report_refgrouped_fr_complete_intersection/`
- `outputs/embedding_supplementary_report_refgrouped_nr/`
- `outputs/embedding_supplementary_report_refgrouped_nr_complete_intersection/`

Use cases:

- Use `refgrouped_complete_intersection` for headline overall conclusions.
- Use `refgrouped_fr` to study full-reference behavior.
- Use `refgrouped_nr` to study no-reference behavior.
- Use non-refgrouped bundles only for historical comparison or ablation context.

### Chart experiments

The chart registry in `charts/manifest.json` exposes `11` experiment workspaces:

- `overall`
- `fr`
- `nr`
- `fr_quality_scope_comparison`
- `fr_vs_nr`
- `layerwise`
- `triplet_overall`
- `triplet_fr`
- `triplet_nr`
- `triplet_fr_vs_nr`
- `triplet_layerwise`

Each workspace usually contains:

- `sources/` - pointers to upstream reports
- `aggregates/` - chart-ready TSV/JSON summaries
- `figures/` - individual vector figures
- `exports/` - bundled paper/supplementary PDFs

## 8. Main Findings Visible in Stored Artifacts

This section summarizes the main results that are already committed as artifacts.

### 8.1 Overall headline ranking (ref-grouped, complete-intersection)

Source:

- `outputs/embedding_supplementary_report_refgrouped_complete_intersection/best_models_report.txt`

Top quality models in the canonical overall slice:

1. `dinov2_base` - `0.9161`
2. `clip_vit_l14` - `0.9073`
3. `vit_tiny` - `0.9055`
4. `clip_vit_l14_336` - `0.9035`
5. `clip_vit_b16` - `0.8935`

Top alignment models in the same slice:

1. `dinov2_large` - `0.9134`
2. `dinov2_small` - `0.8892`
3. `dinov2_base` - `0.8883`
4. `siglip_large` - `0.8867`
5. `vit_mae_large` - `0.8847`

Per-dataset winners are heterogeneous rather than dominated by one family:

- `AGIQA-3K` -> `internvit_6b_v25`
- `CSIQ` -> `fastvithd_15b`
- `LIVEC` -> `vit_mae_huge`
- `PieAPP` -> `fastvithd_05b`
- `SPAQ` -> `siglip_so400m`
- `TID2013` -> `fastvithd_15b`
- `kadid10k` -> `fastvithd_05b`
- `koniq10k` -> `internvit_6b_v25`

Interpretation:

- DINOv2 and CLIP are strong overall generalists in the retained headline slice.
- Dataset-specific winners vary enough that no single backbone family is universally dominant.

### 8.2 FR and NR behavior are sharply different

Sources:

- `outputs/embedding_supplementary_report_refgrouped_fr/best_models_report.txt`
- `outputs/embedding_supplementary_report_refgrouped_nr/best_models_report.txt`
- `charts/experiments/fr_vs_nr/aggregates/paper/quality_fr_nr_gap.tsv`

FR quality leaders:

1. `fastvithd_05b` - `0.9938`
2. `fastvithd_15b` - `0.9922`
3. `fastvithd_7b` - `0.9693`

NR quality leaders:

1. `internvit_6b_v25` - `0.9882`
2. `clip_vit_l14_336` - `0.9680`
3. `internvit_300m_v25` - `0.9615`

Examples of large FR-vs-NR quality shifts from `quality_fr_nr_gap.tsv`:

- `siglip2_so400m_naflex`: `+0.6037` toward FR
- `siglip2_base_naflex`: `+0.5874` toward FR
- `internvit_300m`: `-0.5455` toward NR
- `internvit_300m_v25`: `-0.5305` toward NR

Interpretation:

- FR and NR should not be treated as a single homogeneous evaluation regime.
- Some families are excellent on FR but weak on NR, while others show the opposite pattern.

### 8.3 Triplet agreement highlights a partly different frontier

Source:

- `charts/experiments/triplet_overall/aggregates/paper/triplet_model_ranking.tsv`

Top triplet models in the stored overall ranking:

1. `clip_vit_l14_336` - `0.9390`
2. `siglip_so400m` - `0.9341`
3. `clip_vit_l14` - `0.9294`
4. `fastvithd_15b` - `0.9285`
5. `dinov2_base` - `0.9259`

Interpretation:

- The triplet frontier overlaps with the quality frontier but is not identical.
- CLIP, SigLIP, FastViTHD, and DINOv2 remain consistently strong families under triplet-order agreement.

### 8.4 The FR aggregation choice is not a minor detail

Source:

- `charts/experiments/fr_quality_scope_comparison/aggregates/paper/quality_scope_summary.json`

Key numbers from the stored FR aggregation comparison:

- `models_compared`: `75`
- `datasets_compared`: `4`
- `top_overlap_count`: `0`
- `spearman_rank`: `0.2424`
- `layer_changed_models`: `71`
- `mean_abs_delta_quality`: `0.1424`
- `mean_abs_rank_delta`: `21.87`

Interpretation:

- Using global pair aggregation on FR datasets materially changes the ranking.
- The repository's FR-specific grouped results should be preferred over non-grouped FR interpretations.

### 8.5 Profiling artifacts exist, but they are environment-sensitive

Sources:

- `logs/full_flops_20260324_patch_check.tsv`
- `logs/full_flops_20260324_v2/`

What is stored:

- Per-layer latency and throughput
- Approximate FLOPs / MACs
- Optional parity checks against normal extractor outputs

Important caveat:

- These numbers depend on hardware, CUDA stack, dtype, and model-loader behavior.
- Treat them as useful comparative measurements, not universally portable constants.

### 8.6 Training artifacts are currently mostly infrastructure/proof-of-pipeline

Source:

- `outputs/training_runs/smoke_regression_after_scale/run_summary.json`

The committed training run under `outputs/training_runs/` is a smoke-scale regression run, not a final large benchmark.

Interpretation:

- The training code is substantial and operational.
- The stored training artifacts in this repository snapshot should be interpreted as validation/prototyping evidence rather than final headline research results.

## 9. Methodological Caveats

These caveats are important for any downstream analysis.

### 9.1 FR quality must respect reference grouping

For FR datasets, using all pairs globally can inflate pair counts and distort rankings. The dedicated FR scope comparison experiment exists precisely because this issue is large enough to change conclusions.

Recommended rule:

- Prefer `refgrouped` FR-derived results over non-grouped FR results.

### 9.2 Complete-intersection filtering matters

Some models have incomplete dataset coverage. If the goal is a fair cross-model headline ranking, prefer complete-intersection slices such as:

- `outputs/embedding_supplementary_report_refgrouped_complete_intersection/`

### 9.3 `ARNIQA` and `MANIQA` are excluded from headline non-IQA rankings

The canonical supplementary bundle explicitly excludes these families from the main non-IQA comparison.

Reason:

- They are quality-specialized reference models and serve better as comparators or alignment anchors than as part of the generic backbone leaderboard.

### 9.4 External data is required for full recomputation

The repository expects IQA datasets outside the repo, typically under `/mnt/l/IQA`.

Implication:

- Stored tables and PDFs are analyzable directly from the repository.
- Full recomputation from raw data requires restoring the external datasets.

### 9.5 Some loaders are fragile by design

The extractor stack supports:

- Hugging Face models
- timm models
- remote-code models such as InternViT
- vendored GitHub-based models such as FastViTHD and MANIQA

This breadth is a strength, but it also makes exact environment replication harder.

## 10. Reproducibility Guide

### 10.1 What is easy to reproduce from the current repository snapshot

- Read headline results from committed `TSV`, `TXT`, and `PDF` artifacts.
- Rebuild chart bundles from existing upstream report artifacts with `scripts/build_charts.py`.
- Inspect chart source-to-export provenance through `charts/manifest.json` and per-experiment manifests.

### 10.2 What is reproducible if datasets and model access are restored

- Embedding extraction
- Quality/alignment/triplet recomputation
- Chart generation from newly generated reports
- Training and profiling runs

Prerequisites:

- IQA datasets under a compatible root such as `/mnt/l/IQA`
- model downloads into `weights/`
- a Python/CUDA environment similar to the one described in `README.md`, `Dockerfile`, and `scripts/setup_encoders_env.sh`

### 10.3 What is less portable

- Exact latency/FLOPs numbers across different hardware
- cluster launcher behavior in `multinode_train_all.slurm` and `train_all_array.slurm`
- remote-code model behavior if upstream implementations change

## 11. Minimal Reproduction Commands

These are the shortest commands needed to understand the intended workflow. For fuller operational detail, see `README.md`.

### Environment

Use either:

- `Dockerfile`
- `scripts/setup_encoders_env.sh`

### Prefetch weights

```bash
conda run -n encoders python scripts/prefetch_weights.py --weights-dir ./weights
```

### Smoke test

```bash
conda run -n encoders python scripts/smoke_test_all.py \
  --datasets-root /mnt/l/IQA \
  --weights-dir ./weights \
  --max-models 12 \
  --device cuda
```

### Extract embeddings

```bash
conda run -n encoders python scripts/extract_embeddings.py \
  --datasets-root /mnt/l/IQA \
  --dataset koniq10k \
  --weights-dir ./weights \
  --output-dir ./outputs \
  --batch-size 32 \
  --num-workers 4
```

### Quality report

```bash
conda run -n encoders python scripts/report_embedding_quality.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --target-field normalized_score \
  --embedding-distance cos,l2 \
  --score-distance abs,sq \
  --corr-metrics pcc,scc,kcc
```

### Alignment report

```bash
conda run -n encoders python scripts/report_embedding_alignment.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --distance-metrics cos,l2 \
  --corr-metrics pcc,scc,kcc
```

### Triplet report

```bash
conda run -n encoders python scripts/report_embedding_triplet_accuracy.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --target-field normalized_score \
  --embedding-distance cos,l2
```

### Build charts

```bash
conda run -n encoders python scripts/build_charts.py
```

## 12. What a Research Agent Should Trust First

Prefer these when summarizing the repository:

- `outputs/embedding_supplementary_report_refgrouped_complete_intersection/`
- `outputs/embedding_supplementary_report_refgrouped_fr/`
- `outputs/embedding_supplementary_report_refgrouped_nr/`
- `charts/experiments/overall/exports/`
- `charts/experiments/fr/exports/`
- `charts/experiments/nr/exports/`
- `charts/experiments/triplet_overall/exports/`
- `charts/experiments/fr_quality_scope_comparison/aggregates/paper/`

Use with caution:

- `outputs/embedding_supplementary_report/` when drawing FR-sensitive conclusions
- `logs/smoke_*.json` and `logs/repro_*.json` as diagnostics rather than final results
- `outputs/training_runs/` as infrastructure validation unless a run is clearly marked as a true benchmark

Usually ignore for research-summary purposes unless implementation detail is needed:

- `weights/hf/`
- `weights/github/`
- `qualitybackbones.sqsh`
- `__pycache__/` and committed cache artifacts

## 13. Auxiliary and Diagnostic Scripts

These scripts are useful, but they are not the main analytical pipeline.

| Script | Purpose |
| --- | --- |
| `scripts/report_extraction_progress.py` | Audits extraction completeness across datasets/models. |
| `scripts/audit_training_heads.py` | Audits leave-one-dataset-out training coverage. |
| `scripts/audit_dataset_indices.py` | Checks dataset CSV/path integrity. |
| `scripts/fix_dataset_case_paths.py` | Repairs case-only path mismatches in dataset CSVs. |
| `scripts/check_maniqa_embedding_parity.py` | Verifies MANIQA extractor stability across refactors. |
| `scripts/plot_vit_large_old_new_cos_heatmap.py` | One-off old-vs-new embedding comparison. |
| `scripts/run_full_flops_profile_shards.py` | Batch wrapper for profiling shards. |

## 14. Limitations and Technical Debt

The repository is powerful, but there are several structural caveats that matter for maintenance and analysis.

- No standard root packaging metadata such as `pyproject.toml` or `requirements.txt`.
- Sparse test coverage; the visible unit tests are mainly `tests/unit/test_profiling.py` and `tests/unit/test_triplet_evaluation.py`.
- Large, monolithic modules such as `src/quality_backbones/extractors.py`, `src/quality_backbones/evaluation.py`, and `src/quality_backbones/training.py`.
- Cluster launchers are tied to a specific environment and container layout.
- The repository mixes source code, heavy caches, weights, and generated results in one tree.
- Some exact results depend on remote-code or vendored third-party model implementations.

## 15. Short Interpretation Summary

If a research agent needs a concise summary of the repository, this is the safest version:

- `QualityBackbones` is a large comparative framework for evaluating vision backbones as IQA-relevant embedding spaces.
- The strongest stored headline results are the ref-grouped, complete-intersection quality/alignment bundles under `outputs/embedding_supplementary_report_refgrouped_complete_intersection/`.
- In that canonical slice, DINOv2 and CLIP are the strongest general overall families, but dataset-level winners remain heterogeneous.
- FR and NR behave very differently; FR especially requires reference-grouped evaluation.
- Triplet analysis broadly agrees with the quality story but emphasizes CLIP, SigLIP, FastViTHD, and DINOv2.
- The repository contains substantial training and profiling infrastructure, but the most clearly publication-facing stored artifacts are the supplementary report bundles and chart experiments.
