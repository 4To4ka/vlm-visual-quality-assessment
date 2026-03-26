# Charts

Paper-facing chart workspace for ACM Multimedia 2026.

Structure:

- `registry/` - stable experiment, figure, and style registries.
- `experiments/<slug>/sources/` - pointers to upstream quality/alignment reports.
- `experiments/<slug>/aggregates/` - chart-ready TSV tables.
- `experiments/<slug>/figures/` - individual vector figures.
- `experiments/<slug>/exports/` - bundled PDFs, indices, and figure manifests.

Workflow notes:

- The chart pipeline is matplotlib-first and aligned with the `research-charts` skill.
- Each experiment now writes `exports/figures_manifest.json` with figure-level provenance metadata.

Current experiment set:

- `overall` - all retained datasets.
- `fr` - full-reference datasets only.
- `nr` - no-reference datasets only.
- `fr_quality_scope_comparison` - FR quality under correct within-group vs incorrect global aggregation.
- `fr_vs_nr` - FR/NR gap analysis.
- `layerwise` - per-model layerwise quality/alignment bundles.
- `triplet_overall` - all retained datasets under triplet agreement.
- `triplet_fr` - full-reference triplet agreement only.
- `triplet_nr` - no-reference triplet agreement only.
- `triplet_fr_vs_nr` - FR/NR triplet gap analysis.
- `triplet_layerwise` - per-model triplet layerwise bundles.

Hybrid storage policy:

- Commit lightweight registry/config/manifest/TSV artifacts.
- Keep heavyweight previews, caches, and large exploratory bundles generated.

Build:

```bash
conda run -n encoders python scripts/build_charts.py
```
