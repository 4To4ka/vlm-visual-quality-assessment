# QualityBackbones

Unified embedding extraction for multiple vision backbone families with:

- Local-only weight/cache management under `weights/`
- Compact per-layer embedding export in original layer order
- Single-image and batch extraction with GPU support
- Dataset ingestion from `/mnt/l/IQA/*/data.csv`

## Quick Start

1) Create environment and install dependencies (already executed in this workspace):

```bash
conda create -y --name encoders --clone encoders_heads
conda run -n encoders python -m pip install --upgrade pip setuptools wheel
conda run -n encoders python -m pip install "transformers>=4.56.0" "timm>=1.0.20" "huggingface_hub>=0.24.0" accelerate safetensors pillow numpy pandas pyarrow tqdm pyyaml h5py einops sentencepiece scipy scikit-image opencv-python ftfy regex tensorboardX torchsummary numba
```

2) Prefetch model weights into `weights/`:

```bash
conda run -n encoders python scripts/prefetch_weights.py --weights-dir ./weights
```

Notes:
- InternViT models are loaded through a local patched copy of remote code with `use_flash_attn=False`.
- MANIQA models auto-clone the official repository and auto-download official checkpoints.

3) Smoke-test loading/extraction on a few images:

```bash
conda run -n encoders python scripts/smoke_test_all.py \
  --datasets-root /mnt/l/IQA \
  --weights-dir ./weights \
  --max-models 12 \
  --device cuda
```

Runtime precision:
- On CUDA, extractors use `bfloat16` autocast by default.

4) Run extraction on a dataset (compact output):

```bash
conda run -n encoders python scripts/extract_embeddings.py \
  --datasets-root /mnt/l/IQA \
  --dataset koniq10k \
  --weights-dir ./weights \
  --output-dir ./outputs \
  --batch-size 32 \
  --num-workers 4 \
  --max-samples 256
```

## Truncated Encoder Profiling

Profile cumulative latency and approximate FLOPs for the current exported layer boundaries of each visual encoder.
The profiler runs prefix-forward variants up to each exported `layer_name` and can optionally verify that the
truncated output exactly matches the existing extractor output for the same layer.

Example:

```bash
conda run -n encoders python scripts/profile_layer_performance.py \
  --datasets-root /mnt/l/IQA \
  --dataset koniq10k \
  --weights-dir ./weights \
  --models vit_base clip_vit_b16 swin_tiny \
  --batch-size 1 \
  --warmup 5 \
  --iters 20 \
  --verify-parity \
  --table-out ./logs/layer_profile.tsv \
  --json-out ./logs/layer_profile.json
```

Artifacts are written under `outputs/profile_runs/<timestamp>/` by default, including `results.jsonl`,
`progress.json`, and `report.json`.

## Output Format (schema v1)

For each dataset/model:

- `outputs/{dataset}/{model_key}/meta.json`
- `outputs/{dataset}/{model_key}/index.parquet`
- `outputs/{dataset}/{model_key}/layers.h5`

`layers.h5` contains one dataset per layer in forward order: `layer_000`, `layer_001`, ...
with shape `[N, D_i]` and file attrs including `layer_names`.

## Extraction Progress Report

To analyze extraction coverage across datasets/models and list missing/error runs:

```bash
conda run -n encoders python scripts/report_extraction_progress.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs
```

Optional JSON export:

```bash
conda run -n encoders python scripts/report_extraction_progress.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --json-out ./logs/extraction_progress.json
```

Status definitions per `outputs/{dataset}/{model}`:

- `success`: `meta.json`, `index.parquet`, and `layers.h5` exist
- `error`: `error.json` exists and success artifacts are incomplete
- `missing`: model directory does not exist (not run yet)
- `incomplete`: directory exists but neither `success` nor `error`

## Exact Pairwise Embedding Quality Report

Compute exact layer-wise IQA proxy by correlating:

- pairwise embedding distances (`cos`, `l2`, `l1`)
- pairwise target distances (`abs` or `sq` over chosen target field)

Supported correlation metrics:

- `pcc` (Pearson)
- `scc` (Spearman)
- `kcc` (Kendall tau-b)

Pair scope modes:

- `--pair-scope auto` (default): use `within_ref` for FR datasets and `global` otherwise
- `--pair-scope global`: compare all sampled pairs in the dataset
- `--pair-scope within_ref`: compare pairs only within the same `ref_filename`, then average scores across references

The report uses existing extraction outputs (`index.parquet` + `layers.h5`).
By default, it runs on the full dataset (`--sample-limit` omitted).

Long-run behavior:

- each completed `dataset/model/layer` is appended immediately to `results.jsonl`
- live run state is written to `progress.json`
- human-readable logs go to `run.log`
- final aggregated report is written to `report.json`
- resume is supported via `--run-dir ... --resume`

Example:

```bash
conda run -n encoders python scripts/report_embedding_quality.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --datasets CSIQ \
  --models vit_base \
  --layers canonical_embedding \
  --target-field normalized_score \
  --embedding-distance cos,l2 \
  --score-distance abs,sq \
  --corr-metrics pcc,scc,kcc \
  --jobs auto \
  --table-out ./logs/embedding_quality.tsv \
  --json-out ./logs/embedding_quality.json
```

To limit the number of images while still computing all pairs exactly inside the subset:

```bash
--sample-limit 2048
```

Long-run / resumable example:

```bash
conda run -n encoders python scripts/report_embedding_quality.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --target-field normalized_score \
  --embedding-distance cos,l2 \
  --score-distance abs,sq \
  --corr-metrics pcc,scc,kcc \
  --jobs auto \
  --progress log \
  --heartbeat-sec 60 \
  --run-dir ./outputs/embedding_quality_runs/full_exact \
  --table-out ./logs/embedding_quality_full.tsv \
  --json-out ./logs/embedding_quality_full.json
```

Resume the same run:

```bash
conda run -n encoders python scripts/report_embedding_quality.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --target-field normalized_score \
  --embedding-distance cos,l2 \
  --score-distance abs,sq \
  --corr-metrics pcc,scc,kcc \
  --jobs auto \
  --progress log \
  --heartbeat-sec 60 \
  --run-dir ./outputs/embedding_quality_runs/full_exact \
  --resume
```

## Exact Embedding Triplet Accuracy Report

Compute exact triplet-order agreement between embedding distances and absolute target differences.
For each ordered triplet `(i, j, k)`, the report checks:

- `dist(i, j) < dist(i, k)`
- `|score_i - score_j| < |score_i - score_k|`

and reports the fraction of matches per `(dataset, model, layer, embedding_distance)`.

Grouping follows the same pair-scope logic as `report_embedding_quality.py`:

- `--pair-scope auto` (default): use `within_ref` for FR datasets and `global` otherwise
- `--pair-scope global`: compare all sampled items together
- `--pair-scope within_ref`: compare triplets only within the same `ref_filename`, then average scores across references

Example:

```bash
conda run -n encoders python scripts/report_embedding_triplet_accuracy.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --datasets CSIQ \
  --models vit_base \
  --layers canonical_embedding \
  --target-field normalized_score \
  --embedding-distance cos,l2 \
  --pair-scope auto \
  --jobs auto \
  --table-out ./logs/embedding_triplet.tsv \
  --json-out ./logs/embedding_triplet.json
```

As with the pairwise quality report, `--sample-limit` applies before grouping and the exact metric is computed inside the sampled subset.

## Exact Embedding Alignment Report

Compute exact alignment between pairwise distances from one embedding space and another.
This is useful for questions like: how similar are the geometry of a generic encoder and
the geometry of a reference IQA encoder such as `MANIQA` or `ARNIQA`.

The runner compares:

- candidate embedding distances
- reference embedding distances

and reports exact `pcc`, `scc`, `kcc` for each `(dataset, reference layer, candidate layer, distance metric)`.

Alignment uses the same `--pair-scope auto|global|within_ref` behavior as the quality report.

Example: compare all non-IQA families against selected IQA references, using the last layer of each reference model:

```bash
conda run -n encoders python scripts/report_embedding_alignment.py \
  --datasets CSIQ koniq10k \
  --reference-models arniqa_csiq maniqa_koniq10k \
  --reference-layers last \
  --exclude-families ARNIQA MANIQA \
  --distance-metrics cos,l2 \
  --corr-metrics pcc,scc,kcc \
  --jobs auto \
  --progress log \
  --run-dir ./outputs/embedding_alignment_runs/main \
  --table-out ./logs/embedding_alignment_main.tsv \
  --json-out ./logs/embedding_alignment_main.json
```

Self-consistency sanity check (same model/layer against itself should be `1.0`):

```bash
conda run -n encoders python scripts/report_embedding_alignment.py \
  --datasets CSIQ \
  --reference-models vit_base \
  --candidate-models vit_base \
  --reference-layers last \
  --candidate-layers last \
  --sample-limit 32 \
  --distance-metrics cos,l2 \
  --corr-metrics pcc,scc,kcc
```

## Training (Embeddings and Encoder Fine-Tuning)

Training is implemented with **PyTorch + Lightning** in `scripts/train_embeddings.py`.

Install training dependencies into the project env:

```bash
conda run -n encoders python -m pip install "lightning>=2.4,<3"
conda run -n encoders python -m pip install tensorboard
conda run -n encoders python -m pip install peft
```

### Train mode

Choose training backend with `--train-mode`:

- `--train-mode embeddings` (default): train head on pre-extracted features (`--feature-source` required)
- `--train-mode encoder`: train/fine-tune encoder + head directly from images (`--encoder-model` required)

Encoder mode currently supports model loaders:

- `hf_auto_image`
- `timm_cnn_features`
- `timm_vit_blocks`

`fastvithd`, `arniqa`, and `maniqa` are intentionally excluded from encoder fine-tuning mode.

### Feature selection

Use one or more `--feature-source` arguments:

- `model_key:all` or `model_key:*` -> all layers for a model
- `model_key:layer_name` -> one named layer
- `model_key:0,3,7` -> selected layer indices
- `model_key:0-11,canonical_embedding` -> mix ranges + named layers

This supports:

- one layer
- multiple layers
- all layers
- multiple layers across multiple models

### Encoder tuning options

When using `--train-mode encoder`:

- `--encoder-model <model_key>` picks backbone from `manifest.py`
- `--tune-mode full` enables full fine-tune
- `--tune-mode frozen` freezes encoder and trains only head
- `--tune-mode lora` injects LoRA adapters (HF encoder loaders only)
- `--head-type linear|mlp` selects prediction head
- `--init-head-from <checkpoint>` initializes head from `.ckpt` or `state_dict`

### Optimizer and scheduler controls

- `--optimizer adamw|sgd`
- `--scheduler none|cosine|linear`
- `--warmup-steps` or `--warmup-ratio`
- `--min-lr-ratio`
- `--early-stopping-patience` (0 disables; stops on `val_loss`)
- `--encoder-lr`, `--head-lr` (separate LR in encoder mode)
- `--encoder-weight-decay`, `--head-weight-decay`
- `--max-grad-norm`, `--accumulate-grad-batches`

### Dataset and split control

- `--train-source dataset[:fraction]` (repeatable)
- `--val-source dataset[:fraction]` (repeatable, optional)

If `--val-source` is omitted, validation is built from train sources via `--split-policy`:

- `predefined` -> uses `split` column (`train,training` vs `val,validation,valid,test` by default)
- `random` -> random split by `--val-ratio`
- `group` -> group-aware split by `--group-field`

### Target field selection

Set `--target-field` to:

- top-level columns (for example `subjective_score`)
- metadata values via `metadata.<key>` (for example `metadata.MOS`)

To inspect fields before training:

```bash
conda run -n encoders python scripts/train_embeddings.py \
  --train-source kadid10k \
  --feature-source resnet50:canonical_embedding \
  --list-target-fields
```

### Training modes

- `--task regression` -> scalar target regression
- `--task pairwise` -> N=2 Siamese-style ranking
- `--task listwise --num-choices N` -> N>=2 candidate ranking

For ranking targets:

- hard winner target: `--pairwise-target hard` / `--listwise-target hard`
- soft score matching: `--pairwise-target soft` / `--listwise-target soft`
- soft-target temperature (GT smoothing only): `--target-temperature`
- CLIP-style learnable model logit scale for ranking: `--learnable-logit-scale`
- learnable logit scale init as temperature: `--logit-temperature-init` (default `0.07`)
- clamp for learnable logit scale: `--logit-scale-max` (default `100`)

### Logs and artifacts

Each run directory (for example `outputs/training_runs/<run_name>/`) includes:

- `run_summary.json` and `result.json`
- `checkpoints/` (best and last checkpoint)
- `logs/` (Lightning CSV logger)
- `tb/` (TensorBoard event files, enabled by default; disable with `--no-tensorboard`)
- `epoch_metrics.csv` + `epoch_metrics.jsonl` (epoch-level metrics)
- `epoch_predictions/epoch_XXX.csv` + `epoch_predictions/epoch_XXX.jsonl` (val predictions per epoch)

The single-node launcher `train_all.slurm`, the multi-node launcher `multinode_train_all.slurm`,
and the job-array launcher `train_all_array.slurm` use a nested layout for easier browsing:

- `outputs/training_runs_loo_all/<val_dataset>/<model_key>/<layer_dir>/loo_<run_idx>/`

Use `multinode_train_all.slurm` when you want to spread the same plan across all allocated GPUs
on 2+ nodes; for example: `sbatch --nodes=3 multinode_train_all.slurm`.

To resume strictly from an existing `outputs/training_runs_loo_all/plan.tsv` without rebuilding it,
submit `sbatch --nodes=3 multinode_train_all.slurm --reuse-plan`.

Use `train_all_array.slurm` when you want to spread the same plan across multiple 8-GPU nodes via
independent array jobs instead of a shared multi-node `srun` step. Each array task consumes one node
and launches 8 local GPU workers, so the total shard count is `8 * SLURM_ARRAY_TASK_COUNT`.

Examples:

```bash
sbatch --array=0-15%4 train_all_array.slurm
sbatch --array=0-15%4 train_all_array.slurm --reuse-plan
sbatch --array=3,7,11 train_all_array.slurm --reuse-plan
```

Control prediction artifact size with `--val-predictions-max-rows`.

Open TensorBoard:

```bash
conda run -n encoders tensorboard --logdir ./outputs/training_runs --port 6006
```

### Examples

Regression on one layer:

```bash
conda run -n encoders python scripts/train_embeddings.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --train-source kadid10k \
  --feature-source resnet50:canonical_embedding \
  --target-field normalized_score \
  --task regression \
  --max-epochs 10 \
  --early-stopping-patience 2 \
  --batch-size 256
```

Pairwise (N=2) on multiple layers from one model:

```bash
conda run -n encoders python scripts/train_embeddings.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --train-source kadid10k \
  --feature-source vit_small:block_008,block_010,canonical_embedding \
  --task pairwise \
  --pairwise-target soft \
  --target-temperature 2.0 \
  --learnable-logit-scale \
  --logit-temperature-init 0.07 \
  --max-epochs 10
```

Listwise (N>=2) on all layers from two models and mixed datasets:

```bash
conda run -n encoders python scripts/train_embeddings.py \
  --datasets-root /mnt/l/IQA \
  --outputs-root ./outputs \
  --train-source kadid10k \
  --train-source koniq10k:0.6 \
  --val-source koniq10k:0.2 \
  --feature-source resnet50:all \
  --feature-source vit_small:all \
  --target-field normalized_score \
  --task listwise \
  --num-choices 4 \
  --listwise-target soft \
  --target-temperature 2.0 \
  --accelerator auto \
  --devices auto
```

Encoder mode: full fine-tune on a CNN backbone:

```bash
conda run -n encoders python scripts/train_embeddings.py \
  --train-mode encoder \
  --datasets-root /mnt/l/IQA \
  --weights-dir ./weights \
  --train-source kadid10k \
  --encoder-model resnet18 \
  --task regression \
  --target-field subjective_score \
  --tune-mode full \
  --head-type linear \
  --max-epochs 10
```

Encoder mode: LoRA on a transformer backbone:

```bash
conda run -n encoders python scripts/train_embeddings.py \
  --train-mode encoder \
  --datasets-root /mnt/l/IQA \
  --weights-dir ./weights \
  --train-source kadid10k \
  --encoder-model vit_base \
  --task regression \
  --target-field subjective_score \
  --tune-mode lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --head-type mlp \
  --max-epochs 10
```

For multi-GPU runs, set `--devices` and `--strategy` (for example `--devices 4 --strategy ddp`).

## MANIQA-Based Training with Swappable Encoder

You can train a MANIQA-style IQA model (TAB + Swin + patch-weighted scoring) while swapping only
the vision encoder.

Supported encoder scope:

- Transformer encoders from manifest with loaders:
  - `timm_vit_blocks`
  - `hf_auto_image` (transformer families)
  - `hf_auto_image_remote`
  - `hf_swin_vision`
  - `hf_clip_vision`
  - `hf_siglip_vision`
  - `hf_siglip2_vision`
- `fastvithd_*` encoders are intentionally excluded.

List supported encoder keys:

```bash
conda run -n encoders python scripts/train_maniqa.py \
  --list-encoders
```

Example run:

```bash
conda run -n encoders python scripts/train_maniqa.py \
  --datasets-root /mnt/l/IQA \
  --weights-dir ./weights \
  --train-source koniq10k \
  --target-field normalized_score \
  --encoder-model vit_small \
  --max-epochs 10 \
  --batch-size 16
```

Optional encoder-layer control:

```bash
--encoder-layer-indices -4,-3,-2,-1
```

If not provided, the script auto-selects layers from the upper half of encoder depth.

Artifacts are written to `outputs/maniqa_training_runs/<run_name>/` with the same structure as
embedding training (`run_summary.json`, `result.json`, checkpoints, CSV/TB logs, epoch metrics/predictions).

## MANIQA Embedding Parity Check (Before/After Refactors)

Use this script to snapshot MANIQA extractor embeddings and verify they are unchanged after code updates.

Create baseline snapshot:

```bash
conda run -n encoders python scripts/check_maniqa_embedding_parity.py \
  --datasets-root /mnt/l/IQA \
  --dataset koniq10k \
  --weights-dir ./weights \
  --max-samples 3 \
  --device cpu \
  --seed 42 \
  --snapshot logs/maniqa_parity/baseline_before_changes.json
```

Compare current outputs to baseline:

```bash
conda run -n encoders python scripts/check_maniqa_embedding_parity.py \
  --datasets-root /mnt/l/IQA \
  --dataset koniq10k \
  --weights-dir ./weights \
  --max-samples 3 \
  --device cpu \
  --seed 42 \
  --compare logs/maniqa_parity/baseline_before_changes.json \
  --report logs/maniqa_parity/compare_after_changes.json
```
