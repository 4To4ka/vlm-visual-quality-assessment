# Research Output

## Scope and sources

- This report follows `ResearchREADME.md` and uses the repository's strongest stored artifacts for model selection.
- `Quality Overall` source: `outputs/embedding_supplementary_report_refgrouped_complete_intersection/quality_model_ranking.tsv`
- `Quality NR` source: `outputs/embedding_supplementary_report_refgrouped_nr_complete_intersection/quality_model_ranking.tsv`
- `Triplet Overall` source: `charts/experiments/triplet_overall/aggregates/paper/triplet_model_ranking.tsv`
- `Triplet NR` source: `charts/experiments/triplet_nr/aggregates/paper/triplet_model_ranking.tsv`
- As documented in `ResearchREADME.md`, the quality leaderboards are the canonical ref-grouped complete-intersection slices. Triplet does not have a stored complete-intersection counterpart, so the final shortlist is chosen from the 73-model intersection shared by all four leaderboards.
- `q` is the rank quantile inside each published leaderboard, where `1.000` is best-in-table. Quality tables contain 73 models; triplet tables contain 75 models.
- `Recommended layer` is the single compromise layer that best preserves performance across overall/NR quality and overall/NR triplet layer profiles.

## Requested top-10 leaderboards

### Quality Overall

| Rank | Model | Family | Best layer | Score |
| --- | --- | --- | --- | --- |
| 1 | `dinov2_base` | DINOv2 | `hidden_state_003` | 0.9161 |
| 2 | `clip_vit_l14` | CLIP | `hidden_state_020` | 0.9073 |
| 3 | `vit_tiny` | ViT | `block_003` | 0.9055 |
| 4 | `clip_vit_l14_336` | CLIP | `hidden_state_020` | 0.9035 |
| 5 | `clip_vit_b16` | CLIP | `hidden_state_010` | 0.8935 |
| 6 | `siglip_so400m` | SigLIP | `hidden_state_012` | 0.8928 |
| 7 | `dinov2_small` | DINOv2 | `hidden_state_004` | 0.8875 |
| 8 | `clip_vit_b32` | CLIP | `hidden_state_010` | 0.8853 |
| 9 | `dinov2_giant` | DINOv2 | `hidden_state_030` | 0.8836 |
| 10 | `siglip_large` | SigLIP | `hidden_state_012` | 0.8812 |

### Quality NR

| Rank | Model | Family | Best layer | Score |
| --- | --- | --- | --- | --- |
| 1 | `internvit_6b_v25` | InternViT | `hidden_state_043` | 0.9879 |
| 2 | `clip_vit_l14_336` | CLIP | `hidden_state_020` | 0.9672 |
| 3 | `internvit_300m_v25` | InternViT | `hidden_state_005` | 0.9616 |
| 4 | `clip_vit_l14` | CLIP | `hidden_state_019` | 0.9596 |
| 5 | `dinov2_base` | DINOv2 | `hidden_state_003` | 0.9451 |
| 6 | `dinov2_large` | DINOv2 | `hidden_state_015` | 0.9449 |
| 7 | `dinov2_giant` | DINOv2 | `hidden_state_021` | 0.9301 |
| 8 | `internvit_300m` | InternViT | `hidden_state_005` | 0.9188 |
| 9 | `dinov2_small` | DINOv2 | `hidden_state_003` | 0.9170 |
| 10 | `siglip_so400m` | SigLIP | `hidden_state_010` | 0.9157 |

### Triplet Overall

| Rank | Model | Family | Best layer | Score |
| --- | --- | --- | --- | --- |
| 1 | `clip_vit_l14_336` | CLIP | `hidden_state_020` | 0.9390 |
| 2 | `siglip_so400m` | SigLIP | `hidden_state_012` | 0.9341 |
| 3 | `clip_vit_l14` | CLIP | `hidden_state_020` | 0.9294 |
| 4 | `fastvithd_15b` | FastViTHD | `block_041` | 0.9285 |
| 5 | `dinov2_base` | DINOv2 | `hidden_state_003` | 0.9259 |
| 6 | `fastvithd_05b` | FastViTHD | `block_038` | 0.9253 |
| 7 | `clip_vit_b16` | CLIP | `hidden_state_010` | 0.9104 |
| 8 | `dinov2_giant` | DINOv2 | `hidden_state_029` | 0.9013 |
| 9 | `fastvithd_7b` | FastViTHD | `block_040` | 0.8997 |
| 10 | `dinov2_small` | DINOv2 | `hidden_state_004` | 0.8957 |

### Triplet NR

| Rank | Model | Family | Best layer | Score |
| --- | --- | --- | --- | --- |
| 1 | `internvit_6b_v25` | InternViT | `hidden_state_041` | 0.9863 |
| 2 | `clip_vit_l14` | CLIP | `hidden_state_019` | 0.9692 |
| 3 | `clip_vit_l14_336` | CLIP | `hidden_state_020` | 0.9683 |
| 4 | `internvit_300m_v25` | InternViT | `hidden_state_005` | 0.9529 |
| 5 | `dinov2_base` | DINOv2 | `hidden_state_005` | 0.9448 |
| 6 | `dinov2_large` | DINOv2 | `hidden_state_016` | 0.9427 |
| 7 | `siglip_so400m` | SigLIP | `hidden_state_012` | 0.9376 |
| 8 | `dinov2_giant` | DINOv2 | `hidden_state_020` | 0.9296 |
| 9 | `dinov2_small` | DINOv2 | `hidden_state_003` | 0.9272 |
| 10 | `internvit_300m` | InternViT | `hidden_state_006` | 0.9022 |

## Final top-10 models for further e2e training

### Selection logic

- Priority 1: strong ranks across all four requested leaderboards, not only one regime.
- Priority 2: keep models with repeatable layer behavior across the four layer-profile tables.
- Priority 3: reserve one slot for the clearest NR specialist, because `ResearchREADME.md` shows that NR and FR/overall behavior diverge sharply.

This yields a shortlist dominated by CLIP, DINOv2, and SigLIP as the strongest balanced families, plus one explicit NR-first InternViT candidate.

| Final | Model | Family | Recommended layer | Avg q | Quality Overall | Quality NR | Triplet Overall | Triplet NR | Why keep it |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `clip_vit_l14_336` | CLIP | `hidden_state_020` | 0.979 | `#4`, `q=0.958`, peak `hidden_state_020` | `#2`, `q=0.986`, peak `hidden_state_020` | `#1`, `q=1.000`, peak `hidden_state_020` | `#3`, `q=0.973`, peak `hidden_state_020` | Best fully balanced generalist; same layer wins every board. |
| 2 | `clip_vit_l14` | CLIP | `hidden_state_020` | 0.976 | `#2`, `q=0.986`, peak `hidden_state_020` | `#4`, `q=0.958`, peak `hidden_state_019` | `#3`, `q=0.973`, peak `hidden_state_020` | `#2`, `q=0.986`, peak `hidden_state_019` | Elite on all four boards; layer 19/20 is highly stable. |
| 3 | `dinov2_base` | DINOv2 | `hidden_state_003` | 0.960 | `#1`, `q=1.000`, peak `hidden_state_003` | `#5`, `q=0.944`, peak `hidden_state_003` | `#5`, `q=0.946`, peak `hidden_state_003` | `#5`, `q=0.946`, peak `hidden_state_005` | Strongest DINOv2 all-around and the cleanest general-purpose anchor. |
| 4 | `siglip_so400m` | SigLIP | `hidden_state_012` | 0.929 | `#6`, `q=0.931`, peak `hidden_state_012` | `#10`, `q=0.875`, peak `hidden_state_010` | `#2`, `q=0.986`, peak `hidden_state_012` | `#7`, `q=0.919`, peak `hidden_state_012` | Best SigLIP blend; especially strong for triplet structure. |
| 5 | `dinov2_giant` | DINOv2 | `hidden_state_021` | 0.905 | `#9`, `q=0.889`, peak `hidden_state_030` | `#7`, `q=0.917`, peak `hidden_state_021` | `#8`, `q=0.905`, peak `hidden_state_029` | `#8`, `q=0.905`, peak `hidden_state_020` | High ceiling across all four regimes; good large-scale DINOv2 option. |
| 6 | `dinov2_small` | DINOv2 | `hidden_state_004` | 0.895 | `#7`, `q=0.917`, peak `hidden_state_004` | `#9`, `q=0.889`, peak `hidden_state_003` | `#10`, `q=0.878`, peak `hidden_state_004` | `#9`, `q=0.892`, peak `hidden_state_003` | Compact but still top-10 in every requested leaderboard. |
| 7 | `clip_vit_b16` | CLIP | `hidden_state_009` | 0.884 | `#5`, `q=0.944`, peak `hidden_state_010` | `#12`, `q=0.847`, peak `hidden_state_009` | `#7`, `q=0.919`, peak `hidden_state_010` | `#14`, `q=0.824`, peak `hidden_state_009` | Reliable CLIP fallback with very stable mid-depth layers. |
| 8 | `dinov2_large` | DINOv2 | `hidden_state_018` | 0.870 | `#15`, `q=0.806`, peak `hidden_state_020` | `#6`, `q=0.931`, peak `hidden_state_015` | `#15`, `q=0.811`, peak `hidden_state_020` | `#6`, `q=0.932`, peak `hidden_state_016` | Strong NR-heavy DINOv2 variant; good if NR is important. |
| 9 | `siglip_large` | SigLIP | `hidden_state_012` | 0.856 | `#10`, `q=0.875`, peak `hidden_state_012` | `#11`, `q=0.861`, peak `hidden_state_011` | `#13`, `q=0.838`, peak `hidden_state_012` | `#12`, `q=0.851`, peak `hidden_state_011` | Very even cross-regime profile; no bad board. |
| 10 | `internvit_6b_v25` | InternViT | `hidden_state_043` | 0.811 | `#38`, `q=0.486`, peak `hidden_state_045` | `#1`, `q=1.000`, peak `hidden_state_043` | `#19`, `q=0.757`, peak `hidden_state_045` | `#1`, `q=1.000`, peak `hidden_state_041` | Explicit NR specialist; too strong on both NR boards to ignore for e2e. |

## Practical interpretation

- Best balanced trio to start with: `clip_vit_l14_336`, `clip_vit_l14`, `dinov2_base`.
- Best family-level hedge: keep several DINOv2 scales (`base`, `small`, `large`, `giant`) because their strong layers stay near the front across all regimes.
- Best non-CLIP/non-DINOv2 addition: `siglip_so400m`.
- If you want a pure balanced shortlist with no explicit NR-specialist slot, replace `internvit_6b_v25` with `vit_tiny`.
- If you want an NR-first extension beyond this top-10, the next model to test is `internvit_300m_v25`.
