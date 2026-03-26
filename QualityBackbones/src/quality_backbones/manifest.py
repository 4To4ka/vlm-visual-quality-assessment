from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    size: str
    source: str
    loader: str
    pretrained_id: str | None = None
    timm_name: str | None = None
    regressor_dataset: str | None = None
    modality: str = "image"
    enabled: bool = True
    notes: str = ""


MODEL_SPECS: tuple[ModelSpec, ...] = (
    # ResNet (HF)
    ModelSpec("resnet18", "ResNet", "18", "hf", "hf_auto_image", pretrained_id="microsoft/resnet-18"),
    ModelSpec("resnet34", "ResNet", "34", "hf", "hf_auto_image", pretrained_id="microsoft/resnet-34"),
    ModelSpec("resnet50", "ResNet", "50", "hf", "hf_auto_image", pretrained_id="microsoft/resnet-50"),
    ModelSpec("resnet101", "ResNet", "101", "hf", "hf_auto_image", pretrained_id="microsoft/resnet-101"),
    ModelSpec("resnet152", "ResNet", "152", "hf", "hf_auto_image", pretrained_id="microsoft/resnet-152"),
    # ResNeXt (timm)
    ModelSpec("resnext50_32x4d", "ResNeXt", "50-32x4d", "timm", "timm_cnn_features", timm_name="resnext50_32x4d.a1_in1k"),
    ModelSpec("resnext101_32x8d", "ResNeXt", "101-32x8d", "timm", "timm_cnn_features", timm_name="resnext101_32x8d.fb_swsl_ig1b_ft_in1k"),
    # ConvNeXt (HF)
    ModelSpec("convnext_tiny", "ConvNeXt", "tiny", "hf", "hf_auto_image", pretrained_id="facebook/convnext-tiny-224"),
    ModelSpec("convnext_small", "ConvNeXt", "small", "hf", "hf_auto_image", pretrained_id="facebook/convnext-small-224"),
    ModelSpec("convnext_base", "ConvNeXt", "base", "hf", "hf_auto_image", pretrained_id="facebook/convnext-base-224"),
    ModelSpec("convnext_large", "ConvNeXt", "large", "hf", "hf_auto_image", pretrained_id="facebook/convnext-large-224"),
    # Swin Transformer (HF)
    ModelSpec("swin_tiny", "Swin", "tiny", "hf", "hf_swin_vision", pretrained_id="microsoft/swin-tiny-patch4-window7-224"),
    ModelSpec("swin_small", "Swin", "small", "hf", "hf_swin_vision", pretrained_id="microsoft/swin-small-patch4-window7-224"),
    ModelSpec("swin_base", "Swin", "base", "hf", "hf_swin_vision", pretrained_id="microsoft/swin-base-patch4-window7-224"),
    ModelSpec("swin_large", "Swin", "large", "hf", "hf_swin_vision", pretrained_id="microsoft/swin-large-patch4-window7-224"),
    # Swin Transformer V2 (HF)
    ModelSpec("swinv2_tiny", "SwinV2", "tiny", "hf", "hf_swin_vision", pretrained_id="microsoft/swinv2-tiny-patch4-window8-256"),
    ModelSpec("swinv2_small", "SwinV2", "small", "hf", "hf_swin_vision", pretrained_id="microsoft/swinv2-small-patch4-window8-256"),
    ModelSpec("swinv2_base", "SwinV2", "base", "hf", "hf_swin_vision", pretrained_id="microsoft/swinv2-base-patch4-window8-256"),
    ModelSpec("swinv2_large", "SwinV2", "large", "hf", "hf_swin_vision", pretrained_id="microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"),
    # FastViT (timm)
    ModelSpec("fastvit_t8", "FastViT", "t8", "timm", "timm_cnn_features", timm_name="fastvit_t8.apple_in1k"),
    ModelSpec("fastvit_t12", "FastViT", "t12", "timm", "timm_cnn_features", timm_name="fastvit_t12.apple_in1k"),
    ModelSpec("fastvit_s12", "FastViT", "s12", "timm", "timm_cnn_features", timm_name="fastvit_s12.apple_in1k"),
    ModelSpec("fastvit_sa12", "FastViT", "sa12", "timm", "timm_cnn_features", timm_name="fastvit_sa12.apple_in1k"),
    ModelSpec("fastvit_sa24", "FastViT", "sa24", "timm", "timm_cnn_features", timm_name="fastvit_sa24.apple_in1k"),
    ModelSpec("fastvit_sa36", "FastViT", "sa36", "timm", "timm_cnn_features", timm_name="fastvit_sa36.apple_in1k"),
    ModelSpec("fastvit_ma36", "FastViT", "ma36", "timm", "timm_cnn_features", timm_name="fastvit_ma36.apple_in1k"),
    # VGG (timm)
    ModelSpec("vgg11", "VGG", "11", "timm", "timm_cnn_features", timm_name="vgg11.tv_in1k"),
    ModelSpec("vgg11_bn", "VGG", "11-bn", "timm", "timm_cnn_features", timm_name="vgg11_bn.tv_in1k"),
    ModelSpec("vgg13", "VGG", "13", "timm", "timm_cnn_features", timm_name="vgg13.tv_in1k"),
    ModelSpec("vgg13_bn", "VGG", "13-bn", "timm", "timm_cnn_features", timm_name="vgg13_bn.tv_in1k"),
    ModelSpec("vgg16", "VGG", "16", "timm", "timm_cnn_features", timm_name="vgg16.tv_in1k"),
    ModelSpec("vgg16_bn", "VGG", "16-bn", "timm", "timm_cnn_features", timm_name="vgg16_bn.tv_in1k"),
    ModelSpec("vgg19", "VGG", "19", "timm", "timm_cnn_features", timm_name="vgg19.tv_in1k"),
    ModelSpec("vgg19_bn", "VGG", "19-bn", "timm", "timm_cnn_features", timm_name="vgg19_bn.tv_in1k"),
    # ViT (HF + timm)
    ModelSpec("vit_tiny", "ViT", "tiny", "timm", "timm_vit_blocks", timm_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k"),
    ModelSpec("vit_small", "ViT", "small", "timm", "timm_vit_blocks", timm_name="vit_small_patch16_224.augreg_in21k_ft_in1k"),
    ModelSpec("vit_base", "ViT", "base", "hf", "hf_auto_image", pretrained_id="google/vit-base-patch16-224"),
    ModelSpec("vit_large", "ViT", "large", "hf", "hf_auto_image", pretrained_id="google/vit-large-patch16-224"),
    ModelSpec("vit_huge", "ViT", "huge", "hf", "hf_auto_image", pretrained_id="google/vit-huge-patch14-224-in21k"),
    # MAE (HF)
    ModelSpec("vit_mae_base", "MAE", "base", "hf", "hf_auto_image", pretrained_id="facebook/vit-mae-base"),
    ModelSpec("vit_mae_large", "MAE", "large", "hf", "hf_auto_image", pretrained_id="facebook/vit-mae-large"),
    ModelSpec("vit_mae_huge", "MAE", "huge", "hf", "hf_auto_image", pretrained_id="facebook/vit-mae-huge"),
    # CLIP vision (HF)
    ModelSpec("clip_vit_b32", "CLIP", "base-patch32", "hf", "hf_clip_vision", pretrained_id="openai/clip-vit-base-patch32"),
    ModelSpec("clip_vit_b16", "CLIP", "base-patch16", "hf", "hf_clip_vision", pretrained_id="openai/clip-vit-base-patch16"),
    ModelSpec("clip_vit_l14", "CLIP", "large-patch14", "hf", "hf_clip_vision", pretrained_id="openai/clip-vit-large-patch14"),
    ModelSpec("clip_vit_l14_336", "CLIP", "large-patch14-336", "hf", "hf_clip_vision", pretrained_id="openai/clip-vit-large-patch14-336"),
    # DINO v1 (HF)
    ModelSpec("dino_vits8", "DINO", "vits8", "hf", "hf_auto_image", pretrained_id="facebook/dino-vits8"),
    ModelSpec("dino_vits16", "DINO", "vits16", "hf", "hf_auto_image", pretrained_id="facebook/dino-vits16"),
    ModelSpec("dino_vitb8", "DINO", "vitb8", "hf", "hf_auto_image", pretrained_id="facebook/dino-vitb8"),
    ModelSpec("dino_vitb16", "DINO", "vitb16", "hf", "hf_auto_image", pretrained_id="facebook/dino-vitb16"),
    # DINOv2 (HF)
    ModelSpec("dinov2_small", "DINOv2", "small", "hf", "hf_auto_image", pretrained_id="facebook/dinov2-small"),
    ModelSpec("dinov2_base", "DINOv2", "base", "hf", "hf_auto_image", pretrained_id="facebook/dinov2-base"),
    ModelSpec("dinov2_large", "DINOv2", "large", "hf", "hf_auto_image", pretrained_id="facebook/dinov2-large"),
    ModelSpec("dinov2_giant", "DINOv2", "giant", "hf", "hf_auto_image", pretrained_id="facebook/dinov2-giant"),
    # DINOv3 (timm)
    ModelSpec("dinov3_vits16", "DINOv3", "vits16", "timm", "timm_vit_blocks", timm_name="vit_small_patch16_dinov3.lvd1689m"),
    ModelSpec("dinov3_vits16plus", "DINOv3", "vits16plus", "timm", "timm_vit_blocks", timm_name="vit_small_plus_patch16_dinov3.lvd1689m"),
    ModelSpec("dinov3_vitb16", "DINOv3", "vitb16", "timm", "timm_vit_blocks", timm_name="vit_base_patch16_dinov3.lvd1689m"),
    ModelSpec("dinov3_vitl16", "DINOv3", "vitl16", "timm", "timm_vit_blocks", timm_name="vit_large_patch16_dinov3.lvd1689m"),
    ModelSpec("dinov3_vith16plus", "DINOv3", "vith16plus", "timm", "timm_vit_blocks", timm_name="vit_huge_plus_patch16_dinov3.lvd1689m"),
    ModelSpec("dinov3_vit7b16", "DINOv3", "vit7b16", "timm", "timm_vit_blocks", timm_name="vit_7b_patch16_dinov3.lvd1689m"),
    # SigLIP (HF)
    ModelSpec("siglip_base", "SigLIP", "base", "hf", "hf_siglip_vision", pretrained_id="google/siglip-base-patch16-224"),
    ModelSpec("siglip_large", "SigLIP", "large", "hf", "hf_siglip_vision", pretrained_id="google/siglip-large-patch16-256"),
    ModelSpec("siglip_so400m", "SigLIP", "so400m", "hf", "hf_siglip_vision", pretrained_id="google/siglip-so400m-patch14-384"),
    # SigLIP2 (HF)
    ModelSpec("siglip2_base", "SigLIP2", "base", "hf", "hf_siglip2_vision", pretrained_id="google/siglip2-base-patch16-224"),
    ModelSpec("siglip2_large", "SigLIP2", "large", "hf", "hf_siglip2_vision", pretrained_id="google/siglip2-large-patch16-384"),
    ModelSpec("siglip2_so400m", "SigLIP2", "so400m", "hf", "hf_siglip2_vision", pretrained_id="google/siglip2-so400m-patch14-384"),
    ModelSpec("siglip2_base_naflex", "SigLIP2", "base-naflex", "hf", "hf_siglip2_vision", pretrained_id="google/siglip2-base-patch16-naflex"),
    ModelSpec("siglip2_so400m_naflex", "SigLIP2", "so400m-naflex", "hf", "hf_siglip2_vision", pretrained_id="google/siglip2-so400m-patch16-naflex"),
    # I-JEPA (HF)
    ModelSpec("ijepa_vith14", "I-JEPA", "vith14", "hf", "hf_auto_image", pretrained_id="facebook/ijepa_vith14_1k"),
    # InternViT (HF remote code)
    ModelSpec("internvit_300m", "InternViT", "300M", "hf", "hf_auto_image_remote", pretrained_id="OpenGVLab/InternViT-300M-448px", notes="Local patched remote-code load without flash-attn"),
    ModelSpec("internvit_300m_v25", "InternViT", "300M-V2_5", "hf", "hf_auto_image_remote", pretrained_id="OpenGVLab/InternViT-300M-448px-V2_5", notes="Local patched remote-code load without flash-attn"),
    ModelSpec("internvit_6b_v25", "InternViT", "6B-V2_5", "hf", "hf_auto_image_remote", pretrained_id="OpenGVLab/InternViT-6B-448px-V2_5", notes="Local patched remote-code load without flash-attn"),
    # FastViTHD (apple/ml-fastvlm repo architecture + checkpoints)
    ModelSpec(
        "fastvithd_05b",
        "FastViTHD",
        "0.5B",
        "github",
        "fastvlm_fastvithd",
        pretrained_id="apple/FastVLM-0.5B",
        notes="Loads FastViTHD vision tower from apple/ml-fastvlm with per-block embeddings",
    ),
    ModelSpec(
        "fastvithd_15b",
        "FastViTHD",
        "1.5B",
        "github",
        "fastvlm_fastvithd",
        pretrained_id="apple/FastVLM-1.5B",
        notes="Loads FastViTHD vision tower from apple/ml-fastvlm with per-block embeddings",
    ),
    ModelSpec(
        "fastvithd_7b",
        "FastViTHD",
        "7B",
        "github",
        "fastvlm_fastvithd",
        pretrained_id="apple/FastVLM-7B",
        notes="Loads FastViTHD vision tower from apple/ml-fastvlm with per-block embeddings",
    ),
    # ARNIQA torch.hub variants
    ModelSpec("arniqa_live", "ARNIQA", "regressor-live", "github", "arniqa_torchhub", regressor_dataset="live"),
    ModelSpec("arniqa_csiq", "ARNIQA", "regressor-csiq", "github", "arniqa_torchhub", regressor_dataset="csiq"),
    ModelSpec("arniqa_tid2013", "ARNIQA", "regressor-tid2013", "github", "arniqa_torchhub", regressor_dataset="tid2013"),
    ModelSpec("arniqa_kadid10k", "ARNIQA", "regressor-kadid10k", "github", "arniqa_torchhub", regressor_dataset="kadid10k"),
    ModelSpec("arniqa_flive", "ARNIQA", "regressor-flive", "github", "arniqa_torchhub", regressor_dataset="flive"),
    ModelSpec("arniqa_spaq", "ARNIQA", "regressor-spaq", "github", "arniqa_torchhub", regressor_dataset="spaq"),
    ModelSpec("arniqa_clive", "ARNIQA", "regressor-clive", "github", "arniqa_torchhub", regressor_dataset="clive"),
    ModelSpec("arniqa_koniq10k", "ARNIQA", "regressor-koniq10k", "github", "arniqa_torchhub", regressor_dataset="koniq10k"),
    # MANIQA (GitHub, checkpoint-driven)
    ModelSpec("maniqa_pipal22", "MANIQA", "PIPAL22", "github", "maniqa_repo", enabled=True, notes="Auto-clone MANIQA repo + auto-download checkpoint"),
    ModelSpec("maniqa_kadid10k", "MANIQA", "KADID10K", "github", "maniqa_repo", enabled=True, notes="Auto-clone MANIQA repo + auto-download checkpoint"),
    ModelSpec("maniqa_koniq10k", "MANIQA", "KONIQ10K", "github", "maniqa_repo", enabled=True, notes="Auto-clone MANIQA repo + auto-download checkpoint"),
    # Explicitly skipped (video-only)
    ModelSpec("vjepa_skipped", "V-JEPA", "n/a", "github", "skipped", modality="video", enabled=False, notes="Skipped: video-only"),
    ModelSpec("vjepa2_skipped", "V-JEPA2", "n/a", "hf", "skipped", modality="video", enabled=False, notes="Skipped: video-only"),
    ModelSpec("dover_skipped", "DOVER", "n/a", "github", "skipped", modality="video", enabled=False, notes="Skipped: video-only"),
)


def iter_enabled_image_model_specs() -> Iterable[ModelSpec]:
    for spec in MODEL_SPECS:
        if spec.enabled and spec.modality == "image":
            yield spec


def get_model_spec(key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown model key: {key}")
