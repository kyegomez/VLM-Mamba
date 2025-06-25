import torch
from vlm_mamba import (
    VLMamba,
    VLMambaConfig,
    LoRAConfig,
    LanguageConfig,
    BridgerConfig,
    VisionConfig,
)
from loguru import logger

# --- Example Usage ---
if __name__ == "__main__":
    # --- Base Configuration ---
    model_dim = 256  # Smaller dim for quick example
    n_layers = 4  # Fewer layers for speed

    # Create configurations using Pydantic classes
    vision_cfg = VisionConfig(
        img_size=224,
        patch_size=16,
        in_chans=3,
        d_model=model_dim,
        n_layers=n_layers,
    )
    language_cfg = LanguageConfig(
        vocab_size=10000,
        d_model=model_dim,
        n_layers=n_layers,
    )
    bridger_cfg = BridgerConfig(d_model=model_dim, expansion_factor=4)

    # --- Dummy Data ---
    batch_size = 2
    dummy_image = torch.randn(
        batch_size, 3, vision_cfg.img_size, vision_cfg.img_size
    )
    dummy_text = torch.randint(
        0, language_cfg.vocab_size, (batch_size, 64)
    )

    # --- Example 1: Standard VLMamba with Encoder ---
    logger.info("-" * 50)
    logger.info(
        "Example 1: Standard VLMamba with VisionMamba encoder"
    )
    config_encoder = VLMambaConfig(
        vision=vision_cfg,
        language=language_cfg,
        bridger=bridger_cfg,
        vision_mode="encoder",
    )
    model_encoder = VLMamba(config_encoder)
    with torch.no_grad():
        output_logits = model_encoder(dummy_image, dummy_text)
    logger.success(
        f"Encoder model output shape: {output_logits.shape}"
    )

    # --- Example 2: VLMamba with Fuyu-style Processor ---
    logger.info("-" * 50)
    logger.info("Example 2: VLMamba with Fuyu-style image processor")
    config_fuyu = VLMambaConfig(
        vision=vision_cfg,
        language=language_cfg,
        bridger=bridger_cfg,
        vision_mode="fuyu",
    )
    model_fuyu = VLMamba(config_fuyu)
    with torch.no_grad():
        output_logits_fuyu = model_fuyu(dummy_image, dummy_text)
    logger.success(
        f"Fuyu-style model output shape: {output_logits_fuyu.shape}"
    )

    # --- Example 3: VLMamba with LoRA enabled ---
    logger.info("-" * 50)
    logger.info(
        "Example 3: VLMamba with LoRA for efficient fine-tuning"
    )
    lora_cfg = LoRAConfig(rank=8, alpha=16)
    config_lora = VLMambaConfig(
        vision=vision_cfg,
        language=language_cfg,
        bridger=bridger_cfg,
        vision_mode="encoder",
        lora=lora_cfg,
    )
    model_lora = VLMamba(config_lora)

    total_params = sum(p.numel() for p in model_lora.parameters())
    trainable_params = sum(
        p.numel() for p in model_lora.parameters() if p.requires_grad
    )

    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(
        f"Trainable (LoRA) parameters: {trainable_params / 1e3:.2f}K"
    )
    logger.info(
        f"Trainable percentage: {100 * trainable_params / total_params:.4f}%"
    )

    with torch.no_grad():
        output_logits_lora = model_lora(dummy_image, dummy_text)
    logger.success(
        f"LoRA model output shape: {output_logits_lora.shape}"
    )
