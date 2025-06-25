#
# A production-grade, enhanced implementation of a Vision-Language Model using Mamba.
#
# To run this code, you'll need to install the required packages:
#
# pip install torch torchvision loguru einops pydantic
# pip install mamba-ssm
#
# This version includes:
#   - Optional LoRA (Low-Rank Adaptation) for efficient fine-tuning.
#   - An alternative "Fuyu-style" image processor that bypasses a deep vision encoder.
#   - Comprehensive type hinting for clarity and robustness.
#   - Detailed docstrings for all modules and methods.
#   - Loguru for structured and informative logging.
#   - Pydantic configuration classes for type safety and validation.
#

from __future__ import annotations
import math
from typing import Optional, Literal

import torch
import torch.nn as nn
from loguru import logger
from einops import rearrange
from pydantic import BaseModel, Field
from pydantic.v1 import validator

# Attempt to import Mamba, providing a clear error message if it's not installed.
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None
    logger.warning(
        "Mamba package not found. Please install with 'pip install mamba-ssm'."
    )

# --- Pydantic Configuration Classes ---


class VisionConfig(BaseModel):
    """Configuration for the vision processing component."""

    img_size: int = Field(
        default=224,
        description="Size of input images (height and width)",
    )
    patch_size: int = Field(
        default=16, description="Size of each square patch"
    )
    in_chans: int = Field(
        default=3, description="Number of input image channels"
    )
    d_model: int = Field(
        default=768, description="Model dimensionality"
    )
    n_layers: int = Field(
        default=12,
        description="Number of layers in the vision encoder",
    )

    @validator("img_size")
    def validate_img_size(cls, v):
        if v <= 0 or v % 16 != 0:
            raise ValueError(
                "img_size must be positive and divisible by 16"
            )
        return v

    @validator("patch_size")
    def validate_patch_size(cls, v):
        if v <= 0 or v > 32:
            raise ValueError("patch_size must be positive and <= 32")
        return v

    @validator("d_model")
    def validate_d_model(cls, v):
        if v <= 0 or v % 64 != 0:
            raise ValueError(
                "d_model must be positive and divisible by 64"
            )
        return v


class LanguageConfig(BaseModel):
    """Configuration for the language model component."""

    vocab_size: int = Field(
        default=50257, description="Size of the vocabulary"
    )
    d_model: int = Field(
        default=768, description="Model dimensionality"
    )
    n_layers: int = Field(
        default=12,
        description="Number of layers in the language model",
    )

    @validator("vocab_size")
    def validate_vocab_size(cls, v):
        if v <= 0:
            raise ValueError("vocab_size must be positive")
        return v

    @validator("d_model")
    def validate_d_model(cls, v):
        if v <= 0 or v % 64 != 0:
            raise ValueError(
                "d_model must be positive and divisible by 64"
            )
        return v


class BridgerConfig(BaseModel):
    """Configuration for the vision-language bridge component."""

    d_model: int = Field(
        default=768, description="Model dimensionality"
    )
    expansion_factor: int = Field(
        default=4, description="Expansion factor for the FFN"
    )

    @validator("d_model")
    def validate_d_model(cls, v):
        if v <= 0 or v % 64 != 0:
            raise ValueError(
                "d_model must be positive and divisible by 64"
            )
        return v

    @validator("expansion_factor")
    def validate_expansion_factor(cls, v):
        if v <= 0 or v > 8:
            raise ValueError(
                "expansion_factor must be positive and <= 8"
            )
        return v


class LoRAConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation)."""

    rank: int = Field(
        description="Rank of the low-rank adaptation matrices"
    )
    alpha: float = Field(
        description="Scaling factor for the LoRA update"
    )

    @validator("rank")
    def validate_rank(cls, v):
        if v <= 0 or v > 256:
            raise ValueError("rank must be positive and <= 256")
        return v

    @validator("alpha")
    def validate_alpha(cls, v):
        if v <= 0:
            raise ValueError("alpha must be positive")
        return v


class VLMambaConfig(BaseModel):
    """Complete configuration for the VLMamba model."""

    vision: VisionConfig = Field(
        description="Vision processing configuration"
    )
    language: LanguageConfig = Field(
        description="Language model configuration"
    )
    bridger: BridgerConfig = Field(
        description="Bridge component configuration"
    )
    vision_mode: Literal["encoder", "fuyu"] = Field(
        default="encoder",
        description="Vision processing mode: 'encoder' for VisionMamba, 'fuyu' for lightweight processor",
    )
    lora: Optional[LoRAConfig] = Field(
        default=None,
        description="LoRA configuration for efficient fine-tuning",
    )


# --- LoRA Implementation ---


class LoRALinear(nn.Module):
    """
    A Low-Rank Adaptation (LoRA) layer that can be used to replace a standard nn.Linear layer.
    This allows for efficient fine-tuning by only training the low-rank matrices.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int,
        alpha: float,
    ):
        """
        Args:
            linear_layer (nn.Linear): The original linear layer to be adapted.
            rank (int): The rank of the low-rank adaptation matrices.
            alpha (float): The scaling factor for the LoRA update.
        """
        super().__init__()
        self.linear = linear_layer

        self.lora_down = nn.Linear(
            self.linear.in_features, rank, bias=False
        )
        self.lora_up = nn.Linear(
            rank, self.linear.out_features, bias=False
        )
        self.scaling = alpha / rank

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(
            self.lora_down.weight, a=math.sqrt(5)
        )
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combines the output of the original layer with the LoRA update.
        """
        original_output = self.linear(x)
        lora_update = self.lora_up(self.lora_down(x)) * self.scaling
        return original_output + lora_update


# --- Vision Processing Modules ---


class FuyuImageProcessor(nn.Module):
    """
    A Fuyu-style image processor.

    This processor avoids a deep vision encoder. Instead, it patchifies the image
    and applies a single linear projection, treating image patches as a sequence
    of tokens. This is a lighter-weight approach to vision processing.
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        img_size: int,
        in_chans: int = 3,
    ):
        """
        Args:
            d_model (int): The dimensionality of the model's embeddings.
            patch_size (int): The size of each square patch.
            img_size (int): The size (height and width) of the input image.
            in_chans (int): The number of image channels.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_chans * patch_size * patch_size

        self.projection = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )
        logger.info("Initialized Fuyu-style Image Processor.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes an image into a sequence of embeddings.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: A sequence of visual embeddings of shape [B, num_patches, d_model].
        """
        p = self.patch_size
        # Use einops to rearrange the image into a sequence of patches
        patches = rearrange(
            x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p
        )

        projected_patches = self.projection(patches)
        projected_patches += self.pos_embed
        return projected_patches


class VisionMamba(nn.Module):
    """
    The Vision Encoder component of the VLM, using Mamba.
    Processes an image by patchifying it and running it through Mamba blocks.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        d_model: int = 768,
        n_layers: int = 12,
    ):
        super().__init__()
        if Mamba is None:
            raise RuntimeError("Mamba SSM package is required.")

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_chans,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        logger.info(
            f"Initialized VisionMamba Encoder with {n_layers} layers."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if not (H == self.img_size and W == self.img_size):
            raise ValueError(
                f"Input image size ({H}x{W}) doesn't match model's expected size ({self.img_size}x{self.img_size})."
            )

        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# --- Core Language and Bridging Modules ---


class BridgerFFN(nn.Module):
    """
    A Feed-Forward Network to bridge the vision and language domains.
    """

    def __init__(self, d_model: int = 768, expansion_factor: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.GELU(),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LanguageMamba(nn.Module):
    """
    The Language Model component, using Mamba.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layers: int = 12,
    ):
        super().__init__()
        if Mamba is None:
            raise RuntimeError("Mamba SSM package is required.")

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        logger.info(
            f"Initialized LanguageMamba with {n_layers} layers."
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) * (self.d_model**0.5)
        return x


# --- Main VLM Class ---


class VLMamba(nn.Module):
    """
    The complete Vision-Language Mamba model, enhanced with modular vision processing
    and optional LoRA for efficient fine-tuning.
    """

    def __init__(
        self,
        config: VLMambaConfig,
    ):
        """
        Args:
            config (VLMambaConfig): Complete configuration for the VLMamba model.
        """
        super().__init__()

        if config.vision_mode == "encoder":
            self.vision_processor = VisionMamba(
                **config.vision.dict()
            )
        elif config.vision_mode == "fuyu":
            # Fuyu mode needs d_model, patch_size, img_size from vision_config
            fuyu_cfg = {
                k: v
                for k, v in config.vision.dict().items()
                if k
                in ["d_model", "patch_size", "img_size", "in_chans"]
            }
            self.vision_processor = FuyuImageProcessor(**fuyu_cfg)
        else:
            raise ValueError(
                f"Unknown vision_mode: {config.vision_mode}"
            )

        self.bridger = BridgerFFN(**config.bridger.dict())
        self.language_model = LanguageMamba(**config.language.dict())

        if config.lora:
            self.add_lora(config.lora)
            self.freeze_non_lora_params()

    def add_lora(self, lora_config: LoRAConfig):
        """
        Recursively replaces targeted nn.Linear layers with LoRALinear layers.
        """
        rank = lora_config.rank
        alpha = lora_config.alpha

        for name, module in self.named_modules():
            # Target the 'in_proj' and 'x_proj' layers within Mamba blocks
            if isinstance(module, Mamba):
                logger.debug(f"Applying LoRA to Mamba block: {name}")
                module.in_proj = LoRALinear(
                    module.in_proj, rank, alpha
                )

        logger.success(
            f"Successfully applied LoRA with rank={rank} and alpha={alpha}."
        )

    def freeze_non_lora_params(self):
        """
        Freezes all parameters in the model except for the LoRA parameters.
        """
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        logger.info(
            "Froze all non-LoRA parameters for efficient fine-tuning."
        )

    def forward(
        self, image: torch.Tensor, text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        The main forward pass for the VLM.
        """
        vision_features = self.vision_processor(image)
        bridged_vision_features = self.bridger(vision_features)
        text_features = self.language_model.embedding(text_tokens)

        combined_features = torch.cat(
            [bridged_vision_features, text_features], dim=1
        )

        x = combined_features
        for layer in self.language_model.layers:
            x = layer(x)
        x = self.language_model.norm(x)

        return self.language_model.lm_head(x)


# # --- Example Usage ---
# if __name__ == "__main__":
#     if Mamba is None:
#         logger.error("Cannot run example because Mamba SSM is not installed.")
#     else:
#         # --- Base Configuration ---
#         model_dim = 256  # Smaller dim for quick example
#         n_layers = 4  # Fewer layers for speed

#         # Create configurations using Pydantic classes
#         vision_cfg = VisionConfig(
#             img_size=224,
#             patch_size=16,
#             in_chans=3,
#             d_model=model_dim,
#             n_layers=n_layers,
#         )
#         language_cfg = LanguageConfig(
#             vocab_size=10000,
#             d_model=model_dim,
#             n_layers=n_layers,
#         )
#         bridger_cfg = BridgerConfig(
#             d_model=model_dim,
#             expansion_factor=4
#         )

#         # --- Dummy Data ---
#         batch_size = 2
#         dummy_image = torch.randn(
#             batch_size, 3, vision_cfg.img_size, vision_cfg.img_size
#         )
#         dummy_text = torch.randint(0, language_cfg.vocab_size, (batch_size, 64))

#         # --- Example 1: Standard VLMamba with Encoder ---
#         logger.info("-" * 50)
#         logger.info("Example 1: Standard VLMamba with VisionMamba encoder")
#         config_encoder = VLMambaConfig(
#             vision=vision_cfg,
#             language=language_cfg,
#             bridger=bridger_cfg,
#             vision_mode="encoder",
#         )
#         model_encoder = VLMamba(config_encoder)
#         with torch.no_grad():
#             output_logits = model_encoder(dummy_image, dummy_text)
#         logger.success(f"Encoder model output shape: {output_logits.shape}")

#         # --- Example 2: VLMamba with Fuyu-style Processor ---
#         logger.info("-" * 50)
#         logger.info("Example 2: VLMamba with Fuyu-style image processor")
#         config_fuyu = VLMambaConfig(
#             vision=vision_cfg,
#             language=language_cfg,
#             bridger=bridger_cfg,
#             vision_mode="fuyu",
#         )
#         model_fuyu = VLMamba(config_fuyu)
#         with torch.no_grad():
#             output_logits_fuyu = model_fuyu(dummy_image, dummy_text)
#         logger.success(f"Fuyu-style model output shape: {output_logits_fuyu.shape}")

#         # --- Example 3: VLMamba with LoRA enabled ---
#         logger.info("-" * 50)
#         logger.info("Example 3: VLMamba with LoRA for efficient fine-tuning")
#         lora_cfg = LoRAConfig(rank=8, alpha=16)
#         config_lora = VLMambaConfig(
#             vision=vision_cfg,
#             language=language_cfg,
#             bridger=bridger_cfg,
#             vision_mode="encoder",
#             lora=lora_cfg,
#         )
#         model_lora = VLMamba(config_lora)

#         total_params = sum(p.numel() for p in model_lora.parameters())
#         trainable_params = sum(
#             p.numel() for p in model_lora.parameters() if p.requires_grad
#         )

#         logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
#         logger.info(f"Trainable (LoRA) parameters: {trainable_params / 1e3:.2f}K")
#         logger.info(
#             f"Trainable percentage: {100 * trainable_params / total_params:.4f}%"
#         )

#         with torch.no_grad():
#             output_logits_lora = model_lora(dummy_image, dummy_text)
#         logger.success(f"LoRA model output shape: {output_logits_lora.shape}")
