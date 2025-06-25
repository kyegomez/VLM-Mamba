"""
VLM-Mamba: The First State Space Model-Based Vision-Language Model

A novel Vision-Language Model built entirely on State Space Models (SSMs),
eliminating the need for attention mechanisms while maintaining competitive performance.

Author: Kye Gomez
License: MIT
"""

from vlm_mamba.model import (
    VLMamba,
    VLMambaConfig,
    VisionConfig,
    LanguageConfig,
    BridgerConfig,
    LoRAConfig,
    VisionMamba,
    FuyuImageProcessor,
    LanguageMamba,
    BridgerFFN,
    LoRALinear,
)

__version__ = "0.1.0"
__author__ = "Kye Gomez"
__email__ = "kye@apac.ai"

__all__ = [
    "VLMamba",
    "VLMambaConfig",
    "VisionConfig",
    "LanguageConfig",
    "BridgerConfig",
    "LoRAConfig",
    "VisionMamba",
    "FuyuImageProcessor",
    "LanguageMamba",
    "BridgerFFN",
    "LoRALinear",
]
