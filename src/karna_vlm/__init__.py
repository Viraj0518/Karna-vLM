"""
Karna VLM — A compact, customizable vision-language model platform.

The intelligence lives in three places:
1. A strong frozen vision encoder (perception substrate)
2. A compact learned multimodal bridge (the product moat)
3. A smaller but capable decoder LLM (the language interface)
"""

__version__ = "0.1.0"

from karna_vlm.models.vlm_model import KarnaVLM, KarnaVLMConfig

__all__ = ["KarnaVLM", "KarnaVLMConfig", "__version__"]
