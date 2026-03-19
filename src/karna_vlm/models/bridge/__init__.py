"""
Multimodal bridge modules — the core proprietary moat of Karna VLM.

The bridge compresses and transforms vision encoder patch features into
a compact sequence of tokens that the decoder LLM can consume.

Bridge variants:
- LinearProjector: Simple linear projection (baseline)
- QFormerLiteBridge: Learned query-based bridge (default product path)
- ResamplerBridge: Perceiver-resampler style bridge
- GatedBridge: Gated linear projection with learnable gates
- InstructionConditionedBridge: Bridge conditioned on text instructions
"""

from karna_vlm.models.bridge.bridge_interface import BridgeInterface, BridgeOutput
from karna_vlm.models.bridge.linear_projector import LinearProjector
from karna_vlm.models.bridge.qformer_lite import QFormerLiteBridge
from karna_vlm.models.bridge.resampler import ResamplerBridge
from karna_vlm.models.bridge.gated_bridge import GatedBridge
from karna_vlm.models.bridge.instruction_conditioned import InstructionConditionedBridge

__all__ = [
    "BridgeInterface",
    "BridgeOutput",
    "LinearProjector",
    "QFormerLiteBridge",
    "ResamplerBridge",
    "GatedBridge",
    "InstructionConditionedBridge",
]
