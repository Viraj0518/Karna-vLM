"""
KarnaVLM — the unified vision-language model.

This is the top-level model class that orchestrates:
1. Vision encoding (frozen backbone)
2. Multimodal bridging (the trainable moat)
3. Language decoding (compact LLM with optional LoRA)
4. Prompt packing (multimodal input assembly)

It exposes the premium API surface:
- KarnaVLM.from_config() / from_pretrained()
- model.generate()
- model.encode_image()
- model.bridge_image()
- model.attach_adapter()
- model.load_domain_pack()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import yaml
from PIL import Image

from karna_vlm.models.vision.encoder_interface import VisionEncoderInterface, VisionEncoderOutput
from karna_vlm.models.bridge.bridge_interface import BridgeInterface, BridgeOutput
from karna_vlm.models.decoder.decoder_interface import DecoderInterface, DecoderOutput
from karna_vlm.models.prompt_packing.packer import PromptPacker, PackedSequence

logger = logging.getLogger(__name__)


# ── Bridge registry ───────────────────────────────────────────────
BRIDGE_REGISTRY: dict[str, type] = {}


def _register_bridges() -> None:
    """Lazily populate the bridge registry."""
    if BRIDGE_REGISTRY:
        return
    from karna_vlm.models.bridge.linear_projector import LinearProjector
    from karna_vlm.models.bridge.qformer_lite import QFormerLiteBridge
    from karna_vlm.models.bridge.resampler import ResamplerBridge
    from karna_vlm.models.bridge.gated_bridge import GatedBridge
    from karna_vlm.models.bridge.instruction_conditioned import InstructionConditionedBridge

    BRIDGE_REGISTRY.update({
        "linear": LinearProjector,
        "qformer_lite": QFormerLiteBridge,
        "resampler": ResamplerBridge,
        "gated": GatedBridge,
        "instruction_conditioned": InstructionConditionedBridge,
    })


# ── Vision encoder registry ──────────────────────────────────────
VISION_REGISTRY: dict[str, type] = {}


def _register_vision() -> None:
    if VISION_REGISTRY:
        return
    from karna_vlm.models.vision.siglip_encoder import SigLIPEncoder
    from karna_vlm.models.vision.clip_encoder import CLIPEncoder

    VISION_REGISTRY.update({
        "siglip": SigLIPEncoder,
        "clip": CLIPEncoder,
    })


@dataclass
class KarnaVLMConfig:
    """Configuration for KarnaVLM.

    All fields have sensible defaults. Override via YAML config files
    or direct construction.
    """

    # Model identity
    model_name: str = "karna-vlm-small"
    model_family: str = "small"  # tiny | small | mid

    # Vision encoder
    vision_backend: str = "siglip"
    vision_model: str = "google/siglip-base-patch16-224"
    vision_freeze: bool = True
    vision_select_layer: int = -1

    # Bridge
    bridge_type: str = "qformer_lite"  # linear | qformer_lite | resampler | gated | instruction_conditioned
    bridge_dim: int = 512
    bridge_num_queries: int = 64
    bridge_num_layers: int = 4
    bridge_num_heads: int = 8
    bridge_ffn_ratio: float = 4.0
    bridge_dropout: float = 0.1

    # Decoder (US-origin, open weights + architecture)
    decoder_model: str = "google/gemma-2-2b"
    decoder_freeze: bool = True
    decoder_dtype: str = "bfloat16"

    # Prompt packing
    max_length: int = 2048
    image_token: str = "<image>"

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50000

    # LoRA
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Extra
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "KarnaVLMConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save config to a YAML file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


class KarnaVLM(nn.Module):
    """Karna Vision-Language Model.

    The unified model orchestrating vision encoder, bridge, and decoder
    into a single forward pass. This is the primary API surface.

    Usage:
        config = KarnaVLMConfig(model_family="tiny")
        model = KarnaVLM(config)

        # Encode and generate
        output = model.generate(images=[pil_image], prompt="Describe this image.")
    """

    def __init__(self, config: KarnaVLMConfig) -> None:
        super().__init__()
        self.config = config
        _register_bridges()
        _register_vision()

        # ── Build components ──────────────────────────────────────
        logger.info("Building KarnaVLM: %s", config.model_name)

        # 1. Vision encoder
        self.vision_encoder: VisionEncoderInterface = self._build_vision_encoder()
        vision_dim = self.vision_encoder.get_output_dim()

        # 2. Decoder (build before bridge so we know decoder_dim)
        self.decoder: DecoderInterface = self._build_decoder()
        decoder_dim = self.decoder.get_hidden_dim()

        # 3. Bridge
        self.bridge: BridgeInterface = self._build_bridge(vision_dim, decoder_dim)

        # 4. Prompt packer
        self.packer = PromptPacker(
            tokenizer=self.decoder.tokenizer,
            embed_fn=self.decoder.embed_tokens,
            image_token=config.image_token,
            max_length=config.max_length,
        )

        # 5. Optional LoRA
        if config.use_lora:
            self._apply_lora()

        # Log parameter counts
        self._log_param_counts()

    # ── Builders ──────────────────────────────────────────────────

    def _build_vision_encoder(self) -> VisionEncoderInterface:
        """Instantiate the vision encoder from config."""
        cls = VISION_REGISTRY.get(self.config.vision_backend)
        if cls is None:
            raise ValueError(
                f"Unknown vision backend '{self.config.vision_backend}'. "
                f"Available: {list(VISION_REGISTRY.keys())}"
            )
        return cls(
            model_name=self.config.vision_model,
            freeze=self.config.vision_freeze,
            select_layer=self.config.vision_select_layer,
        )

    def _build_decoder(self) -> DecoderInterface:
        """Instantiate the decoder LLM from config."""
        from karna_vlm.models.decoder.hf_decoder import HFDecoder

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return HFDecoder(
            model_name=self.config.decoder_model,
            freeze=self.config.decoder_freeze,
            torch_dtype=dtype_map.get(self.config.decoder_dtype, torch.bfloat16),
        )

    def _build_bridge(self, vision_dim: int, decoder_dim: int) -> BridgeInterface:
        """Instantiate the bridge from config."""
        cls = BRIDGE_REGISTRY.get(self.config.bridge_type)
        if cls is None:
            raise ValueError(
                f"Unknown bridge type '{self.config.bridge_type}'. "
                f"Available: {list(BRIDGE_REGISTRY.keys())}"
            )

        # Build kwargs based on bridge type
        kwargs: dict[str, Any] = {
            "vision_dim": vision_dim,
            "decoder_dim": decoder_dim,
        }

        if self.config.bridge_type in ("qformer_lite", "resampler", "instruction_conditioned"):
            kwargs.update({
                "bridge_dim": self.config.bridge_dim,
                "num_queries": self.config.bridge_num_queries,
                "num_layers": self.config.bridge_num_layers,
                "num_heads": self.config.bridge_num_heads,
                "ffn_ratio": self.config.bridge_ffn_ratio,
                "dropout": self.config.bridge_dropout,
            })
        elif self.config.bridge_type in ("linear", "gated"):
            kwargs["dropout"] = self.config.bridge_dropout

        return cls(**kwargs)

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the decoder."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.decoder.model = get_peft_model(self.decoder.model, lora_config)
            logger.info("LoRA applied to decoder with r=%d", self.config.lora_r)
        except ImportError:
            logger.warning("peft not installed — skipping LoRA. Install with: pip install peft")

    def _log_param_counts(self) -> None:
        """Log parameter counts for each component."""
        def count(module: nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        v_total, v_train = count(self.vision_encoder)
        b_total, b_train = count(self.bridge)
        d_total, d_train = count(self.decoder)

        logger.info(
            "Parameter counts:\n"
            "  Vision:  %dM total, %dK trainable\n"
            "  Bridge:  %dM total, %dM trainable\n"
            "  Decoder: %dM total, %dK trainable",
            v_total // 1_000_000, v_train // 1_000,
            b_total // 1_000_000, b_train // 1_000_000,
            d_total // 1_000_000, d_train // 1_000,
        )

    # ── Core API ──────────────────────────────────────────────────

    def encode_image(self, images: list[Image.Image]) -> VisionEncoderOutput:
        """Encode PIL images through the vision encoder.

        Args:
            images: List of PIL images.

        Returns:
            VisionEncoderOutput with patch features.
        """
        pixel_values = self.vision_encoder.preprocess(images)
        return self.vision_encoder(pixel_values)

    def bridge_image(
        self,
        vision_output: VisionEncoderOutput,
        instruction_embeds: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        """Transform vision features through the bridge.

        Args:
            vision_output: Output from encode_image().
            instruction_embeds: Optional instruction embeddings.

        Returns:
            BridgeOutput with decoder-compatible visual tokens.
        """
        return self.bridge(vision_output, instruction_embeds)

    def forward_train(
        self,
        pixel_values: torch.Tensor,
        prompts: list[str],
        responses: list[str],
    ) -> "DecoderOutput":
        """Training forward pass with full gradient flow through bridge.

        Accepts raw pixel values + text strings and handles the complete
        pipeline (vision → bridge → prompt packing → decode) in one call.
        Labels are generated by the packer from ``responses``, ensuring
        perfect alignment between the packed sequence length and the label
        tensor.

        This is the method that ``VLMTrainer`` should use when paired with
        ``VLMTrainingCollator``.

        Args:
            pixel_values: [B, C, H, W] image tensors (on model device).
            prompts: List of B prompt strings (may contain ``<image>``).
            responses: List of B response strings (used to build labels).

        Returns:
            DecoderOutput with .loss computed from properly-aligned labels.
        """
        B = pixel_values.shape[0]

        # Vision encoding (frozen by default)
        vision_out = self.vision_encoder(pixel_values)
        # Bridge (trainable — gradients flow here)
        bridge_out = self.bridge(vision_out)
        image_features = bridge_out.projected_features  # [B, num_tokens, decoder_dim]

        # Pack each example with proper label alignment
        packed_list = []
        for i in range(B):
            packed = self.packer.pack(
                text=prompts[i],
                image_embeds=image_features[i],
                labels_text=responses[i],
                mask_image_in_labels=True,
                mask_prompt_in_labels=True,
            )
            packed_list.append(packed)

        # Pad to max length in batch
        max_len = max(p.inputs_embeds.shape[1] for p in packed_list)
        dim = packed_list[0].inputs_embeds.shape[-1]
        device = packed_list[0].inputs_embeds.device
        dtype = packed_list[0].inputs_embeds.dtype

        batched_embeds = torch.zeros(B, max_len, dim, device=device, dtype=dtype)
        batched_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        batched_labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)

        for i, p in enumerate(packed_list):
            length = p.inputs_embeds.shape[1]
            batched_embeds[i, :length] = p.inputs_embeds[0]
            batched_mask[i, :length] = p.attention_mask[0]
            if p.labels is not None:
                batched_labels[i, :length] = p.labels[0]

        return self.decoder(
            inputs_embeds=batched_embeds,
            attention_mask=batched_mask,
            labels=batched_labels,
        )

    def forward(
        self,
        images: Optional[list[Image.Image]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        """Full forward pass through the VLM.

        Supports multiple input modes:
        1. images + text: Full pipeline (encode, bridge, pack, decode)
        2. pixel_values + input_ids: Pre-processed inputs
        3. inputs_embeds: Pre-packed embeddings (training mode)

        Args:
            images: PIL images to process.
            pixel_values: Pre-processed image tensors [B, C, H, W].
            text: Text prompt (may contain <image> placeholder).
            input_ids: Pre-tokenized text [B, seq_len].
            attention_mask: [B, seq_len].
            labels: [B, seq_len] for training loss.
            inputs_embeds: Pre-assembled [B, seq_len, hidden_dim].

        Returns:
            DecoderOutput with logits and optional loss.
        """
        # If inputs_embeds already provided, skip encoding
        if inputs_embeds is not None:
            return self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Encode images
        image_features = None
        if images is not None:
            vision_out = self.encode_image(images)
            bridge_out = self.bridge_image(vision_out)
            image_features = bridge_out.projected_features  # [B, num_tokens, decoder_dim]
        elif pixel_values is not None:
            vision_out = self.vision_encoder(pixel_values)
            bridge_out = self.bridge_image(vision_out)
            image_features = bridge_out.projected_features

        # Cast bridge output to decoder dtype (bridge=float32, decoder may be bfloat16)
        if image_features is not None:
            decoder_dtype = next(self.decoder.parameters()).dtype
            image_features = image_features.to(dtype=decoder_dtype)

        # Pack prompt
        if text is not None and image_features is not None:
            # Single image for now; first in batch
            packed = self.packer.pack(
                text=text,
                image_embeds=image_features[0],  # [num_tokens, dim]
            )
            return self.decoder(
                inputs_embeds=packed.inputs_embeds,
                attention_mask=packed.attention_mask,
                labels=labels,
            )

        # Text-only fallback
        if input_ids is not None:
            text_embeds = self.decoder.embed_tokens(input_ids)
            return self.decoder(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        raise ValueError("Must provide either (images + text), (pixel_values + input_ids), or inputs_embeds")

    @torch.no_grad()
    def generate(
        self,
        images: Optional[list[Image.Image]] = None,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate text from images + prompt.

        Args:
            images: PIL images to describe/analyze.
            prompt: Text instruction/question.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling threshold.
            do_sample: Whether to sample or use greedy decoding.

        Returns:
            Generated text string.
        """
        self.eval()

        # Encode image
        image_features = None
        if images is not None:
            vision_out = self.encode_image(images)
            bridge_out = self.bridge_image(vision_out)
            image_features = bridge_out.projected_features[0]  # [num_tokens, dim]
            # Cast to decoder dtype (bridge is float32, decoder may be bfloat16)
            decoder_dtype = next(self.decoder.parameters()).dtype
            image_features = image_features.to(dtype=decoder_dtype)

        # Pack prompt
        packed = self.packer.pack(text=prompt, image_embeds=image_features)

        # Use HF generate with inputs_embeds
        # We need to manually run generation since inputs_embeds isn't directly
        # supported by all HF generate() implementations
        generated_ids = self._generate_from_embeds(
            inputs_embeds=packed.inputs_embeds,
            attention_mask=packed.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        # Decode
        text_output = self.decoder.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        return text_output

    def _generate_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation from input embeddings.

        Simple implementation; for production use the HF generate() pipeline
        with KV caching.
        """
        B = inputs_embeds.shape[0]
        device = inputs_embeds.device

        generated = torch.zeros(B, 0, dtype=torch.long, device=device)
        past_kv = None
        current_embeds = inputs_embeds
        current_mask = attention_mask

        eos_id = self.decoder.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            output = self.decoder(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                use_cache=True,
                past_key_values=past_kv,
            )

            # Get logits for last position
            next_logits = output.logits[:, -1, :]  # [B, vocab]

            if do_sample and temperature > 0:
                next_logits = next_logits / temperature
                # Top-p filtering
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                probs = torch.softmax(sorted_logits, dim=-1)
                next_token_sorted = torch.multinomial(probs, num_samples=1)
                next_token = sorted_idx.gather(-1, next_token_sorted)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check EOS
            if (next_token == eos_id).all():
                break

            # For next iteration, only feed the new token embedding
            current_embeds = self.decoder.embed_tokens(next_token)
            current_mask = torch.cat([
                current_mask,
                torch.ones(B, 1, dtype=torch.long, device=device),
            ], dim=1)
            past_kv = output.past_key_values

        return generated

    # ── Adapter & domain pack API ─────────────────────────────────

    def attach_adapter(
        self,
        adapter_path: Union[str, Path],
        adapter_name: str = "default",
    ) -> None:
        """Attach a LoRA adapter to the decoder.

        Args:
            adapter_path: Path to saved PEFT adapter.
            adapter_name: Name for this adapter (for stacking).
        """
        try:
            from peft import PeftModel
            self.decoder.model = PeftModel.from_pretrained(
                self.decoder.model,
                adapter_path,
                adapter_name=adapter_name,
            )
            logger.info("Attached adapter '%s' from %s", adapter_name, adapter_path)
        except ImportError:
            raise ImportError("peft required for adapter loading. Install: pip install peft")

    def load_domain_pack(self, pack_path: Union[str, Path]) -> None:
        """Load a domain adaptation pack (bridge weights + optional adapter).

        A domain pack contains:
        - bridge_weights.pt: Trained bridge weights for the domain
        - adapter/ (optional): LoRA adapter for the decoder
        - config.yaml: Domain-specific settings

        Args:
            pack_path: Path to the domain pack directory.
        """
        pack_dir = Path(pack_path)

        # Load bridge weights
        bridge_path = pack_dir / "bridge_weights.pt"
        if bridge_path.exists():
            state_dict = torch.load(bridge_path, map_location="cpu", weights_only=True)
            self.bridge.load_state_dict(state_dict)
            logger.info("Loaded bridge weights from %s", bridge_path)

        # Load adapter if present
        adapter_dir = pack_dir / "adapter"
        if adapter_dir.exists():
            self.attach_adapter(adapter_dir, adapter_name=pack_dir.name)

        # Load domain config
        config_path = pack_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                domain_config = yaml.safe_load(f)
            logger.info("Domain config loaded: %s", domain_config.get("domain_name", "unknown"))

    # ── Serialization ─────────────────────────────────────────────

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model components to disk.

        Saves:
        - config.yaml
        - bridge_weights.pt (always)
        - decoder adapter (if LoRA active)
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.to_yaml(save_dir / "config.yaml")

        # Save bridge (always trainable, always save)
        torch.save(self.bridge.state_dict(), save_dir / "bridge_weights.pt")

        # Save decoder adapter if applicable
        if self.config.use_lora and hasattr(self.decoder.model, "save_pretrained"):
            adapter_dir = save_dir / "decoder_adapter"
            self.decoder.model.save_pretrained(adapter_dir)

        logger.info("Model saved to %s", save_dir)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "KarnaVLM":
        """Load a saved KarnaVLM.

        Args:
            path: Directory containing saved model.

        Returns:
            Loaded KarnaVLM instance.
        """
        load_dir = Path(path)
        config = KarnaVLMConfig.from_yaml(load_dir / "config.yaml")
        model = cls(config)

        # Load bridge weights
        bridge_path = load_dir / "bridge_weights.pt"
        if bridge_path.exists():
            state_dict = torch.load(bridge_path, map_location="cpu", weights_only=True)
            model.bridge.load_state_dict(state_dict)

        # Load decoder adapter
        adapter_dir = load_dir / "decoder_adapter"
        if adapter_dir.exists():
            model.attach_adapter(adapter_dir)

        return model

    @classmethod
    def from_config(cls, config: Union[KarnaVLMConfig, str, Path]) -> "KarnaVLM":
        """Create a KarnaVLM from a config object or YAML path.

        Args:
            config: KarnaVLMConfig instance or path to YAML config.

        Returns:
            New KarnaVLM instance.
        """
        if isinstance(config, (str, Path)):
            config = KarnaVLMConfig.from_yaml(config)
        return cls(config)
