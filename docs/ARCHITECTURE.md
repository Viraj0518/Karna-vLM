# Karna VLM Architecture

## Overview

Karna VLM is a compact, customizable vision-language model platform. Its intelligence lives in three places:

```
                    ┌──────────────────────────────────────────────────┐
                    │                  Karna VLM                       │
                    │                                                  │
 ┌─────────┐       │  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
 │  IMAGE   │──────▶│  │  Vision  │───▶│  Bridge  │───▶│ Decoder  │───▶│ TEXT
 └─────────┘       │  │ Encoder  │    │ (MOAT)   │    │   LLM    │   │ OUTPUT
                    │  │ (frozen) │    │(trained) │    │ (+LoRA)  │   │
 ┌─────────┐       │  └──────────┘    └──────────┘    └──────────┘   │
 │  TEXT    │──────▶│                                                  │
 │ PROMPT   │       │  Prompt Packing: image tokens + text tokens     │
 └─────────┘       └──────────────────────────────────────────────────┘
```

## Components

### 1. Vision Encoder (Frozen)

The perception substrate. Extracts patch-level features from images.

- **SigLIP** (default): Strong patch representations, no CLS token
- **CLIP / EVA-CLIP**: Alternative backbone with CLS token

**Key properties:**
- Frozen by default (no gradient computation)
- Outputs patch-level features: `[B, num_patches, hidden_dim]`
- Swappable via `VisionEncoderInterface`

### 2. Bridge (The Moat) — Trainable

The core proprietary intelligence. Compresses and transforms vision features into decoder-compatible tokens.

| Bridge | Description | Params | Best For |
|--------|------------|--------|----------|
| `LinearProjector` | Two-layer MLP per patch | ~2M | Baseline, fast iteration |
| `QFormerLiteBridge` | Learned query cross-attention | ~15M | Production default |
| `ResamplerBridge` | Perceiver-style latent queries | ~12M | Efficient compression |
| `GatedBridge` | Gated linear with learned gates | ~3M | Lightweight + interpretable |
| `InstructionConditionedBridge` | Query bridge + instruction awareness | ~20M | Complex instruction tasks |

### 3. Decoder LLM (Mostly Frozen)

Compact language model that generates text from the mixed multimodal sequence.

- **Tiny:** Qwen2-0.5B (~500M params)
- **Small:** Qwen2-1.5B (~1.5B params)
- **Mid:** Qwen2.5-3B (~3B params)

Frozen by default. Fine-tuned via LoRA adapters on attention projections.

## Training Pipeline

```
Stage 1: Bootstrap Alignment
├── Train: Bridge only
├── Data: Image-caption pairs
├── LR: 1e-3 (high, bridge-only)
└── Goal: Align vision-decoder spaces

Stage 2: Multitask Instruction Tuning
├── Train: Bridge + Decoder LoRA
├── Data: Mixed instruction-following
├── LR: 2e-5
└── Goal: Follow multimodal instructions

Stage 3: Domain Specialization
├── Train: Bridge LoRA + Decoder domain adapter
├── Data: Domain-specific (medical, legal, OCR, etc.)
├── Output: Domain pack (bridge weights + adapter)
└── Goal: Expert performance in a domain

Stage 4: Distillation & Compression
├── Methods: KD, quantization, pruning
├── Output: Deployment-ready model
└── Goal: Minimize latency and VRAM
```

## Model Family

All variants share the same API, training pipeline, dataset interface, and evaluation harness.

| Variant | Vision | Bridge | Decoder | Total | VRAM |
|---------|--------|--------|---------|-------|------|
| Tiny/Edge | SigLIP-Base | QFormer-Lite (256d, 32q) | Qwen2-0.5B | ~0.6B | ~2GB |
| Small | SigLIP-Base | QFormer-Lite (512d, 64q) | Qwen2-1.5B | ~1.7B | ~4GB |
| Mid | SigLIP-SO400M | Instruction-Conditioned (768d, 96q) | Qwen2.5-3B | ~3.8B | ~8GB |

## Customization

### Domain Packs

```
domain_pack/
├── bridge_weights.pt   # Domain-trained bridge
├── adapter/            # LoRA adapter for decoder
└── config.yaml         # Domain metadata
```

### Adapter Stacking

Multiple LoRA adapters can be composed for multi-domain deployment.
