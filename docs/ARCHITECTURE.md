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

Compact language model that generates text from the mixed multimodal sequence. All decoders are **US-origin, open-weights, open-architecture** — safe for government and regulated environments.

- **Tiny:** Llama 3.2-1B (Meta, ~1B params)
- **Small:** Gemma 2-2B (Google, ~2B params)
- **Mid:** Llama 3.2-3B (Meta, ~3B params)

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
| Tiny/Edge | SigLIP-Base | QFormer-Lite (256d, 32q) | Llama 3.2-1B (Meta) | ~1.2B | ~3GB |
| Small | SigLIP-Base | QFormer-Lite (512d, 64q) | Gemma 2-2B (Google) | ~2.3B | ~5GB |
| Mid | SigLIP-SO400M | Instruction-Conditioned (768d, 96q) | Llama 3.2-3B (Meta) | ~3.5B | ~8GB |

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
