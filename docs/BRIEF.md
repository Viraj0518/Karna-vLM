# Karna VLM — Technical Brief

**From:** Viraj  
**Re:** Custom Vision-Language Model for On-Premise Deployment  
**Date:** March 2026  

---

## The Short Version

I built a proof-of-concept for a small, trainable vision-language model that we can fine-tune on our own data, run on our own hardware, and fully audit end-to-end. No cloud APIs. No black boxes. No Chinese-origin model dependencies. Total training cost: under $200 on a single GPU.

---

## Why This Matters

Right now, if we want AI to understand images — read documents, analyze scans, extract structured data — we're stuck choosing between:

1. **Cloud APIs** (GPT-4V, Gemini Vision) — great quality, but data leaves our network. That's a non-starter for anything sensitive.
2. **Open-source monoliths** (LLaVA-7B, InternVL-8B) — 7-8 billion parameters, need expensive GPU clusters to train, and most use Chinese-origin base models (Qwen, Yi, DeepSeek).

Neither works for us.

## What I Built

A modular VLM split into three components:

| Component | What It Does | Size | Trainable? |
|-----------|-------------|------|------------|
| **Vision Encoder** (SigLIP, Google) | Looks at the image | 86M params | No — already pre-trained on 400M images |
| **Bridge** (custom, ours) | Translates vision → language | 18.5M params | **Yes — this is what we train** |
| **Decoder** (Gemma 2, Google) | Generates text output | 2.6B params | Minimal (LoRA adapters, ~4M params) |

The key insight: we only train 18.5 million parameters instead of billions. That's why it fits on one GPU and trains in hours instead of weeks.

**All components are US-origin, open-weights, open-architecture.** SigLIP and Gemma 2 from Google, Llama 3.2 from Meta. No Qwen, no restricted licenses.

## Current Status

- ✅ Full architecture implemented and unit tested (88/88 tests pass)
- ✅ Production model loads and runs end-to-end with real weights (Gemma 2-2B confirmed)
- ✅ 4-stage training pipeline scaffolded (bootstrap → instruction tuning → domain specialization → compression)
- ✅ LoRA fine-tuning, evaluation suite, safety system, REST API — all implemented
- ❌ Bridge not yet trained on real data (outputs gibberish currently — expected for a POC)
- ❌ No benchmark numbers yet

**What's needed to go from POC → trained model:** Training data (image-caption pairs) + one A100 GPU for ~8 hours. Estimated compute cost: $50-200 depending on dataset size.

## What This Enables

Once trained, this model can be domain-specialized for specific verticals by swapping lightweight adapter files (~80MB each):

- **Document extraction** — read forms, invoices, handwritten notes into structured JSON
- **Medical image analysis** — generate radiology reports, classify pathology slides, screen dermatology images
- **Visual QA** — answer questions about images in a chat interface
- **Quality inspection** — flag defects in manufacturing images

Domain adaptation takes a few hours of additional training on domain-specific data. Multiple domains can share the same base model — just swap the adapter.

## The Numbers

| Metric | Value |
|--------|-------|
| Total parameters | 2.7B |
| Trainable parameters | 18.5M (0.7%) |
| Training compute | ~$50-200 (single A100, 4-8 hours) |
| Inference VRAM | ~6GB (fits on any modern workstation GPU) |
| Adapter size | ~80MB per domain |
| All US-origin components | ✅ |
| On-premise deployable | ✅ |
| Fully auditable | ✅ |

## Repo

Everything is at [github.com/Viraj0518/Karna-VLM](https://github.com/Viraj0518/Karna-VLM) — code, architecture docs, deep technical writeup, audit report. Happy to walk through any of it.

---

*Let me know if you want to see a demo or discuss next steps on getting training data and compute allocated.*
