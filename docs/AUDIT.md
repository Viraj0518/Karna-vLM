# Karna VLM — Code Audit

**Auditor:** Excaliber ⚔️  
**Date:** 2026-03-20  
**Scope:** Full codebase — every file in `Karna-VLM/`  
**Verdict:** Strong architecture, clean code, production-ready scaffolding. Several real gaps before first training run.

---

## Executive Summary

Karna VLM is a custom VLM training platform (`src/karna_vlm/`). Architecturally excellent — modular, typed, well-tested — but has never been trained (no checkpoints, no data, no benchmark results). The OCR API has been split into its own repository.

| Area | Grade | Notes |
|------|-------|-------|
| **Architecture** | A | Clean 3-component design, every piece swappable via interfaces |
| **Code Quality** | A- | Typed, documented, consistent style. Minor issues below |
| **Test Coverage** | B+ | 8 test files covering bridges, LoRA, packing, safety, eval, config. No integration tests |
| **Training Pipeline** | B | 4-stage pipeline is well-designed but never executed. Missing data, no results |
| **Production Readiness (VLM)** | C+ | Can't generate until trained. No pre-trained weights exist |
| **Safety** | B+ | Policy system, content filters, model cards — but safety rules are shallow |
| **Documentation** | B | README is good, ARCHITECTURE.md solid, but VLM training guide is missing |

---

## Architecture Audit

### What's Right

1. **Three clean abstractions**: `VisionEncoderInterface`, `BridgeInterface`, `DecoderInterface` — any component can be swapped without touching others
2. **Bridge as moat**: The design insight of concentrating trainable intelligence in the bridge (~15M params) while freezing vision (~86M) and decoder (~500M-3B) is sound and economically smart
3. **Five bridge variants**: Linear, QFormer-Lite, Resampler, Gated, InstructionConditioned — each solves a different use case with proper registries
4. **4-stage training pipeline**: Bootstrap → Multitask → Domain → Distill is the industry-standard approach (LLaVA, InternVL, etc.)
5. **Prompt packing**: Proper multimodal sequence assembly with image token insertion, label masking, batch padding
6. **Evaluation suite**: Caption (BLEU-4, ROUGE-L), VQA (exact/relaxed/soft), Grounding (IoU), Instruction (format+relevance), Latency, Ablation framework, Report generation
7. **Domain pack system**: Portable `bridge_weights.pt` + `adapter/` + `config.yaml` — elegant for vertical deployment

### What Needs Fixing

#### Critical (Blocks Training)

1. **No training data pipeline exists**
   - `VLMDataset` reads JSONL but no sample data or download scripts
   - No `data/` directory, no manifest files registered
   - Missing: CC3M/SBU download script, LLaVA-Instruct conversion, data validation
   - **Fix:** Create `scripts/prepare_data.py` with dataset downloaders and JSONL converters

2. **`VLMCollator` calls bridge in `__call__`** — bridge should be part of the model forward pass, not the collator
   - Collator does `self.bridge(vision_out)` with `torch.no_grad()` during data loading
   - This means bridge doesn't receive gradients through the collator path
   - The trainer uses `model(inputs_embeds=...)` which bypasses `encode_image()` and `bridge_image()`
   - **Fix:** Collator should return raw pixel values; model forward should handle full pipeline
   - Current workaround: It works if you use `model(images=..., text=...)` directly (bypasses collator)

3. **Forward pass label alignment bug**
   - In `vlm_model.py::forward()`, when `text` + `image_features` are provided, `packed` is created but `labels` from the function arg may not align with the packed sequence length
   - The packer creates labels internally only when `labels_text` is passed, but `forward()` doesn't pass `labels_text`
   - **Fix:** Training path should go through collator → packer with `labels_text=response`, not through `forward(labels=...)`

#### High Priority

4. **`LoRALinear` matrix multiplication order is wrong**
   - `lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling`
   - `lora_A` shape: `(r, in_features)`, `lora_B` shape: `(out_features, r)`
   - `x @ lora_A.T` gives `(*, in) @ (in, r) = (*, r)` ✅
   - `(*, r) @ lora_B.T` gives `(*, r) @ (r, out)` = `(*, out)` ✅
   - Actually this is correct. The naming is non-standard (usually A is down-projection, B is up-projection) but the math works. LoRA B is zero-initialized so initial output matches original. **No fix needed** — just document the convention.

5. **Autoregressive generation loop is naive**
   - `_generate_from_embeds()` implements manual token-by-token generation
   - No KV cache optimization for the initial `inputs_embeds` pass (HF's `generate()` doesn't accept `inputs_embeds` for all models)
   - Performance will be 5-10x slower than HF's optimized generation with proper KV caching
   - **Fix:** For production, use vLLM or implement proper KV cache handling. Current impl is fine for dev/eval.

6. **`SafetyAction` comparison uses string comparison**
   - `if rule.action.value > action.value` — this compares strings lexicographically ("refuse" > "allow" works by luck, but "warn" > "refuse" is wrong)
   - **Fix:** Use enum ordering or numeric severity levels

#### Medium Priority

8. **No gradient checkpointing support**
   - With 3B decoder, VRAM will be tight for batch_size > 1
   - Bridge + decoder LoRA training should offer `torch.utils.checkpoint`
   - **Fix:** Add `gradient_checkpointing: bool` to `TrainingConfig`

9. **Vision encoder `preprocess()` sends data to model device inside DataLoader workers**
   - `pixel_values.to(device=device, dtype=...)` in `preprocess()` will fail with multi-worker DataLoader (workers don't have GPU access)
   - **Fix:** Preprocessing should stay on CPU; move to device in the training loop

10. **`ChatSession._build_prompt()` includes ALL images as `<image>` tokens**
    - But `model.generate()` only takes `self.current_image` (single image)
    - Multi-turn chat with multiple images will have mismatched image tokens vs actual images
    - **Fix:** Track image count and pass all images to generate, or only include `<image>` for the latest

11. **PromptPacker label construction is brittle**
    - `labels_text` handling concatenates prompt + response then takes last N tokens
    - If tokenization of the combined text differs from individual tokenization (which it will for BPE), labels may be misaligned
    - **Fix:** Tokenize prompt and response separately, mask prompt positions

12. **Config `extra` dict is unused throughout the codebase**
    - `KarnaVLMConfig.extra` is defined but never read
    - Could be useful for passing bridge-specific args but nothing consumes it
    - **Minor** — leave as extension point

13. **No `__all__` exports in subpackages**
    - Most `__init__.py` files are empty
    - Makes it unclear what the public API is
    - **Fix:** Add `__all__` to each subpackage

#### Low Priority

14. ~~`fix_model.py` is a one-off script~~ — **RESOLVED**: OCR files removed, repo is VLM-only now

---

## Test Audit

| Test File | Tests | What's Covered | What's Missing |
|-----------|-------|---------------|----------------|
| `test_bridge_shapes.py` | 10 | All 5 bridge shapes, attention weights, gradient flow | Batch size edge cases, very long sequences |
| `test_config.py` | 3 | Default config, YAML round-trip, config file loading | Invalid config handling |
| `test_data.py` | 8 | VLMSample, DatasetManifest, Templates, Mixtures | VLMDataset loading (needs real files), HFVLMDataset |
| `test_evaluation.py` | 8 | VQA, Grounding, Instruction evaluators | Caption evaluator, Latency benchmark |
| `test_lora.py` | 6 | LoRALinear shapes, frozen original, gradient flow, zero-init | LoRAManager, adapter save/load |
| `test_prompt_packing.py` | 7 | Text-only, image insertion, auto-prepend, truncation, batching | Label alignment, multi-image |
| `test_safety.py` | 6 | SafetyPolicy, ContentFilter, ModelCard | Output filtering, audit log persistence |
| `__init__.py` | 0 | — | — |

**Missing test categories:**
- Integration test: full model forward pass (even with tiny config)
- Training loop smoke test (1 batch, 1 step)
- API endpoint tests (FastAPI TestClient)
- Checkpoint save/load round-trip
- Domain pack save/load

---

## Security Audit

| Finding | Severity | Location |
|---------|----------|----------|
| CORS `allow_origins=["*"]` in VLM API | Medium | `api/server.py` |
| No auth/API key on VLM API | Medium | `api/server.py` |
| No rate limiting on VLM API | Low | `api/server.py` |
| `trust_remote_code=True` in decoder | Low | `hf_decoder.py` — needed for some HF models |
| `weights_only=True` used consistently in `torch.load()` | ✅ Good | All checkpoint loading |

| Prompt injection via user text → VLM | Low | Inherent to VLM design |

---

## Performance Considerations

1. **Bridge parameter counts are accurate**: QFormer-Lite 512d/64q/4L ≈ 15M params ✅
2. **VRAM estimates look right**: Tiny ~2GB, Small ~4GB, Mid ~8GB (frozen backbones + bridge + LoRA)
3. **Generation will be slow** without vLLM/TGI — manual loop is 5-10x slower
4. **SigLIP-Base is a solid choice** — good patch representations, 86M params, fast
5. **Llama 3.2 + Gemma 2 decoder family** — 1B/2B/3B gives a nice size ladder, all US-origin open weights
6. **Prompt packing handles variable-length sequences correctly** — no wasted compute on padding

---

## Recommendations (Prioritized)

### Must Do Before First Training Run
1. Create `scripts/prepare_data.py` — download CC3M-subset, convert to JSONL schema
2. Fix collator-bridge interaction — return pixel values from collator, run bridge in model forward
3. Create `scripts/train_stage1.py` — end-to-end training script that ties config + data + trainer
4. Add one integration test — instantiate tiny model, run one forward + backward pass

### Should Do Before Release
5. Split repo or at minimum split requirements (OCR API vs VLM platform)
6. Add gradient checkpointing
7. Fix `SafetyAction` enum comparison
8. Add proper KV cache generation (or document vLLM as production inference path)
9. Add API auth middleware
10. Create `TRAINING.md` — step-by-step guide from zero to trained model

### Nice to Have
11. Add HuggingFace Hub integration (push/pull models)
12. Add ONNX export pipeline (stage 4)
13. Add WandB integration in trainer
14. Multi-image support in forward pass
15. Flash attention support in bridge cross-attention

---

## File Inventory

| Category | Files | Lines (approx) |
|----------|-------|----------------|
| Model core | 12 | ~2,200 |
| Training | 7 | ~950 |
| Data | 6 | ~750 |
| Evaluation | 7 | ~900 |
| Inference | 4 | ~550 |
| Safety | 4 | ~550 |
| Utils | 4 | ~200 |
| API (VLM) | 1 | ~200 |
| Configs | 5 | ~120 |
| Tests | 8 | ~450 |
| **Total** | **58** | **~6,870** |

This is a well-scoped, well-structured codebase. The architecture decisions (frozen encoder, trainable bridge, LoRA decoder) are industry-validated. The gap is execution: no one has pressed "go" on training yet.

> **Note:** OCR API (`api.py`, `Dockerfile`, `prompts/`, `scripts/`, `index.html`) has been moved to its own repository. This repo is VLM platform only.
