<p align="center">
  <h1 align="center">Karna VLM</h1>
  <p align="center"><strong>A compact, customizable vision-language model you can actually train yourself.</strong></p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#training">Training</a> •
    <a href="#evaluation">Evaluation</a> •
    <a href="#deployment">Deployment</a>
  </p>
</p>

---

## What Is This?

Karna VLM is a **vision-language model platform** designed so that a small team (or one person) can train, customize, and deploy a production-grade VLM on a single GPU.

The core idea: instead of training a monolithic 7B+ model, we split the problem into three components and only train the small one:

```
Image → [Frozen Vision Encoder] → [Trainable Bridge] → [Frozen Decoder LLM] → Text
              86M params              ~15M params            1-3B params
             (SigLIP)              (the secret sauce)     (Llama 3.2 / Gemma 2)
```

The **bridge** is the only component you fully train. It's a compact neural network (~15M parameters) that learns to translate visual features into language. The vision encoder and decoder stay frozen — you fine-tune the decoder with LoRA adapters (~0.5-2% of its parameters).

**Why this matters:**
- Train on **1 GPU** instead of 8-64
- Full training run in **hours** instead of weeks
- Customize for your domain (medical, legal, OCR, finance) by swapping the bridge + adapter
- Deploy the whole thing in **2-8GB VRAM**

> Built for teams that want to own their vision-language model without renting a GPU cluster.

---

## Architecture

### The Three Components

| Component | What It Does | Default | Trainable? | Params | Origin |
|-----------|-------------|---------|-----------|--------|--------|
| **Vision Encoder** | Converts images into patch features | SigLIP-Base | ❄️ Frozen | ~86M | Google (US) |
| **Bridge** | Compresses visual features into decoder-compatible tokens | QFormer-Lite | ✅ Fully trained | ~15M | Ours |
| **Decoder LLM** | Generates text from the multimodal token sequence | Gemma 2-2B | LoRA only | ~2B (0.5% trainable) | Google (US) |

> **All components are US-origin, open-weights, open-architecture.** No Qwen, no restricted-license models. Safe for government and regulated environments.

### How a Forward Pass Works

```
1. Image (224×224 RGB)
   ↓
2. SigLIP extracts 196 patch features → [B, 196, 768]
   ↓
3. QFormer-Lite bridge:
   - 64 learned query tokens cross-attend to 196 patches
   - 4 transformer layers refine the queries
   - Output projection → [B, 64, decoder_dim]
   ↓
4. Prompt Packer interleaves image tokens + text tokens:
   [<text_embed> ... <image_token_1> ... <image_token_64> ... <text_embed> ...]
   ↓
5. Decoder LLM (Gemma 2 / Llama 3.2) generates text autoregressively
   ↓
6. "A golden retriever playing fetch in a park"
```

### The Bridge (The Moat)

The bridge is where the intelligence lives. We provide 5 variants:

| Bridge | How It Works | Params | Best For |
|--------|-------------|--------|----------|
| **LinearProjector** | MLP per patch (no compression) | ~2M | Fast baseline, debugging |
| **QFormerLiteBridge** | Learned queries cross-attend to patches | ~15M | **Production default** |
| **ResamplerBridge** | Perceiver-style latent queries | ~12M | Efficient compression |
| **GatedBridge** | Sigmoid gates select important features | ~3M | Lightweight + interpretable |
| **InstructionConditionedBridge** | Queries attend to vision AND instruction text | ~20M | Complex tasks, highest quality |

All bridges implement `BridgeInterface` — swap them by changing one config field.

### Model Family

| Variant | Vision | Bridge | Decoder | Total | VRAM | Use Case |
|---------|--------|--------|---------|-------|------|----------|
| **Tiny** | SigLIP-Base | QFormer-Lite (256d, 32q) | Llama 3.2-1B (Meta) | ~1.2B | ~3GB | Edge, mobile, prototyping |
| **Small** | SigLIP-Base | QFormer-Lite (512d, 64q) | Gemma 2-2B (Google) | ~2.3B | ~5GB | Production default |
| **Mid** | SigLIP-SO400M | InstructionConditioned (768d, 96q) | Llama 3.2-3B (Meta) | ~3.5B | ~8GB | Maximum quality |

---

## Quick Start

### Install

```bash
git clone https://github.com/Viraj0518/Karna-VLM.git
cd Karna-VLM
pip install -e ".[all]"    # everything: train + eval + serve + export + dev
# Or just what you need:
# pip install -e "."           # inference only
# pip install -e ".[train]"   # + training deps (deepspeed, wandb)
# pip install -e ".[serve]"   # + API server deps (fastapi, uvicorn)
```

### Generate Text from an Image

```python
from karna_vlm import KarnaVLM, KarnaVLMConfig
from PIL import Image

config = KarnaVLMConfig(model_family="small")
model = KarnaVLM(config)

image = Image.open("photo.jpg")
output = model.generate(images=[image], prompt="Describe this image in detail.")
print(output)
```

### Load from YAML Config

```python
model = KarnaVLM.from_config("configs/model_tiny.yaml")
```

### Load a Trained Model

```python
model = KarnaVLM.from_pretrained("outputs/my_trained_model/")
```

### CLI

```bash
karna generate photo.jpg --prompt "What's in this image?"
karna serve --config configs/model_small.yaml --port 8080
karna info --config configs/model_tiny.yaml
```

---

## Training

Training happens in 4 stages. Each stage builds on the previous one.

### Stage 1: Bootstrap Alignment

**Goal:** Teach the bridge to map vision features into the decoder's language space.  
**What's trained:** Bridge only (~15M params). Encoder and decoder are frozen.  
**Data:** Image-caption pairs (CC3M, LAION subset, SBU, etc.)  
**Typical time:** 4-8 hours on 1× A100

```python
from karna_vlm import KarnaVLM, KarnaVLMConfig
from karna_vlm.training.stage1_bootstrap import run_stage1, Stage1Config
from karna_vlm.data import VLMDataset
from torch.utils.data import DataLoader

# 1. Build model
config = KarnaVLMConfig(model_family="small")
model = KarnaVLM(config)

# 2. Load data (JSONL format — see Data Format section below)
train_dataset = VLMDataset("data/captions_train.jsonl", image_root="data/images/")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

# 3. Train
metrics = run_stage1(model, train_loader, config=Stage1Config(
    output_dir="outputs/stage1",
    learning_rate=1e-3,    # high LR — bridge learns fast
    num_epochs=1,
    max_steps=20000,
))
print(f"Stage 1 complete — loss: {metrics['train_loss']:.4f}")
```

### Stage 2: Multitask Instruction Tuning

**Goal:** Teach the model to follow instructions across diverse tasks.  
**What's trained:** Bridge + decoder LoRA adapters.  
**Data:** Mixed instruction-following data (VQA, captioning, OCR, chat, grounding)

```python
from karna_vlm.training.stage2_multitask import run_stage2, Stage2Config

# Load Stage 1 checkpoint
model = KarnaVLM.from_pretrained("outputs/stage1/best/")

# Instruction-following dataset
train_dataset = VLMDataset("data/instruct_train.jsonl", image_root="data/images/")
eval_dataset = VLMDataset("data/instruct_val.jsonl", image_root="data/images/")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)

metrics = run_stage2(model, train_loader, eval_loader, config=Stage2Config(
    output_dir="outputs/stage2",
    learning_rate=2e-5,    # lower LR — decoder LoRA is sensitive
    num_epochs=3,
    lora_r=16,
    lora_alpha=32,
))
```

### Stage 3: Domain Specialization (Optional)

**Goal:** Fine-tune for a specific vertical — medical, legal, finance, OCR, etc.  
**Output:** A portable "domain pack" (bridge weights + LoRA adapter + config)

```python
from karna_vlm.training.stage3_domain import run_stage3, Stage3Config

model = KarnaVLM.from_pretrained("outputs/stage2/best/")
medical_data = VLMDataset("data/medical_train.jsonl", image_root="data/medical/")
medical_loader = DataLoader(medical_data, batch_size=4, shuffle=True)

metrics = run_stage3(model, medical_loader, config=Stage3Config(
    domain_name="medical_xray",
    output_dir="outputs/stage3_medical",
    num_epochs=5,
    lora_r=8,
))
# Produces: outputs/stage3_medical/domain_pack_medical_xray/
```

Load a domain pack at inference time:

```python
model = KarnaVLM.from_pretrained("outputs/stage2/best/")
model.load_domain_pack("outputs/stage3_medical/domain_pack_medical_xray/")
output = model.generate(images=[xray_image], prompt="Analyze this chest X-ray.")
```

### Stage 4: Distillation & Compression (Optional)

**Goal:** Compress the model for deployment — quantization, knowledge distillation.

```python
from karna_vlm.training.stage4_distill import quantize_bridge

# INT8 dynamic quantization of the bridge
model.bridge = quantize_bridge(model.bridge, bits=8, method="dynamic")
model.save_pretrained("outputs/stage4_quantized/")
```

### Data Format

All training data uses a simple JSONL format:

```json
{
    "image_path": "images/001.jpg",
    "conversations": [
        {"role": "user", "content": "What is shown in this image?"},
        {"role": "assistant", "content": "A golden retriever playing in a park."}
    ],
    "task_type": "vqa",
    "source_dataset": "visual_qa_v2"
}
```

Supported task types: `caption`, `vqa`, `instruction`, `chat`, `grounding`, `ocr`, `classification`, `structured_extraction`.

### Multi-Task Mixtures

Train on multiple datasets with weighted sampling:

```python
from karna_vlm.data import VLMDataset, DatasetMixture, MixtureComponent

mixture = DatasetMixture(components=[
    MixtureComponent(dataset=VLMDataset("data/captions.jsonl"), weight=2.0, name="captions"),
    MixtureComponent(dataset=VLMDataset("data/vqa.jsonl"), weight=1.0, name="vqa"),
    MixtureComponent(dataset=VLMDataset("data/ocr.jsonl"), weight=1.5, name="ocr"),
])
# Samples proportional to weights: 44% captions, 22% vqa, 33% ocr
```

### LoRA Management

```python
from karna_vlm.training.lora import LoRAManager, LoRAConfig

lora_mgr = LoRAManager(model)

# Apply to decoder (standard — uses PEFT library)
lora_mgr.apply_decoder_lora(LoRAConfig(r=16, alpha=32, target_modules=["q_proj", "v_proj"]))

# Apply to bridge (custom — wraps bridge Linear layers)
lora_mgr.apply_bridge_lora(LoRAConfig(r=8, alpha=16))

# Check what's trainable
print(lora_mgr.get_trainable_summary())
# → {'bridge': 15000000, 'decoder': 800000, 'other': 0}

# Save/load specific adapters
lora_mgr.save_adapter("default", "adapters/medical/")
lora_mgr.load_adapter("default", "adapters/medical/")
```

### Per-Component Learning Rates

```python
from karna_vlm.training.optim import build_optimizer, build_cosine_schedule

optimizer = build_optimizer(
    model,
    base_lr=1e-4,
    bridge_lr=1e-3,        # bridge trains faster
    decoder_lr=2e-5,       # decoder LoRA needs gentle LR
    weight_decay=0.01,
)
scheduler = build_cosine_schedule(optimizer, warmup_steps=500, total_steps=50000)
```

### Checkpointing

```python
from karna_vlm.training.checkpointing import CheckpointManager

ckpt_mgr = CheckpointManager("outputs/", max_checkpoints=3)

# Save (auto-prunes old checkpoints)
ckpt_mgr.save(model, optimizer, scheduler, step=1000, loss=0.42)

# Load
meta = ckpt_mgr.load(model, "outputs/checkpoint-1000/")

# Resume from latest
latest = ckpt_mgr.get_latest()
```

---

## Evaluation

Every evaluator works standalone — no model required for metric computation.

### Caption Quality

```python
from karna_vlm.evaluation import CaptionEvaluator

evaluator = CaptionEvaluator()
metrics = evaluator.evaluate(
    predictions=["A dog playing in the park"],
    references=[["A golden retriever playing fetch", "Dog in a park"]],
)
print(f"BLEU-4: {metrics.bleu_4:.3f}, ROUGE-L: {metrics.rouge_l:.3f}")
```

### VQA Accuracy

```python
from karna_vlm.evaluation import VQAEvaluator

evaluator = VQAEvaluator()
metrics = evaluator.evaluate(
    predictions=["blue", "2", "yes"],
    ground_truths=["blue", ["2", "two"], "yes"],
)
print(f"Exact: {metrics.exact_match:.3f}, Relaxed: {metrics.relaxed_accuracy:.3f}")
```

### Visual Grounding

```python
from karna_vlm.evaluation import GroundingEvaluator

evaluator = GroundingEvaluator()
metrics = evaluator.evaluate(
    pred_boxes=[[0.1, 0.2, 0.5, 0.8]],
    gt_boxes=[[0.1, 0.2, 0.5, 0.8]],
)
print(f"mIoU: {metrics.mean_iou:.3f}, P@50: {metrics.precision_at_50:.3f}")
```

### Latency Benchmarking

```python
from karna_vlm.evaluation import LatencyBenchmark

bench = LatencyBenchmark(model, warmup_runs=3, num_runs=10)
metrics = bench.benchmark(max_new_tokens=64)
print(f"Throughput: {metrics.tokens_per_second:.1f} tok/s")
print(f"VRAM: {metrics.peak_vram_mb:.0f}MB")
print(f"Vision: {metrics.vision_encode_ms:.1f}ms, Bridge: {metrics.bridge_ms:.1f}ms")
```

### Ablation Studies

```python
from karna_vlm.evaluation.ablations import AblationStudy, AblationResult

study = AblationStudy(study_name="Bridge Comparison", baseline_name="qformer_lite")
study.add_result(AblationResult(
    name="qformer_lite", config_diff={"bridge": "qformer_lite"},
    metrics={"vqa_acc": 0.72, "caption_cider": 1.05},
))
study.add_result(AblationResult(
    name="resampler", config_diff={"bridge": "resampler"},
    metrics={"vqa_acc": 0.70, "caption_cider": 1.01},
))
print(study.summary())
```

### Full Evaluation Report

```python
from karna_vlm.evaluation.reports import EvalReport

report = EvalReport(
    model_name="karna-vlm-small",
    vqa_metrics={"exact_match": 0.72, "relaxed": 0.81},
    caption_metrics={"bleu_4": 0.35, "rouge_l": 0.52},
    latency_metrics={"total_ms": 450, "tokens_per_sec": 42},
)
report.save("eval_results/report.json")
print(report.summary())
```

---

## Inference

### Single Image

```python
from karna_vlm.inference import generate
output = generate(model, image=pil_image, prompt="What's happening here?")
```

### Multi-Turn Chat

```python
from karna_vlm.inference.chat import ChatSession

chat = ChatSession(model, system_prompt="You are a helpful assistant.")
response1 = chat.chat("What do you see in this image?", image=photo)
response2 = chat.chat("What color is the car?")  # remembers the image
response3 = chat.chat("Now look at this one", image=photo2)  # new image
```

### Structured Data Extraction

```python
from karna_vlm.inference.structured_output import StructuredExtractor

extractor = StructuredExtractor(model)

# Extract specific fields
result = extractor.extract_key_value(invoice_image, keys=["vendor", "date", "total", "items"])
print(result.data)  # {"vendor": "Acme Inc", "date": "2024-03-15", ...}

# Extract table
result = extractor.extract_table(table_image)
print(result.data)  # [{"Name": "Alice", "Age": "30"}, {"Name": "Bob", "Age": "25"}]

# Open-ended JSON extraction
result = extractor.extract_json(document_image, schema_hint="name, date, amount")
print(f"Confidence: {result.confidence}, Valid JSON: {result.format_valid}")
```

### Visualize Bridge Attention

```python
from karna_vlm.inference.visualize import visualize_bridge_attention

vision_out = model.encode_image([image])
bridge_out = model.bridge_image(vision_out)

heatmap = visualize_bridge_attention(
    bridge_out.attention_weights,
    image,
    query_idx=0,  # which query token to visualize
    output_path="attention_map.png",
)
```

---

## Safety & Governance

### Content Safety

```python
from karna_vlm.safety import SafetyPolicy

policy = SafetyPolicy()
result = policy.check_input("How to make a bomb")
print(result.safe)          # False
print(result.should_block)  # True
print(result.message)       # "I cannot provide instructions..."

# Add custom rules
from karna_vlm.safety.policy import SafetyRule, SafetyAction
policy.add_rule(SafetyRule(
    name="pii_detection",
    patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],  # SSN pattern
    action=SafetyAction.REDACT,
))
```

### Content Filtering

```python
from karna_vlm.safety.filters import ContentFilter

filt = ContentFilter(max_image_size=4096, min_image_size=16)
result = filt.filter_image(image)   # validates dimensions, format
result = filt.filter_output(text)   # detects repetition/hallucination
```

### Model Cards

```python
from karna_vlm.safety.model_card import generate_default_card

card = generate_default_card(model.config)
card.eval_results = {"vqa_accuracy": 0.72, "caption_bleu4": 0.35}
card.save("MODEL_CARD.md")
```

### Dataset Governance

```python
from karna_vlm.data.schemas import DatasetManifest, License, TaskType
from karna_vlm.data.manifests import register_manifest, validate_commercial_safety

manifest = DatasetManifest(
    name="my_dataset",
    license=License.CC_BY_4,
    num_samples=50000,
    task_types=[TaskType.VQA, TaskType.CAPTION],
)
register_manifest(manifest)

# Check commercial safety before training
safety = validate_commercial_safety(["my_dataset", "other_dataset"])
# → {"my_dataset": True, "other_dataset": False}
```

---

## Deployment

### As a Python Library

```python
model = KarnaVLM.from_pretrained("my_model/")
output = model.generate(images=[image], prompt="Describe this.")
```

### As a REST API

```python
from karna_vlm.api import create_app
from karna_vlm.safety import SafetyPolicy

model = KarnaVLM.from_pretrained("my_model/")
app = create_app(model, safety_policy=SafetyPolicy())
# uvicorn app:app --host 0.0.0.0 --port 8080
```

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Generate text from base64 image + prompt |
| `POST` | `/generate/upload` | Generate text from uploaded file |
| `POST` | `/extract` | Structured data extraction |
| `GET` | `/health` | Health check |

### Via CLI

```bash
karna serve --config configs/model_small.yaml --port 8080
```

---

## Project Structure

```
karna-vlm/
├── src/karna_vlm/                  # VLM platform (pip-installable)
│   ├── models/
│   │   ├── vlm_model.py            # KarnaVLM — the main class
│   │   ├── vision/                  # SigLIP, CLIP encoder backends
│   │   ├── bridge/                  # 5 bridge variants (the moat)
│   │   ├── decoder/                 # HuggingFace causal LM wrapper
│   │   ├── projector/               # Structured output + grounding heads
│   │   └── prompt_packing/          # Multimodal sequence assembly
│   ├── training/
│   │   ├── trainer.py               # Core training loop (AMP, grad accum, checkpointing)
│   │   ├── stage1_bootstrap.py      # Bridge-only captioning alignment
│   │   ├── stage2_multitask.py      # Bridge + LoRA instruction tuning
│   │   ├── stage3_domain.py         # Domain specialization → domain pack
│   │   ├── stage4_distill.py        # Quantization + knowledge distillation
│   │   ├── lora.py                  # LoRA management (decoder + bridge)
│   │   ├── checkpointing.py         # Component-level checkpoint manager
│   │   └── optim.py                 # Per-component LR, cosine scheduling
│   ├── data/
│   │   ├── schemas.py               # VLMSample, DatasetManifest, TaskType
│   │   ├── datasets.py              # VLMDataset (JSONL), HFVLMDataset
│   │   ├── mixtures.py              # Weighted multi-task sampling
│   │   ├── collators.py             # Batch collation with packing
│   │   ├── templates.py             # 9 prompt templates (VQA, chat, OCR, etc.)
│   │   └── manifests.py             # Dataset governance + license tracking
│   ├── evaluation/
│   │   ├── caption.py               # BLEU-4, ROUGE-L
│   │   ├── vqa.py                   # Exact match, relaxed, VQA accuracy
│   │   ├── grounding.py             # IoU, precision@50/75
│   │   ├── instruction.py           # Format compliance, relevance
│   │   ├── latency.py               # Throughput, VRAM, component timing
│   │   ├── ablations.py             # Systematic variant comparison
│   │   └── reports.py               # Unified eval reports
│   ├── inference/
│   │   ├── generate.py              # Single + batch generation
│   │   ├── chat.py                  # Multi-turn chat sessions
│   │   ├── structured_output.py     # JSON, key-value, table extraction
│   │   └── visualize.py             # Attention heatmaps, grounding boxes
│   ├── safety/
│   │   ├── policy.py                # Configurable safety rules
│   │   ├── filters.py               # Image + text content filters
│   │   └── model_card.py            # Model card generation
│   ├── api/server.py                # FastAPI inference server
│   ├── utils/                       # Distributed, profiling, logging, I/O
│   └── cli.py                       # CLI entry point
│
├── configs/
│   ├── model_tiny.yaml              # ~600M params, ~2GB VRAM
│   ├── model_small.yaml             # ~1.7B params, ~4GB VRAM
│   ├── model_mid.yaml               # ~3.8B params, ~8GB VRAM
│   ├── train_stage1.yaml            # Stage 1 hyperparameters
│   └── train_stage2.yaml            # Stage 2 hyperparameters
├── tests/                           # Test suite
├── docs/
│   ├── ARCHITECTURE.md              # Detailed architecture docs
│   └── AUDIT.md                     # Full code audit
└── pyproject.toml                   # Package config + deps
```

---

## Configuration

All model and training configs are YAML:

```yaml
# configs/model_small.yaml
model_name: karna-vlm-small
model_family: small
vision_backend: siglip
vision_model: google/siglip-base-patch16-224
vision_freeze: true
bridge_type: qformer_lite       # linear | qformer_lite | resampler | gated | instruction_conditioned
bridge_dim: 512
bridge_num_queries: 64
bridge_num_layers: 4
decoder_model: google/gemma-2-2b
decoder_freeze: true
max_length: 2048
use_lora: false
lora_r: 16
lora_alpha: 32
```

Load and modify at runtime:

```python
config = KarnaVLMConfig.from_yaml("configs/model_small.yaml")
config.bridge_type = "resampler"
config.bridge_num_queries = 32
model = KarnaVLM(config)
```

---

## Design Principles

1. **Bridge is the moat** — most trainable intelligence lives in the bridge, not the decoder
2. **Freeze by default** — vision encoder always frozen, decoder frozen until LoRA
3. **PEFT over full finetuning** — LoRA adapters instead of training billions of parameters
4. **Every component is swappable** — vision, bridge, decoder each behind clean interfaces
5. **Evaluation is first-class** — 6 evaluators + ablation framework + report generation
6. **Safety is built in** — content policy, filters, model cards, dataset governance
7. **Domain packs are portable** — `bridge_weights.pt` + `adapter/` + `config.yaml` = deployable vertical
8. **One GPU is enough** — the whole point is accessible training

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

Apache 2.0

---

## Links

- **GitHub:** [Viraj0518/Karna-VLM](https://github.com/Viraj0518/Karna-VLM)
