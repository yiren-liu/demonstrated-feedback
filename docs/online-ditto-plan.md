# Online DITTO: Real-Time Personalization from 2-3 Writing Samples

**Goal**: Adapt DITTO into an online learning system where a user submits 2-3 writing samples and receives a personalized model within 5 minutes.

**Status**: Draft plan / Feasibility analysis

---

## 1. Current Pipeline Profile

Reference: the default config (`configs/ditto-mistral-7b-instruct.yaml`) running on a single GPU.

| Phase | Steps | Wall-clock (A100) | Bottleneck |
|-------|-------|--------------------|------------|
| Model load | — | 30-60 s | Disk I/O / HF Hub |
| SFT (LoRA) | 30 (early-stops ~15) | 2-3 min | Moderate; small data converges fast |
| Resample ×4 | 10 completions/prompt × 1024 tok | 12-20 min | **Dominant** (~60-70% of total) |
| DPO training | 40 steps | 5-8 min | Reference log-prob doubles fwd cost |
| Save adapter | — | <5 s | Negligible (~100-200 MB LoRA) |
| **Total** | | **~20-30 min** | Generation is the bottleneck |

### Why generation is slow

`dataset_utils.py:107` — hardcoded batch size of **2 prompts**, each producing `num_return_sequences=10` completions of up to **1024 tokens** via naive `model.generate()`. With 4 resample rounds (`resample_rate=10`, `ditto_max_steps=40`), this runs 4× for every training job.

---

## 2. Target Budget

| Constraint | Value |
|------------|-------|
| User-facing latency | **< 5 min** end-to-end (< 2 min stretch goal) |
| Input data | 2-3 (prompt, output) demonstrations |
| Hardware | 1-2× A100-80 GB (or equivalent: H100, 4× A6000) |
| Quality floor | No worse than 90% of offline DITTO on style-match score |

---

## 3. Optimization Plan

### 3.1 Tier 1 — Config Knob Changes (no code edits, no quality risk)

These are pure wins that exploit the small data regime (2-3 samples).

| Knob | Default | Online | Rationale |
|------|---------|--------|-----------|
| `max_steps` (SFT) | 30 | **10** | 2-3 examples memorized in <10 steps |
| `sft_stop_loss` | 1.0 | **1.2** | Relax threshold to stop earlier |
| `ditto_max_steps` | 40 | **20** | Halve DPO steps |
| `resample_rate` | 10 | **10** | 2 resample rounds instead of 4 |
| `bootstrap_count` | 10 | **4** | 60% fewer generations per prompt |
| `per_device_train_batch_size` | 4 | 4 | Keep same |
| `ditto_per_device_train_batch_size` | 4 | 4 | Keep same |

**Expected savings**: ~50% wall-clock reduction (15 min → ~8 min). Not enough alone.

### 3.2 Tier 2 — Targeted Code Changes (moderate effort, high impact)

#### A. Increase generation batch size

**File**: `scripts/dataset_utils.py:107`

```python
# BEFORE (hardcoded batch=2)
for i in tqdm(range(0, len(prompt_text), 2), desc="Generating"):
    batch_prompts = prompt_text[i:i+2]

# AFTER (configurable, default to all prompts at once for small N)
gen_batch_size = min(len(prompt_text), self.gen_batch_size)  # e.g. 8
for i in tqdm(range(0, len(prompt_text), gen_batch_size), desc="Generating"):
    batch_prompts = prompt_text[i:i+gen_batch_size]
```

With 2-3 prompts and `bootstrap_count=4`, this produces only 8-12 sequences per resample — fits easily in a single batch on 80 GB.

**Impact**: Eliminates the generation loop entirely for 2-3 samples. ~2× speedup on generation.

#### B. Reduce max generation length

**File**: `scripts/dataset_utils.py:98`

```python
# BEFORE
max_gen_tokens = 1024

# AFTER — configurable, default to match typical output length
max_gen_tokens = self.max_gen_tokens  # set to 512 or match median demo length
```

Most writing demonstrations are 200-500 tokens. Generating 1024 tokens wastes compute on padding/early-EOS sequences.

**Impact**: ~1.5-2× generation speedup.

#### C. Pre-load model (keep warm)

Create a persistent model server process that holds the base Mistral-7B in GPU memory. New training jobs attach a fresh LoRA adapter without reloading the base model.

```python
# Sketch: model_server.py
class WarmModelPool:
    def __init__(self, model_name, num_gpus=1):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_session(self, user_id: str) -> TrainingSession:
        """Attach fresh LoRA adapters to the shared base model."""
        # Note: need locking for single-GPU; or dedicated GPU per session
        ...
```

**Impact**: Saves 30-60 s per request (one-time load amortized to zero).

#### D. vLLM for generation steps

Replace `model.generate()` in `resample()` with a vLLM-backed generation call. vLLM's paged attention and continuous batching yields 3-5× throughput over HF generate.

Two integration approaches:

| Approach | Complexity | Speedup |
|----------|-----------|---------|
| **Offline**: export LoRA, call vLLM subprocess | Low | 3-4× but adds export overhead |
| **In-process**: use vLLM's `LLM` class with LoRA support | Medium | 4-5×, no export overhead |

vLLM natively supports LoRA adapters since v0.4+. The in-process approach is preferred:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def resample_vllm(self, step):
    sampling = SamplingParams(
        temperature=1.0, top_k=50,
        max_tokens=self.max_gen_tokens, n=self.bootstrap_count
    )
    outputs = self.vllm_engine.generate(
        prompts=prompt_text,
        sampling_params=sampling,
        lora_request=LoRARequest("ditto", 1, self.lora_path)
    )
```

**Caveat**: vLLM holds its own copy of model weights, so this requires either (a) a separate generation GPU, or (b) unloading the training model during generation.

**Impact**: 3-5× generation speedup. Combined with Tier 1 + 2A/2B, generation drops from ~15 min to ~1-2 min.

### 3.3 Tier 3 — Architecture Changes (higher effort, enables scaling)

#### E. Dual-GPU pipeline (train on GPU0, generate on GPU1)

```
GPU 0 (Training)          GPU 1 (Generation / vLLM)
─────────────────         ─────────────────────────
Load base model           Load base model + vLLM engine
SFT (10 steps)            [idle]
  → export LoRA ─────────→ Load LoRA, generate samples
DPO (10 steps)  ←──────── Return comparisons
  → export LoRA ─────────→ Resample with updated policy
DPO (10 steps)  ←──────── Return comparisons
Save final adapter        [idle]
```

This overlaps generation with training, further hiding the generation latency.

**Impact**: Near-complete overlap of gen and training. Total time dominated by the slower of the two.

#### F. Smaller / quantized base model

| Model | Params | Gen speed vs 7B | Quality risk |
|-------|--------|-----------------|--------------|
| Mistral-7B (bf16) | 7B | 1× (baseline) | None |
| Mistral-7B (AWQ 4-bit) | 7B | ~1.5-2× | Low (generation only) |
| Phi-3-mini | 3.8B | ~2× | Moderate — needs validation |
| Gemma-2-2B | 2B | ~3-4× | Higher — needs validation |

For the generation step only, 4-bit quantization is low-risk since we're sampling diverse completions, not doing precision tasks.

#### G. Adapter serving for multi-user scale

For serving personalized models to many users simultaneously:

| Users | Strategy | Memory overhead |
|-------|----------|-----------------|
| 1-10 | Hot LoRA adapters in GPU memory | ~1 GB total |
| 10-100 | Adapter bank with LRU eviction | ~10 GB + CPU swap |
| 100-1000 | S-LoRA / Punica batched multi-LoRA serving | Shared base + paged adapters |
| 1000+ | Adapter store (S3/Redis) + on-demand loading | Negligible GPU per cold user |

Reference: [S-LoRA (Sheng et al. 2023)](https://arxiv.org/abs/2311.03285) — batched serving of thousands of LoRA adapters on shared base models.

---

## 4. Projected Timeline

### Conservative estimate (Tier 1 + 2A + 2B + 2C, single A100-80GB)

| Phase | Time |
|-------|------|
| Model load | 0 s (pre-loaded) |
| SFT (10 steps, 2-3 samples) | ~30 s |
| Resample #1 (4 completions × 2-3 prompts × 512 tok, batched) | ~45 s |
| DPO steps 1-10 | ~30 s |
| Resample #2 (same) | ~45 s |
| DPO steps 11-20 | ~30 s |
| Save adapter | ~3 s |
| **Total** | **~3 min** |

### Aggressive estimate (+ Tier 2D vLLM + Tier 3E dual-GPU)

| Phase | Time |
|-------|------|
| Model load | 0 s (pre-loaded) |
| SFT (10 steps) | ~30 s |
| Resample #1 via vLLM (overlapped) | ~15 s |
| DPO steps 1-10 | ~25 s |
| Resample #2 via vLLM (overlapped) | ~15 s |
| DPO steps 11-20 | ~25 s |
| **Total** | **~1.5-2 min** |

---

## 5. Quality Validation Plan

Reducing steps and bootstrap count may degrade alignment quality. Validation is essential before deploying any online config.

### Ablation matrix

| Experiment | SFT steps | DPO steps | Resamples | Bootstrap | Expected time |
|------------|-----------|-----------|-----------|-----------|---------------|
| Baseline (offline) | 30 | 40 | 4 | 10 | ~25 min |
| Online-A | 10 | 20 | 2 | 4 | ~3 min |
| Online-B | 10 | 15 | 2 | 3 | ~2.5 min |
| Online-C | 10 | 10 | 1 | 4 | ~1.5 min |
| Minimal | 5 | 10 | 1 | 2 | ~1 min |

**Evaluation protocol**: Run each config on 10 CMCC authors, 3 seeds (SEEN condition). Compare GPT-4 style-match scores (1-5 scale) against the offline baseline. Accept configs scoring >= 90% of baseline mean.

### Metrics to track

- GPT-4 style-match score (primary)
- Training wall-clock time (primary)
- Generation diversity (self-BLEU across bootstrap samples)
- DPO loss convergence (does it plateau before `max_steps`?)

---

## 6. Implementation Roadmap

### Phase 1: Validate reduced configs (1-2 days)

1. Create `configs/ditto-online.yaml` with Tier 1 knobs
2. Run ablation matrix on CMCC SEEN-s0 across 10 authors
3. Compare quality scores against offline baseline
4. Identify the Pareto-optimal (quality, latency) config

### Phase 2: Code optimizations (2-3 days)

1. Make generation batch size configurable in `dataset_utils.py`
2. Make `max_gen_tokens` configurable (pass from YAML config)
3. Add adaptive `max_gen_tokens` based on median demo length
4. Create `WarmModelPool` for persistent base model
5. Benchmark each change independently

### Phase 3: vLLM integration (3-5 days)

1. Add vLLM as optional dependency
2. Implement `resample_vllm()` in `dataset_utils.py` as alternative to `resample()`
3. Handle LoRA adapter handoff between training (PEFT) and generation (vLLM)
4. Benchmark end-to-end latency
5. Address weight synchronization between training updates and vLLM LoRA

### Phase 4: Online serving layer (1 week)

1. Build API wrapper (`FastAPI` or `Ray Serve`)
2. Implement adapter bank with LRU eviction
3. Request queuing and GPU scheduling
4. End-to-end integration test: upload samples → wait → generate with personalized model
5. Load testing with concurrent users

### Phase 5: Scale-out (if needed)

1. Multi-GPU pipeline (train + generate in parallel)
2. S-LoRA integration for high-concurrency serving
3. Adapter persistence (save to object store, reload on demand)

---

## 7. Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Quality degrades with fewer steps/bootstrap | Medium | High | Ablation study in Phase 1; fall back to conservative config |
| vLLM LoRA sync issues with PEFT | Medium | Medium | Keep HF generate as fallback; vLLM LoRA well-supported since v0.4 |
| GPU memory pressure with concurrent users | Low (1-10 users) / High (100+) | High | Adapter eviction + S-LoRA for scale |
| 2-3 samples insufficient for some users | Medium | Medium | Detect low-quality outputs, prompt for more samples |
| Base model updates break LoRA adapters | Low | High | Pin model version; version adapters with base model hash |

---

## 8. Open Questions

1. **Quality floor**: What's the minimum DPO steps / bootstrap count before quality drops noticeably? (Answered by Phase 1 ablation.)
2. **Warm start**: Can we skip SFT entirely and initialize DITTO from the base model for even faster adaptation? (Some evidence this works for in-distribution tasks.)
3. **Incremental updates**: If a user submits additional samples later, can we continue training from the existing adapter instead of retraining from scratch?
4. **Model choice**: Is Mistral-7B the right base for online use, or would a faster model (Phi-3, Gemma-2) with comparable quality be better?
5. **Serving topology**: Single large GPU vs. multiple smaller GPUs for handling concurrent personalization requests?
