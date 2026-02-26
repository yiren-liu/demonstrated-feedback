# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DITTO (Demonstration ITerated Task Optimization)** — a method for aligning language models with demonstrated feedback from the paper "Show, Don't Tell: Aligning Language Models with Demonstrated Feedback" (Shaikh, Lam et al., Stanford). DITTO enables few-shot (<10 demonstrations) alignment by treating user demonstrations as preferred examples, generating synthetic comparison data via the model's own policy, and optimizing with DPO.

## Key Commands

### Setup
```bash
conda create -n ditto python=3.10 && conda activate ditto
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
cd alignment-handbook && git checkout 606d2e954fd17999af40e6fb4f712055ca11b2f0 && pip install . && cd ..
pip install -r requirements.txt
```

### Train a single DITTO model
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file configs/multi_gpu.yaml \
    scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml \
    --train_pkl=benchmarks/cmcc/processed/cmcc_seen_s0_train.pkl \
    --train_author_key=0 \
    --output_dir=outputs/scenario_experiment/seen-s0-a0
```

### Generate from a trained model
```bash
CUDA_VISIBLE_DEVICES=0 python generate_scenario.py \
    --model_dir outputs/scenario_experiment/seen-s0-a0/ditto \
    --test_pkl benchmarks/cmcc/processed/cmcc_seen_s0_test.pkl \
    --author_key 0 \
    --output_json outputs/scenario_experiment/seen-s0-a0/generations.json \
    --num_samples 3
```

### Evaluate with GPT-4
```bash
python eval_scenario.py --batch \
    --results_dir outputs/scenario_experiment \
    --output_csv outputs/scenario_experiment/eval_results.csv
```

### Run full scenario experiment
```bash
bash run_scenario_experiment.sh          # full experiment (all authors × conditions)
bash run_scenario_experiment.sh --dry    # print commands without executing
```

Environment variables for `run_scenario_experiment.sh`:
- `DATA_DIR` — path to processed pickle files (default: `benchmarks/cmcc/processed`)
- `UNSEEN_GT_TAGS` — space-separated genre-topic holdout tags

## Architecture

### Training Pipeline (two-phase)

1. **SFT Phase** (`scripts/sft_trainer.py`): Supervised fine-tuning on author demonstrations using LoRA. Early-stops when loss drops below `sft_stop_loss` (default 1.0). Creates an "sft" LoRA adapter.

2. **DITTO Phase** (`scripts/ditto_trainer.py`): DPO-based preference optimization. Initializes a "ditto" LoRA adapter from the SFT adapter, then iteratively:
   - Generates samples from the current policy
   - Constructs comparison pairs with three mixing ratios: expert vs current (70%), expert vs replay (20%), inter-policy (10%)
   - Runs DPO training on the synthetic comparisons
   - `ResampleCallback` regenerates comparisons at configurable intervals

`scripts/run_ditto.py` orchestrates both phases. `scripts/dataset_utils.py` provides `DPODataCollatorWithPadding` for online comparison generation and tokenization.

### Data Format

All datasets use pickle files with structure: `{author_id: [{"prompt": str, "output": str}, ...]}`. Preprocessing scripts in `benchmarks/` create scenario-aware train/test splits.

### Experiment Conditions (Scenario Experiment)

| Condition | Description | Variants |
|-----------|-------------|----------|
| **SEEN** | Random train/test split | 3 seeds (s0, s1, s2) |
| **UNSEEN** | Held-out topics never in training | 3 holdout pairs: is (Iraq, Gender Disc.), cg (Catholic Church, Gay Marriage), mp (Privacy, Marijuana) |
| **UNSEEN-GT** | Held-out (genre, topic) atomic tasks | Tag-based, e.g. "bi-cm-dp-sg" |

Each condition trains 10 authors (IDs 0-9), generating 3 samples per test prompt, evaluated by GPT-4 on a 1-5 style-matching scale.

### Config

`configs/ditto-mistral-7b-instruct.yaml` — all hyperparameters for both SFT and DITTO phases. Base model is `mistralai/Mistral-7B-Instruct-v0.2` with LoRA (r=32, alpha=64). `configs/multi_gpu.yaml` — accelerate config (single GPU, bf16).

### Output Structure
```
outputs/scenario_experiment/{condition}-{variant}-a{author}/
├── ditto/              # LoRA adapter weights
├── generations.json    # model outputs (prompt + reference + generations)
└── eval_results.csv    # GPT-4 ratings
```

### Key Dependencies

Pinned versions that matter: `transformers==4.41.1`, `trl==0.8.6`, `peft==0.9.0`, `accelerate==0.30.1`. The `alignment-handbook` git submodule (HuggingFace) must be installed at commit `606d2e9`.

### API Keys

OpenAI API key (for GPT-4 evaluation) is loaded from `.env`. Not needed for training or generation.
