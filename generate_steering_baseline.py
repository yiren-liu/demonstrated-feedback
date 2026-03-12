"""
Steering-vector baseline for the genre-holdout experiment.

Learns a style direction in activation space from author demonstrations
and applies it during generation.

Pipeline:
1. Load base model (Llama-3.1-8B or Mistral-7B)
2. Load author training data from pickle
3. Generate style-agnostic (negative) responses from base model
4. Train steering vector from (personalized, style_agnostic) pairs
5. Generate steered test outputs
6. Write eval-compatible JSON

Usage:
    python generate_steering_baseline.py \
        --train_pkl benchmarks/cmcc/processed/genre_holdout/cmcc_seen_s0_train.pkl \
        --test_pkl  benchmarks/cmcc/processed/genre_holdout/cmcc_seen_s0_test.pkl \
        --author_key 0 \
        --output_json outputs/genre_holdout_steering_exp/llama-3.1-8b/seen-s0-a0/generations.json \
        --num_samples 3
"""

import argparse
import json
import os
import pickle

import torch
from steering_vectors import train_steering_vector
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline_utils import load_author_data, write_output_json


def is_mistral(model_id):
    return "mistral" in model_id.lower()


def setup_tokenizer(tokenizer, model_id):
    """Configure tokenizer: set Mistral chat template if needed, set pad token."""
    if is_mistral(model_id):
        from generate_scenario import MISTRAL_CHAT_TEMPLATE
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def format_chat(tokenizer, prompt, output=None, add_generation_prompt=True):
    """Format a prompt (and optional output) as a chat string."""
    msgs = [{"role": "user", "content": prompt}]
    if output is not None:
        msgs.append({"role": "assistant", "content": output})
        add_generation_prompt = False
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
    )


def generate_style_agnostic(model, tokenizer, prompts, cache_path=None,
                            max_new_tokens=1024):
    """Generate style-agnostic (negative) responses from the base model.

    One response per prompt, low temperature for stable negatives.
    Results are cached to disk to avoid redundant compute on re-runs.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached style-agnostic responses from {cache_path}")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        if len(cached) == len(prompts):
            return cached
        print(f"  Cache has {len(cached)} entries but need {len(prompts)}, regenerating...")

    print(f"Generating {len(prompts)} style-agnostic responses...")
    responses = []
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            text = format_chat(tokenizer, prompt)
            inputs = tokenizer(
                [text], return_tensors="pt", padding=True, truncation=True,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            gen_text = tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True,
            )
            responses.append(gen_text)
            if (i + 1) % 5 == 0 or i == len(prompts) - 1:
                print(f"  Style-agnostic: {i + 1}/{len(prompts)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(responses, f, indent=2)
        print(f"  Cached style-agnostic responses -> {cache_path}")

    return responses


def train_sv(model, tokenizer, train_samples, style_agnostic_responses,
             layers=None, cache_path=None):
    """Train a steering vector from (personalized, style-agnostic) pairs.

    The steering vector is cached to disk as a .pt file.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached steering vector from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    # Build training pairs: (positive_text, negative_text)
    training_pairs = []
    for sample, neg_response in zip(train_samples, style_agnostic_responses):
        prompt = sample["prompt"]
        pos_output = sample["output"]

        pos_text = format_chat(tokenizer, prompt, output=pos_output)
        neg_text = format_chat(tokenizer, prompt, output=neg_response)

        training_pairs.append((pos_text, neg_text))

    print(f"Training steering vector from {len(training_pairs)} pairs...")
    sv = train_steering_vector(
        model,
        tokenizer,
        training_pairs,
        layers=layers,
        show_progress=True,
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(sv, cache_path)
        print(f"  Cached steering vector -> {cache_path}")

    return sv


def generate_steered(model, tokenizer, steering_vector, prompts, references,
                     num_samples=3, multiplier=0.15, max_new_tokens=1024,
                     temperature=1.0):
    """Generate steered outputs for test prompts."""
    results = []
    with torch.inference_mode():
        with steering_vector.apply(model, multiplier=multiplier):
            for i, prompt in enumerate(prompts):
                text = format_chat(tokenizer, prompt)
                inputs = tokenizer(
                    [text], return_tensors="pt", padding=True, truncation=True,
                ).to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    num_return_sequences=num_samples,
                    pad_token_id=tokenizer.eos_token_id,
                )

                prompt_len = inputs["input_ids"].shape[1]
                gens = []
                for k in range(num_samples):
                    gen_text = tokenizer.decode(
                        outputs[k][prompt_len:], skip_special_tokens=True,
                    )
                    gens.append(gen_text)

                results.append({
                    "prompt": prompt,
                    "reference": references[i],
                    "generations": gens,
                })
                if (i + 1) % 5 == 0 or i == len(prompts) - 1:
                    print(f"  Steered generation: {i + 1}/{len(prompts)} prompts")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Steering-vector baseline for genre-holdout experiment",
    )
    parser.add_argument("--train_pkl", type=str, required=True,
                        help="Path to training pickle file")
    parser.add_argument("--test_pkl", type=str, required=True,
                        help="Path to test pickle file")
    parser.add_argument("--author_key", type=int, required=True,
                        help="Author key in the pickle")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to write generation results as JSON")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to generate per test prompt")
    parser.add_argument("--model_id", type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model HF identifier")
    parser.add_argument("--multiplier", type=float, default=0.15,
                        help="Steering vector multiplier (default: 0.15)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices for steering vector (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for test generation")
    args = parser.parse_args()

    # ── output directory for caching ──
    out_dir = os.path.dirname(args.output_json)
    os.makedirs(out_dir, exist_ok=True)

    # ── load model ──
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer = setup_tokenizer(tokenizer, args.model_id)
    model.eval()

    # ── load author data ──
    train_samples = load_author_data(args.train_pkl, args.author_key)
    print(f"Loaded {len(train_samples)} training samples for author {args.author_key}")

    with open(args.test_pkl, "rb") as f:
        test_data = pickle.load(f)
    test_samples = test_data[args.author_key]
    test_prompts = [item["prompt"] for item in test_samples]
    test_references = [item.get("output") for item in test_samples]
    print(f"Loaded {len(test_prompts)} test prompts")

    # ── generate style-agnostic negatives ──
    train_prompts = [s["prompt"] for s in train_samples]
    sa_cache = os.path.join(out_dir, "style_agnostic_cache.json")
    style_agnostic = generate_style_agnostic(
        model, tokenizer, train_prompts,
        cache_path=sa_cache, max_new_tokens=args.max_new_tokens,
    )

    # ── train steering vector ──
    sv_cache = os.path.join(out_dir, "steering_vector.pt")
    sv = train_sv(
        model, tokenizer, train_samples, style_agnostic,
        layers=args.layers, cache_path=sv_cache,
    )

    # ── generate steered test outputs ──
    print(f"Generating steered outputs (multiplier={args.multiplier})...")
    results = generate_steered(
        model, tokenizer, sv, test_prompts, test_references,
        num_samples=args.num_samples, multiplier=args.multiplier,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )

    # ── write output ──
    method_desc = (
        f"steering-vector (model={args.model_id}, "
        f"multiplier={args.multiplier}, layers={args.layers})"
    )
    write_output_json(
        args.output_json, args.author_key, method_desc,
        args.test_pkl, args.num_samples, results,
    )


if __name__ == "__main__":
    main()
