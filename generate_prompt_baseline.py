"""
Prompt-based baseline: summarize author style, then generate with style profile.

Phase 1: Feed all training samples to an LLM to produce a ~150-200 word style profile.
Phase 2: Use that profile as system prompt for generation on test prompts.

Usage:
    python generate_prompt_baseline.py \
        --train_pkl benchmarks/cmcc/processed/genre_holdout/cmcc_seen_s0_train.pkl \
        --test_pkl  benchmarks/cmcc/processed/genre_holdout/cmcc_seen_s0_test.pkl \
        --author_key 0 \
        --output_json outputs/genre_holdout_prompt_exp/seen-s0-a0/generations.json \
        --num_samples 3 \
        --backend openai --openai_model gpt-4o
"""

import argparse
import json
import os
import pickle

from tqdm import tqdm

from baseline_utils import (
    add_backend_args,
    get_backend,
    load_author_data,
    write_output_json,
)

SUMMARIZE_PROMPT = """\
Analyze the following writing samples from a single author. Produce a concise \
style profile (150-200 words) that captures the author's distinctive writing \
characteristics. Focus on: tone, vocabulary level, sentence structure, \
formality, use of rhetoric, emotional register, and any recurring patterns.

Writing samples:
{samples}

Style profile:"""


def summarize_style(train_data, backend):
    """Generate a style profile from training samples."""
    samples_text = "\n\n".join(
        f"--- Sample {i+1} ---\nPrompt: {ex['prompt']}\nResponse: {ex['output']}"
        for i, ex in enumerate(train_data)
    )
    prompt = SUMMARIZE_PROMPT.format(samples=samples_text)
    messages = [{"role": "user", "content": prompt}]
    results = backend.generate(messages, num_samples=1, max_new_tokens=512)
    return results[0]


def main():
    parser = argparse.ArgumentParser(description="Prompt-based baseline generation")
    parser.add_argument("--train_pkl", type=str, required=True)
    parser.add_argument("--test_pkl", type=str, required=True)
    parser.add_argument("--author_key", type=int, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--no_cache", action="store_true",
                        help="Skip style profile cache and regenerate")
    add_backend_args(parser)
    add_backend_args(parser, prefix="summarizer")
    args = parser.parse_args()

    # Load data
    train_data = load_author_data(args.train_pkl, args.author_key)
    with open(args.test_pkl, "rb") as f:
        test_data = pickle.load(f)[args.author_key]

    # Phase 1: Style summarization (with caching)
    output_dir = os.path.dirname(args.output_json)
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "style_profile.json")

    if not args.no_cache and os.path.exists(cache_path):
        print(f"Loading cached style profile from {cache_path}")
        with open(cache_path) as f:
            style_profile = json.load(f)["style_profile"]
    else:
        summarizer_backend_name = args.summarizer_backend or args.backend
        summarizer_openai_model = args.summarizer_openai_model or args.openai_model
        summarizer = get_backend(summarizer_backend_name, summarizer_openai_model)
        print(f"Generating style profile with {summarizer_backend_name} backend...")
        style_profile = summarize_style(train_data, summarizer)
        with open(cache_path, "w") as f:
            json.dump({"style_profile": style_profile}, f, indent=2)
        print(f"Cached style profile to {cache_path}")

    print(f"Style profile ({len(style_profile.split())} words):\n{style_profile[:200]}...")

    # Phase 2: Generation
    backend = get_backend(args.backend, args.openai_model)

    system_prompt = (
        "You are a writing assistant. Write in the following style:\n\n"
        f"{style_profile}\n\n"
        "Respond to the user's prompt using this writing style."
    )

    results = []
    for item in tqdm(test_data, desc="Generating"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["prompt"]},
        ]
        generations = backend.generate(messages, num_samples=args.num_samples)
        results.append({
            "prompt": item["prompt"],
            "reference": item.get("output"),
            "generations": generations,
        })

    # Write output
    method_desc = f"prompt-baseline (backend={args.backend})"
    write_output_json(args.output_json, args.author_key, method_desc,
                      args.test_pkl, args.num_samples, results)


if __name__ == "__main__":
    main()
