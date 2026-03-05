"""
RAG-based baseline: retrieve top-k similar training examples as few-shot demos.

For each test prompt, embeds it and retrieves the most similar training examples
by cosine similarity, then uses them as in-context demonstrations for generation.

Usage:
    python generate_rag_baseline.py \
        --train_pkl benchmarks/cmcc/processed/genre_holdout/cmcc_seen_s0_train.pkl \
        --test_pkl  benchmarks/cmcc/processed/genre_holdout/cmcc_seen_s0_test.pkl \
        --author_key 0 \
        --output_json outputs/genre_holdout_rag_exp/seen-s0-a0/generations.json \
        --num_samples 3 --top_k 3 \
        --backend openai --openai_model gpt-4o
"""

import argparse
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from baseline_utils import (
    add_backend_args,
    get_backend,
    load_author_data,
    write_output_json,
)


def build_rag_system_prompt(retrieved_examples):
    """Format retrieved examples into a few-shot system prompt."""
    parts = [
        "You are a writing assistant. Match the writing style shown in the "
        "following examples when responding to the user's prompt.\n"
    ]
    for i, ex in enumerate(retrieved_examples, 1):
        parts.append(f"--- Example {i} ---")
        parts.append(f"Prompt: {ex['prompt']}")
        parts.append(f"Response: {ex['output']}\n")
    parts.append(
        "Now respond to the user's prompt in the same writing style as the examples above."
    )
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="RAG-based baseline generation")
    parser.add_argument("--train_pkl", type=str, required=True)
    parser.add_argument("--test_pkl", type=str, required=True)
    parser.add_argument("--author_key", type=int, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    add_backend_args(parser)
    args = parser.parse_args()

    # Load data
    train_data = load_author_data(args.train_pkl, args.author_key)
    with open(args.test_pkl, "rb") as f:
        test_data = pickle.load(f)[args.author_key]

    # Embed training prompts
    print(f"Embedding {len(train_data)} training prompts with {args.embed_model}...")
    embedder = SentenceTransformer(args.embed_model)
    train_prompts = [ex["prompt"] for ex in train_data]
    train_embeddings = embedder.encode(train_prompts, normalize_embeddings=True)

    # Initialize generation backend
    backend = get_backend(args.backend, args.openai_model)

    # Build all message lists (retrieval is fast, do it upfront)
    all_messages = []
    for item in test_data:
        test_prompt = item["prompt"]
        test_emb = embedder.encode([test_prompt], normalize_embeddings=True)
        similarities = np.dot(train_embeddings, test_emb.T).squeeze()
        top_indices = np.argsort(similarities)[-args.top_k:][::-1]
        retrieved = [train_data[i] for i in top_indices]
        system_prompt = build_rag_system_prompt(retrieved)
        all_messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_prompt},
        ])

    # Generate for each test prompt
    results = []
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Generating"):
        generations = backend.generate(all_messages[idx], num_samples=args.num_samples)
        results.append({
            "prompt": item["prompt"],
            "reference": item.get("output"),
            "generations": generations,
        })

    # Write output
    method_desc = f"rag-baseline (backend={args.backend}, top_k={args.top_k}, embed={args.embed_model})"
    write_output_json(args.output_json, args.author_key, method_desc,
                      args.test_pkl, args.num_samples, results)


if __name__ == "__main__":
    main()
