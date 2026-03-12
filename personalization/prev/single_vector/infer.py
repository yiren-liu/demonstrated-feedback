import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch

from utils import (
    get_model_and_tokenizer,
    generate_batch,
)
from utils.data_utils import make_test_dataset, save_responses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference with steering vectors for personalization")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset identifier (e.g., v3-gpt-4o-mini-1-500)")
    parser.add_argument("--vector", type=str, required=True,
                        help="Vector name (e.g., v5-gpt-5-mini-update_style_vector_sv_layer_all)")
    parser.add_argument("--multiplier", type=float, default=0.15,
                        help="Steering multiplier (default: 0.15)")
    parser.add_argument("--test", action="store_true",
                        help="Print 3 sample steered results (reference and prediction)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "dataset" / args.dataset / "val.json"

    # 1. Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # 2. Load test dataset
    chats, references, prompts = make_test_dataset(
        tokenizer, dataset_path)

    # Limit to 5 samples if test mode is enabled
    if args.test:
        chats = chats[:3]
        references = references[:3]
        prompts = prompts[:3]
        print(f"\nTEST MODE: Processing only 3 samples")

    # 3. Load steering vector
    vector_dir = Path(__file__).resolve().parent / "vectors"
    vector_path = vector_dir / f"{args.vector}.pt"
    steering_vector = torch.load(str(vector_path), weights_only=False)
    multiplier = args.multiplier

    # 4. Define output paths
    results_dir = Path(__file__).resolve().parent / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    without_path = results_dir / f"{args.dataset}_without_steering.csv"
    with_path = results_dir / \
        f"{args.vector}_multiplier{str(args.multiplier)}.csv"

    # Check if results already exist
    if without_path.exists() and with_path.exists() and not args.test:
        print(f"\nSkipping generation - results already exist:")
        print(f"  - {without_path}")
        print(f"  - {with_path}")
        return

    # 5. Generate responses without steering
    if without_path.exists():
        print(
            f"\nSkipping generation without steering - results already exist at {without_path}")
    else:
        responses_without_steering = generate_batch(
            model=model,
            tokenizer=tokenizer,
            chats=chats,
            description="Generating without steering",
            use_steering=False,
        )

        if not args.test:
            # Save responses without steering
            save_responses(
                output_path=without_path,
                prompts=prompts,
                references=references,
                predictions=responses_without_steering,
            )

    # 6. Generate responses with steering
    if with_path.exists() and not args.test:
        print(
            f"\nSkipping generation with steering - results already exist at {with_path}")
    else:
        responses_with_steering = generate_batch(
            model=model,
            tokenizer=tokenizer,
            chats=chats,
            description="Generating with steering vector: " + args.vector,
            use_steering=True,
            steering_vector=steering_vector,
            multiplier=multiplier,
        )

        # Save responses with steering
        if not args.test:
            save_responses(
                output_path=with_path,
                prompts=prompts,
                references=references,
                predictions=responses_with_steering,
                message="\nSaving responses with steering...",
            )

        # Print test samples if --test flag is enabled
        if args.test:
            print("\n" + "="*80)
            print("TEST MODE: Displaying steered results")
            print("="*80)
            for i in range(len(responses_with_steering)):
                print(f"\n--- Sample {i+1} ---")
                print(f"Prompt: {prompts[i]}")
                print(f"\nReference: {references[i]}")
                # print(f"\nWithout Steering: {responses_without_steering[i]}")
                print(f"\nWith Steering: {responses_with_steering[i]}")
                print("-" * 80)


if __name__ == "__main__":
    main()
