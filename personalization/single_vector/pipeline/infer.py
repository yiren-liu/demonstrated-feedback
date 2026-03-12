import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch

from single_vector.utils import (
    get_model_and_tokenizer,
    generate_batch,
)
from single_vector.utils.data_utils import make_test_dataset, save_responses


def run_inference(
    dataset: str,
    vector: str,
    model,
    tokenizer,
    data_file: str = "val.json",
    multiplier: float = 0.15,
    test_mode: bool = False,
    batch_size: int = 1,
) -> None:
    """Run inference with steering vector.

    Args:
        dataset: Dataset identifier (e.g., v3-gpt-4o-mini-1-500)
        vector: Vector name (e.g., v5-gpt-5-mini-update_style_vector_sv_layer_all)
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        data_file: Data file to use (e.g., val.json or test.json)
        multiplier: Steering multiplier (default: 0.15)
        test_mode: If True, only process 3 samples and print results
        batch_size: Batch size for generation (default: 1)
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_path = base_dir / "dataset" / dataset / data_file

    # 2. Load test dataset
    chats, references, prompts = make_test_dataset(
        tokenizer, dataset_path)

    # Limit to 3 samples if test mode is enabled
    if test_mode:
        chats = chats[:3]
        references = references[:3]
        prompts = prompts[:3]
        print(f"\nTEST MODE: Processing only 3 samples")

    # 3. Load steering vector
    vector_dir = Path(__file__).resolve().parent.parent / "outputs" / "vectors"
    vector_path = vector_dir / f"{vector}.pt"
    steering_vector = torch.load(str(vector_path), weights_only=False)

    # 4. Define output paths
    # Determine the split name (val or test) from data_file
    split_name = Path(data_file).stem  # e.g., "val" or "test"
    results_dir = Path(__file__).resolve().parent.parent / \
        "outputs" / "results" / split_name

    without_path = results_dir / f"{dataset}_without_steering.csv"
    with_path = results_dir / \
        f"{dataset}_multiplier{str(multiplier)}.csv"

    # Create parent directories for output files (handles nested paths in dataset name)
    without_path.parent.mkdir(parents=True, exist_ok=True)
    with_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if results already exist
    if without_path.exists() and with_path.exists() and not test_mode:
        print(f"\nSkipping generation - results already exist:")
        print(f"  - {without_path}")
        print(f"  - {with_path}")
        return

    # # 5. Generate responses without steering
    # if without_path.exists():
    #     print(
    #         f"\nSkipping generation without steering - results already exist at {without_path}")
    # else:
    #     responses_without_steering = generate_batch(
    #         model=model,
    #         tokenizer=tokenizer,
    #         chats=chats,
    #         description="Generating without steering",
    #         use_steering=False,
    #     )

    #     if not test_mode:
    #         # Save responses without steering
    #         save_responses(
    #             output_path=without_path,
    #             prompts=prompts,
    #             references=references,
    #             predictions=responses_without_steering,
    #         )

    # 6. Generate responses with steering
    if with_path.exists() and not test_mode:
        print(
            f"\nSkipping generation with steering - results already exist at {with_path}")
    else:
        responses_with_steering = generate_batch(
            model=model,
            tokenizer=tokenizer,
            chats=chats,
            description="Generating with steering vector: " + vector,
            use_steering=True,
            steering_vector=steering_vector,
            multiplier=multiplier,
            batch_size=batch_size,
        )

        # Save responses with steering
        if not test_mode:
            save_responses(
                output_path=with_path,
                prompts=prompts,
                references=references,
                predictions=responses_with_steering,
                message="\nSaving responses with steering...",
            )

        # Print test samples if test mode is enabled
        if test_mode:
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
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation (default: 8)")
    args = parser.parse_args()

    # Load model and tokenizer for standalone usage
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer()

    run_inference(
        dataset=args.dataset,
        vector=args.vector,
        model=model,
        tokenizer=tokenizer,
        multiplier=args.multiplier,
        test_mode=args.test,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
