import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from vanilla_llm.utils import (
    get_model_and_tokenizer,
    generate_batch,
)
from vanilla_llm.utils.data_utils import make_test_dataset, save_responses


def run_inference(
    dataset: str,
    model,
    tokenizer,
    data_file: str = "val.json",
    test_mode: bool = False,
    n_shot: int = 0,
) -> None:
    """Run vanilla LLM inference without steering."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_path = base_dir / "dataset" / dataset / data_file
    train_path = base_dir / "dataset" / dataset / "train.json"

    # Load test dataset
    chats, references, prompts = make_test_dataset(
        tokenizer, dataset_path, train_path, n_shot)

    # Limit to 3 samples if test mode is enabled
    if test_mode:
        chats = chats[:3]
        references = references[:3]
        prompts = prompts[:3]
        print(f"\nTEST MODE: Processing only 3 samples")

    # Define output paths
    # Determine the split name (val or test) from data_file
    split_name = Path(data_file).stem  # e.g., "val" or "test"
    results_dir = Path(__file__).resolve().parent.parent / \
        "outputs" / "results" / split_name

    # Include n_shot in filename if using few-shot prompting
    shot_suffix = f"_{n_shot}shot" if n_shot > 0 else ""
    output_path = results_dir / f"{dataset}_vanilla{shot_suffix}.csv"

    # Create parent directories for output files (handles nested paths in dataset name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if results already exist
    if output_path.exists() and not test_mode:
        print(f"\nSkipping generation - results already exist:")
        print(f"  - {output_path}")
        return

    # Generate responses
    responses = generate_batch(
        model=model,
        tokenizer=tokenizer,
        chats=chats,
        description="Generating responses",
    )

    if not test_mode:
        # Save responses
        save_responses(
            output_path=output_path,
            prompts=prompts,
            references=references,
            predictions=responses,
        )
    else:
        # Print test samples if test mode is enabled
        print("\n" + "="*80)
        print("TEST MODE: Displaying results")
        print("="*80)
        for i in range(len(responses)):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {prompts[i]}")
            print(f"\nReference: {references[i]}")
            print(f"\nGenerated: {responses[i]}")
            print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vanilla LLM inference without steering")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset identifier (e.g., email/persona_9)")
    parser.add_argument("--test", action="store_true",
                        help="Print 3 sample results (reference and prediction)")
    args = parser.parse_args()

    # Load model and tokenizer for standalone usage
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer()

    run_inference(
        dataset=args.dataset,
        model=model,
        tokenizer=tokenizer,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()
