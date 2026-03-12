import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from utils import get_model_and_tokenizer, build_messages, generate_text


def main():
    # Configuration
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_path = Path(__file__).resolve().parent / \
        "v5-gpt-5-mini-update-v2" / "dataset.json"

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} entries")

    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    model, tokenizer = get_model_and_tokenizer(model_name)

    # Process each entry in the dataset
    print("\nGenerating style_agnostic text for each prompt...")
    for i, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        # Skip if style_agnostic already exists
        if "style_agnostic" in entry and entry["style_agnostic"]:
            continue

        # Get the prompt
        prompt = entry.get("prompt", "")
        if not prompt:
            print(f"\nWarning: Entry {i+1} has no prompt, skipping...")
            continue

        # Build messages using existing utility
        messages = build_messages(prompt)

        # Convert to chat format
        chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate style-agnostic text using existing utility
        try:
            style_agnostic = generate_text(
                model=model,
                tokenizer=tokenizer,
                chat=chat,
                use_steering=False,
            )
            entry["style_agnostic"] = style_agnostic

            # Save after each generation to avoid losing progress
            if (i + 1) % 10 == 0:  # Save every 10 entries
                with open(dataset_path, 'w') as f:
                    json.dump(dataset, f, indent=2)

        except Exception as e:
            print(f"\nError processing entry {i+1}: {e}")
            continue

    # Final save
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"\nCompleted! Dataset saved to {dataset_path}")
    print(f"Processed {len(dataset)} entries.")


if __name__ == "__main__":
    main()
