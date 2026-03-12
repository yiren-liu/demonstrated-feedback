import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .model_utils import build_messages


def _process_train_examples(
    tokenizer,
    examples: List[dict],
):
    dataset: List[Tuple[str, str]] = []

    for example in examples:
        prompt = example.get("prompt")
        pos = example.get("personalized")
        neg = example.get("style_agnostic")

        if not prompt or not pos or not neg:
            continue

        # User personalized text
        pos_msgs = build_messages(prompt, pos)
        pos_text = tokenizer.apply_chat_template(
            pos_msgs, tokenize=False, add_generation_prompt=False
        )

        # LLM generated style-agnostic text
        neg_msgs = build_messages(prompt, neg)
        neg_text = tokenizer.apply_chat_template(
            neg_msgs, tokenize=False, add_generation_prompt=False
        )

        dataset.append((pos_text, neg_text))

    return dataset


def make_train_dataset_from_path(
    tokenizer, dataset_path: Path
):
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return _process_train_examples(tokenizer, data)


def make_train_dataset_from_data(
    tokenizer,
    dataset: List[dict],
):
    """Create training dataset from cluster examples."""
    return _process_train_examples(tokenizer, dataset)


def load_train_rows(dataset_path: Path):
    prompts: List[str] = []
    personalized: List[str] = []
    style_agnostic: List[str] = []

    with dataset_path.open(encoding="utf-8") as jsonfile:
        records = json.load(jsonfile)

    for row in records:
        prompt = row.get("prompt")
        personalized_text = row.get("personalized")
        style_agnostic_text = row.get("style_agnostic")

        prompts.append(prompt)
        personalized.append(personalized_text)
        style_agnostic.append(style_agnostic_text)

    return {
        "prompts": prompts,
        "personalized": personalized,
        "style_agnostic": style_agnostic,
    }


def make_test_dataset(
    tokenizer,
    dataset_path: Path,
):
    """Load and process test dataset for inference."""
    dataset: List[Tuple[str, str, str]] = []
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
        for row in data:
            prompt = row.get("prompt")
            pos = row.get("personalized")

            msgs = build_messages(prompt)
            chat = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

            dataset.append((chat, pos, prompt))

    print(f"Loaded {len(dataset)} test samples.")
    chats, references, prompts = zip(*dataset)
    return list(chats), list(references), list(prompts)


def save_responses(
    output_path: Path,
    prompts: List[str],
    references: List[str],
    predictions: List[str],
    selection_reasons: Optional[List[str]] = None,
    message: Optional[str] = None,
):
    """Save model responses to a CSV file."""
    if message is None:
        if selection_reasons:
            message = f"\nSaving responses with cluster steering to {output_path}..."
        else:
            message = "\nSaving responses without steering..."

    print(message)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if selection_reasons:
        # Save with selection reasons
        for i, (prediction, reference, reason) in enumerate(zip(predictions, references, selection_reasons)):
            rows.append({
                "index": i,
                "prompt": prompts[i],
                "reference": reference,
                "prediction": prediction,
                "selected_vector": reason,
            })
        fieldnames = ["index", "prompt", "reference",
                      "prediction", "selected_vector"]
    else:
        # Save without selection reasons
        for i, (prediction, reference) in enumerate(zip(predictions, references)):
            rows.append({
                "index": i,
                "prompt": prompts[i],
                "reference": reference,
                "prediction": prediction,
            })
        fieldnames = ["index", "prompt", "reference", "prediction"]

    with output_path.open("w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
