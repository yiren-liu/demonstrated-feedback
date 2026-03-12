import argparse
import torch
from pathlib import Path
import torch.nn.functional as F

from utils import get_model_and_tokenizer


def analyze_steering_vector(steering_vector, model, tokenizer, multiplier=1.0, top_k=50):
    """
    Analyze steering vector by running a forward pass with minimal input
    and examining the output probability distribution with and without steering.
    """
    device = model.device

    # Minimal prompt - just the beginning of assistant response
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat, return_tensors="pt",
                       add_special_tokens=False).to(device)

    with torch.no_grad():
        # Run forward pass WITHOUT steering
        outputs_without = model(**inputs)
        logits_without = outputs_without.logits[0, -1, :]  # [vocab_size]

        # Run forward pass WITH steering
        with steering_vector.apply(model, multiplier=multiplier):
            outputs_with = model(**inputs)
            logits_with = outputs_with.logits[0, -1, :]  # [vocab_size]

    # Compute logit differences
    logit_diff = logits_with - logits_without  # [vocab_size]

    # Get top-k tokens with largest positive change (most boosted)
    top_diff_values, top_diff_indices = torch.topk(logit_diff, k=top_k)
    boosted_tokens = [
        (tokenizer.decode([idx.item()]), diff.item(),
         logits_without[idx].item(), logits_with[idx].item())
        for idx, diff in zip(top_diff_indices, top_diff_values)
    ]

    # Get top-k tokens with largest negative change (most suppressed)
    bottom_diff_values, bottom_diff_indices = torch.topk(
        logit_diff, k=top_k, largest=False)
    suppressed_tokens = [
        (tokenizer.decode([idx.item()]), diff.item(),
         logits_without[idx].item(), logits_with[idx].item())
        for idx, diff in zip(bottom_diff_indices, bottom_diff_values)
    ]

    return boosted_tokens, suppressed_tokens


def print_analysis(boosted_tokens, suppressed_tokens, top_n=30):
    """Pretty print the analysis results."""

    print("\n" + "="*80)
    print("STEERING VECTOR TOKEN ANALYSIS - LOGIT CHANGES")
    print("="*80)

    print(f"\nTop {top_n} MOST BOOSTED tokens (largest positive change):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Token':<30} {'Change':<12} {'Without':<12} {'With':<12}")
    print("-" * 100)

    for i, (token, diff, logit_without, logit_with) in enumerate(boosted_tokens[:top_n], 1):
        display_token = repr(token)[1:-1]
        print(
            f"{i:<6} {display_token:<30} {diff:>+11.4f} {logit_without:>11.4f} {logit_with:>11.4f}")

    print(f"\nTop {top_n} MOST SUPPRESSED tokens (largest negative change):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Token':<30} {'Change':<12} {'Without':<12} {'With':<12}")
    print("-" * 100)

    for i, (token, diff, logit_without, logit_with) in enumerate(suppressed_tokens[:top_n], 1):
        display_token = repr(token)[1:-1]
        print(
            f"{i:<6} {display_token:<30} {diff:>+11.4f} {logit_without:>11.4f} {logit_with:>11.4f}")

    print("="*100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze what tokens a steering vector pushes toward")
    parser.add_argument("--vector", type=str, required=True,
                        help="Vector name (e.g., basics_style_vector_sv_layer_all)")

    args = parser.parse_args()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer()

    # Load steering vector
    vector_dir = Path(__file__).resolve().parent / "vectors"
    vector_path = vector_dir / f"{args.vector}.pt"

    if not vector_path.exists():
        raise FileNotFoundError(f"Steering vector not found: {vector_path}")

    print(f"Loading steering vector from: {vector_path}")
    steering_vector = torch.load(str(vector_path), weights_only=False)

    # Analyze the steering vector
    print(f"\nAnalyzing steering vector...")
    boosted_tokens, suppressed_tokens = analyze_steering_vector(
        steering_vector=steering_vector,
        model=model,
        tokenizer=tokenizer,
        multiplier=0.15,
        top_k=100
    )

    # Print results
    print_analysis(boosted_tokens, suppressed_tokens, top_n=30)


if __name__ == "__main__":
    main()
