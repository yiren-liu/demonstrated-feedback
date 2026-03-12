import argparse
import torch
from pathlib import Path
from typing import Optional

# Steering-vectors API
from steering_vectors import SteeringVector, train_steering_vector

from single_vector.utils import get_model_and_tokenizer, make_train_dataset_from_path


def run_extraction(
    dataset: str,
    model,
    tokenizer,
    layers: Optional[list[int]] = None,
) -> str:
    """Run extraction for single vector.

    Args:
        dataset: Dataset identifier (e.g., v3-gpt-4o-mini-1-500)
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        layers: Specific layers to train on (None for all layers)

    Returns:
        Path to the saved vector file
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_path = base_dir / "dataset" / dataset / "train.json"

    # Check if output already exists
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "vectors"
    name = (
        "layer_all"
        if layers is None
        else f"layer_{'_'.join(map(str, layers))}"
    )
    out_path = out_dir / f"{dataset}_style_vector_sv_{name}.pt"

    if out_path.exists():
        print(f"Steering vector already exists at {out_path}")
        print("Skipping extraction.")
        return str(out_path)

    # 2. Load dataset and prepare positive/negative pairs
    training_samples = make_train_dataset_from_path(tokenizer, dataset_path)
    print(f"Loaded {len(training_samples)} training samples.")

    # 3. Train steering vector
    print("Training steering vector with steering-vectors...")
    steering_vector: SteeringVector = train_steering_vector(
        model,
        tokenizer,
        training_samples,
        layers=layers,
        show_progress=True,
        # aggregator=fp32_pca_aggregator()
    )

    # 4. Save the steering vector object to disk (PyTorch serialization)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(steering_vector, out_path)
    print(f"Saved steering vector to {out_path}")

    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract steering vectors for personalization")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset identifier (e.g., v3-gpt-4o-mini-1-500)")
    parser.add_argument("--layers", type=int, nargs="+", default=[],
                        help="Layers to train the steering vector on (e.g., 15 20)")
    args = parser.parse_args()

    dataset = args.dataset
    layers = args.layers if args.layers else None

    # Load model and tokenizer for standalone usage
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer()

    run_extraction(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
    )


if __name__ == "__main__":
    main()
