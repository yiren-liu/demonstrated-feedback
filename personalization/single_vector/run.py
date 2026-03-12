import argparse
import logging
from datetime import datetime
from pathlib import Path

# Import the pipeline run functions
from single_vector.pipeline.extract import run_extraction
from single_vector.pipeline.infer import run_inference
from single_vector.utils.model_utils import get_model_and_tokenizer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_extraction_step(dataset, layers, model, tokenizer, logger):
    """Run the vector extraction pipeline step."""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: VECTOR EXTRACTION")
    logger.info("="*70)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Layers: {layers if layers else 'all'}")

    vectors_path = run_extraction(
        dataset=dataset,
        layers=layers,
        model=model,
        tokenizer=tokenizer,
    )

    logger.info("✓ Vector extraction completed successfully")
    logger.info(f"Vectors saved to: {vectors_path}")

    return vectors_path


def run_inference_step(dataset, vector, multiplier, test_mode, model, tokenizer, data_file, step_name, logger, batch_size):
    """Run the inference pipeline step.

    Args:
        dataset: Dataset identifier (e.g., v3-gpt-4o-mini-1-500)
        vector: Vector name (e.g., v5-gpt-5-mini-update_style_vector_sv_layer_all)
        multiplier: Steering multiplier (default: 0.15)
        test_mode: If True, only process 3 samples and print results
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        data_file: Data file to use (e.g., val.json or test.json)
        step_name: Name of the step (e.g., "STEP 2: INFERENCE ON VAL")
        logger: Logger object
        batch_size: Batch size for generation (default: 8)
    """
    logger.info("\n" + "="*70)
    logger.info(f"{step_name}")
    logger.info("="*70)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Vector: {vector}")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Multiplier: {multiplier}")
    logger.info(f"Test mode: {test_mode}")
    logger.info(f"Batch size: {batch_size}")
    run_inference(
        dataset=dataset,
        vector=vector,
        data_file=data_file,
        multiplier=multiplier,
        test_mode=test_mode,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    logger.info(f"✓ {step_name} completed successfully")


def get_personas_from_folder(folder_path):
    """Get list of persona folders from a dataset folder."""
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    personas = []
    for item in sorted(folder.iterdir()):
        if item.is_dir() and item.name.startswith("persona_"):
            # Check if train.json exists
            if (item / "train.json").exists():
                personas.append(item.name)

    return personas


def main():
    parser = argparse.ArgumentParser(
        description="Run single-vector pipeline: extraction → inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset and general settings
    parser.add_argument("--dataset", type=str, default="email",
                        help="Dataset identifier or folder path (e.g., 'email' for dataset/email/)")
    parser.add_argument("--vector", type=str, default=None,
                        help="Vector name for inference (required if --skip-extract is used)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue to next step even if current step fails")

    # Pipeline control
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip vector extraction step")
    parser.add_argument("--skip-infer-val", action="store_true",
                        help="Skip inference on validation set")
    parser.add_argument("--skip-infer-test", action="store_true",
                        help="Skip inference on test set")

    # Extraction arguments
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers for steering vectors (default: all layers)")

    # Inference arguments
    parser.add_argument("--multiplier", type=float, default=0.15,
                        help="Steering vector multiplier (default: 0.15)")
    parser.add_argument("--test", action="store_true",
                        help="Run inference in test mode (3 samples)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation (default: 8)")

    args = parser.parse_args()
    logger = setup_logging()

    # Validate batch size is greater than 0
    if args.batch_size <= 0:
        logger.error("Error: --batch-size must be greater than 0.")
        return 1

    # Validate that at least one step is enabled
    if args.skip_extract and args.skip_infer_val and args.skip_infer_test:
        logger.error("Error: All pipeline steps are skipped. Nothing to do.")
        return 1

    # Validate vector is provided when extraction is skipped
    if args.skip_extract and not args.vector:
        logger.error(
            "Error: --vector is required when --skip-extract is used.")
        return 1

    # Check if dataset is a folder containing personas
    base_dir = Path(__file__).resolve().parent.parent
    dataset_folder = base_dir / "dataset" / args.dataset

    if dataset_folder.exists() and dataset_folder.is_dir():
        # Check if this is a folder with personas
        personas = get_personas_from_folder(dataset_folder)
        if personas:
            logger.info(f"\n" + "="*70)
            logger.info(
                f"Found {len(personas)} personas in folder: {args.dataset}")
            logger.info(f"Personas: {', '.join(personas)}")
            logger.info("="*70)

            # Process each persona
            failed_personas = []
            for persona in personas:
                persona_dataset = f"{args.dataset}/{persona}"
                logger.info(f"\n{'='*70}")
                logger.info(f"Processing {persona_dataset}")
                logger.info(f"{'='*70}")

                try:
                    result = run_persona_pipeline(
                        dataset=persona_dataset,
                        args=args,
                        logger=logger,
                        # batch_size=args.batch_size,
                    )
                    if result != 0:
                        failed_personas.append(persona)
                        if not args.continue_on_error:
                            return result
                except Exception as e:
                    logger.error(f"Failed to process {persona}: {str(e)}")
                    failed_personas.append(persona)
                    if not args.continue_on_error:
                        raise

            # Summary
            logger.info("\n" + "="*70)
            logger.info("ALL PERSONAS PROCESSING COMPLETE!")
            logger.info("="*70)
            logger.info(f"Total personas: {len(personas)}")
            logger.info(f"Successful: {len(personas) - len(failed_personas)}")
            if failed_personas:
                logger.info(f"Failed: {', '.join(failed_personas)}")
            logger.info("="*70)

            return 1 if failed_personas else 0

    # Single dataset processing (original behavior)
    return run_persona_pipeline(dataset=args.dataset, args=args, logger=logger)


def run_persona_pipeline(dataset: str, args, logger):
    """Run pipeline for a single persona/dataset."""
    start_time = datetime.now()

    try:
        logger.info("\n" + "="*70)
        logger.info("SINGLE-VECTOR PIPELINE")
        logger.info("="*70)
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Layers: {args.layers if args.layers else 'all'}")
        logger.info(f"Batch size: {args.batch_size}")
        steps = []
        if not args.skip_extract:
            steps.append("Extract")
        if not args.skip_infer_val:
            steps.append("Infer-Val")
        if not args.skip_infer_test:
            steps.append("Infer-Test")
        logger.info(f"Steps: {' → '.join(steps)}")

        # Load model and tokenizer once if needed for extraction or inference
        model = None
        tokenizer = None
        if not args.skip_extract or not args.skip_infer_val or not args.skip_infer_test:
            logger.info("\n" + "="*70)
            logger.info("LOADING MODEL AND TOKENIZER")
            logger.info("="*70)
            model, tokenizer = get_model_and_tokenizer()
            logger.info("✓ Model and tokenizer loaded successfully")

        # Determine vector name
        vector_name = args.vector

        # Step 1: Vector Extraction
        if not args.skip_extract:
            try:
                vectors_path = run_extraction_step(
                    dataset=dataset,
                    layers=args.layers,
                    model=model,
                    tokenizer=tokenizer,
                    logger=logger
                )
                # Construct vector name from dataset and layers
                layer_name = (
                    "layer_all"
                    if args.layers is None
                    else f"layer_{'_'.join(map(str, args.layers))}"
                )
                vector_name = f"{dataset}_style_vector_sv_{layer_name}"
            except Exception as e:
                logger.error(f"✗ Vector extraction failed: {str(e)}")
                if not args.continue_on_error:
                    raise
                return 1
        else:
            # Use the provided vector name
            logger.info(f"Using provided vector: {vector_name}")

        # Step 2: Inference on Val
        if not args.skip_infer_val:
            try:
                run_inference_step(
                    dataset=dataset,
                    vector=vector_name,
                    multiplier=args.multiplier,
                    test_mode=args.test,
                    batch_size=args.batch_size,
                    model=model,
                    tokenizer=tokenizer,
                    data_file="val.json",
                    step_name="STEP 2: INFERENCE ON VAL",
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"✗ Inference on Val failed: {str(e)}")
                if not args.continue_on_error:
                    raise
                return 1

        # Step 3: Inference on Test
        if not args.skip_infer_test:
            try:
                run_inference_step(
                    dataset=dataset,
                    vector=vector_name,
                    multiplier=args.multiplier,
                    test_mode=args.test,
                    batch_size=args.batch_size,
                    model=model,
                    tokenizer=tokenizer,
                    data_file="test.json",
                    step_name="STEP 3: INFERENCE ON TEST",
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"✗ Inference on Test failed: {str(e)}")
                if not args.continue_on_error:
                    raise
                return 1

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total duration: {duration}")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Vector: {vector_name}")
        if not args.test:
            if not args.skip_infer_val:
                logger.info(
                    f"Val results saved to: single_vector/outputs/results/val/")
            if not args.skip_infer_test:
                logger.info(
                    f"Test results saved to: single_vector/outputs/results/test/")
        logger.info("="*70)

        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\nPipeline failed: {str(e)}")
        logger.exception("Full error details:")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
