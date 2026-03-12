import argparse
import logging
from datetime import datetime
from pathlib import Path

# Import the pipeline run functions
from vanilla_llm.pipeline.infer import run_inference
from vanilla_llm.utils.model_utils import get_model_and_tokenizer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_inference_step(dataset, test_mode, model, tokenizer, data_file, step_name, logger, n_shot=0):
    """Run the inference pipeline step."""
    logger.info("\n" + "="*70)
    logger.info(f"{step_name}")
    logger.info("="*70)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Test mode: {test_mode}")
    logger.info(f"N-shot: {n_shot}")

    run_inference(
        dataset=dataset,
        data_file=data_file,
        test_mode=test_mode,
        model=model,
        tokenizer=tokenizer,
        n_shot=n_shot,
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
        description="Run vanilla LLM inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset and general settings
    parser.add_argument("--dataset", type=str, default="email",
                        help="Dataset identifier or folder path (e.g., 'email' for dataset/email/)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue to next step even if current step fails")

    # Pipeline control
    parser.add_argument("--skip-infer-val", action="store_true",
                        help="Skip inference on validation set")
    parser.add_argument("--skip-infer-test", action="store_true",
                        help="Skip inference on test set")

    # Inference arguments
    parser.add_argument("--test", action="store_true",
                        help="Run inference in test mode (3 samples)")
    parser.add_argument("--n-shot", type=int, default=0,
                        help="Number of examples to include in few-shot prompting (default: 0 for zero-shot)")

    args = parser.parse_args()
    logger = setup_logging()

    # Validate that at least one step is enabled
    if args.skip_infer_val and args.skip_infer_test:
        logger.error("Error: All pipeline steps are skipped. Nothing to do.")
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
                        logger=logger
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
        logger.info("VANILLA LLM PIPELINE")
        logger.info("="*70)
        logger.info(f"Dataset: {dataset}")
        steps = []
        if not args.skip_infer_val:
            steps.append("Infer-Val")
        if not args.skip_infer_test:
            steps.append("Infer-Test")
        logger.info(f"Steps: {' → '.join(steps)}")

        # Load model and tokenizer once
        logger.info("\n" + "="*70)
        logger.info("LOADING MODEL AND TOKENIZER")
        logger.info("="*70)
        model, tokenizer = get_model_and_tokenizer()
        logger.info("✓ Model and tokenizer loaded successfully")

        # Step 1: Inference on Val
        if not args.skip_infer_val:
            try:
                run_inference_step(
                    dataset=dataset,
                    test_mode=args.test,
                    model=model,
                    tokenizer=tokenizer,
                    data_file="val.json",
                    step_name="STEP 1: INFERENCE ON VAL",
                    logger=logger,
                    n_shot=args.n_shot
                )
            except Exception as e:
                logger.error(f"✗ Inference on Val failed: {str(e)}")
                if not args.continue_on_error:
                    raise
                return 1

        # Step 2: Inference on Test
        if not args.skip_infer_test:
            try:
                run_inference_step(
                    dataset=dataset,
                    test_mode=args.test,
                    model=model,
                    tokenizer=tokenizer,
                    data_file="test.json",
                    step_name="STEP 2: INFERENCE ON TEST",
                    logger=logger,
                    n_shot=args.n_shot
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
        if not args.test:
            if not args.skip_infer_val:
                logger.info(
                    f"Val results saved to: vanilla_llm/outputs/results/val/")
            if not args.skip_infer_test:
                logger.info(
                    f"Test results saved to: vanilla_llm/outputs/results/test/")
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
