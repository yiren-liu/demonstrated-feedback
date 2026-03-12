import argparse
import logging
from datetime import datetime
from pathlib import Path

# Import the pipeline run functions
from multi_vectors.pipeline.cluster_v2 import cluster_by_scenario
from multi_vectors.pipeline.extract import run_extraction
from multi_vectors.pipeline.infer import run_inference
from multi_vectors.utils.model_utils import get_model_and_tokenizer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_clustering_step(dataset, output_dir, logger):
    """Run the clustering pipeline step."""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: CLUSTERING BY SCENARIO")
    logger.info("="*70)
    logger.info(f"Dataset: {dataset}")
    logger.info("Clustering method: Group by scenario_description")

    clusters_filename = cluster_by_scenario(dataset=dataset, output_dir=output_dir)

    logger.info(f"✓ Clustering completed successfully")
    logger.info(f"Clusters saved to: {clusters_filename}")

    return clusters_filename


def run_extraction_step(clusters_file, dataset, layers, model, tokenizer, output_dir, logger):
    """Run the vector extraction pipeline step."""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: VECTOR EXTRACTION")
    logger.info("="*70)
    logger.info(f"Clusters file: {clusters_file}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Layers: {layers if layers else 'all'}")

    vectors_path = run_extraction(
        clusters_file=clusters_file,
        dataset=dataset,
        layers=layers,
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info("✓ Vector extraction completed successfully")
    logger.info(f"Vectors saved to: {vectors_path}")

    return vectors_path


def run_inference_step(clusters_file, vectors_path, dataset, layers, k, temperature, multiplier, min_similarity, test_mode, model, tokenizer, context_embedding_model, data_file, step_name, output_dir, batch_size, logger):
    """Run the inference pipeline step."""
    logger.info("\n" + "="*70)
    logger.info(f"{step_name}")
    logger.info("="*70)
    logger.info(f"Clusters file: {clusters_file}")
    logger.info(f"Vectors path: {vectors_path}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Layers: {layers if layers else 'all'}")
    logger.info(f"K (nearest neighbors): {k}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Multiplier: {multiplier}")
    logger.info(f"Min similarity: {min_similarity}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Test mode: {test_mode}")
    logger.info(f"Context embedding model: {context_embedding_model}")

    run_inference(
        clusters_file=clusters_file,
        vectors_path=vectors_path,
        dataset=dataset,
        data_file=data_file,
        layers=layers,
        k=k,
        temperature=temperature,
        multiplier=multiplier,
        min_similarity=min_similarity,
        test_mode=test_mode,
        model=model,
        tokenizer=tokenizer,
        embedding_model=context_embedding_model,
        output_dir=output_dir,
        logger=logger,
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
        description="Run multi-vectors pipeline: clustering → extraction → inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset and general settings
    parser.add_argument("--dataset", type=str, default="email",
                        help="Dataset identifier or folder path (e.g., 'email' for dataset/email/)")
    parser.add_argument("--output", type=str, default="multi_vectors/outputs",
                        help="Output directory for results (default: multi_vectors/outputs)")
    parser.add_argument("--clusters-file", type=str, default=None,
                        help="Clusters metadata file name (required if --skip-cluster is used)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue to next step even if current step fails")

    # Model settings
    parser.add_argument("--language-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Language model for extraction and inference (default: meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--context-embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Embedding model for context matching in inference (default: sentence-transformers/all-mpnet-base-v2)")

    # Pipeline control
    parser.add_argument("--skip-cluster", action="store_true",
                        help="Skip clustering step")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip vector extraction step")
    parser.add_argument("--skip-infer-val", action="store_true",
                        help="Skip inference on validation set")
    parser.add_argument("--skip-infer-test", action="store_true",
                        help="Skip inference on test set")

    # Extraction and inference arguments (shared)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers for steering vectors (default: all layers)")

    # Inference arguments
    parser.add_argument("--k", type=int, default=3,
                        help="Number of nearest neighbors for KNN (default: 3)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for softmax weighting (default: 1.0)")
    parser.add_argument("--multiplier", type=float, default=0.15,
                        help="Steering vector multiplier (default: 0.15)")
    parser.add_argument("--min-similarity", type=float, default=0.35,
                        help="Minimum similarity threshold (default: 0.35)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference (default: 8)")
    parser.add_argument("--test", action="store_true",
                        help="Run inference in test mode (single example)")

    args = parser.parse_args()
    logger = setup_logging()

    # Validate that at least one step is enabled
    if args.skip_cluster and args.skip_extract and args.skip_infer_val and args.skip_infer_test:
        logger.error("Error: All pipeline steps are skipped. Nothing to do.")
        return 1

    # Validate clusters-file is provided when clustering is skipped
    if args.skip_cluster and not args.clusters_file:
        logger.error(
            "Error: --clusters-file is required when --skip-cluster is used.")
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
        logger.info("MULTI-VECTORS PIPELINE")
        logger.info("="*70)
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Clustering: By scenario_description")
        logger.info(f"Layers: {args.layers if args.layers else 'all'}")
        steps = []
        if not args.skip_cluster:
            steps.append("Cluster")
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
            logger.info(f"Language model: {args.language_model}")
            model, tokenizer = get_model_and_tokenizer(args.language_model)
            logger.info("✓ Model and tokenizer loaded successfully")

        # Determine clusters file and vectors path
        clusters_file = args.clusters_file
        vectors_path = None

        # Step 1: Clustering
        if not args.skip_cluster:
            try:
                clusters_file = run_clustering_step(
                    dataset=dataset,
                    output_dir=args.output,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"✗ Clustering failed: {str(e)}")
                if not args.continue_on_error:
                    raise
                return 1
        else:
            # Use the provided clusters file
            logger.info(f"Using provided clusters file: {clusters_file}")

        # Step 2: Vector Extraction
        if not args.skip_extract:
            try:
                vectors_path = run_extraction_step(
                    clusters_file=clusters_file,
                    dataset=args.dataset,
                    layers=args.layers,
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=args.output,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"✗ Vector extraction failed: {str(e)}")
                if not args.continue_on_error:
                    raise
                return 1

        # Step 3: Inference on Val
        if not args.skip_infer_val:
            try:
                run_inference_step(
                    clusters_file=clusters_file,
                    vectors_path=vectors_path,
                    dataset=dataset,
                    layers=args.layers,
                    k=args.k,
                    temperature=args.temperature,
                    multiplier=args.multiplier,
                    min_similarity=args.min_similarity,
                    test_mode=args.test,
                    model=model,
                    tokenizer=tokenizer,
                    context_embedding_model=args.context_embedding_model,
                    data_file="val.json",
                    step_name="STEP 3: INFERENCE ON VAL",
                    output_dir=args.output,
                    batch_size=args.batch_size,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"✗ Inference on Val failed: {str(e)}")
                if not args.continue_on_error:
                    raise
                return 1

        # Step 4: Inference on Test
        if not args.skip_infer_test:
            try:
                run_inference_step(
                    clusters_file=clusters_file,
                    vectors_path=vectors_path,
                    dataset=dataset,
                    layers=args.layers,
                    k=args.k,
                    temperature=args.temperature,
                    multiplier=args.multiplier,
                    min_similarity=args.min_similarity,
                    test_mode=args.test,
                    model=model,
                    tokenizer=tokenizer,
                    context_embedding_model=args.context_embedding_model,
                    data_file="test.json",
                    step_name="STEP 4: INFERENCE ON TEST",
                    output_dir=args.output,
                    batch_size=args.batch_size,
                    logger=logger
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
        logger.info(f"Clusters file: {clusters_file}")
        if not args.test:
            if not args.skip_infer_val:
                logger.info(
                    f"Val results saved to: {args.output}/results/val/")
            if not args.skip_infer_test:
                logger.info(
                    f"Test results saved to: {args.output}/results/test/")
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
