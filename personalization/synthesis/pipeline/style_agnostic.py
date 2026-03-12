import logging
import os
from collections import defaultdict
from synthesis.modules.style_agnostic_generator import StyleAgnosticGenerator
from synthesis.utils.files import save_json, load_json, ensure_directory, get_persona_dir

logger = logging.getLogger(__name__)


def generate_style_agnostic(
    model_name: str,
    tasks: list,
    output_dir: str,
    batch_size: int = 10,
    writing_type: str = "email"
) -> list:
    """Generate style-agnostic text for tasks from a single persona."""
    logger.info(
        f"Starting Round 6: Generating style-agnostic text for {len(tasks)} tasks")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Batch size: {batch_size}")

    if not tasks:
        logger.warning("No tasks provided")
        return []

    # Get persona info from first task (all tasks should be from same persona)
    persona_name = tasks[0].get('persona', {}).get('name', '')
    persona_id = None

    # Load personas to get ID
    personas_path = os.path.join(output_dir, "personas.json")
    personas = load_json(personas_path, default=[])
    for p in personas:
        if p['persona']['name'] == persona_name:
            persona_id = p['id']
            break

    if persona_id is None:
        logger.error(f"Could not find persona ID for {persona_name}")
        return tasks

    # Check if this persona already has a dataset with style_agnostic
    persona_dir = get_persona_dir(output_dir, persona_id)
    dataset_file = os.path.join(persona_dir, "dataset.json")
    existing_dataset = load_json(dataset_file, default=[])

    # If we already have dataset with style_agnostic for this persona, return it
    if existing_dataset and all("style_agnostic" in task and task["style_agnostic"] for task in existing_dataset):
        logger.info(
            f"Found existing dataset with style_agnostic for persona {persona_id}")
        return existing_dataset

    logger.info(
        f"Generating style_agnostic text for {len(tasks)} tasks of persona {persona_id}")

    # Initialize generator with batch_size for efficient batched inference
    generator = StyleAgnosticGenerator(
        model_name=model_name,
        verbose=False,
        writing_type=writing_type,
        batch_size=batch_size
    )

    # Process all tasks with batched inference
    # The generator handles left-padding and batching internally
    total = len(tasks)
    updated_tasks = generator(tasks)

    # Save results to persona dataset file
    logger.info(f"Saving persona {persona_id} dataset...")
    ensure_directory(persona_dir)
    save_json(dataset_file, updated_tasks, indent=4)

    # Count how many were generated
    generated_count = sum(
        1 for task in updated_tasks if "style_agnostic" in task and task["style_agnostic"])
    logger.info(
        f"✓ Round 6 complete! Generated {generated_count}/{total} style-agnostic texts for persona {persona_id}")

    # Update centralized dataset.json file
    dataset_path = os.path.join(output_dir, "dataset.json")
    all_dataset = load_json(dataset_path, default=[])

    # Remove any existing tasks for this persona from centralized file
    all_dataset = [t for t in all_dataset if t.get(
        'persona', {}).get('name') != persona_name]

    # Add this persona's dataset
    all_dataset.extend(updated_tasks)

    # Save centralized file
    save_json(dataset_path, all_dataset, indent=4)
    logger.info(f"Updated centralized dataset.json")

    logger.info(
        f"Saved final dataset for persona {persona_id} ({len(updated_tasks)} entries)")

    return updated_tasks
