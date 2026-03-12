import logging
import asyncio
import os
from collections import defaultdict
from synthesis.modules.prompt_generator import AsyncPromptGenerator
from synthesis.utils.files import save_json, load_json, ensure_directory, get_persona_dir

logger = logging.getLogger(__name__)


async def generate_prompts_async(
    model_name: str,
    tasks: list,
    output_dir: str,
    max_concurrency: int = 10,
    batch_size: int = 20,
    writing_type: str = "email"
) -> list:
    """Generate prompts for tasks from a single persona."""
    logger.info(
        f"Starting Round 5: Generating task specifications for {len(tasks)} tasks")
    logger.info(
        f"Using batch size: {batch_size}, max concurrency: {max_concurrency}")

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

    # Check if this persona already has a dataset with prompts
    persona_dir = get_persona_dir(output_dir, persona_id)
    dataset_file = os.path.join(persona_dir, "dataset.json")
    existing_dataset = load_json(dataset_file, default=[])

    # If we already have dataset with prompts for this persona, return it
    if existing_dataset and all("prompt" in task and task["prompt"] for task in existing_dataset):
        logger.info(
            f"Found existing dataset with prompts for persona {persona_id}")
        return existing_dataset

    logger.info(
        f"Generating prompts for {len(tasks)} tasks of persona {persona_id}")

    # Initialize generator
    generator = AsyncPromptGenerator(
        model_name=model_name,
        prompt_path="prompt_generator.yaml",
        temperature=1,
        max_tokens=4096,
        verbose=False,
        max_concurrency=max_concurrency
    )

    # Process in batches to allow saving progress
    total = len(tasks)
    updated_tasks = tasks.copy()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_tasks = updated_tasks[batch_start:batch_end]

        logger.info(f"Processing batch {batch_start+1}-{batch_end}...")

        # Generate prompts for this batch
        batch_results = await generator(batch_tasks, writing_type)

        # Update the tasks list with results (includes prompts)
        for i, result in enumerate(batch_results):
            updated_tasks[batch_start + i] = result

    # Count how many prompts were generated
    prompts_generated = sum(
        1 for task in updated_tasks if "prompt" in task and task["prompt"])
    logger.info(
        f"✓ Round 5 complete! Generated {prompts_generated}/{total} task specifications for persona {persona_id}")

    # Note: We don't save to dataset.json yet - that happens in Round 6
    # But we can save progress to a temporary file if needed
    logger.info(
        "Note: Prompts will be saved to dataset.json in Round 6")

    return updated_tasks
