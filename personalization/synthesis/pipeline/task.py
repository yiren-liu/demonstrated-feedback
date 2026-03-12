import os
import json
import logging
from collections import defaultdict
from synthesis.modules import AsyncTaskGenerator
from synthesis.utils.files import load_json, save_json, ensure_directory, get_persona_dir

logger = logging.getLogger(__name__)


async def generate_tasks_async(
    model_name: str,
    styled_scenarios: list,
    output_dir: str,
    n_tasks: int = 25,
    max_concurrency: int = 10
) -> list:
    """Generate tasks for scenarios from a single persona."""
    # Ensure output directory exists
    ensure_directory(output_dir)

    # Get persona ID from first scenario (all scenarios should be from same persona)
    if not styled_scenarios:
        logger.warning("No scenarios provided")
        return []

    persona_id = styled_scenarios[0].get('id')

    # Check if this persona already has tasks
    persona_dir = get_persona_dir(output_dir, persona_id)
    tasks_file = os.path.join(persona_dir, "tasks.json")
    existing_tasks = load_json(tasks_file, default=[])

    # Calculate expected total tasks for this persona
    expected_total = len(styled_scenarios) * n_tasks

    # If we already have tasks for this persona, return them
    if len(existing_tasks) == expected_total:
        logger.info(
            f"Found {len(existing_tasks)} existing tasks for persona {persona_id} "
            f"({len(styled_scenarios)} scenarios × {n_tasks} tasks)"
        )
        return existing_tasks

    logger.info(
        f"Generating tasks for {len(styled_scenarios)} scenarios of persona {persona_id}")

    # Initialize async task generator
    task_generator = AsyncTaskGenerator(
        model_name=model_name,
        n_tasks=n_tasks,
        max_concurrency=max_concurrency,
        # verbose=True,
        temperature=1,
        max_tokens=4096*4
    )

    # Generate tasks for all scenarios of this persona (parallelized at scenario level)
    logger.info(
        f"Generating {n_tasks} tasks per scenario for {len(styled_scenarios)} scenarios "
        f"with max concurrency {max_concurrency}..."
    )
    new_tasks = await task_generator(styled_scenarios)

    # Save the tasks to persona directory
    ensure_directory(persona_dir)
    save_json(tasks_file, new_tasks, indent=4)
    logger.info(f"Saved {len(new_tasks)} tasks for persona {persona_id}")

    # Also update the centralized tasks.json file
    tasks_path = os.path.join(output_dir, "tasks.json")
    all_tasks = load_json(tasks_path, default=[])

    # Remove any existing tasks for this persona from centralized file
    # We identify tasks by persona name since that's embedded in each task
    if new_tasks:
        persona_name = new_tasks[0]['persona']['name']
        all_tasks = [t for t in all_tasks if t.get(
            'persona', {}).get('name') != persona_name]

        # Add new tasks
        all_tasks.extend(new_tasks)

        # Save centralized file
        save_json(tasks_path, all_tasks, indent=4)
        logger.info(f"Updated centralized tasks.json")

    # Log sample for debugging
    if new_tasks:
        logger.debug("Sample generated task:")
        sample = new_tasks[0]
        logger.debug(json.dumps({
            'prompt': sample.get('prompt', '')[:100] + '...',
            'task_instruction': sample.get('task_instruction', '')[:100] + '...',
            'personalized': sample.get('personalized', '')[:100] + '...'
        }, indent=2))

    return new_tasks
