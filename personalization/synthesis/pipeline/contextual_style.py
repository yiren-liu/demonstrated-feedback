import os
import json
import logging
import asyncio
from collections import defaultdict
from synthesis.modules import ContextualStyleGenerator
from synthesis.modules.contextual_style_generator import AsyncContextualStyleGenerator
from synthesis.utils.files import load_json, save_json, ensure_directory, get_persona_dir

logger = logging.getLogger(__name__)


async def generate_contextual_styles_async(
    model_name: str,
    persona_with_style: dict,
    n_scenarios: int,
    output_dir: str,
    max_concurrency: int = 10,
    writing_type: str = "email"
) -> list:
    """Generate contextual styles for a single persona."""
    # Ensure output directory exists
    ensure_directory(output_dir)

    persona_id = persona_with_style.get('id')

    # Check if this persona already has scenarios
    persona_dir = get_persona_dir(output_dir, persona_id)
    styles_file = os.path.join(persona_dir, "styles.json")
    existing_scenarios = load_json(styles_file, default=[])

    # If we already have scenarios for this persona, return them
    if len(existing_scenarios) == n_scenarios:
        logger.info(
            f"Found {len(existing_scenarios)} existing scenarios for persona {persona_id}")
        return existing_scenarios

    logger.info(
        f"Generating {n_scenarios} scenarios for persona {persona_id}")

    # Initialize async contextual style generator
    contextual_style_generator = AsyncContextualStyleGenerator(
        model_name=model_name,
        max_concurrency=max_concurrency
    )

    # Generate scenarios with contextual writing styles for this persona
    logger.info(
        f"Generating {n_scenarios} {writing_type} scenarios for persona {persona_id} "
        f"with max concurrency {max_concurrency}...")
    new_scenarios = await contextual_style_generator([persona_with_style], n_scenarios, writing_type)

    # Save the scenarios to persona directory
    ensure_directory(persona_dir)
    save_json(styles_file, new_scenarios, indent=4)
    logger.info(
        f"Saved {len(new_scenarios)} scenarios for persona {persona_id}")

    # Also update the centralized styles.json file
    styles_path = os.path.join(output_dir, "styles.json")
    all_scenarios = load_json(styles_path, default=[])

    # Remove any existing scenarios for this persona from centralized file
    all_scenarios = [s for s in all_scenarios if s.get('id') != persona_id]

    # Add new scenarios
    all_scenarios.extend(new_scenarios)

    # Save centralized file
    save_json(styles_path, all_scenarios, indent=4)
    logger.info(f"Updated centralized styles.json")

    # Log sample for debugging
    if new_scenarios:
        logger.debug("Sample generated scenario:")
        sample = new_scenarios[0]
        logger.debug(json.dumps({
            'scenario_description': sample.get('scenario_description', '')[:100] + '...',
            'detailed_style': sample.get('detailed_style', '')[:100] + '...',
            'linguistic_patterns': sample.get('linguistic_patterns', '')[:100] + '...'
        }, indent=2))

    return new_scenarios
