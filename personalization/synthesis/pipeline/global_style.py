import os
import json
import logging
from synthesis.modules import GlobalStyleGenerator
from synthesis.utils.files import load_json, save_json, ensure_directory, get_persona_dir

logger = logging.getLogger(__name__)


def generate_global_styles(model_name: str, persona: dict, output_dir: str) -> dict:
    """Generate global style for a single persona."""
    # Ensure output directory exists
    ensure_directory(output_dir)

    # Check if this persona already has a global style
    if 'global_style' in persona:
        logger.info(f"Persona {persona.get('id')} already has global style")
        return persona

    logger.info(f"Generating global style for persona {persona.get('id')}")

    # Initialize global style generator
    global_style_generator = GlobalStyleGenerator(model_name=model_name)

    # Generate global style (pass as single-item list to generator)
    personas_with_styles = global_style_generator([persona])
    updated_persona = personas_with_styles[0]

    # Load all personas from centralized file
    personas_path = os.path.join(output_dir, "personas.json")
    all_personas = load_json(personas_path, default=[])

    # Update this specific persona in the centralized list
    for i, p in enumerate(all_personas):
        if p.get('id') == updated_persona.get('id'):
            all_personas[i] = updated_persona
            break

    # Save the updated personas list with global styles (centralized file)
    save_json(personas_path, all_personas, indent=4)
    logger.info(
        f"Updated persona {updated_persona.get('id')} saved to: {personas_path}")

    # Also save this persona to its own directory
    persona_id = updated_persona.get('id')
    persona_dir = get_persona_dir(output_dir, persona_id)
    ensure_directory(persona_dir)
    persona_file = os.path.join(persona_dir, "persona.json")
    save_json(persona_file, updated_persona, indent=4)

    return updated_persona
