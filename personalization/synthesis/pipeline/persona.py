import os
import json
import logging
from synthesis.modules import PersonaTemplateSampler, PersonaGenerator
from synthesis.utils.files import load_json, save_json, ensure_directory, get_persona_dir

logger = logging.getLogger(__name__)


def generate_personas(model_name: str, n_personas: int, output_dir: str) -> list:
    # Ensure output directory exists
    ensure_directory(output_dir)

    # Load existing personas if they exist
    personas_path = os.path.join(output_dir, "personas.json")
    personas = load_json(personas_path, default=[])

    # If we already have enough personas, return them
    if len(personas) >= n_personas:
        return personas

    # Initialize generators for creating new personas
    persona_sampler = PersonaTemplateSampler()  # Creates diverse persona templates
    # Generates full personas from templates
    persona_generator = PersonaGenerator(model_name=model_name)

    # Track what seed data we've already used to ensure diversity
    used_seeds = [p.get('seed') for p in personas]
    n_remaining = n_personas - len(personas)

    # Generate diverse persona templates
    logger.info("Creating Diverse Persona Templates...")
    persona_templates = persona_sampler(n_remaining, used_seeds)

    # Generate full personas from templates using the specified model
    logger.info("Generating Personas...")
    new_personas = persona_generator(persona_templates)
    # Update IDs of the new personas
    max_prev_id = max([p.get('id', 0) for p in personas], default=0)
    for p in new_personas:
        p['id'] = max_prev_id + 1
        max_prev_id += 1
    personas += new_personas

    # Log the generated personas for debugging (at debug level to avoid spam)
    logger.debug(json.dumps(personas, indent=2))

    # Save the updated personas list (centralized file)
    save_json(personas_path, personas, indent=4)

    # Also save each persona to its own directory
    for persona in personas:
        persona_id = persona.get('id')
        persona_dir = get_persona_dir(output_dir, persona_id)
        ensure_directory(persona_dir)
        persona_file = os.path.join(persona_dir, "persona.json")
        save_json(persona_file, persona, indent=4)
        logger.debug(f"Saved persona {persona_id} to {persona_dir}")

    return personas
