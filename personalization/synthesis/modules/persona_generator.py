import logging
from synthesis.utils.parsing import parse_json
from synthesis.utils.generation import Generator

logger = logging.getLogger(__name__)


class PersonaGenerator(Generator):
    def __init__(
        self,
        batch_size=4,
        model_name="claude-3-5-sonnet-20241022",
        prompt_path="persona_generator.yaml",
        temperature=0.7,
        max_tokens=4096,
        verbose=False
    ):
        super().__init__(
            model_name,
            prompt_path,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose
        )
        self.batch_size = batch_size

    def __call__(self, persona_templates, max_retries=3):
        personas = []
        for i in range(0, len(persona_templates), self.batch_size):
            batch = persona_templates[i:i + self.batch_size]
            batch_input_str = "\n".join(
                map(lambda x: f"{x[0] + 1}. {x[1]}", enumerate(batch)))

            logger.info(
                f"Processing persona batch {i//self.batch_size + 1}/{(len(persona_templates)-1)//self.batch_size + 1}")
            logger.debug(f"Batch input: {batch_input_str}")

            last_error = None
            for attempt in range(max_retries):
                try:
                    output = super().__call__(
                        seed_descriptions=batch_input_str
                    )

                    processed_output = parse_json(output)["personas"]
                    for persona_template, output in zip(batch, processed_output):
                        persona_entry = {
                            'id': persona_template['id'],
                            'seed': persona_template['seed'],
                            'persona': {
                                'name': output['name'],
                                'age': output['age'],
                                'gender': output['gender'],
                                'demographics': output['demographics'],
                                'profession': output['profession'],
                                'education': output['education'],
                                'socioeconomic_status': output['socioeconomic_status'],
                                'brief_background': output['brief_background'],
                                'interests_lifestyle': output['interests_lifestyle']
                            }
                        }
                        personas.append(persona_entry)

                    logger.info(
                        f"Successfully processed {len(processed_output)} personas in batch")
                    break  # Success, exit retry loop

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for persona batch {i//self.batch_size + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        logger.info(
                            f"Retrying... ({attempt + 2}/{max_retries})")
                        continue
                    else:
                        logger.error(
                            f"Error processing persona batch {i//self.batch_size + 1} after {max_retries} attempts: {str(e)}")
                        logger.warning("Continuing with next batch...")

        logger.info(f"Generated {len(personas)} personas total")
        return personas
