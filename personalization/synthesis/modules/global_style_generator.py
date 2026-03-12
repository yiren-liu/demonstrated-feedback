import logging
from synthesis.utils.parsing import parse_json
from synthesis.utils.generation import Generator

logger = logging.getLogger(__name__)


class GlobalStyleGenerator(Generator):
    def __init__(
        self,
        model_name="claude-3-5-sonnet-20241022",
        prompt_path="global_style_generator.yaml",
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

    def generate_style(
        self,
        persona: dict,
        max_retries: int = 10
    ) -> dict:
        """Generate global writing style for a single persona."""
        last_error = None
        for attempt in range(max_retries):
            try:
                p = persona['persona']
                output = super().__call__(
                    name=p['name'],
                    age=p['age'],
                    gender=p['gender'],
                    demographics=p['demographics'],
                    profession=p['profession'],
                    education=p['education'],
                    socioeconomic_status=p['socioeconomic_status'],
                    brief_background=p['brief_background'],
                    interests_lifestyle=p['interests_lifestyle']
                )

                parsed_output = parse_json(output)
                return parsed_output

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for global style generation: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
                    continue

        logger.error(
            f"Error generating global style after {max_retries} attempts: {str(last_error)}")
        raise last_error

    def __call__(self, personas: list) -> list:
        """Generate global writing styles for a list of personas."""
        total = len(personas)
        logger.info(f"Generating global writing styles for {total} personas")

        for idx, persona in enumerate(personas):
            try:
                logger.info(
                    f"Processing persona {idx + 1}/{total} "
                    f"(ID: {persona.get('id', 'unknown')}, Name: {persona.get('persona', {}).get('name', 'unknown')})"
                )

                # Generate global writing style for this persona
                global_style = self.generate_style(persona=persona)

                # Add global style to persona
                persona['global_style'] = global_style

                # Log progress every 10 personas
                if (idx + 1) % 10 == 0:
                    logger.info(f"Completed {idx + 1}/{total} personas")

            except Exception as e:
                logger.error(
                    f"Failed to generate style for persona {idx + 1} "
                    f"(ID: {persona.get('id', 'unknown')}, Name: {persona.get('persona', {}).get('name', 'unknown')}): {str(e)}"
                )
                logger.warning("Continuing with next persona...")
                continue

        logger.info(
            f"Successfully generated global styles for {len([p for p in personas if 'global_style' in p])} personas")
        return personas
