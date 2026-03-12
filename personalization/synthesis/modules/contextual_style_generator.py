import logging
import asyncio
from typing import List, Dict, Any
from synthesis.utils.parsing import parse_json
from synthesis.utils.generation import Generator, AsyncGenerator

logger = logging.getLogger(__name__)


class ContextualStyleGenerator(Generator):
    def __init__(
        self,
        model_name="claude-3-5-sonnet-20241022",
        prompt_path="contextual_style_generator.yaml",
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

    def generate_scenarios(
        self,
        persona: dict,
        global_style: dict,
        n_scenarios: int = 20,
        writing_type: str = "email",
        max_retries: int = 10
    ) -> list:
        """Generate diverse writing scenarios with contextual writing styles."""
        last_error = None
        for attempt in range(max_retries):
            try:
                output = super().__call__(
                    name=persona['name'],
                    age=persona['age'],
                    gender=persona['gender'],
                    demographics=persona['demographics'],
                    profession=persona['profession'],
                    education=persona['education'],
                    socioeconomic_status=persona['socioeconomic_status'],
                    brief_background=persona['brief_background'],
                    interests_lifestyle=persona['interests_lifestyle'],
                    tone=global_style.get('tone', ''),
                    syntax=global_style.get('syntax', ''),
                    lexicon=global_style.get('lexicon', ''),
                    linguistic_patterns=global_style.get(
                        'linguistic_patterns', ''),
                    n_scenarios=n_scenarios,
                    writing_type=writing_type
                )

                parsed_output = parse_json(output)
                scenarios = parsed_output.get('scenarios', [])
                return scenarios

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for scenario generation: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
                    continue

        logger.error(
            f"Error generating scenarios after {max_retries} attempts: {str(last_error)}")
        raise last_error

    def __call__(self, personas_with_styles: list, n_scenarios: int = 20, writing_type: str = "email") -> list:
        """Generate scenarios with contextual writing styles for a list of personas."""
        all_scenarios = []
        total_personas = len(personas_with_styles)

        logger.info(
            f"Generating {n_scenarios} {writing_type} scenarios for each of {total_personas} personas")

        for idx, persona_data in enumerate(personas_with_styles):
            try:
                logger.info(
                    f"Processing persona {idx + 1}/{total_personas}: {persona_data['persona']['name']}"
                )

                # Generate scenarios for this persona
                scenarios = self.generate_scenarios(
                    persona=persona_data['persona'],
                    global_style=persona_data['global_style'],
                    n_scenarios=n_scenarios,
                    writing_type=writing_type
                )

                # Add persona and global_style to each scenario
                for scenario in scenarios:
                    scenario_with_persona = {
                        'persona': persona_data['persona'],
                        'global_style': persona_data['global_style'],
                        'scenario_description': scenario.get('scenario_description', ''),
                        'detailed_style': scenario.get('detailed_style', ''),
                        'linguistic_patterns': scenario.get('linguistic_patterns', '')
                    }
                    # Preserve any additional fields from the persona_data
                    for key in persona_data:
                        if key not in ['persona', 'global_style']:
                            scenario_with_persona[key] = persona_data[key]

                    all_scenarios.append(scenario_with_persona)

                logger.info(
                    f"Generated {len(scenarios)} scenarios for persona {idx + 1}")

            except Exception as e:
                logger.error(
                    f"Failed to generate scenarios for persona {idx + 1} "
                    f"({persona_data['persona']['name']}): {str(e)}"
                )
                logger.warning("Continuing with next persona...")
                continue

        logger.info(
            f"Successfully generated {len(all_scenarios)} total scenarios across {total_personas} personas")
        return all_scenarios


class AsyncContextualStyleGenerator(AsyncGenerator):
    def __init__(
        self,
        model_name="claude-3-5-sonnet-20241022",
        prompt_path="contextual_style_generator.yaml",
        temperature=0.7,
        max_tokens=4096,
        verbose=False,
        max_concurrency=10
    ):
        super().__init__(
            model_name,
            prompt_path,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            max_concurrency=max_concurrency
        )

    async def generate_scenarios(
        self,
        persona: dict,
        global_style: dict,
        n_scenarios: int = 20,
        writing_type: str = "email",
        max_retries: int = 10
    ) -> list:
        """Generate scenarios for a single persona with retry logic."""
        last_error = None
        for attempt in range(max_retries):
            try:
                batch_items = [{
                    'name': persona['name'],
                    'age': persona['age'],
                    'gender': persona['gender'],
                    'demographics': persona['demographics'],
                    'profession': persona['profession'],
                    'education': persona['education'],
                    'socioeconomic_status': persona['socioeconomic_status'],
                    'brief_background': persona['brief_background'],
                    'interests_lifestyle': persona['interests_lifestyle'],
                    'tone': global_style.get('tone', ''),
                    'syntax': global_style.get('syntax', ''),
                    'lexicon': global_style.get('lexicon', ''),
                    'linguistic_patterns': global_style.get('linguistic_patterns', ''),
                    'n_scenarios': n_scenarios,
                    'writing_type': writing_type
                }]

                results = await self.generate_batch(batch_items)
                output, error = results[0]

                if error is not None:
                    raise Exception(f"Generation error: {error}")

                parsed_output = parse_json(output)
                scenarios = parsed_output.get('scenarios', [])
                return scenarios

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for scenario generation ({persona['name']}): {str(e)}")
                logger.debug(f"Output that failed to parse: {repr(output) if 'output' in locals() else 'N/A'}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
                    continue

        logger.error(
            f"Error generating scenarios for {persona['name']} after {max_retries} attempts: {str(last_error)}")
        raise last_error

    async def __call__(self, personas_with_styles: List[Dict[str, Any]], n_scenarios: int = 20, writing_type: str = "email", max_retries: int = 10) -> List[Dict[str, Any]]:
        """Generate scenarios with contextual writing styles for a list of personas using batched async calls."""
        total_personas = len(personas_with_styles)
        logger.info(
            f"Generating {n_scenarios} {writing_type} scenarios for each of {total_personas} personas with max concurrency {self.max_concurrency}")

        # Prepare batch items
        batch_items = []
        for persona_data in personas_with_styles:
            persona = persona_data['persona']
            global_style = persona_data['global_style']
            batch_items.append({
                'name': persona['name'],
                'age': persona['age'],
                'gender': persona['gender'],
                'demographics': persona['demographics'],
                'profession': persona['profession'],
                'education': persona['education'],
                'socioeconomic_status': persona['socioeconomic_status'],
                'brief_background': persona['brief_background'],
                'interests_lifestyle': persona['interests_lifestyle'],
                'tone': global_style.get('tone', ''),
                'syntax': global_style.get('syntax', ''),
                'lexicon': global_style.get('lexicon', ''),
                'linguistic_patterns': global_style.get('linguistic_patterns', ''),
                'n_scenarios': n_scenarios,
                'writing_type': writing_type
            })

        # Generate all scenarios in batch
        results = await self.generate_batch(batch_items)

        # Combine results with original persona data
        all_scenarios = []
        for idx, (persona_data, (output, error)) in enumerate(zip(personas_with_styles, results)):
            if error is None and output is not None:
                last_parse_error = None
                for attempt in range(max_retries):
                    try:
                        # Parse the JSON output
                        result = parse_json(output)
                        scenarios = result.get('scenarios', [])

                        # Add persona and global_style to each scenario
                        for scenario in scenarios:
                            scenario_with_persona = {
                                'persona': persona_data['persona'],
                                'global_style': persona_data['global_style'],
                                'scenario_description': scenario.get('scenario_description', ''),
                                'detailed_style': scenario.get('detailed_style', ''),
                                'linguistic_patterns': scenario.get('linguistic_patterns', '')
                            }
                            # Preserve any additional fields from the persona_data
                            for key in persona_data:
                                if key not in ['persona', 'global_style']:
                                    scenario_with_persona[key] = persona_data[key]

                            all_scenarios.append(scenario_with_persona)

                        logger.debug(
                            f"Successfully generated {len(scenarios)} scenarios for persona {idx + 1}/{total_personas} "
                            f"({persona_data['persona']['name']})"
                        )
                        break  # Success, exit retry loop
                    except Exception as parse_error:
                        last_parse_error = parse_error
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed to parse result for persona {idx + 1} "
                            f"({persona_data['persona']['name']}): {str(parse_error)}"
                        )
                        logger.error(f"Original output that failed to parse: {repr(output)}")
                        logger.error(f"Error type: {type(parse_error).__name__}")
                        if attempt < max_retries - 1:
                            logger.info(
                                f"Regenerating for persona {persona_data['persona']['name']} (attempt {attempt + 2}/{max_retries})")
                            # Regenerate scenarios for this persona
                            try:
                                scenarios = await self.generate_scenarios(
                                    persona=persona_data['persona'],
                                    global_style=persona_data['global_style'],
                                    n_scenarios=n_scenarios,
                                    writing_type=writing_type,
                                    max_retries=1  # Single attempt here since we're already in retry loop
                                )
                                # Try parsing again with new output
                                for scenario in scenarios:
                                    scenario_with_persona = {
                                        'persona': persona_data['persona'],
                                        'global_style': persona_data['global_style'],
                                        'scenario_description': scenario.get('scenario_description', ''),
                                        'detailed_style': scenario.get('detailed_style', ''),
                                        'linguistic_patterns': scenario.get('linguistic_patterns', '')
                                    }
                                    # Preserve any additional fields from the persona_data
                                    for key in persona_data:
                                        if key not in ['persona', 'global_style']:
                                            scenario_with_persona[key] = persona_data[key]

                                    all_scenarios.append(scenario_with_persona)

                                logger.debug(
                                    f"Successfully regenerated {len(scenarios)} scenarios for persona {idx + 1}/{total_personas} "
                                    f"({persona_data['persona']['name']})"
                                )
                                break  # Success, exit retry loop
                            except Exception as regen_error:
                                logger.warning(
                                    f"Regeneration attempt failed: {str(regen_error)}")
                                last_parse_error = regen_error
                                continue
                        else:
                            # Final attempt failed
                            logger.error(
                                f"Failed to parse result for persona {idx + 1} "
                                f"({persona_data['persona']['name']}) after {max_retries} attempts: {str(last_parse_error)}"
                            )
                            logger.warning("Skipping this persona...")
            else:
                logger.error(
                    f"Failed to generate scenarios for persona {idx + 1} "
                    f"({persona_data['persona']['name']}): {error}"
                )
                logger.warning("Skipping this persona...")

        logger.info(
            f"Successfully generated {len(all_scenarios)} total scenarios across {total_personas} personas")
        return all_scenarios
