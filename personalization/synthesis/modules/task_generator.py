import logging
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json_repair

from synthesis.utils.generation import Generator, AsyncGenerator

logger = logging.getLogger(__name__)


class WritingTask(BaseModel):
    """Single writing task with instructions and generated content."""
    task_instructions: str
    task_details: str
    writing_content: Optional[str] = None


class WritingTasks(BaseModel):
    """Collection of writing tasks for a scenario."""
    tasks: List[WritingTask]


class TaskGenerator(Generator):
    def __init__(
        self,
        model_name="gpt-4o-mini",
        prompt_path="task_generator.yaml",
        n_tasks=25,
        temperature=1,
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
        self.n_tasks = n_tasks

    def generate_tasks_for_scenario(
        self,
        persona: dict,
        global_style: dict,
        scenario: dict,
        max_retries: int = 3
    ) -> List[Dict[str, str]]:
        """Generate N_TASKS writing tasks for a single scenario."""
        last_error = None
        for attempt in range(max_retries):
            try:
                # Call LLM with structured output
                output = super().__call__(
                    # Persona fields
                    name=persona['name'],
                    age=persona['age'],
                    gender=persona['gender'],
                    demographics=persona['demographics'],
                    profession=persona['profession'],
                    education=persona['education'],
                    socioeconomic_status=persona['socioeconomic_status'],
                    brief_background=persona['brief_background'],
                    interests_lifestyle=persona['interests_lifestyle'],
                    # Style fields
                    tone=global_style['tone'],
                    syntax=global_style['syntax'],
                    lexicon=global_style['lexicon'],
                    linguistic_patterns=global_style['linguistic_patterns'],
                    # Scenario fields
                    scenario_description=scenario.get(
                        'scenario_description', scenario.get('scenario', '')),
                    detailed_style=scenario.get(
                        'detailed_style', scenario.get('contextual_style', '')),
                    scenario_linguistic_patterns=scenario.get(
                        'linguistic_patterns', ''),
                    # Task configuration
                    n_tasks=self.n_tasks
                )

                # Parse structured output
                if isinstance(output, str):
                    # If output is string, assume it's the raw tasks list
                    # This shouldn't happen with structured parsing, but handle gracefully
                    logger.warning(
                        "Received string output instead of structured data")
                    return []

                # Extract tasks from structured response
                tasks = []
                if hasattr(output, 'tasks'):
                    for task in output.tasks:
                        tasks.append({
                            'task_instructions': task.task_instructions,
                            'task_details': task.task_details,
                            'writing_content': task.writing_content
                        })

                return tasks

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for task generation: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
                    continue

        logger.error(
            f"Error generating tasks after {max_retries} attempts: {str(last_error)}")
        raise last_error

    def __call__(self, styled_scenarios: list) -> list:
        """Generate N_TASKS writing tasks for each styled scenario."""
        all_tasks = []
        total = len(styled_scenarios)

        logger.info(
            f"Generating {self.n_tasks} tasks per scenario for {total} scenarios")

        for idx, scenario_data in enumerate(styled_scenarios):
            try:
                logger.info(
                    f"Processing scenario {idx + 1}/{total} "
                    f"(Scenario ID: {scenario_data.get('id')}, Persona ID: {scenario_data.get('persona_id')})"
                )

                # Generate tasks for this scenario
                tasks = self.generate_tasks_for_scenario(
                    persona=scenario_data['persona'],
                    global_style=scenario_data.get('global_style', {}),
                    scenario=scenario_data
                )

                # Embed full context in each task (for flattened output)
                for task in tasks:
                    task_entry = {
                        # Context fields (embedded)
                        'persona': scenario_data['persona'],
                        'general_writing_style': scenario_data.get('global_style', {}),
                        'contextual_writing_style': {
                            'scenario_description': scenario_data.get('scenario_description', ''),
                            'detailed_style': scenario_data.get('detailed_style', ''),
                            'linguistic_patterns': scenario_data.get('linguistic_patterns', '')
                        },
                        # Task fields (primary keys)
                        'task_instruction': task['task_instructions'],
                        'task_details': task['task_details'],
                        'personalized': task['writing_content'],
                    }
                    all_tasks.append(task_entry)

                logger.info(
                    f"Generated {len(tasks)} tasks for scenario {idx + 1}")

            except Exception as e:
                logger.error(
                    f"Failed to generate tasks for scenario {idx + 1} "
                    f"(Scenario ID: {scenario_data.get('id')}): {str(e)}"
                )
                logger.warning("Continuing with next scenario...")
                continue

        logger.info(
            f"Successfully generated {len(all_tasks)} total tasks across {total} scenarios")
        return all_tasks


class AsyncTaskGenerator(AsyncGenerator):
    def __init__(
        self,
        model_name="gpt-4o-mini",
        prompt_path="task_generator.yaml",
        n_tasks=25,
        temperature=1,
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
        self.n_tasks = n_tasks

    async def generate_tasks_for_scenario(
        self,
        persona: dict,
        global_style: dict,
        scenario: dict,
        scenario_idx: int,
        max_retries: int = 3
    ) -> List[Dict[str, str]]:
        """Generate N_TASKS writing tasks for a single scenario asynchronously."""
        logger.debug(f"  → Processing scenario {scenario_idx}...")

        last_error = None
        for attempt in range(max_retries):
            try:
                output = await self.generate_single(
                    # Persona fields
                    name=persona['name'],
                    age=persona['age'],
                    gender=persona['gender'],
                    demographics=persona['demographics'],
                    profession=persona['profession'],
                    education=persona['education'],
                    socioeconomic_status=persona['socioeconomic_status'],
                    brief_background=persona['brief_background'],
                    interests_lifestyle=persona['interests_lifestyle'],
                    # Style fields
                    tone=global_style.get('tone', ''),
                    syntax=global_style.get('syntax', ''),
                    lexicon=global_style.get('lexicon', ''),
                    linguistic_patterns=global_style.get(
                        'linguistic_patterns', ''),
                    # Scenario fields
                    scenario_description=scenario.get(
                        'scenario_description', scenario.get('scenario', '')),
                    detailed_style=scenario.get(
                        'detailed_style', scenario.get('contextual_style', '')),
                    scenario_linguistic_patterns=scenario.get(
                        'linguistic_patterns', ''),
                    # Task configuration
                    n_tasks=self.n_tasks
                )
                # print(output)

                # Parse structured output
                tasks = []
                output = json_repair.loads(output)
                output = WritingTasks(**output)
                tasks = [
                    {
                        'task_instructions': task.task_instructions,
                        'task_details': task.task_details,
                        'writing_content': task.writing_content
                    }
                    for task in output.tasks
                ]

                logger.debug(
                    f"  ✓ Completed scenario {scenario_idx}: {len(tasks)} tasks generated")
                return tasks

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for scenario {scenario_idx}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

        logger.error(
            f"Failed to generate tasks for scenario {scenario_idx} after {max_retries} attempts")
        return []  # Return empty list instead of raising to continue with other scenarios

    async def __call__(self, styled_scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate N_TASKS tasks for all scenarios in parallel."""
        total = len(styled_scenarios)
        logger.info(
            f"Generating {self.n_tasks} tasks per scenario for {total} scenarios "
            f"(parallel processing with max concurrency {self.max_concurrency})"
        )

        # Create async tasks for all scenarios (parallel execution)
        async_tasks = [
            self.generate_tasks_for_scenario(
                persona=scenario_data['persona'],
                global_style=scenario_data.get('global_style', {}),
                scenario=scenario_data,
                scenario_idx=idx + 1
            )
            for idx, scenario_data in enumerate(styled_scenarios)
        ]

        # Execute all tasks in parallel with asyncio.gather
        scenario_results = await asyncio.gather(*async_tasks)

        # Flatten results and embed context
        all_tasks = []
        for scenario_data, tasks in zip(styled_scenarios, scenario_results):
            for task in tasks:
                task_entry = {
                    # Context fields (embedded)
                    'persona': scenario_data['persona'],
                    'general_writing_style': scenario_data.get('global_style', {}),
                    'contextual_writing_style': {
                        'scenario_description': scenario_data.get('scenario_description', ''),
                        'detailed_style': scenario_data.get('detailed_style', ''),
                        'linguistic_patterns': scenario_data.get('linguistic_patterns', '')
                    },
                    # Task fields (primary keys)
                    'task_instruction': task['task_instructions'],
                    'task_details': task['task_details'],
                    'personalized': task['writing_content'],
                }
                all_tasks.append(task_entry)

        logger.info(
            f"✓ All {total} scenarios completed! Generated {len(all_tasks)} total tasks "
            f"(expected: {total * self.n_tasks})"
        )
        return all_tasks
