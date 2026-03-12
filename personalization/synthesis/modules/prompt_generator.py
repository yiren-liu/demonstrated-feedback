import logging
import asyncio
from typing import List, Dict, Any, Optional

from synthesis.utils.generation import Generator, AsyncGenerator

logger = logging.getLogger(__name__)


class PromptGenerator(Generator):
    def __init__(
        self,
        model_name="gpt-5-mini",
        prompt_path="prompt_generator.yaml",
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

    def generate_prompt(
        self,
        personalized_content: str,
        writing_type: str = "email",
        max_retries: int = 3
    ) -> Optional[str]:
        """Generate a task specification from personalized content."""
        if not personalized_content:
            logger.warning(
                "Empty personalized content, skipping prompt generation")
            return None

        last_error = None
        for attempt in range(max_retries):
            try:
                # Call LLM to generate prompt
                output = super().__call__(
                    personalized_content=personalized_content,
                    writing_type=writing_type
                )

                # The output should be a string (the task specification)
                if isinstance(output, str):
                    return output.strip()
                else:
                    logger.warning(f"Unexpected output type: {type(output)}")
                    return str(output).strip()

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for prompt generation: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
                    continue

        logger.error(
            f"Error generating prompt after {max_retries} attempts: {str(last_error)}")
        return None

    def __call__(self, tasks: list, writing_type: str = "email") -> list:
        """Generate prompts for all tasks."""
        total = len(tasks)
        logger.info(f"Generating task specifications for {total} tasks")

        updated_tasks = []
        prompts_generated = 0

        for idx, task_entry in enumerate(tasks):
            try:
                # Skip if already has prompt
                if "prompt" in task_entry and task_entry["prompt"]:
                    logger.debug(
                        f"[{idx+1}/{total}] Skipping - already has prompt")
                    updated_tasks.append(task_entry)
                    continue

                personalized = task_entry.get("personalized", "")

                # Generate prompt
                prompt = self.generate_prompt(personalized, writing_type)

                if prompt:
                    task_entry["prompt"] = prompt
                    prompts_generated += 1
                    logger.info(
                        f"[{idx+1}/{total}] Generated: {prompt[:80]}...")
                else:
                    logger.warning(
                        f"[{idx+1}/{total}] Failed to generate prompt")

                updated_tasks.append(task_entry)

            except Exception as e:
                logger.error(
                    f"Failed to process task {idx + 1}: {str(e)}"
                )
                updated_tasks.append(task_entry)
                continue

        logger.info(
            f"Successfully generated {prompts_generated}/{total} task specifications")
        return updated_tasks


class AsyncPromptGenerator(AsyncGenerator):
    def __init__(
        self,
        model_name="gpt-5-mini",
        prompt_path="prompt_generator.yaml",
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

    async def generate_prompt(
        self,
        personalized_content: str,
        writing_type: str,
        task_idx: int,
        total: int,
        max_retries: int = 3
    ) -> Optional[str]:
        """Generate a task specification from personalized content asynchronously."""
        if not personalized_content:
            logger.debug(f"[{task_idx+1}/{total}] Empty content, skipping")
            return None

        logger.debug(
            f"[{task_idx+1}/{total}] Generating task specification...")

        last_error = None
        for attempt in range(max_retries):
            try:
                output = await self.generate_single(
                    personalized_content=personalized_content,
                    writing_type=writing_type
                )

                # The output should be a string (the task specification)
                if isinstance(output, str):
                    prompt = output.strip()
                    logger.debug(
                        f"[{task_idx+1}/{total}] Generated: {prompt[:80]}...")
                    return prompt
                else:
                    logger.warning(
                        f"[{task_idx+1}/{total}] Unexpected output type: {type(output)}")
                    return str(output).strip()

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{task_idx+1}/{total}] Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

        logger.error(
            f"[{task_idx+1}/{total}] Failed after {max_retries} attempts")
        return None

    async def __call__(self, tasks: List[Dict[str, Any]], writing_type: str = "email") -> List[Dict[str, Any]]:
        """Generate prompts for all tasks in parallel."""
        total = len(tasks)
        logger.info(
            f"Generating task specifications for {total} tasks "
            f"(parallel processing with max concurrency {self.max_concurrency})"
        )

        # Create async tasks for all entries (parallel execution)
        async_tasks = [
            self.generate_prompt(
                personalized_content=task_entry.get("personalized", ""),
                writing_type=writing_type,
                task_idx=idx,
                total=total
            )
            for idx, task_entry in enumerate(tasks)
        ]

        # Execute all tasks in parallel with asyncio.gather
        prompts = await asyncio.gather(*async_tasks)

        # Update tasks with generated prompts
        updated_tasks = []
        prompts_generated = 0

        for task_entry, prompt in zip(tasks, prompts):
            # Skip if already has prompt
            if "prompt" in task_entry and task_entry["prompt"]:
                updated_tasks.append(task_entry)
                continue

            if prompt:
                task_entry["prompt"] = prompt
                prompts_generated += 1

            updated_tasks.append(task_entry)

        logger.info(
            f"✓ All {total} tasks completed! Generated {prompts_generated} task specifications"
        )
        return updated_tasks
