import argparse
import logging
import os
import asyncio
from synthesis.pipeline.persona import generate_personas
from synthesis.pipeline.global_style import generate_global_styles
from synthesis.pipeline.contextual_style import generate_contextual_styles_async
from synthesis.pipeline.task import generate_tasks_async
from synthesis.pipeline.prompt import generate_prompts_async
from synthesis.pipeline.style_agnostic import generate_style_agnostic
from synthesis.utils.files import load_json, get_persona_dir


def is_persona_complete(output_dir: str, persona_id: int) -> bool:
    """Check if a persona has all required files indicating completion."""
    persona_dir = get_persona_dir(output_dir, persona_id)
    required_files = ['persona.json',
                      'styles.json', 'tasks.json', 'dataset.json']

    for filename in required_files:
        file_path = os.path.join(persona_dir, filename)
        if not os.path.exists(file_path):
            return False

    return True


async def process_single_persona(
    persona_id: int,
    persona: dict,
    args,
    logger
):
    """Process a single persona through the entire pipeline (Rounds 2-6)."""
    try:
        logger.info("="*70)
        logger.info(f"Processing Persona {persona_id}")
        logger.info("="*70)

        # Round 2: Generate global style for this persona
        logger.info(
            f"[Persona {persona_id}] Step 2/6: Generating global style...")
        persona = generate_global_styles(args.model, persona, args.output_dir)
        logger.info(
            f"[Persona {persona_id}] Successfully generated global style")

        # Round 3: Generate scenarios with contextual styles
        logger.info(
            f"[Persona {persona_id}] Step 3/6: Generating {args.n_scenarios} scenarios...")
        scenarios = await generate_contextual_styles_async(
            args.model, persona, args.n_scenarios, args.output_dir,
            args.max_concurrency, args.writing_type
        )
        logger.info(
            f"[Persona {persona_id}] Successfully generated {len(scenarios)} scenarios")

        # Round 4: Generate tasks for each scenario
        logger.info(f"[Persona {persona_id}] Step 4/6: Generating tasks...")
        tasks = await generate_tasks_async(
            args.model, scenarios, args.output_dir, args.n_tasks, args.max_concurrency
        )
        logger.info(
            f"[Persona {persona_id}] Successfully generated {len(tasks)} tasks")

        # Round 5: Generate prompts (task specifications)
        logger.info(
            f"[Persona {persona_id}] Step 5/6: Generating task specifications...")
        tasks = await generate_prompts_async(
            args.model, tasks, args.output_dir, args.max_concurrency,
            writing_type=args.writing_type
        )
        logger.info(
            f"[Persona {persona_id}] Successfully generated task specifications")

        # Round 6: Generate style-agnostic text
        if not args.skip_style_agnostic:
            logger.info(
                f"[Persona {persona_id}] Step 6/6: Generating style-agnostic text...")
            tasks = generate_style_agnostic(
                args.local_model, tasks, args.output_dir,
                writing_type=args.writing_type)
            logger.info(
                f"[Persona {persona_id}] Successfully generated style-agnostic text")
        else:
            logger.info(
                f"[Persona {persona_id}] Step 6/6: Skipped (style-agnostic generation)")

        logger.info("="*70)
        logger.info(f"PERSONA {persona_id} COMPLETE!")
        logger.info(f"  Scenarios: {len(scenarios)}")
        logger.info(f"  Tasks: {len(tasks)}")
        logger.info("="*70)

        return {
            'persona_id': persona_id,
            'persona': persona,
            'scenarios': len(scenarios),
            'tasks': len(tasks),
            'success': True
        }

    except Exception as e:
        logger.error(f"[Persona {persona_id}] Failed: {str(e)}")
        logger.exception(f"[Persona {persona_id}] Full error details:")
        return {
            'persona_id': persona_id,
            'success': False,
            'error': str(e)
        }


async def main_async(args, logger):
    """Async main function that processes personas sequentially."""
    results = []
    successful_personas = 0
    failed_personas = 0

    # Round 1: Generate all personas first
    logger.info("="*70)
    logger.info("ROUND 1: GENERATING ALL PERSONAS")
    logger.info("="*70)
    logger.info(f"Generating {args.n_personas} personas...")
    all_personas = generate_personas(
        args.model, args.n_personas, args.output_dir)
    logger.info(f"Successfully generated {len(all_personas)} personas")
    logger.info("="*70)

    # Rounds 2-6: Process each persona sequentially
    for persona_id in range(1, args.n_personas + 1):
        # Check if persona is already complete
        if is_persona_complete(args.output_dir, persona_id):
            logger.info("="*70)
            logger.info(f"Persona {persona_id} already complete - skipping")
            logger.info("="*70)
            results.append({
                'persona_id': persona_id,
                'success': True,
                'skipped': True
            })
            successful_personas += 1
            continue

        # Get the persona object for this ID
        persona = [p for p in all_personas if p.get('id') == persona_id][0]
        result = await process_single_persona(persona_id, persona, args, logger)
        results.append(result)

        if result['success']:
            successful_personas += 1
        else:
            failed_personas += 1
            logger.warning(
                f"Persona {persona_id} failed but continuing with next persona...")

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(
        f"Successfully processed: {successful_personas}/{args.n_personas} personas")

    # Calculate totals from successful personas
    total_scenarios = sum(r.get('scenarios', 0)
                          for r in results if r['success'])
    total_tasks = sum(r.get('tasks', 0) for r in results if r['success'])

    logger.info(f"Total scenarios generated: {total_scenarios}")
    logger.info(f"Total tasks generated: {total_tasks}")
    logger.info(
        f"Expected per persona: {args.n_scenarios} scenarios × {args.n_tasks} tasks")
    logger.info("="*70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse personas, scenarios, and tasks (v2 - task-based pipeline)")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for the generated data", required=True)
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="Model to use for generation")
    parser.add_argument("--n_personas", type=int, default=5,
                        help="Number of personas to generate (default: 5)")
    parser.add_argument("--n_scenarios", type=int, default=20,
                        help="Number of scenarios to generate per persona (default: 20)")
    parser.add_argument("--n_tasks", type=int, default=25,
                        help="Number of tasks to generate per scenario (default: 25)")
    parser.add_argument("--writing-type", type=str, default="email",
                        help="Type of writing to generate (default: email)")
    parser.add_argument("--max-concurrency", type=int, default=20,
                        help="Maximum number of concurrent API calls (default: 20)")
    parser.add_argument("--local-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Local model to use for style-agnostic generation (default: meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--skip-style-agnostic", action="store_true",
                        help="Skip Round 6 (style-agnostic generation)")
    args = parser.parse_args()

    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting generation with parameters:")
        logger.info(f"  Output directory: {args.output_dir}")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Max concurrency: {args.max_concurrency}")
        logger.info(f"  Number of personas: {args.n_personas}")
        logger.info(f"  Number of scenarios per persona: {args.n_scenarios}")
        logger.info(f"  Number of tasks per scenario: {args.n_tasks}")
        logger.info(
            f"  Expected total tasks: {args.n_personas * args.n_scenarios * args.n_tasks}")

        # Run async main function
        results = asyncio.run(main_async(args, logger))

        # If any personas failed, raise an error at the end
        failed = [r for r in results if not r['success']]
        if failed:
            raise Exception(
                f"{len(failed)} persona(s) failed to process completely")

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        logger.exception("Full error details:")
        raise


if __name__ == "__main__":
    main()
