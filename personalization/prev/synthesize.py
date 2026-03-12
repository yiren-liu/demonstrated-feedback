import json
import asyncio
import random
from typing import List, Dict, Any
from openai import AsyncOpenAI
from pydantic import BaseModel


# Initialize the async client
client = AsyncOpenAI(
    api_key=open("./openai.key").read().strip(),
)

# Hyperparameters
N_PERSONAS = 5
N_SCENARIOS = 20
N_TASKS = 25


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Persona(BaseModel):
    name: str
    age: int
    gender: str
    demographics: str
    profession: str
    education: str
    socioeconomic_status: str
    brief_background: str
    interests_lifestyle: str


class Personas(BaseModel):
    personas: List[Persona]


class WritingStyle(BaseModel):
    tone: str
    syntax: str
    lexicon: str
    linguistic_patterns: str


class ContextualWritingStyle(BaseModel):
    scenario_description: str
    detailed_style: str
    linguistic_patterns: str


class ContextualWritingStyles(BaseModel):
    scenarios: List[ContextualWritingStyle]


class WritingTask(BaseModel):
    task_instructions: str
    task_details: str
    writing_content: str


class WritingTasks(BaseModel):
    tasks: List[WritingTask]


class StyleAgnosticEmail(BaseModel):
    email: str


test_persona = Persona(
    name='Roy "Big Dog" Callahan',
    age=62,
    gender="Male",
    demographics="Rural; Caucasian/White; Lifelong resident of Harlan County, KY (Appalachian region); Widowed.",
    profession="Self-Employed Owner/Operator of a Small Landscaping/Hauling Business (Semi-Retired Trucker)",
    education="High School Diploma (Local Harlan County High School)",
    socioeconomic_status="Lower-Middle Income; Struggling to maintain retirement savings.",
    brief_background="Roy spent 40 years on the road, hauling everything from lumber to produce across the country. He bought his own big rig 20 years ago, eventually semi-retiring a few years ago due to minor health issues. He is deeply connected to his community, is an excellent mechanic, and spends most of his free time documenting the forgotten coal mining history and local genealogy of his county. He values reliability, plain language, and self-sufficiency.",
    interests_lifestyle="Restoring antique tractors, Ham Radio operation (Call Sign: K4BD), discussing local politics and infrastructure repair, gospel music, grilling, and visiting flea markets for antique mining memorabilia."
)

test_style = WritingStyle(
    tone="No-nonsense, informative, and slightly guarded until trust is established. Focuses on actionable information or verifiable facts.",
    syntax="Predominantly simple and compound sentences. Minimal use of subordinate clauses. Avoids ambiguity or unnecessary flourish.",
    lexicon='Utilizes vocational terminology (trucking, mechanics, hauling, equipment, logistics) and plain, Anglo-Saxon vocabulary. Prefers established American measurements (feet, pounds, acres).',
    linguistic_patterns='Heavy use of contractions ("ain\'t," "y\'all," "it\'s"). Frequent use of pragmatic markers or simple interjections common in conversation ("Well now," "Listen here," "Shoot").'
)

# ============================================================================
# ROUND 1: GENERATE PERSONAS
# ============================================================================


async def generate_personas():
    """Generate 3 diverse personas."""
    print("\n" + "="*70)
    print("ROUND 1: Generating Personas")
    print("="*70)

    prompt = """Try to synthesize {N_PERSONAS} completely different personas with different demographic and profession backgrounds.

Return in a structured format."""

    response = await client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        seed=42,
        response_format=Personas
    )

    personas = response.choices[0].message.parsed.personas

    # Print personas
    for i, persona in enumerate(personas, 1):
        print(f"\n{'─'*60}")
        print(f"PERSONA {i}: {persona.name}")
        print(f"{'─'*60}")
        print(f"Age: {persona.age}")
        print(f"Profession: {persona.profession}")
        print(f"Demographics: {persona.demographics}")

    return personas


# ============================================================================
# ROUND 2: GENERATE GENERAL WRITING STYLE
# ============================================================================

async def generate_writing_style(persona: Persona):
    """Generate general writing style for a persona."""
    print(f"\n{'─'*60}")
    print(f"ROUND 2: Generating Writing Style for {persona.name}")
    print(f"{'─'*60}")

    user_prompt = f"""Think of a few personal writing styles that reasonably fit this background. 
They should be general styles for daily communication to any audience and are NOT specific to any particular scenarios. 
Make sure to include some specific linguistic patterns. Return the style summary directly without explanations. 

- Name: {persona.name}
- Age: {persona.age}
- Gender: {persona.gender}
- Demographics: {persona.demographics}
- Profession: {persona.profession}
- Education: {persona.education}
- Socioeconomic Status: {persona.socioeconomic_status}
- Background: {persona.brief_background}
- Interests/Lifestyle: {persona.interests_lifestyle}
"""

    response = await client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1,
        seed=42,
        response_format=WritingStyle
    )

    style = response.choices[0].message.parsed
    print(f"Tone: {style.tone}")

    return style


# ============================================================================
# ROUND 3: GENERATE SCENARIO-SPECIFIC WRITING STYLES
# ============================================================================

async def generate_scenarios(persona: Persona, style: WritingStyle):
    """Generate N_SCENARIOS scenario-specific writing styles."""
    print(f"\n{'─'*60}")
    print(f"ROUND 3: Generating Scenarios for {persona.name}")
    print(f"{'─'*60}")

    user_prompt = f"""You are embodying this person:

**Background:**
- Name: {persona.name}
- Age: {persona.age}
- Gender: {persona.gender}
- Demographics: {persona.demographics}
- Profession: {persona.profession}
- Education: {persona.education}
- Socioeconomic Status: {persona.socioeconomic_status}
- Background: {persona.brief_background}
- Interests/Lifestyle: {persona.interests_lifestyle}

**General Writing Style:**
- Tone: {style.tone}
- Syntax: {style.syntax}
- Lexicon: {style.lexicon}
- Linguistic Patterns: {style.linguistic_patterns}

Now imagine you are this person with the aforementioned background and general writing styles. 
Now think of {N_SCENARIOS} diverse scenarios in which you need to write an email for communication. Do not include any specific details about the scenarios. It should be high-level, e.g., "to a friend.", "to a business partner.", "to a family member.", etc.
Tell me more detailed writing styles you will use for these {N_SCENARIOS} scenarios. They should be reasonable and coherent to your general writing styles.
Avoid using numbered lists and bullet points in any scenario.
"""  # Note: This is already inside an f-string

    response = await client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1,
        seed=42,
        response_format=ContextualWritingStyles
    )

    scenarios = response.choices[0].message.parsed.scenarios
    print(f"Generated {len(scenarios)} scenarios (expecting {N_SCENARIOS})")

    return scenarios


# ============================================================================
# ROUND 4: GENERATE TASKS AND EMAILS (PARALLELIZED)
# ============================================================================

async def generate_tasks_for_scenario(
    persona: Persona,
    style: WritingStyle,
    scenario: ContextualWritingStyle,
    scenario_idx: int
):
    """Generate N_TASKS task instructions and emails for a specific scenario."""
    print(
        f"  → Processing scenario {scenario_idx}: {scenario.scenario_description[:50]}...")

    user_prompt = f"""You are embodying this person:

**Background:**
- Name: {persona.name}
- Age: {persona.age}
- Gender: {persona.gender}
- Demographics: {persona.demographics}
- Profession: {persona.profession}
- Education: {persona.education}
- Socioeconomic Status: {persona.socioeconomic_status}
- Background: {persona.brief_background}
- Interests/Lifestyle: {persona.interests_lifestyle}

**General Writing Style:**
- Tone: {style.tone}
- Syntax: {style.syntax}
- Lexicon: {style.lexicon}
- Linguistic Patterns: {style.linguistic_patterns}

**Task Specifications:**
Task Audience: {scenario.scenario_description}
Task Writing Style: {scenario.detailed_style}
Task Linguistic Patterns: {scenario.linguistic_patterns}

For the above scenario, generate {N_TASKS} diverse task instructions that are reasonable to your background and the task audience. 
Put additional specification as the task details. Make sure the task details is self-contained and all the information in the email is covered by the task instructions and task details. 
The task instruction and task details should be style-neutral and short.
For each task, write the email based on the corresponding task writing style. Make sure the tone, format, and content all fit your background and general writing style.
"""  # Note: This is already inside an f-string

    response = await client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1,
        seed=42,
        response_format=WritingTasks
    )

    writing_tasks = response.choices[0].message.parsed.tasks
    print(
        f"  ✓ Completed scenario {scenario_idx}: {len(writing_tasks)} tasks generated (expecting {N_TASKS})")

    return {
        "scenario": scenario,
        "tasks": writing_tasks
    }


async def generate_all_tasks(persona: Persona, style: WritingStyle, scenarios: List[ContextualWritingStyle]):
    """Generate tasks and emails for all scenarios in parallel (N_TASKS tasks per scenario)."""
    print(f"\n{'─'*60}")
    print(f"ROUND 4: Generating Tasks (Parallelized)")
    print(f"{'─'*60}")
    print(
        f"Processing {len(scenarios)} scenarios in parallel ({N_TASKS} tasks each)...")

    # Create tasks for all scenarios
    tasks = [
        generate_tasks_for_scenario(persona, style, scenario, idx)
        for idx, scenario in enumerate(scenarios, 1)
    ]

    # Run all tasks in parallel
    results = await asyncio.gather(*tasks)

    print(f"\n✓ All {len(scenarios)} scenarios completed!")
    return results

# ============================================================================
# ROUND 5: GENERATE STYLE-AGNOSTIC EMAILS (PARALLELIZED)
# ============================================================================


async def generate_style_agnostic(task_prompt: str):
    """Generate style-agnostic email for a task."""

    user_prompt = f"""Generate an email for the following task:
    {task_prompt}

    Make sure to write the email only based on the task instructions. Do not add any additional information. Return the email directly without explanations.
    """

    response = await client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1,
        seed=42,
        response_format=StyleAgnosticEmail
    )

    return response.choices[0].message.parsed.email


async def generate_all_style_agnostic(scenario_results: List[Dict[str, Any]]):
    """Generate style-agnostic emails for all tasks."""
    print(f"\n{'─'*60}")
    print(f"ROUND 5: Generating Style-Agnostic Emails")
    print(f"{'─'*60}")
    print(f"Processing {len(scenario_results)} scenarios in parallel...")

    # Create tasks for all tasks (outer loop first, then inner loop)
    tasks = [
        generate_style_agnostic(
            task.task_instructions + " " + task.task_details)
        for result in scenario_results
        for task in result["tasks"]
    ]

    # Run all tasks in parallel
    results = await asyncio.gather(*tasks)

    print(f"\n✓ All {len(results)} style-agnostic emails generated!")
    return results

# ============================================================================
# MAIN PIPELINE
# ============================================================================


async def main():
    """Main pipeline to generate all data."""
    """
    print("\n" + "="*70)
    print("PERSONA SYNTHESIS PIPELINE")
    print("="*70)
    
    # Round 1: Generate personas
    personas = await generate_personas()
    
    # For demonstration, process just the first persona
    # You can loop through all personas if needed
    persona = personas[0]
    
    # Round 2: Generate general writing style
    style = await generate_writing_style(persona)
    """

    persona = test_persona
    style = test_style

    # Round 3: Generate scenarios
    scenarios = await generate_scenarios(persona, style)

    # Round 4: Generate tasks and emails (parallelized)
    scenario_results = await generate_all_tasks(persona, style, scenarios)

    # Round 5: Generate style-agnostic emails (parallelized)
    style_agnostic_results = await generate_all_style_agnostic(scenario_results)

    # ============================================================================
    # SAVE TO JSON
    # ============================================================================

    print(f"\n{'='*70}")
    print("SAVING TO JSON")
    print("="*70)

    # Flatten the data structure - each entry is a task with full context
    output_data = []

    # Index to track position in style_agnostic_results
    style_agnostic_idx = 0

    for result in scenario_results:
        scenario = result["scenario"]
        for task in result["tasks"]:
            entry = {
                # Task fields (primary keys)
                "prompt": task.task_instructions + " " + task.task_details,
                "task_instruction": task.task_instructions,
                "task_details": task.task_details,
                "personalized": task.writing_content,
                "style_agnostic": style_agnostic_results[style_agnostic_idx],

                "persona": {
                    "name": persona.name,
                    "age": persona.age,
                    "gender": persona.gender,
                    "demographics": persona.demographics,
                    "profession": persona.profession,
                    "education": persona.education,
                    "socioeconomic_status": persona.socioeconomic_status,
                    "brief_background": persona.brief_background,
                    "interests_lifestyle": persona.interests_lifestyle
                },
                "general_writing_style": {
                    "tone": style.tone,
                    "syntax": style.syntax,
                    "lexicon": style.lexicon,
                    "linguistic_patterns": style.linguistic_patterns
                },
                "scenarios": {
                    "scenario_description": scenario.scenario_description,
                    "detailed_style": scenario.detailed_style,
                    "linguistic_patterns": scenario.linguistic_patterns,
                }
            }
            output_data.append(entry)
            style_agnostic_idx += 1

    output_file = "./synthesis_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Data saved to: {output_file}")
    print(f"  - Persona: {persona.name}")
    print(f"  - Scenarios: {len(scenario_results)}")
    print(f"  - Total tasks: {sum(len(r['tasks']) for r in scenario_results)}")
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
