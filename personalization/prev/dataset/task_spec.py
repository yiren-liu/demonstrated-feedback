import json
import os
import asyncio
from pathlib import Path
from openai import AsyncOpenAI

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# System prompt for task specification extraction
SYSTEM_PROMPT = """Generate a task specification that begins with “Write an email” so another LLM can draft a new message that achieves the same goal as the main body of a given email. List the key objectives and facts in the task specification. Do not include any stylistic elements, personal opinions, or writing habits."""


async def generate_task_spec(personalized_email):
    """Generate task specification from personalized email"""
    try:
        response = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": personalized_email}
            ],
            temperature=1,
            seed=42,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating task spec: {e}")
        return None


async def process_entry(idx, entry, total):
    """Process a single entry asynchronously."""
    # Skip if already has prompt
    if "prompt" in entry:
        print(f"[{idx+1}/{total}] Skipping - already has prompt")
        return idx, None

    personalized = entry.get("personalized", "")
    if not personalized:
        print(f"[{idx+1}/{total}] Skipping - no personalized email")
        return idx, None

    print(f"[{idx+1}/{total}] Generating task spec...")
    task_spec = await generate_task_spec(personalized)

    if task_spec:
        print(f"[{idx+1}/{total}] Generated: {task_spec[:80]}...")
        return idx, task_spec
    else:
        print(f"[{idx+1}/{total}] Failed to generate")
        return idx, None


async def process_dataset_async(input_path, output_path, batch_size=20):
    """Process dataset and add task specifications with async batch processing."""
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total = len(dataset)
    print(f"Found {total} entries to process")
    print(f"Using batch size: {batch_size}")

    # Process entries in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_entries = dataset[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start+1}-{batch_end}...")

        # Create tasks for this batch
        tasks = [
            process_entry(batch_start + i, entry, total)
            for i, entry in enumerate(batch_entries)
        ]

        # Process batch concurrently
        results = await asyncio.gather(*tasks)

        # Update dataset with results
        for idx, task_spec in results:
            if task_spec:
                dataset[idx]["prompt"] = task_spec

        # Save progress after each batch
        print(f"Saving progress after batch {batch_start+1}-{batch_end}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nComplete! Processed {total} entries")


def process_dataset(input_path, output_path, batch_size=20):
    """Wrapper function to run async processing."""
    asyncio.run(process_dataset_async(input_path, output_path, batch_size))


if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent / \
        "v5-gpt-5-mini-update-v2" / "dataset.json"
    process_dataset(dataset_path, dataset_path)
