from typing import List, Optional, Dict, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)


def generate_text(model, tokenizer, chat, use_steering=True, steering_vector=None, multiplier=1.0):
    inputs = tokenizer(chat, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    def _gen():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            # do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if use_steering:
        with steering_vector.apply(model, multiplier=multiplier):
            return _gen()
    else:
        return _gen()


def generate_text_batch(
    model,
    tokenizer,
    chats: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> List[str]:
    """Generate text for a batch of chats with left-padding for decoder-only models."""
    if not chats:
        return []

    # Save original padding side and restore after
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Ensure pad token is set (use eos_token if not available)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize batch with padding
    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    ).to(model.device)

    # Generate with attention mask
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode each response, removing the input tokens
    input_length = inputs["input_ids"].shape[1]
    responses = []
    for output in out:
        response = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        responses.append(response.strip())

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    return responses


def get_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cpu":
        print("WARNING: Running on CPU - this will be very slow!")

    return model, tokenizer


def build_messages(prompt: str = "", output: str | None = None, writing_type: str = "email"):
    user_prompt = f"Help me write an {writing_type}: {prompt} Return the {writing_type} only."

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]
    if output is not None:
        msgs.append({"role": "assistant", "content": output})
    return msgs


class StyleAgnosticGenerator:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        verbose: bool = False,
        writing_type: str = "email",
        batch_size: int = 1
    ):
        self.model_name = model_name
        self.verbose = verbose
        self.writing_type = writing_type
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the local model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = get_model_and_tokenizer(self.model_name)
        logger.info("Model loaded successfully")

    def _build_chat(self, prompt: str) -> str:
        """Build chat string from a prompt."""
        messages = build_messages(prompt, writing_type=self.writing_type)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate_style_agnostic(
        self,
        prompt: str,
        task_idx: int,
        total: int,
        max_retries: int = 3
    ) -> Optional[str]:
        """Generate style-agnostic text from a prompt."""
        if not prompt:
            logger.debug(f"[{task_idx+1}/{total}] Empty prompt, skipping")
            return None

        chat = self._build_chat(prompt)

        last_error = None
        for attempt in range(max_retries):
            try:
                # Generate style-agnostic text (no steering)
                style_agnostic = generate_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    chat=chat,
                    use_steering=False,
                )

                if style_agnostic:
                    if self.verbose:
                        logger.info(
                            f"[{task_idx+1}/{total}] Generated: {style_agnostic[:80]}...")
                    return style_agnostic.strip()
                else:
                    logger.warning(
                        f"[{task_idx+1}/{total}] Empty generation result")

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{task_idx+1}/{total}] Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... ({attempt + 2}/{max_retries})")
                    continue

        logger.error(
            f"[{task_idx+1}/{total}] Failed after {max_retries} attempts: {str(last_error)}")
        return None

    def generate_style_agnostic_batch(
        self,
        prompts: List[str],
        max_retries: int = 3
    ) -> List[Optional[str]]:
        """Generate style-agnostic text for a batch of prompts."""
        if not prompts:
            return []

        # Build chats for all prompts
        chats = [self._build_chat(prompt) for prompt in prompts]

        last_error = None
        for attempt in range(max_retries):
            try:
                # Generate style-agnostic text in batch
                results = generate_text_batch(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    chats=chats,
                )
                return results

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Batch generation attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying batch... ({attempt + 2}/{max_retries})")
                    continue

        logger.error(f"Batch generation failed after {max_retries} attempts: {str(last_error)}")
        # Return None for all prompts on failure
        return [None] * len(prompts)

    def __call__(self, tasks: List[Dict[str, Any]], batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate style-agnostic text for all tasks.
        
        Args:
            tasks: List of task dictionaries containing prompts
            batch_size: Override instance batch_size if provided
        """
        total = len(tasks)
        logger.info(f"Generating style-agnostic text for {total} tasks")

        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        updated_tasks = []
        generated_count = 0

        if effective_batch_size <= 1:
            # Single-item generation (original behavior)
            for idx, task_entry in enumerate(tqdm(tasks, desc="Processing tasks")):
                try:
                    # Skip if already has style_agnostic
                    if "style_agnostic" in task_entry and task_entry["style_agnostic"]:
                        logger.debug(
                            f"[{idx+1}/{total}] Skipping - already has style_agnostic")
                        updated_tasks.append(task_entry)
                        continue

                    # Get the prompt
                    prompt = task_entry.get("prompt", "")
                    if not prompt:
                        logger.warning(
                            f"[{idx+1}/{total}] No prompt found, skipping")
                        updated_tasks.append(task_entry)
                        continue

                    # Generate style-agnostic text
                    style_agnostic = self.generate_style_agnostic(
                        prompt=prompt,
                        task_idx=idx,
                        total=total
                    )

                    if style_agnostic:
                        task_entry["style_agnostic"] = style_agnostic
                        generated_count += 1
                    else:
                        logger.warning(
                            f"[{idx+1}/{total}] Failed to generate style_agnostic")

                    updated_tasks.append(task_entry)

                except Exception as e:
                    logger.error(
                        f"[{idx+1}/{total}] Failed to process task: {str(e)}")
                    updated_tasks.append(task_entry)
                    continue
        else:
            # Batched generation with left-padding
            logger.info(f"Using batched generation with batch_size={effective_batch_size}")

            # First pass: identify which tasks need generation
            tasks_to_process = []
            task_indices = []
            for idx, task_entry in enumerate(tasks):
                if "style_agnostic" in task_entry and task_entry["style_agnostic"]:
                    logger.debug(f"[{idx+1}/{total}] Skipping - already has style_agnostic")
                    continue
                prompt = task_entry.get("prompt", "")
                if not prompt:
                    logger.warning(f"[{idx+1}/{total}] No prompt found, skipping")
                    continue
                tasks_to_process.append(prompt)
                task_indices.append(idx)

            # Process in batches
            results_map = {}
            for batch_start in tqdm(range(0, len(tasks_to_process), effective_batch_size), 
                                    desc="Processing batches"):
                batch_end = min(batch_start + effective_batch_size, len(tasks_to_process))
                batch_prompts = tasks_to_process[batch_start:batch_end]
                batch_indices = task_indices[batch_start:batch_end]

                # Generate for this batch
                batch_results = self.generate_style_agnostic_batch(batch_prompts)

                # Map results back to task indices
                for idx, result in zip(batch_indices, batch_results):
                    results_map[idx] = result

            # Second pass: update tasks with results
            for idx, task_entry in enumerate(tasks):
                if idx in results_map:
                    result = results_map[idx]
                    if result:
                        task_entry["style_agnostic"] = result
                        generated_count += 1
                    else:
                        logger.warning(f"[{idx+1}/{total}] Failed to generate style_agnostic")
                updated_tasks.append(task_entry)

        logger.info(
            f"Successfully generated {generated_count}/{total} style-agnostic texts")
        return updated_tasks
