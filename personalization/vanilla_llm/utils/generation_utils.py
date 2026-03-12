from typing import List
from tqdm import tqdm


def generate_batch(
    model,
    tokenizer,
    chats: List[str],
    description: str,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
):
    """Generate text for a batch of chats.

    For batch_size <= 1, uses single-item generation.
    For batch_size > 1, uses batched generation with left-padding for decoder-only models.
    """
    responses: List[str] = []

    if batch_size <= 1:
        # Use single-item generation
        for chat in tqdm(chats, desc=description):
            inputs = tokenizer(chat, return_tensors="pt",
                               add_special_tokens=False).to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            responses.append(response)
    else:
        # Batched generation with left-padding for decoder-only models
        # Save original padding side and restore after
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        # Ensure pad token is set (use eos_token if not available)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Process in batches
        for i in tqdm(range(0, len(chats), batch_size), desc=description):
            batch_chats = chats[i:i + batch_size]

            # Tokenize batch with padding
            inputs = tokenizer(
                batch_chats,
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
            for j, output in enumerate(out):
                response = tokenizer.decode(
                    output[input_length:], skip_special_tokens=True)
                responses.append(response)

        # Restore original padding side
        tokenizer.padding_side = original_padding_side

    return responses
