from typing import List, Optional, Dict, Tuple
import torch
from tqdm import tqdm


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


def generate_batch(
    model,
    tokenizer,
    chats: List[str],
    description: str,
    use_steering: bool,
    steering_vector=None,
    multiplier: float = 1.0,
    batch_size: int = 1,
):
    """Generate text for a batch of chats with optional steering.

    For batch_size <= 1, uses generate_text for single-item generation.
    For batch_size > 1, [TODO]
    """
    responses: List[str] = []

    if batch_size <= 1:
        # Use single-item generation
        for chat in tqdm(chats, desc=description):
            responses.append(
                generate_text(
                    model,
                    tokenizer,
                    chat,
                    use_steering=use_steering,
                    steering_vector=steering_vector,
                    multiplier=multiplier,
                )
            )

    # TODO: Implement batch generation for batch_size > 1

    return responses
