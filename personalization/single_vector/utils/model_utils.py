from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple


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


SYS_PROMPT = "You are a helpful assistant."


def build_messages(prompt: str = "", output: str | None = None):
    user_prompt = f"Help me write an email: {prompt} Return the email body only."

    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if output is not None:
        msgs.append({"role": "assistant", "content": output})
    return msgs

