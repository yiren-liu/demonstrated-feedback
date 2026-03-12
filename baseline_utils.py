"""
Shared utilities for prompt-based and RAG-based baselines.

Provides a unified generation backend abstraction for both local Mistral-7B-Instruct
(no adapter) and OpenAI API models, plus common data loading and output helpers.
"""

import argparse
import json
import os
import pickle

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_scenario import MISTRAL_CHAT_TEMPLATE

load_dotenv()


# ── Data helpers ──

def load_author_data(pkl_path, author_key):
    """Load training samples for a single author from a pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data[author_key]


def write_output_json(output_path, author_key, method_desc, test_pkl, num_samples, results):
    """Write eval-compatible JSON output matching generate_scenario.py format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "author_key": author_key,
            "model_dir": method_desc,
            "test_pkl": test_pkl,
            "num_samples": num_samples,
            "results": results,
        }, f, indent=2)
    print(f"Wrote {len(results)} prompts x {num_samples} samples -> {output_path}")


# ── Backend abstraction ──

class MistralBackend:
    """Local Mistral-7B-Instruct generation (no adapter)."""

    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def generate(self, messages, num_samples=1, max_new_tokens=1024):
        """Generate completions from a list of chat messages.

        Args:
            messages: list of dicts with 'role' and 'content' keys
            num_samples: number of completions to return
            max_new_tokens: max tokens per completion

        Returns:
            list[str] of length num_samples
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True
        ).to("cuda")

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for i in range(num_samples):
            gen_ids = outputs[i][prompt_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(gen_text)
        return results


class OpenAIBackend:
    """OpenAI API generation backend with concurrent requests."""

    def __init__(self, model="gpt-4o", max_workers=8):
        import openai
        self.client = openai.OpenAI()
        self.model = model
        self.max_workers = max_workers

    def _single_completion(self, messages, max_new_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1.0,
            max_completion_tokens=max_new_tokens,
        )
        return response.choices[0].message.content.strip()

    def generate(self, messages, num_samples=1, max_new_tokens=1024*5):
        """Generate completions via OpenAI chat API (concurrent).

        Args:
            messages: list of dicts with 'role' and 'content' keys
            num_samples: number of completions to return
            max_new_tokens: max tokens per completion

        Returns:
            list[str] of length num_samples
        """
        if num_samples == 1:
            return [self._single_completion(messages, max_new_tokens)]

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(num_samples, self.max_workers)) as pool:
            futures = [
                pool.submit(self._single_completion, messages, max_new_tokens)
                for _ in range(num_samples)
            ]
            return [f.result() for f in futures]


def get_backend(name, openai_model="gpt-4o"):
    """Factory to create a generation backend.

    Args:
        name: 'mistral' or 'openai'
        openai_model: model name for OpenAI backend (ignored for mistral)
    """
    if name == "mistral":
        return MistralBackend()
    elif name == "openai":
        return OpenAIBackend(model=openai_model)
    else:
        raise ValueError(f"Unknown backend: {name}")


def add_backend_args(parser, prefix=""):
    """Add --backend and --openai_model arguments to an argparse parser."""
    p = f"{prefix}_" if prefix else ""
    parser.add_argument(f"--{p}backend", type=str, default="mistral",
                        choices=["mistral", "openai"],
                        help=f"Generation backend (default: mistral)")
    parser.add_argument(f"--{p}openai_model", type=str, default="gpt-4o",
                        help=f"OpenAI model name (default: gpt-4o)")
