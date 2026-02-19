"""
Generation script for the held-out scenario experiment.

Unlike generate.py, this accepts explicit paths for the model adapter
and test pickle, and writes structured JSON output for downstream eval.

Usage:
    python generate_scenario.py \
        --model_dir outputs/cmcc-seen-s0-a0/ditto \
        --test_pkl  benchmarks/cmcc/processed/cmcc_seen_s0_test.pkl \
        --author_key 0 \
        --output_json outputs/cmcc-seen-s0-a0/generations.json \
        --num_samples 3
"""

import argparse
import json
import os
import pickle

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MISTRAL_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'].strip() + '\\n\\n' %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = '' %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = system_message + message['content'] %}"
    "{% else %}"
    "{% set content = message['content'] %}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' '  + content.strip() + ' ' + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)


def main():
    parser = argparse.ArgumentParser(description="Scenario experiment generation")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to DITTO adapter directory (e.g. outputs/.../ditto)")
    parser.add_argument("--test_pkl", type=str, required=True,
                        help="Path to test pickle file")
    parser.add_argument("--author_key", type=int, required=True,
                        help="Author key in the pickle")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to write generation results as JSON")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to generate per prompt")
    parser.add_argument("--model_id", type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model HF identifier")
    args = parser.parse_args()

    # ── load model ──
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    base_model = PeftModel.from_pretrained(base_model, args.model_dir)
    base_model.eval()

    generator = pipeline(
        "text-generation",
        model=base_model,
        device="cuda",
        tokenizer=tokenizer,
    )
    generator.tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    # ── load test data ──
    with open(args.test_pkl, "rb") as f:
        data = pickle.load(f)

    spec_dataset = data[args.author_key]

    # ── generate ──
    results = []
    for item in spec_dataset:
        prompt_text = item["prompt"]
        reference = item.get("output")  # ground truth (may be None)

        task = [{"content": prompt_text, "role": "user"}]

        outs = generator(
            task,
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=args.num_samples,
            return_full_text=False,
        )

        generations = [o["generated_text"] for o in outs]

        results.append({
            "prompt": prompt_text,
            "reference": reference,
            "generations": generations,
        })

    # ── write output ──
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "author_key": args.author_key,
            "model_dir": args.model_dir,
            "test_pkl": args.test_pkl,
            "num_samples": args.num_samples,
            "results": results,
        }, f, indent=2)

    print(f"Wrote {len(results)} prompts x {args.num_samples} samples -> {args.output_json}")


if __name__ == "__main__":
    main()
