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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of prompts to process in each pipeline batch")
    args = parser.parse_args()

    # ── load model ──
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    base_model = PeftModel.from_pretrained(base_model, args.model_dir)
    base_model.eval()

    # ── load test data ──
    with open(args.test_pkl, "rb") as f:
        data = pickle.load(f)

    spec_dataset = data[args.author_key]

    # ── generate ──
    prompts = [item["prompt"] for item in spec_dataset]
    references = [item.get("output") for item in spec_dataset]

    results = []
    with torch.inference_mode():
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[batch_start: batch_start + args.batch_size]
            # Apply chat template to each prompt
            batch_texts = [
                tokenizer.apply_chat_template(
                    [{"content": p, "role": "user"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch_prompts
            ]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

            outputs = base_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

            prompt_len = inputs["input_ids"].shape[1]
            for i, prompt in enumerate(batch_prompts):
                idx = batch_start + i
                gens = []
                for k in range(args.num_samples):
                    out_idx = i * args.num_samples + k
                    gen_ids = outputs[out_idx][prompt_len:]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    gens.append(gen_text)
                results.append({
                    "prompt": prompt,
                    "reference": references[idx],
                    "generations": gens,
                })
            print(f"  Processed {min(batch_start + args.batch_size, len(prompts))}/{len(prompts)} prompts")

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
