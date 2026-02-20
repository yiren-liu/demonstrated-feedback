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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

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
    parser.add_argument("--load_8bit", action="store_true",
                        help="Load base model in 8-bit to reduce GPU memory")
    parser.add_argument("--load_4bit", action="store_true",
                        help="Load base model in 4-bit to reduce GPU memory (more efficient than 8-bit)")
    args = parser.parse_args()

    # ── load model (half precision + optional quantization to avoid OOM) ──
    # Note: CPU offload doesn't work with PEFT/LoRA, so we disable it
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = {"torch_dtype": torch_dtype}
    
    if args.load_4bit:
        # 4-bit is more memory efficient and works better with PEFT
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    elif args.load_8bit:
        # 8-bit without CPU offload (PEFT incompatible with CPU offload)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,  # PEFT doesn't support CPU offload
        )
        model_kwargs["device_map"] = "auto"
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if not (args.load_8bit or args.load_4bit):
        base_model = base_model.to("cuda")
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
