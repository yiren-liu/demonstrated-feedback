import os
import json
import yaml
from importlib import resources
import asyncio
from typing import List, Dict, Any

import openai
from anthropic import AnthropicBedrock, Anthropic
# from google import genai
# from google.genai import types
import logging

logger = logging.getLogger(__name__)

openai_client = None
anthropic_client = None
anthropic_bedrock_client = None
together_client = None
gemini_client = None

if os.environ.get("OPENAI_API_KEY") is not None:
    openai_client = openai.Client(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
else:
    logger.warning("OPENAI_API_KEY is not set")

if os.environ.get("AWS_ACCESS_KEY") is not None and os.environ.get("AWS_SECRET_KEY") is not None:
    anthropic_bedrock_client = AnthropicBedrock(
        aws_access_key=os.environ.get("AWS_ACCESS_KEY"),
        aws_secret_key=os.environ.get("AWS_SECRET_KEY"),
        aws_region="us-west-2",
    )
else:
    logger.warning("AWS_ACCESS_KEY or AWS_SECRET_KEY is not set")

if os.environ.get("ANTHROPIC_API_KEY") is not None:
    anthropic_client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
else:
    logger.warning("ANTHROPIC_API_KEY is not set")

if os.environ.get("TOGETHER_API_KEY") is not None:
    together_client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1"
    )
else:
    logger.warning("TOGETHER_API_KEY is not set")

# if os.environ.get("GEMINI_PROJECT_ID") is not None:
#     gemini_client = genai.Client(
#         vertexai=True,
#         project=os.environ.get("GEMINI_PROJECT_ID"),
#         location="us-central1"
#     )
# else:
#     logger.warning("GEMINI_PROJECT_ID is not set")

MODEL_DICTIONARY = {
    "openai": [
        model.id for model in openai_client.models.list().data if openai_client is not None
    ] if openai_client is not None else [],
    "anthropic": [
        "claude-sonnet-4-5-20250929",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-20241022"
    ],
    "anthropic_bedrock": [
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0"
    ],
    "together": [
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        'deepseek-ai/DeepSeek-R1',
        'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
    ],
    "gemini": [
        "gemini-2.0-pro-exp-02-05",
        "gemini-2.0-flash-thinking-exp-01-21"
    ]
}


def anthropic_track_usage(model_name, system, messages, output_text):
    anthropic_usage_file = "anthropic_usage.jsonl"
    input_tokens = anthropic_client.messages.count_tokens(
        model="claude-3-5-sonnet-20241022",
        system=system,
        messages=messages
    ).input_tokens
    output_tokens = anthropic_client.messages.count_tokens(
        model="claude-3-5-sonnet-20241022",
        messages=[{
            "role": "assistant",
            "content": output_text
        }]
    ).input_tokens
    with open(anthropic_usage_file, "a") as f:
        f.write(json.dumps({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }) + "\n")


def generate_openai(model_name, messages, temperature, max_tokens):
    if openai_client is None:
        raise Exception("OPENAI_API_KEY is not set")

    if 'o3' not in model_name:
        output = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature if 'o1' not in model_name and 'gpt-5' not in model_name else 1,
            max_completion_tokens=max_tokens
        )
    else:
        output = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens
        )
    return output.choices[0].message.content


def generate_together(model_name, messages, temperature, max_tokens):
    if together_client is None:
        raise Exception("TOGETHER_API_KEY is not set")

    output = together_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return output.choices[0].message.content


def generate_anthropic(model_name, system, messages, temperature, max_tokens):
    if anthropic_client is None:
        raise Exception("ANTHROPIC_API_KEY is not set")

    if system is not None:
        if "claude-4-5" in model_name:
            output = anthropic_client.messages.create(
                model=model_name,
                system=system,
                messages=messages,
                temperature=1,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 1024
                }
            )
        else:
            output = anthropic_client.messages.create(
                model=model_name,
                system=system,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
    else:
        if "claude-4-5" in model_name:
            output = anthropic_client.messages.create(
                model=model_name,
                messages=messages,
                temperature=1,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 1024
                }
            )
        else:
            output = anthropic_client.messages.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

    if 'claude-4-5' in model_name:
        output_text = ""
        thinking_text = ""

        if len(output.content) > 2:
            logger.debug("Claude output content:")
            logger.debug(output.content)

        for content in output.content:
            # check if has thinking attribute
            if hasattr(content, 'thinking'):
                thinking_text += " " + content.thinking
            elif hasattr(content, 'text'):
                output_text += " " + content.text

        thinking_text = thinking_text.strip()
        output_text = output_text.strip()
        return output_text
    else:
        return output.content[0].text


def generate_anthropic_bedrock(model_name, system, messages, temperature, max_tokens):
    if anthropic_bedrock_client is None:
        raise Exception("AWS_ACCESS_KEY or AWS_SECRET_KEY is not set")

    if system is not None:
        if "claude-3-7" in model_name:
            output = anthropic_bedrock_client.messages.create(
                model=model_name,
                system=system,
                messages=messages,
                temperature=1,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 1024
                }
            )
        else:
            output = anthropic_bedrock_client.messages.create(
                model=model_name,
                system=system,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
    else:
        if "claude-3-7" in model_name:
            output = anthropic_bedrock_client.messages.create(
                model=model_name,
                messages=messages,
                temperature=1,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 1024
                }
            )
        else:
            output = anthropic_bedrock_client.messages.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

    if 'claude-3-7' in model_name:
        output_text = ""
        thinking_text = ""

        if len(output.content) > 2:
            logger.debug("Claude output content:")
            logger.debug(output.content)

        for content in output.content:
            # check if has thinking attribute
            if hasattr(content, 'thinking'):
                thinking_text += " " + content.thinking
            elif hasattr(content, 'text'):
                output_text += " " + content.text

        thinking_text = thinking_text.strip()
        output_text = output_text.strip()
        anthropic_track_usage(model_name, system, messages,
                              thinking_text + " " + output_text)
        return output_text
    else:
        anthropic_track_usage(model_name, system,
                              messages, output.content[0].text)
        return output.content[0].text


# def generate_gemini(model_name, system, messages, temperature, max_tokens):
#     if gemini_client is None:
#         raise Exception("GEMINI_PROJECT_ID is not set")

#     response = gemini_client.models.generate_content(
#         model=model_name,
#         config=types.GenerateContentConfig(
#             system_instruction=system,
#             max_output_tokens=max_tokens,
#             temperature=temperature,
#         ),
#         contents=messages
#     )
#     return response.text

def generate(model_name, system, prompt, temperature=0.0, max_tokens=1024, verbose=False):
    if model_name in MODEL_DICTIONARY['openai']:
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "developer", "content": system})
        output = generate_openai(model_name, messages, temperature, max_tokens)

    elif model_name in MODEL_DICTIONARY['together']:
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        output = generate_together(
            model_name, messages, temperature, max_tokens)

    elif model_name in MODEL_DICTIONARY['anthropic']:
        messages = [{"role": "user", "content": prompt}]
        output = generate_anthropic(
            model_name, system, messages, temperature, max_tokens)

    elif model_name in MODEL_DICTIONARY['anthropic_bedrock']:
        messages = [{"role": "user", "content": prompt}]
        output = generate_anthropic_bedrock(
            model_name, system, messages, temperature, max_tokens)

    # elif model_name in MODEL_DICTIONARY['gemini']:
    #     messages = [prompt]
    #     output = generate_gemini(model_name, system, messages, temperature, max_tokens)

    else:
        raise Exception(f"Model not found: {model_name}")

    if verbose:
        logger.debug("\n\n---\n")
        logger.debug(f"<<<SYSTEM>>>\n{system}\n")
        logger.debug(f"<<<USER>>>\n{prompt}\n")
        logger.debug(f"<<<ASSISTANT>>>\n{output}\n")
        logger.debug("\n---\n")
    return output


class Generator:
    def __init__(self, model_name, prompt_path, temperature=0.0, max_tokens=1024, verbose=False):
        prompt_file = resources.files('synthesis.prompts') / prompt_path
        with prompt_file.open('r') as file:
            prompt = yaml.safe_load(file)

        self.model_name = model_name
        self.system_template = prompt.get('system_prompt', None)
        self.prompt_template = prompt['user_prompt']
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        output = generate(
            self.model_name,
            self.system_template.format(*args, **kwargs),
            self.prompt_template.format(*args, **kwargs),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose
        )
        return output.strip()


def generate_chat(model_name, system, messages, temperature=0.0, max_tokens=1024, verbose=False):
    # Create a copy of the messages to avoid modifying the original list
    messages = messages.copy()

    # Generate response using correct api and format
    if model_name in MODEL_DICTIONARY['openai']:
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        output = generate_openai(model_name, messages, temperature, max_tokens)

    elif model_name in MODEL_DICTIONARY['together']:
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        output = generate_together(
            model_name, messages, temperature, max_tokens)

    elif model_name in MODEL_DICTIONARY['anthropic']:
        output = generate_anthropic(
            model_name, system, messages, temperature, max_tokens)

    elif model_name in MODEL_DICTIONARY['anthropic_bedrock']:
        output = generate_anthropic_bedrock(
            model_name, system, messages, temperature, max_tokens)

    else:
        raise Exception(f"Model not found: {model_name}")

    if verbose:
        logger.debug("\n\n---\n")
        logger.debug(f"<<<SYSTEM>>>\n{system}\n")
        logger.debug(f"<<<INPUT>>>\n{json.dumps(messages, indent=2)}\n")
        logger.debug(f"<<<OUTPUT>>>\n{output}\n")
        logger.debug("\n---\n")
    return output


class GeneratorChat:
    def __init__(self, model_name, prompt_path, initial_message=None, temperature=0.0, max_tokens=1024, verbose=False, **kwargs):
        prompt_file = resources.files('synthesis.prompts') / prompt_path
        with prompt_file.open('r') as file:
            prompt = yaml.safe_load(file)

        self.model_name = model_name
        self.system_prompt = prompt.get('system_prompt', None).format(**kwargs)

        self.chat_history = []
        if initial_message is not None:
            self.chat_history.append(initial_message)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

    def __call__(self, message):
        self.chat_history.append({"role": "user", "content": message})
        output = generate_chat(
            self.model_name,
            self.system_prompt,
            self.chat_history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose
        ).strip()
        self.chat_history.append({"role": "assistant", "content": output})
        return output

    def pop_last_turn(self):
        self.chat_history = self.chat_history[:-2]


# ============================================================================
# ASYNC GENERATION FUNCTIONS
# ============================================================================

async def generate_openai_async(model_name, messages, temperature, max_tokens):
    """Async wrapper for OpenAI generation."""
    if openai_client is None:
        raise Exception("OPENAI_API_KEY is not set")

    # Run the synchronous API call in a thread pool
    loop = asyncio.get_event_loop()
    if 'o3' not in model_name:
        output = await loop.run_in_executor(
            None,
            lambda: openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature if 'o1' not in model_name and 'gpt-5' not in model_name else 1,
                max_completion_tokens=max_tokens
            )
        )
    else:
        output = await loop.run_in_executor(
            None,
            lambda: openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_tokens
            )
        )
    return output.choices[0].message.content


async def generate_together_async(model_name, messages, temperature, max_tokens):
    """Async wrapper for Together generation."""
    if together_client is None:
        raise Exception("TOGETHER_API_KEY is not set")

    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(
        None,
        lambda: together_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    )
    return output.choices[0].message.content


async def generate_anthropic_async(model_name, system, messages, temperature, max_tokens):
    """Async wrapper for Anthropic generation."""
    if anthropic_client is None:
        raise Exception("ANTHROPIC_API_KEY is not set")

    loop = asyncio.get_event_loop()
    
    if system is not None:
        if "claude-4-5" in model_name:
            output = await loop.run_in_executor(
                None,
                lambda: anthropic_client.messages.create(
                    model=model_name,
                    system=system,
                    messages=messages,
                    temperature=1,
                    max_tokens=max_tokens,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                )
            )
        else:
            output = await loop.run_in_executor(
                None,
                lambda: anthropic_client.messages.create(
                    model=model_name,
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
    else:
        if "claude-4-5" in model_name:
            output = await loop.run_in_executor(
                None,
                lambda: anthropic_client.messages.create(
                    model=model_name,
                    messages=messages,
                    temperature=1,
                    max_tokens=max_tokens,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                )
            )
        else:
            output = await loop.run_in_executor(
                None,
                lambda: anthropic_client.messages.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )

    if 'claude-4-5' in model_name:
        output_text = ""
        thinking_text = ""

        if len(output.content) > 2:
            logger.debug("Claude output content:")
            logger.debug(output.content)

        for content in output.content:
            if hasattr(content, 'thinking'):
                thinking_text += " " + content.thinking
            elif hasattr(content, 'text'):
                output_text += " " + content.text

        thinking_text = thinking_text.strip()
        output_text = output_text.strip()
        return output_text
    else:
        return output.content[0].text


async def generate_anthropic_bedrock_async(model_name, system, messages, temperature, max_tokens):
    """Async wrapper for Anthropic Bedrock generation."""
    if anthropic_bedrock_client is None:
        raise Exception("AWS_ACCESS_KEY or AWS_SECRET_KEY is not set")

    loop = asyncio.get_event_loop()
    
    if system is not None:
        if "claude-3-7" in model_name:
            output = await loop.run_in_executor(
                None,
                lambda: anthropic_bedrock_client.messages.create(
                    model=model_name,
                    system=system,
                    messages=messages,
                    temperature=1,
                    max_tokens=max_tokens,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                )
            )
        else:
            output = await loop.run_in_executor(
                None,
                lambda: anthropic_bedrock_client.messages.create(
                    model=model_name,
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
    else:
        if "claude-3-7" in model_name:
            output = await loop.run_in_executor(
                None,
                lambda: anthropic_bedrock_client.messages.create(
                    model=model_name,
                    messages=messages,
                    temperature=1,
                    max_tokens=max_tokens,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 1024
                    }
                )
            )
        else:
            output = await loop.run_in_executor(
                None,
                lambda: anthropic_bedrock_client.messages.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )

    if 'claude-3-7' in model_name:
        output_text = ""
        thinking_text = ""

        if len(output.content) > 2:
            logger.debug("Claude output content:")
            logger.debug(output.content)

        for content in output.content:
            if hasattr(content, 'thinking'):
                thinking_text += " " + content.thinking
            elif hasattr(content, 'text'):
                output_text += " " + content.text

        thinking_text = thinking_text.strip()
        output_text = output_text.strip()
        anthropic_track_usage(model_name, system, messages,
                              thinking_text + " " + output_text)
        return output_text
    else:
        anthropic_track_usage(model_name, system,
                              messages, output.content[0].text)
        return output.content[0].text


async def generate_async(model_name, system, prompt, temperature=0.0, max_tokens=1024, verbose=False):
    """Async version of generate function."""
    if model_name in MODEL_DICTIONARY['openai']:
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        output = await generate_openai_async(model_name, messages, temperature, max_tokens)

    elif model_name in MODEL_DICTIONARY['together']:
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        output = await generate_together_async(model_name, messages, temperature, max_tokens)

    elif model_name in MODEL_DICTIONARY['anthropic']:
        messages = [{"role": "user", "content": prompt}]
        output = await generate_anthropic_async(model_name, system, messages, temperature, max_tokens)

    elif model_name in MODEL_DICTIONARY['anthropic_bedrock']:
        messages = [{"role": "user", "content": prompt}]
        output = await generate_anthropic_bedrock_async(model_name, system, messages, temperature, max_tokens)

    else:
        raise Exception(f"Model not found: {model_name}")

    if verbose:
        logger.debug("\n\n---\n")
        logger.debug(f"<<<SYSTEM>>>\n{system}\n")
        logger.debug(f"<<<USER>>>\n{prompt}\n")
        logger.debug(f"<<<ASSISTANT>>>\n{output}\n")
        logger.debug("\n---\n")
    return output


class AsyncGenerator:
    """Async version of Generator with batching and concurrency control."""
    
    def __init__(self, model_name, prompt_path, temperature=0.0, max_tokens=1024, 
                 verbose=False, max_concurrency=10):
        prompt_file = resources.files('synthesis.prompts') / prompt_path
        with prompt_file.open('r') as file:
            prompt = yaml.safe_load(file)

        self.model_name = model_name
        self.system_template = prompt.get('system_prompt', None)
        self.prompt_template = prompt['user_prompt']
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.max_concurrency = max_concurrency

    async def generate_single(self, *args, **kwargs):
        """Generate a single response asynchronously."""
        output = await generate_async(
            self.model_name,
            self.system_template.format(*args, **kwargs) if self.system_template else None,
            self.prompt_template.format(*args, **kwargs),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose
        )
        return output.strip()

    async def generate_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        Generate responses for a batch of items with concurrency control.
        
        Args:
            items: List of dictionaries containing kwargs for each generation
            
        Returns:
            List of generated responses in the same order as input
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def generate_with_semaphore(item, index):
            async with semaphore:
                try:
                    result = await self.generate_single(**item)
                    return (index, result, None)
                except Exception as e:
                    logger.error(f"Error generating item {index}: {str(e)}")
                    return (index, None, e)
        
        # Create tasks for all items
        tasks = [generate_with_semaphore(item, i) for i, item in enumerate(items)]
        
        # Execute all tasks with progress logging
        results = []
        total = len(tasks)
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            index, result, error = await coro
            results.append((index, result, error))
            completed += 1
            
            if completed % 10 == 0 or completed == total:
                logger.info(f"Completed {completed}/{total} generations")
        
        # Sort results by original index and extract values
        results.sort(key=lambda x: x[0])
        return [(r, e) for _, r, e in results]

    async def __call__(self, *args, **kwargs):
        """Single generation call."""
        return await self.generate_single(*args, **kwargs)
