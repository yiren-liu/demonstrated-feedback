from typing import List, Optional, Dict, Tuple, Sequence, Union
import torch
from tqdm import tqdm

from dataclasses import dataclass
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from steering_vectors.layer_matching import guess_and_enhance_layer_config, collect_matching_layers
from steering_vectors.torch_utils import get_module, untuple_tensor



def _create_additive_hook_batched(
    target_activation_b1d: Tensor,  # shape (B, 1, d), broadcastable over tokens
):
    """Create a forward hook that adds a per-example steering activation."""
    def hook_fn(_m, _inputs, outputs):
        original = untuple_tensor(outputs)  # typically (B, T, d)
        delta = target_activation_b1d.to(original.device)

        # In-place add; broadcasts (B,1,d) over (B,T,d)
        original.add_(delta)
        return outputs

    return hook_fn


@dataclass
class _BatchedSteeringHandle:
    hooks: List[RemovableHandle]

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()


def _patch_model_with_batched_steering_vectors(
    model: nn.Module,
    steering_vectors_batch: Sequence,  # sequence of steering_vectors.SteeringVector
    *,
    multiplier: float = 1.0,
    layer_config=None,
) -> _BatchedSteeringHandle:
    """
    Patch model with per-example steering vectors for a batch.
    Assumes all vectors have the same layer_type and same layer_activations keys.
    """
    assert len(steering_vectors_batch) > 0
    B = len(steering_vectors_batch)

    layer_type = steering_vectors_batch[0].layer_type  # SteeringVector.layer_type exists :contentReference[oaicite:1]{index=1}
    for sv in steering_vectors_batch[1:]:
        if sv.layer_type != layer_type:
            raise ValueError("All steering vectors in the batch must share the same layer_type.")

    # Infer / enhance layer config the same way the library does :contentReference[oaicite:2]{index=2}
    layer_config = guess_and_enhance_layer_config(model, layer_config, layer_type)
    if layer_type not in layer_config:
        raise ValueError(f"layer_type {layer_type} not provided in layer config")

    matcher = layer_config[layer_type]
    matching_layers = collect_matching_layers(model, matcher)

    # Union of layers (usually they’re identical across the batch)
    all_layers = sorted({ln for sv in steering_vectors_batch for ln in sv.layer_activations.keys()})

    hooks: List[RemovableHandle] = []
    for layer_num in all_layers:
        # Stack per-example activations: (B, d)
        vecs_bd = []
        for i, sv in enumerate(steering_vectors_batch):
            if layer_num not in sv.layer_activations:
                raise ValueError(f"Missing layer {layer_num} in steering vector at batch index {i}")
            vecs_bd.append(sv.layer_activations[layer_num].to(model.device))
        vecs_bd = torch.stack(vecs_bd, dim=0)  # (B, d)

        # (B, d) -> (B, 1, d) so it broadcasts across token positions
        target_b1d = (multiplier * vecs_bd).unsqueeze(1)

        layer_name = matching_layers[layer_num]
        module = get_module(model, layer_name)
        handle = module.register_forward_hook(_create_additive_hook_batched(target_b1d))
        hooks.append(handle)

    return _BatchedSteeringHandle(hooks)



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
    batch_size: int = 2,
):
    """Generate text for a batch of chats with optional steering.

    Compatible with your existing experiments:
    - steering_vector can be:
        (a) a single SteeringVector -> shared steering (old behavior), OR
        (b) a List[SteeringVector] aligned with `chats` -> per-example steering
    """
    responses: List[str] = []

    if batch_size <= 1:
        # Use single-item generation (unchanged)
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
        return responses

    # ---- NEW: true batch generation ----
    n = len(chats)

    # assert False, "Not implemented"

    # Helper to decode per example, respecting padding
    def _decode_batch(outputs: Tensor, padded_input_len: int) -> List[str]:
        # outputs: (B, out_len)
        # With left-padding, all prompts end at the same position (padded_input_len)
        out_texts: List[str] = []
        B = outputs.shape[0]
        for i in range(B):
            gen_ids = outputs[i, padded_input_len:]
            out_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        return out_texts

    for start in tqdm(range(0, n, batch_size), desc=description):
        batch_chats = chats[start:start + batch_size]
        B = len(batch_chats)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Left-padding is required for correct generation with decoder-only models
        original_padding_side = tokenizer.padding_side
        if not getattr(model.config, 'is_encoder_decoder', False):
            tokenizer.padding_side = 'left'
        inputs = tokenizer(
            batch_chats,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        tokenizer.padding_side = original_padding_side  # Restore original

        def _gen_batched():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
            return out

        padded_input_len = inputs["input_ids"].shape[1]

        if not use_steering:
            out_ids = _gen_batched()
            responses.extend(_decode_batch(out_ids, padded_input_len))
            continue

        if steering_vector is None:
            raise ValueError("use_steering=True but steering_vector is None")

        # Case A: shared steering vector (old behavior) -> works fine in batch
        if hasattr(steering_vector, "apply") and not isinstance(steering_vector, (list, tuple)):
            with steering_vector.apply(model, multiplier=multiplier):
                out_ids = _gen_batched()
            responses.extend(_decode_batch(out_ids, padded_input_len))
            continue

        # Case B: per-example steering vectors
        if isinstance(steering_vector, (list, tuple)):
            # Allow either: full list aligned with chats, or list matching current batch
            if len(steering_vector) == n:
                batch_vectors = steering_vector[start:start + B]
            elif len(steering_vector) == B:
                batch_vectors = steering_vector
            else:
                raise ValueError(
                    "When passing a list/tuple steering_vector, it must have length == len(chats) "
                    "or length == current batch size."
                )

            handle = _patch_model_with_batched_steering_vectors(
                model,
                batch_vectors,
                multiplier=multiplier,
                layer_config=None,
            )
            try:
                out_ids = _gen_batched()
            finally:
                handle.remove()

            responses.extend(_decode_batch(out_ids, padded_input_len))
            continue

        raise TypeError(
            "steering_vector must be either a single SteeringVector or a list/tuple of SteeringVector."
        )

    return responses
