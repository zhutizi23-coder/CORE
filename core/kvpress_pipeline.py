"""CORE-only query-boundary adapter for the upstream KVPress pipeline."""

from __future__ import annotations

from typing import Optional

import torch
try:
    from flash_attn import flash_attn_func
except ImportError:  # pragma: no cover - portable CPU/test fallback
    flash_attn_func = None
from transformers import AutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    AttentionInterface,
    flash_attention_forward,
)

from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.base_press import BasePress


def _core_flash_attention_forward(module, query_states, key_states, value_states,
                                  attention_mask, **kwargs):
    """Retain post-RoPE Q/K and expose FlashAttention's row LSE to CORE.

    For the production FA2 inference path (unpadded, causal, no dropout),
    ``return_attn_probs=True`` exposes ``softmax_lse`` while FlashAttention
    still returns an empty probability tensor because dropout is disabled.
    Thus no quadratic attention tensor is materialized. Unsupported masking,
    training, sliding-window, and FA3 cases use the upstream implementation
    and leave LSE reuse disabled for that forward.
    """
    module._core_cached_query_states = query_states
    module._core_cached_softmax_lse = None

    dropout = float(kwargs.get("dropout", 0.0))
    scaling = kwargs.get("scaling")
    sliding_window = kwargs.get("sliding_window")
    softcap = kwargs.get("softcap")
    is_causal = kwargs.get("is_causal")
    if is_causal is None:
        is_causal = bool(getattr(module, "is_causal", True))
    expected_scaling = query_states.shape[-1] ** -0.5
    scaling_matches_core = scaling is None or abs(
        float(scaling) - float(expected_scaling)
    ) <= 1e-12
    module_config = getattr(module, "config", None)

    can_reuse_lse = (
        flash_attn_func is not None
        and getattr(module_config, "_attn_implementation", None)
        == "flash_attention_2"
        and attention_mask is None
        and dropout == 0.0
        and bool(is_causal)
        and scaling_matches_core
        and sliding_window is None
        and softcap in (None, 0.0)
        and not torch.is_grad_enabled()
        and query_states.dtype in (torch.float16, torch.bfloat16)
        and key_states.dtype == query_states.dtype
        and value_states.dtype == query_states.dtype
    )
    if can_reuse_lse:
        output, softmax_lse, attention_probs = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=expected_scaling,
            causal=True,
            return_attn_probs=True,
        )
        if attention_probs is not None and attention_probs.numel() != 0:
            raise RuntimeError(
                "FlashAttention unexpectedly materialized attention probabilities "
                "while CORE requested LSE-only inference"
            )
        module._core_cached_softmax_lse = softmax_lse
        return output, None

    return flash_attention_forward(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        **kwargs,
    )


AttentionInterface.register("flash_attention_2", _core_flash_attention_forward)
AttentionInterface.register("flash_attention_3", _core_flash_attention_forward)


class CORETextGenerationPipeline(KVPressTextGenerationPipeline):
    """Keep upstream behavior and add query metadata only for CORE presses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Some model modules initialize attention backends lazily while the
        # checkpoint is loaded. Register once more after model construction so
        # CORE is the final FA2 interface used by every Llama layer.
        AttentionInterface.register(
            "flash_attention_2", _core_flash_attention_forward
        )
        AttentionInterface.register(
            "flash_attention_3", _core_flash_attention_forward
        )
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = (
            _core_flash_attention_forward
        )

    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        enable_thinking: bool = False,
        cache=None,
        query_start_char: Optional[int] = None,
        **kwargs,
    ):
        preprocess, forward, postprocess = super()._sanitize_parameters(
            question=question,
            questions=questions,
            answer_prefix=answer_prefix,
            press=press,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context_length,
            enable_thinking=enable_thinking,
            cache=cache,
            **kwargs,
        )
        preprocess["query_start_char"] = query_start_char
        return preprocess, forward, postprocess

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
        enable_thinking: bool = False,
        query_start_char: Optional[int] = None,
    ):
        inputs = super().preprocess(
            context=context,
            questions=questions,
            answer_prefix=answer_prefix,
            max_context_length=max_context_length,
            enable_thinking=enable_thinking,
        )
        if query_start_char is not None:
            inputs["query_start"] = self._query_start_token_index(
                context, int(query_start_char), inputs["context_ids"].shape[1]
            )
        return inputs

    def _render_context(self, context: str) -> str:
        if self.tokenizer.chat_template is None:
            return getattr(self.tokenizer, "bos_token", "") + context
        separator = "#" * (len(context) + 10)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        return rendered.split(separator)[0]

    def _query_start_token_index(self, context: str, query_start_char: int, length: int) -> int:
        query_start_char = max(0, min(query_start_char, len(context)))
        rendered = self._render_context(context)
        rendered_prefix = self._render_context(context[:query_start_char])
        try:
            encoded = self.tokenizer(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded["offset_mapping"]
            if hasattr(offsets, "tolist"):
                offsets = offsets.tolist()
            offsets = offsets[0] if offsets and isinstance(offsets[0][0], list) else offsets
            boundary_char = len(rendered_prefix)
            boundary = sum(1 for start, end in offsets if end <= boundary_char)
        except (TypeError, ValueError, KeyError):
            boundary = len(self.tokenizer.encode(rendered_prefix, add_special_tokens=False))
        return max(0, min(int(boundary), int(length)))

    def _forward(self, input_tensors, **forward_params):
        query_start = input_tensors.get("query_start")
        press = forward_params.get("press")
        if press is None:
            return super()._forward(input_tensors, **forward_params)
        if hasattr(press, "reset_cache"):
            press.reset_cache()
        if hasattr(press, "begin_sequence"):
            press.begin_sequence(input_tensors["context_ids"].shape[1])
        target = getattr(press, "prefilling_press", press)
        if hasattr(target, "set_token_ids"):
            target.set_token_ids(input_tensors["context_ids"], self.tokenizer.all_special_ids)
        if query_start is not None and hasattr(target, "set_query_boundary"):
            target.set_query_boundary(int(query_start))

        def record_inputs(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            positions = kwargs.get("position_ids")
            if positions is None and kwargs.get("past_key_values") is not None:
                cache = kwargs["past_key_values"]
                if cache.get_seq_length() == 0:
                    positions = torch.arange(ids.shape[1], device=ids.device)[None]
            target.record_input_tokens(ids, positions)

        handle = None
        if hasattr(target, "record_input_tokens"):
            # Both prefill and generation call the decoder model; the hook
            # sees real token IDs and absolute generation position_ids.
            handle = self.model.model.register_forward_pre_hook(record_inputs, with_kwargs=True)
        try:
            return super()._forward(input_tensors, **forward_params)
        finally:
            if handle is not None:
                handle.remove()
            if hasattr(press, "reset_cache"):
                press.reset_cache()

PIPELINE_REGISTRY.register_pipeline(
    "core-kv-press-text-generation",
    pipeline_class=CORETextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
