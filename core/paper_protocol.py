"""Shared protected-token and fixed-budget evaluation for score-based baselines.

This changes the retention protocol, not the baseline's scoring rule. Methods
that merge, mask variable head budgets, or reconstruct values require their own
adapter and are rejected rather than silently replaced by a different method.
"""

from dataclasses import dataclass

import torch

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.prefill_decoding_press import PrefillDecodingPress


def retained_budget(length, ratio, sinks=4):
    if not 0 <= ratio < 1 or length < 1:
        raise ValueError("Expected a nonempty sequence and 0 <= eviction ratio < 1")
    return max(min(sinks, length), int(length * (1 - ratio)))


@dataclass
class PaperScorerPress(ScorerPress):
    base_press: ScorerPress = None
    n_sink_protect: int = 4

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.base_press, ScorerPress):
            raise TypeError("Paper protocol requires a physical score-based retention adapter")

    def post_init_from_model(self, model):
        self.base_press.post_init_from_model(model)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        kwargs = dict(kwargs)
        embeddings = kwargs.get("position_embeddings")
        if embeddings is not None and embeddings[0].shape[-2] != hidden_states.shape[1]:
            end = kwargs.get("position_ids")
            if end is None or not hasattr(module, "rotary_emb"):
                raise RuntimeError("Buffered baseline queries require absolute positions and RoPE")
            offsets = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            positions = end[:, -1:] - hidden_states.shape[1] + 1 + offsets[None]
            kwargs["position_ids"] = positions
            kwargs["position_embeddings"] = module.rotary_emb(hidden_states, positions)
        old_ratio = self.base_press.compression_ratio
        self.base_press.compression_ratio = self.compression_ratio
        try:
            return self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        finally:
            self.base_press.compression_ratio = old_ratio

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        length = keys.shape[2]
        count = retained_budget(length, self.compression_ratio, self.n_sink_protect)
        if count == length:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        sinks = min(self.n_sink_protect, length)
        # Select protected entries explicitly: ties or +inf baseline scores
        # cannot displace a sink from the budget.
        candidates = torch.argsort(scores[..., sinks:], dim=-1, descending=True, stable=True)
        candidates = candidates[..., :count - sinks] + sinks
        prefix = torch.arange(sinks, device=keys.device).expand(*scores.shape[:-1], -1)
        indices = torch.cat((prefix, candidates), dim=-1).sort(dim=-1).values
        gather = indices[..., None].expand(-1, -1, -1, keys.shape[-1])
        return keys.gather(2, gather).contiguous(), values.gather(2, gather).contiguous()


@dataclass
class PaperPrefillDecodingPress(PrefillDecodingPress):
    def begin_sequence(self, context_length):
        self.decoding_press.target_size = retained_budget(
            context_length, self.prefilling_press.compression_ratio,
            self.prefilling_press.n_sink_protect,
        )

    def reset_cache(self):
        self.decoding_press.reset()


def wrap_paper_baseline(press, interval=128):
    if press is None:
        return None
    if type(press).compress is not ScorerPress.compress:
        raise ValueError(
            f"{type(press).__name__} has a custom retention/merge policy; "
            "a verified paper-protocol adapter is required"
        )
    protected = PaperScorerPress(compression_ratio=press.compression_ratio, base_press=press)
    return PaperPrefillDecodingPress(
        prefilling_press=protected,
        decoding_press=DecodingPress(
            base_press=protected, compression_interval=interval,
            target_size=1, hidden_states_buffer_size=interval,
        ),
    )
