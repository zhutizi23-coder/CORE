"""
KVPress bridge for the trained CORE indexer.

Provides press classes compatible with the KVPress framework:

- COREScorerPress: ScorerPress that scores tokens using CORE 14-dim features + MLP
  indexer.  Compatible with DecodingPress for decode-phase compression.
- COREIndexerPress: Backward-compatible press with optional absolute budget support.
  Extends COREScorerPress.
- COREMemoryPress: Wraps COREScorerPress and implements evicted-token weighted
  memory writing. Writes evicted-token information into per-layer M_t/b_t
  fast weights and
  applies a learned gated readout to subsequent attention outputs.

And factory / loading functions:

- load_core_scorer_press(): returns COREScorerPress (usable as DecodingPress base_press)
- load_core_indexer_press(): backward-compatible, returns COREIndexerPress (prefill-only)
- load_core_memory_press(): returns COREMemoryPress (recommended: memory-writing)
- load_core_prefill_decoding_press(): returns PrefillDecodingPress with prefill + decode,
  both phases wrapped in COREMemoryPress when enable_memory_writing is set
"""

from __future__ import annotations
from core.selection import query_aware_span_selection_scores
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from core.cache_metadata import CacheMetadata
from core.config import COREConfig, validate_checkpoint_interface
from core.collector import BackboneStatsCollector
from core.features import COREFeatureExtractor
from core.indexer import (
    COREIndexerMLP,
    COREIndexerStack,
    conditional_evicted_weights,
    student_distribution,
)
from core.train import load_indexer, load_indexer_stack
from kvpress.presses.base_press import BasePress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.prefill_decoding_press import PrefillDecodingPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_key_states, get_prerope_query_states
from transformers.models.llama.modeling_llama import repeat_kv

logger = logging.getLogger(__name__)


@dataclass
class COREPrefillDecodingPress(PrefillDecodingPress):
    """Forward CORE token metadata and coordinate prefill/decode budgets.

    By default the pipeline prefills context separately from the question.
    Full-prompt prefill is available through ``query_aware_prefill``.
    """

    query_aware_prefill: bool = False
    match_prefill_budget: bool = True

    def begin_sequence(self, context_length):
        if self.match_prefill_budget:
            from core.paper_protocol import retained_budget
            self.decoding_press.target_size = retained_budget(
                context_length, self.prefilling_press.compression_ratio
            )

    def set_token_ids(self, token_ids: torch.Tensor, special_token_ids=None):
        if self.prefilling_press is not None and hasattr(self.prefilling_press, "set_token_ids"):
            self.prefilling_press.set_token_ids(token_ids, special_token_ids)

    def record_input_tokens(self, input_ids, position_ids=None):
        self.prefilling_press.record_input_tokens(input_ids, position_ids)

    def set_query_boundary(self, query_start: int):
        if self.prefilling_press is not None and hasattr(self.prefilling_press, "set_query_boundary"):
            self.prefilling_press.set_query_boundary(query_start)

    def reset_cache(self):
        if self.decoding_press is not None:
            self.decoding_press.reset()
        if self.prefilling_press is not None and hasattr(self.prefilling_press, "reset_cache"):
            self.prefilling_press.reset_cache()


@dataclass
class COREDecodingPress(DecodingPress):
    """CORE-only DecodingPress variant accepting the memory wrapper."""

    hidden_states_buffer_size: Optional[int] = None

    def __post_init__(self):
        assert hasattr(self.base_press, "compress"), "CORE decoding press requires a compress-capable base press"
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
        assert self.compression_interval > 0
        assert self.target_size > 0
        if self.hidden_states_buffer_size is None:
            self.hidden_states_buffer_size = self.compression_interval
        if self.hidden_states_buffer_size < self.compression_interval:
            raise ValueError(
                "CORE requires queries from the entire compression interval; "
                "hidden_states_buffer_size must be >= compression_interval"
            )

    def forward_hook(self, module, input, kwargs, output):
        """Run periodic decode compression and consume latent memory.

        ``DecodingPress.forward_hook`` only delegates to ``base_press.compress``
        when its compression interval is reached.  CORE's shared base press
        also owns the latent fast state, whose readout is required on every
        question/decode forward, including steps without cache compression.
        """
        # DecodingPress owns the single pre-write readout. Do not compensate
        # here too: the parent supports memory-capable base presses.
        return super().forward_hook(module, input, kwargs, output)


# ---------------------------------------------------------------------------
# Press classes
# ---------------------------------------------------------------------------


@dataclass
class COREScorerPress(ScorerPress):
    """
    A kvpress-compatible ScorerPress backed by a trained CORE indexer.

    Computes the same 14-dimensional CORE token features used during training,
    runs the MLP indexer to score tokens, and integrates with the KVPress scoring
    and compression pipeline.

    During **prefill**, all 14 features (QK relation + coverage + auxiliary) are
    used. During **decode**, recent buffered queries are matched against the
    cached post-RoPE keys and the phase feature is set to one. Training
    periodically constructs the same decode-window feature distribution.

    The ``score()`` method returns shape ``(batch, num_kv_heads, seq_len)``, with
    the same scores broadcast across all KV heads (CORE features are
    head-agnostic after pooling).

    With ``scoring_stride=1`` the method computes independent scores at every
    layer. Values greater than one enable the separate IndexCache-style score
    reuse mode and must only be used in an explicitly labelled ablation.
    """

    indexer: COREIndexerMLP = field(default_factory=lambda: COREIndexerMLP())
    feature_extractor: Optional[COREFeatureExtractor] = None
    # Student temperature loaded from the training checkpoint.
    # It leaves Top-B ranking unchanged and determines the
    # softmax weights used for memory writing.
    student_temp: float = 1.0
    # Layer index controlling the stride grid. None selects grid offset zero.
    # Set to -1 to score every layer explicitly.
    scoring_layer_idx: Optional[int] = None
    # Stride for optional cross-layer score reuse.
    # 1 = score every layer; N>1 = score every Nth layer and reuse within groups.
    # 4 = every 4th layer scores, the 3 in-between reuse the cached indices.
    scoring_stride: int = 1
    # True selects per-layer COREIndexerStack weights;
    # False shares a single COREIndexerMLP across layers.
    use_indexer_stack: bool = False
    # Force-keep the first N tokens within the retained budget; 0 disables.
    n_sink_protect: int = 0
    # Recency score bonus. Zero disables the bonus.
    recency_alpha: float = 0.0
    recency_window: int = 64
    # Optional span pooling and query-feature adjustment before Top-B selection.
    # The default tokenwise policy leaves learned scores unchanged.
    span_block_size: int = 1
    span_pool_beta: float = 5.0
    span_selection_mode: str = "tokenwise"
    coverage_fraction: float = 0.25
    query_feature_weight: float = 0.0
    lexical_overlap_weight: float = 0.0
    identifier_lexical_weight: float = 0.0
    identifier_min_match_tokens: int = 20
    identifier_context_skip: int = 64
    identifier_span_size: int = 32
    protect_query_tokens: bool = False
    query_aware_prefill: bool = False
    # Prefill query scope. Question scope uses the boundary supplied by KVPress.
    prefill_qk_scope: str = "all"
    capture_diagnostics: bool = False
    _token_ids: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _query_start: Optional[int] = field(default=None, init=False, repr=False)
    _logged_flash_lse_reuse: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.feature_extractor is None:
            raise ValueError("feature_extractor must be provided")
        self.indexer.eval()
        # Cache for the scoring result: kept indices per batch element,
        # keyed by the *original* (pre-compression) sequence length.
        self._cached_kept_indices: Optional[torch.Tensor] = None
        # Preserve continuous calibrated scores for IndexCache reuse. Rebuilding
        # them as binary 0/1 values makes evicted-token memory weights uniform.
        self._cached_scores: Optional[torch.Tensor] = None
        # Unmodified calibrated indexer scores. Selection may use block/query
        # adjustment, but memory writing must keep the distribution learned
        # with student_temp instead of softmaxing the adjusted selection score.
        self._cached_memory_scores: Optional[torch.Tensor] = None
        self._latest_query_feature: Optional[torch.Tensor] = None
        self._cached_original_seq_len: int = 0
        self.cache_metadata = CacheMetadata()
        self.diagnostic_keep_indices: dict[int, torch.Tensor] = {}
        self.diagnostic_scores: dict[int, torch.Tensor] = {}
        self.diagnostic_features: dict[int, torch.Tensor] = {}

    def post_init_from_model(self, model):
        device = next(model.parameters()).device
        self._ensure_device(device)

        language_model = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )
        model_layers = len(language_model.layers)
        expected_layers = int(getattr(self.indexer, "n_layers", model_layers))
        if self.use_indexer_stack and expected_layers != model_layers:
            raise ValueError(
                "CORE checkpoint/model layer mismatch: checkpoint has "
                f"{expected_layers} indexer layers, model has {model_layers}."
            )
        expected_hidden = self.feature_extractor.config.hidden_size
        model_hidden = getattr(model.config, "hidden_size", None)
        if (
            expected_hidden is not None
            and model_hidden is not None
            and int(expected_hidden) != int(model_hidden)
        ):
            raise ValueError(
                "CORE checkpoint/model hidden-size mismatch: checkpoint has "
                f"{expected_hidden}, model has {model_hidden}."
            )

        # Deployment and training use the same groups: stride N scores layers
        # 0, N, 2N, ...; stride 1 performs true per-layer scoring.
        if self.scoring_layer_idx is None:
            self.scoring_layer_idx = -1 if self.scoring_stride <= 1 else 0

        # Align scoring_layer_idx to the stride grid so the scoring layers are
        # evenly distributed (e.g. stride=4, idx=16 -> layers {0,4,8,12,16,...}).
        if self.scoring_stride > 1 and self.scoring_layer_idx >= 0:
            self.scoring_layer_idx = (
                self.scoring_layer_idx // self.scoring_stride * self.scoring_stride
            )

        if self.scoring_stride > 1:
            logger.info(
                "CORE scoring will run every %d layers (scoring layers: "
                "idx %% %d == %d).",
                self.scoring_stride,
                self.scoring_stride,
                self.scoring_layer_idx % self.scoring_stride,
            )
        else:
            logger.info(
                "CORE scoring will run on %s.",
                "every layer" if self.scoring_layer_idx == -1 else f"layer {self.scoring_layer_idx}",
            )

    def _ensure_device(self, device: torch.device):
        device_str = str(device)
        current_device = str(next(self.indexer.parameters()).device)
        feature_device = str(self.feature_extractor.geometry_device)
        if current_device != device_str or feature_device != device_str:
            self.indexer.to(device)
            self.feature_extractor.to(device)
        self.indexer.eval()

    def set_token_ids(self, token_ids: torch.Tensor, special_token_ids=None):
        """Provide context token ids for the sink/special-token feature."""
        self._token_ids = token_ids.detach()
        self.cache_metadata.record(token_ids, torch.arange(token_ids.shape[1], device=token_ids.device)[None])
        if special_token_ids is not None:
            self.feature_extractor.config.special_token_ids = tuple(
                int(x) for x in special_token_ids
            )

    def set_query_boundary(self, query_start: int):
        """Record where the question begins in a query-aware full prompt."""
        self._query_start = int(query_start)

    def record_input_tokens(self, input_ids, position_ids=None):
        self.cache_metadata.record(input_ids, position_ids)

    def retain_metadata(self, layer_idx, indices):
        self.cache_metadata.retain(layer_idx, indices)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        count = max(min(self.n_sink_protect, keys.shape[2]),
                    int(keys.shape[2] * (1 - self.compression_ratio)))
        indices = torch.argsort(scores[:, 0], dim=-1, descending=True, stable=True)[:, :count]
        indices = indices.sort(dim=-1).values
        self.cache_metadata.align(int(module.layer_idx), keys.shape[0], keys.shape[2], keys.device)
        self.retain_metadata(int(module.layer_idx), indices)
        gather = indices[:, None, :, None].expand(-1, keys.shape[1], -1, keys.shape[-1])
        return keys.gather(2, gather).contiguous(), values.gather(2, gather).contiguous()

    def _compute_scores_from_features(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        is_prefill: bool,
        kwargs: Optional[dict] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Run full 14-dim feature extraction + MLP scoring (expensive).

        Prefill Q-K relation features use post-RoPE attention logits to derive
        the causal attention advantage ``log(p)+log|C_t|``. This is the same
        shared extractor used by the training collector.

        Query states are recomputed from the layernormed ``hidden_states``
        fed into the attention module, using the module's ``q_proj`` and the
        upstream ``position_embeddings`` (transformers >= 4.43) or
        ``rotary_emb`` (older), so they align with the cached post-RoPE keys.
        """
        batch_size, num_kv_heads, seq_len, head_dim = keys.shape
        dtype = self.feature_extractor.dtype
        n_q_heads = self.feature_extractor.n_q_heads
        num_groups = self.feature_extractor.num_kv_groups

        positions, aligned_ids, original_length = self.cache_metadata.align(
            layer_idx, batch_size, seq_len, keys.device
        )
        all_scores = []
        all_query_features = []
        for batch_idx in range(batch_size):
            token_ids = aligned_ids[batch_idx]
            if is_prefill:
                value_states = values[batch_idx : batch_idx + 1].squeeze(0)
                # Post-RoPE keys from cache — same source as training.
                key_states_cached = keys[batch_idx : batch_idx + 1].squeeze(0)

                # --- QK features from TRUE logits (matches training) ---
                # Training uses the true post-RoPE Q/K relation features.  The
                # inference path must therefore compute the same six features;
                # silently replacing them with zeros creates a severe
                # train/inference feature mismatch.
                qk_features = None
                if kwargs is not None and hasattr(module, "q_proj"):
                    pos_emb = kwargs.get("position_embeddings", None)
                    pos_ids = kwargs.get("position_ids", None)
                    hs = hidden_states[batch_idx : batch_idx + 1]
                    q_states = getattr(module, "_core_cached_query_states", None)
                    if q_states is None:
                        q_states = BackboneStatsCollector._recompute_post_rope_queries(
                            module, hs, pos_ids, pos_emb
                        )
                    elif q_states.ndim == 4:
                        q_states = q_states[batch_idx]
                    if q_states is not None and q_states.shape[1] > 0:
                        scope = self.prefill_qk_scope

                        if scope == "question":
                            if self._query_start is None:
                                raise RuntimeError(
                                    "CORE prefill_qk_scope='question' requires "
                                    "set_query_boundary() before prefill."
                                )

                            query_start = int(self._query_start)
                            if not (0 <= query_start < q_states.shape[1]):
                                raise RuntimeError(
                                    f"Invalid CORE query boundary: "
                                    f"query_start={query_start}, "
                                    f"seq_len={q_states.shape[1]}"
                                )

                            # The question is appended after the context during
                            # query-aware prefill, so these are exactly the
                            # deployment scoring queries used in training.
                            q_states = q_states[:, query_start:, :]

                        elif scope == "all":
                            pass

                        else:
                            raise ValueError(
                                "prefill_qk_scope must be 'all' or 'question', "
                                f"got {scope!r}"
                            )

                        softmax_lse = getattr(
                            module, "_core_cached_softmax_lse", None
                        )
                        if softmax_lse is not None:
                            if softmax_lse.ndim != 3:
                                softmax_lse = None
                            elif scope == "question":
                                softmax_lse = softmax_lse[
                                    batch_idx : batch_idx + 1,
                                    :,
                                    query_start:,
                                ]
                            else:
                                softmax_lse = softmax_lse[
                                    batch_idx : batch_idx + 1
                                ]
                            if (
                                softmax_lse is not None
                                and softmax_lse.shape[-1] != q_states.shape[1]
                            ):
                                softmax_lse = None
                            if (
                                softmax_lse is not None
                                and not self._logged_flash_lse_reuse
                            ):
                                logger.info(
                                    "CORE reusing FlashAttention softmax_lse; "
                                    "strict QK reduction will run one QK pass"
                                )
                                self._logged_flash_lse_reuse = True

                        qk_features = self._compute_qk_features_from_logits(
                            q_states.to(dtype),
                            key_states_cached.to(dtype),
                            n_q_heads,
                            num_kv_heads,
                            num_groups,
                            head_dim,
                            softmax_lse=softmax_lse,
                        )

                # Use returned attention probabilities when Q/K features are unavailable.
                if qk_features is None and attentions is not None:
                    qk_features = (
                        self.feature_extractor.compute_qk_relation_features_from_attention(
                            attentions[batch_idx].to(dtype)
                        )
                    )

                # Do not silently run a 14-D indexer with its first six inputs
                # replaced by zero.  Failing loudly is much safer than producing
                # apparently valid but feature-mismatched benchmark numbers.
                if qk_features is None:
                    raise RuntimeError(
                        "CORE failed to compute prefill QK relation features. "
                        "Check post-RoPE query reconstruction and the query boundary; "
                        "refusing to replace the six QK features with zeros."
                    )

                cov_features = self.feature_extractor.compute_coverage_features(
                    key_states_cached.to(dtype), layer_idx
                )
                aux_features = self.feature_extractor.compute_auxiliary_features(
                    value_states.to(dtype),
                    seq_len=original_length,
                    is_prefill=True,
                    token_ids=token_ids,
                    token_positions=positions[batch_idx],
                )
            else:
                key_states = keys[batch_idx : batch_idx + 1].squeeze(0)
                value_states = values[batch_idx : batch_idx + 1].squeeze(0)

                # Compute decode QK features from captured queries.
                qk_features = None
                if kwargs is not None and hasattr(module, "q_proj"):
                    pos_emb = kwargs.get("position_embeddings", None)
                    pos_ids = kwargs.get("position_ids", None)
                    hs = hidden_states[batch_idx : batch_idx + 1]
                    if hs.shape[1] > 1 and kwargs.get("cache_position") is not None:
                        # DecodingPress buffers hidden states from the whole
                        # interval, but kwargs only carries RoPE embeddings for
                        # the current token. Reconstruct absolute positions for
                        # every buffered query and recompute their RoPE.
                        # cache_position is commonly derived from the physical
                        # compressed cache length. Cached keys still carry
                        # their original RoPE coordinates, so prefer the
                        # explicit absolute position_ids used for generation.
                        current_position_ids = kwargs.get("position_ids")
                        if current_position_ids is not None:
                            end_pos = int(current_position_ids[0, -1].item())
                        else:
                            end_pos = int(kwargs["cache_position"][-1].item())
                        start_pos = end_pos - hs.shape[1] + 1
                        pos_ids = torch.arange(
                            start_pos, end_pos + 1,
                            device=hs.device, dtype=torch.long,
                        ).unsqueeze(0)
                        pos_emb = None
                    q_states = getattr(module, "_core_cached_query_states", None)
                    if q_states is not None and q_states.ndim == 4:
                        q_states = q_states[batch_idx]
                    if q_states is None or q_states.shape[-2] != hs.shape[1]:
                        q_states = BackboneStatsCollector._recompute_post_rope_queries(
                            module, hs, pos_ids, pos_emb
                        )
                    elif q_states.ndim == 4:
                        q_states = q_states[batch_idx]
                    if q_states is not None and q_states.shape[1] > 0:
                        # Use every query accumulated in the current decode
                        # compression interval, as in decode-window training.
                        q_recent = q_states.to(dtype)
                        qk_features = self._compute_qk_features_from_logits(
                            q_recent,
                            key_states.to(dtype),
                            n_q_heads,
                            num_kv_heads,
                            num_groups,
                            head_dim,
                        )
                if qk_features is None:
                    raise RuntimeError(
                        "CORE failed to reconstruct decode QK features; "
                        "refusing to replace the paper features with zeros"
                    )
                cov_features = self.feature_extractor.compute_coverage_features(
                    key_states.to(dtype), layer_idx
                )
                aux_features = self.feature_extractor.compute_auxiliary_features(
                    value_states.to(dtype),
                    seq_len=original_length,
                    is_prefill=False,
                    token_ids=token_ids,
                    token_positions=positions[batch_idx],
                )

            features = torch.cat([qk_features, cov_features, aux_features], dim=-1)
            all_query_features.append(features[:, 3])
            if self.capture_diagnostics:
                self.diagnostic_features[int(layer_idx)] = (
                    features.detach().float().cpu()
                )
            if self.use_indexer_stack:
                # Per-layer MLP: select this layer's own weights.
                token_scores = self.indexer(features.float(), layer_idx).squeeze(-1)
            else:
                token_scores = self.indexer(features.float()).squeeze(-1)
            all_scores.append(token_scores)

        scores_tensor = torch.stack(all_scores, dim=0)  # (batch, seq_len)
        self._latest_query_feature = torch.stack(all_query_features, dim=0)
        # Use .expand().contiguous() (or just repeat) instead of plain .expand():
        # plain .expand() returns a view with stride 0 along the expanded dim, and
        # the sink-token overwrite in score() (scores[:, :, :n_sink_protect] = ...)
        # then writes to the *same* memory location multiple times, triggering
        # "more than one element of the written-to tensor refers to a single
        # memory location". .contiguous() materialises a real tensor.
        scores_tensor = scores_tensor.unsqueeze(1).expand(-1, num_kv_heads, -1).contiguous()
        return scores_tensor

    def _query_aware_selection_scores(
        self, scores: torch.Tensor, *, is_prefill: bool,
        selection_budget: int | None = None,
    ) -> torch.Tensor:
        """Apply optional selection adjustments during full-prompt prefill.

        The configured policy can pool spans, mix query or lexical features, and
        protect question tokens. The default tokenwise policy and decode path
        return the input scores unchanged.
        """
        if (
            not is_prefill
            or self._query_start is None
            or self._query_start <= 0
            or self._query_start >= scores.shape[-1]
        ):
            return scores
        if (
            int(self.span_block_size) <= 1
            and self.query_feature_weight == 0.0
            and self.lexical_overlap_weight == 0.0
            and not self.protect_query_tokens
        ):
            return scores

        query_start = int(self._query_start)
        adjusted = scores.float()
        raw_mean = adjusted.mean(dim=-1, keepdim=True)
        raw_std = adjusted.std(dim=-1, keepdim=True).clamp_min(1e-6)
        adjusted = (adjusted - raw_mean) / raw_std

        if (
            self.query_feature_weight != 0.0
            and self._latest_query_feature is not None
            and self._latest_query_feature.shape[-1] == scores.shape[-1]
        ):
            query_feature = self._latest_query_feature.to(
                device=scores.device, dtype=torch.float32
            )
            q_mean = query_feature.mean(dim=-1, keepdim=True)
            q_std = query_feature.std(dim=-1, keepdim=True).clamp_min(1e-6)
            query_feature = (query_feature - q_mean) / q_std
            adjusted = adjusted + self.query_feature_weight * query_feature.unsqueeze(1)

        # Early transformer layers can have weak semantic query features.
        # Add an inexpensive lexical anchor for rare query tokens (identifiers,
        # entities, numbers, names). Common prompt words are suppressed by
        # inverse context frequency, so they cannot dominate block selection.
        if (
            self.lexical_overlap_weight != 0.0
            and self._token_ids is not None
            and self._token_ids.shape[-1] == scores.shape[-1]
        ):
            lexical_rows = []
            for batch_idx in range(scores.shape[0]):
                ids_idx = min(batch_idx, self._token_ids.shape[0] - 1)
                ids = self._token_ids[ids_idx].to(scores.device)
                context_ids = ids[:query_start]
                query_ids = ids[query_start:]
                _, inverse, counts = torch.unique(
                    context_ids, return_inverse=True, return_counts=True
                )
                frequency = counts[inverse].to(torch.float32)
                lexical_context = torch.isin(
                    context_ids, query_ids
                ).to(torch.float32)
                lexical_context = (
                    lexical_context / frequency.sqrt().clamp_min(1.0)
                )
                lexical = torch.zeros(
                    scores.shape[-1],
                    device=scores.device,
                    dtype=torch.float32,
                )
                lexical[:query_start] = lexical_context
                lexical_rows.append(lexical)
            lexical = torch.stack(lexical_rows, dim=0)
            lexical_mean = lexical[:, :query_start].mean(
                dim=-1, keepdim=True
            )
            lexical_std = lexical[:, :query_start].std(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            lexical = (lexical - lexical_mean) / lexical_std
            adjusted = adjusted + self.lexical_overlap_weight * lexical.unsqueeze(1)

        span_size = max(1, int(self.span_block_size))

        if span_size > 1 and query_start > 0:
            adjusted = query_aware_span_selection_scores(
                adjusted,
                query_start,
                span_size,
                standardize=False,
                token_ids=self._token_ids,
                identifier_lexical_weight=self.identifier_lexical_weight,
                identifier_min_match_tokens=self.identifier_min_match_tokens,
                identifier_context_skip=self.identifier_context_skip,
                identifier_span_size=self.identifier_span_size,
                protect_query_tokens=self.protect_query_tokens,
                span_pool_beta=self.span_pool_beta,
                span_selection_mode=self.span_selection_mode,
                selection_budget=selection_budget,
                coverage_fraction=self.coverage_fraction,
                n_sink_protect=self.n_sink_protect,
            )

        return adjusted.to(dtype=scores.dtype)

    def _compute_qk_features_from_logits(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        n_q_heads: int,
        n_kv_heads: int,
        num_groups: int,
        head_dim: int,
        softmax_lse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the 6 Q-K relation features from TRUE post-RoPE logits.

        Mirrors the training feature path in ``collector.py``: causal logits
        are normalized to probabilities, converted to
        ``r=log(p)+log|C_t|``, then reduced with the six Q-K statistics.
        """
        return self.feature_extractor.compute_qk_relation_features(
            query_states, key_states, softmax_lse=softmax_lse
        )

    @torch.no_grad()
    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """Compute CORE token scores from the 14-dimensional features and MLP.

        With the default stride of one, each layer computes its own scores.
        Larger strides reuse scores within layer groups when the cache is valid.
        Selection adjustments and sink protection preserve the calibrated scores
        used to construct the memory distribution.
        """
        self._ensure_device(hidden_states.device)

        batch_size, num_kv_heads, seq_len, head_dim = keys.shape

        # Detect prefill vs decode: prefill when hidden_states covers all
        # cached tokens (or more, e.g. before first compression).
        buffer_len = hidden_states.shape[1]
        is_prefill = buffer_len >= seq_len

        # Determine if this is a scoring layer
        layer_idx = getattr(module, "layer_idx", None)
        if self.scoring_stride > 1 and self.scoring_layer_idx >= 0:
            # Stride mode: every `scoring_stride`-th layer scores, the rest
            # reuse the most recent scoring layer's cached indices.
            # Scoring layers: layer_idx % stride == scoring_layer_idx % stride.
            is_scoring_layer = (
                layer_idx is None  # safety fallback
                or (layer_idx % self.scoring_stride
                    == self.scoring_layer_idx % self.scoring_stride)
            )
        else:
            is_scoring_layer = (
                self.scoring_layer_idx == -1  # Score every layer
                or layer_idx == self.scoring_layer_idx
                or layer_idx is None  # safety fallback
            )

        # Invalidate cache on sequence length change (new sample or decode step)
        if seq_len != self._cached_original_seq_len:
            self._cached_kept_indices = None
            self._cached_scores = None
            self._cached_memory_scores = None
            self._cached_original_seq_len = seq_len

        # --- Non-scoring layer: reuse cached scores ---
        if not is_scoring_layer and self._cached_scores is not None:
            # Reuse calibrated scores so both Top-B selection and
            # softmax memory-writing weights remain consistent.
            return self._cached_scores.to(
                device=keys.device, dtype=keys.dtype
            ).unsqueeze(1).expand(-1, num_kv_heads, -1).contiguous()

        # --- Scoring layer (or cache miss): compute full scores ---
        scores = self._compute_scores_from_features(
            module, hidden_states, keys, values, attentions, is_prefill, kwargs,
            layer_idx=layer_idx if layer_idx is not None else 0,
        )
        calibrated_scores = scores

        # Apply the configured recency bonus.
        if self.recency_alpha > 0.0 and seq_len > 1:
            positions = torch.arange(seq_len, device=keys.device, dtype=scores.dtype)
            recency_bonus = self.recency_alpha * torch.exp(
                -(seq_len - 1 - positions) / max(1, self.recency_window)
            )
            scores = scores + recency_bonus.view(1, 1, -1)
            calibrated_scores = scores

        n_kept = int(seq_len * (1 - self.compression_ratio))
        n_kept = max(min(self.n_sink_protect, seq_len), n_kept)
        scores = self._query_aware_selection_scores(
            scores, is_prefill=is_prefill, selection_budget=n_kept
        )

        # Sink token protection: force-keep the first N tokens by saturating
        # their scores so they always land in Top-B.
        if self.n_sink_protect > 0 and self.n_sink_protect < seq_len:
            # The tokenwise policy can return calibrated_scores itself.
            # Protect sinks only in the selection copy, preserving Eq. 6/10
            # allocation mass for memory writing.
            scores = scores.clone()
            scores[:, :, : self.n_sink_protect] = scores.max().detach() + 1.0

        # Cache the top-k indices for reuse on other layers
        if n_kept > 0 and n_kept < seq_len:
            # (batch, n_kept) - take from any head (all identical)
            self._cached_scores = scores[:, 0].detach().clone()
            self._cached_memory_scores = (
                calibrated_scores[:, 0].detach().clone()
            )
            self._cached_kept_indices = scores[:, 0].topk(n_kept, dim=-1).indices
            if self.capture_diagnostics:
                diagnostic_layer = int(layer_idx if layer_idx is not None else -1)
                self.diagnostic_scores[diagnostic_layer] = (
                    calibrated_scores[:, 0].detach().float().cpu()
                )
                self.diagnostic_keep_indices[diagnostic_layer] = (
                    self._cached_kept_indices.detach().cpu()
                )

        return scores

    def reset_cache(self):
        """Clear cached indices (e.g. between samples)."""
        self._cached_kept_indices = None
        self._cached_scores = None
        self._cached_memory_scores = None
        self._latest_query_feature = None
        self._cached_original_seq_len = 0
        self._token_ids = None
        self.cache_metadata.reset()
        self._query_start = None
        self.diagnostic_keep_indices.clear()
        self.diagnostic_scores.clear()
        self.diagnostic_features.clear()


@dataclass
class COREMemoryPress(BasePress):
    """
    KV-cache eviction with learned memory writing and readout.

    Evicted tokens update per-layer fast weights using student distribution
    weights and the trained memory projection. The tied projection is used
    for both memory writes and query reads. The gated readout is added to
    attention outputs without modifying cached values.

    Pass a trained MemoryModule via memory_module or load it from a checkpoint.
    Without a memory module, the press performs eviction without readout.
    """

    base_press: COREScorerPress = None  # type: ignore[assignment]
    memory_module: Optional[nn.Module] = None   # MemoryModule (learnable θ, g)
    memory_decay: float = 0.9       # γ - decay across compression batches
    memory_write_eta: float = 1.0
    memory_uniform_fraction: float = 0.0
    enable_memory_writing: bool = True
    memory_query_space: str = "post_rope_q"
    memory_value_space: str = "pre_o_proj"
    # None uses the learned gate; a scalar forces a controlled ablation.
    # Learned memory contents and fast-state writes remain unchanged.
    memory_gate_override: Optional[float] = None
    query_aware_prefill: bool = False
    requires_decoding_hook: bool = field(default=True, init=False, repr=False)

    def __post_init__(self):
        assert self.base_press is not None, "base_press (COREScorerPress) is required"
        # Per-layer fast weights in d_model space (matches training-side
        # M_bar = proj_k.T @ ev_hs in train.py:train_joint_online). Shapes:
        #   _M[layer_idx]: (B, d_mem, d_model)
        #   _b[layer_idx]: (B, d_mem)
        #   _Z[layer_idx]: (B,) decayed evicted mass
        self._M: dict = {}
        self._b: dict = {}
        self._Z: dict = {}
        self._diagnostic_reads = 0
        self._diagnostic_hooks = 0
        self._diagnostic_state_logged = False
        # GQA repeats every KV head before flattening it into d_model. Cache
        # algebraically folded slow weights so formal stride=1 evaluation does
        # not materialise the redundant representation on every layer/sample.
        self._folded_theta: dict = {}
        self._folded_o_proj: dict = {}

    @property
    def compression_ratio(self) -> float:
        return self.base_press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value: float) -> None:
        self.base_press.compression_ratio = value

    @property
    def student_temp(self) -> float:
        return self.base_press.student_temp

    def post_init_from_model(self, model):
        self.base_press.post_init_from_model(model)

    def set_token_ids(self, token_ids: torch.Tensor, special_token_ids=None):
        self.base_press.set_token_ids(token_ids, special_token_ids)

    def record_input_tokens(self, input_ids, position_ids=None):
        self.base_press.record_input_tokens(input_ids, position_ids)

    def set_query_boundary(self, query_start: int):
        self.base_press.set_query_boundary(query_start)

    def _ensure_device(self, device):
        self.base_press._ensure_device(device)

    def reset_cache(self):
        self.base_press.reset_cache()
        self._M.clear(); self._b.clear(); self._Z.clear()

    def reset_memory(self, layer_idx: Optional[int] = None):
        if layer_idx is None:
            self._M.clear(); self._b.clear(); self._Z.clear()
        else:
            self._M.pop(layer_idx, None); self._b.pop(layer_idx, None)
            self._Z.pop(layer_idx, None)

    def _phi(self, k_full: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Memory write feature map φ(k) = Linear_θ(k) in d_model space.

        ``k_full`` is the merged-key view ``(B, L, d_model)`` (KV heads
        concatenated). Linear_θ operates in d_model space, so we project the
        full per-token key (all KV heads together) rather than per-head - this
        matches the training-side write in train.py (``memory_stack.project``
        on ``ev_hs`` of shape ``(n_evict, d_model)``).

        Falls back to ``elu(normalize(k))+1`` only when no MemoryModule is
        attached.
        """
        if self.memory_module is None:
            k_unit = torch.nn.functional.normalize(k_full.float(), dim=-1)
            return torch.nn.functional.elu(k_unit) + 1.0
        return self.memory_module.project(k_full.float(), layer_idx)

    def _gqa_folded_theta(
        self,
        layer_idx: int,
        num_kv_heads: int,
        num_groups: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Fold Linear_theta over exactly repeated GQA key-head blocks."""
        module_idx = self.memory_module._module_idx(layer_idx)
        theta = self.memory_module.layers[module_idx].theta_dense_weight()
        cache_key = (
            module_idx, num_kv_heads, num_groups, head_dim,
            theta.device, theta.dtype,
        )
        folded = self._folded_theta.get(cache_key)
        if folded is None:
            folded = (
                theta.float()
                .reshape(theta.shape[0], num_kv_heads, num_groups, head_dim)
                .sum(dim=2)
                .reshape(theta.shape[0], num_kv_heads * head_dim)
                .contiguous()
            )
            self._folded_theta[cache_key] = folded
        return folded

    def _gqa_folded_o_proj(
        self,
        module: nn.Module,
        num_kv_heads: int,
        num_groups: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Fold o_proj over exactly repeated GQA value-head blocks."""
        layer_idx = int(getattr(module, "layer_idx", 0))
        weight = module.o_proj.weight
        cache_key = (
            layer_idx, num_kv_heads, num_groups, head_dim,
            weight.device, weight.dtype,
        )
        folded = self._folded_o_proj.get(cache_key)
        if folded is None:
            folded = (
                weight.float()
                .reshape(weight.shape[0], num_kv_heads, num_groups, head_dim)
                .sum(dim=2)
                .reshape(weight.shape[0], num_kv_heads * head_dim)
                .contiguous()
            )
            self._folded_o_proj[cache_key] = folded
        return folded

    @torch.no_grad()
    def compress(
        self, module: nn.Module, hidden_states: torch.Tensor,
        keys: torch.Tensor, values: torch.Tensor,
        attentions: torch.Tensor, kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select retained tokens, summarize evicted tokens, then prune K/V.

        Conditional evicted-token weights determine each write; evicted mass
        weights successive writes in the decayed M, b, Z recurrence. Scores
        are normalized over the full candidate set before conditioning.
        The GQA implementation stores compact KV-head value coordinates and
        folds the corresponding expansion into the output projection.

        The gated readout is applied separately to the attention output.
        """
        self._ensure_device(hidden_states.device)
        if self.compression_ratio == 0:
            return keys, values

        layer_idx = int(getattr(module, "layer_idx", 0))
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_q_heads = self.base_press.feature_extractor.n_q_heads
        num_groups = num_q_heads // num_kv_heads
        d_model = num_q_heads * head_dim

        # State lifetime is controlled by reset_cache() at a sample boundary.
        # Cache shrinkage is normal after eviction and must not reset history.

        scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        # scores: (B, H_kv, L). Top-B is per (B, H_kv); we collapse to a single
        # retention set by averaging per-head scores into (B, L), since the
        # per-layer memory module sees the merged d_model representation and
        # treats all KV heads as one token.
        n_kept = max(
            1,
            min(
                max(
                    int(k_len * (1 - self.compression_ratio)),
                    min(self.base_press.n_sink_protect, k_len),
                ),
                k_len,
            ),
        )
        if n_kept >= k_len:
            return keys, values

        # Pool scores across KV heads for a single per-token eviction decision
        # (consistent with the d_model-space write). Shape (B, L).
        scores_pooled = scores.mean(dim=1)
        keep_indices = torch.argsort(scores_pooled, dim=-1, descending=True, stable=True)[:, :n_kept]
        keep_indices = keep_indices.sort(dim=-1).values  # chronological cache order
        keep_mask = torch.zeros(bsz, k_len, device=keys.device, dtype=torch.bool)
        keep_mask.scatter_(1, keep_indices, True)
        evict_mask = ~keep_mask  # (B, L)
        # Score may be supplied by a custom scorer; align before pruning too.
        self.base_press.cache_metadata.align(layer_idx, bsz, k_len, keys.device)
        self.base_press.retain_metadata(layer_idx, keep_indices)

        # Early-exit paths when memory is disabled / no eviction.
        if not self.enable_memory_writing or self.memory_module is None:
            keep_exp_h = keep_indices.unsqueeze(1).unsqueeze(-1).expand(
                -1, num_kv_heads, -1, head_dim
            )
            return (
                keys.gather(2, keep_exp_h).contiguous(),
                values.gather(2, keep_exp_h).contiguous(),
            )

        # ---- memory write in d_model space (evicted tokens -> M_t, b_t) ----
        # Build the per-token student softmax from the original calibrated
        # indexer scores. Query-aware span adjustment is a retention policy;
        # it must not change the learned p_theta distribution used by memory.
        eps = 1e-8
        tau_m = self.student_temp
        memory_scores = getattr(self.base_press, "_cached_memory_scores", None)
        if memory_scores is None or memory_scores.shape != scores_pooled.shape:
            memory_scores = scores_pooled
        else:
            memory_scores = memory_scores.to(
                device=scores_pooled.device, dtype=scores_pooled.dtype
            )
        # Eq. 6 defines pi_theta on all of C, including protected sinks.
        # They have zero write weight because they are absent from E. Using
        # KL's non-sink normalization here would change event mass in Eq. 11.
        n_protected_sink = min(int(self.base_press.n_sink_protect), k_len)
        p_theta = student_distribution(memory_scores, tau_m)
        p_evict = p_theta * evict_mask.to(dtype=p_theta.dtype)
        pi_E = p_evict.sum(dim=-1)  # (B,) full-cache allocation mass on evicted tokens

        # No hard pi_E threshold.  The theoretical update already contains eps
        # in its denominator, so mu_E should affect the write continuously
        # rather than toggling an entire layer on/off at 1e-8.
        has_evicted = bool(evict_mask.any().item())
        if not has_evicted:
            keep_exp_h = keep_indices.unsqueeze(1).unsqueeze(-1).expand(
                -1, num_kv_heads, -1, head_dim
            )
            return (
                keys.gather(2, keep_exp_h).contiguous(),
                values.gather(2, keep_exp_h).contiguous(),
            )

        if os.environ.get("CORE_MEMORY_DIAGNOSTICS") == "1":
            with torch.no_grad():
                p_theta_fp32 = p_theta.float()
                entropy = -(
                    p_theta_fp32
                    * torch.log(p_theta_fp32.clamp_min(1e-30))
                ).sum(dim=-1).mean()
                p_max = p_theta_fp32.max(dim=-1).values.mean()
                top_k = min(8, p_theta_fp32.shape[-1])
                top_prob, top_idx = torch.topk(p_theta_fp32[0], k=top_k)
                sink_mass = (
                    p_theta_fp32[:, :n_protected_sink].sum(dim=-1).mean()
                    if n_protected_sink > 0
                    else p_theta_fp32.new_zeros(())
                )
                logger.warning(
                    "CORE candidate p_theta top tokens "
                    "layer=%d idx=%s prob=%s",
                    layer_idx,
                    top_idx.detach().cpu().tolist(),
                    [f"{x:.6e}" for x in top_prob.detach().cpu().tolist()],
                )
                logger.warning(
                    "CORE memory write diagnostic "
                    "layer=%d k_len=%d n_kept=%d n_evicted=%d "
                    "n_candidates=%d sink_mass=%.3e pi_E=%.12e "
                    "p_max=%.12e entropy=%.6f score_min=%.6f "
                    "score_max=%.6f score_std=%.6f score_dtype=%s",
                    layer_idx,
                    k_len,
                    n_kept,
                    int(evict_mask.sum().item()),
                    int(k_len),
                    float(sink_mass.item()),
                    float(pi_E.float().mean().item()),
                    float(p_max.item()),
                    float(entropy.item()),
                    float(memory_scores.float().min().item()),
                    float(memory_scores.float().max().item()),
                    float(memory_scores.float().std().item()),
                    str(memory_scores.dtype),
                )

        p_bar = conditional_evicted_weights(
            p_theta,
            evict_mask,
            self.memory_uniform_fraction,
            eps,
        )

        # For pre-o_proj values, fold GQA repetition into theta/o_proj weights.
        # Post-o_proj values use the expanded representation.
        kv_width = num_kv_heads * head_dim
        keys_kv = keys.permute(0, 2, 1, 3).reshape(bsz, k_len, kv_width)
        values_kv = values.permute(0, 2, 1, 3).reshape(bsz, k_len, kv_width)
        if self.memory_value_space == "pre_o_proj":
            values_full = values_kv.float()
            folded_theta = self._gqa_folded_theta(
                layer_idx, num_kv_heads, num_groups, head_dim
            )
            phi_k = torch.nn.functional.linear(
                keys_kv.float(), folded_theta, bias=None
            )
        elif self.memory_value_space == "post_o_proj":
            # Policy-v2 compatibility: these slow weights were trained after
            # mapping every written value through o_proj.
            keys_rep = repeat_kv(keys, num_groups)
            values_rep = repeat_kv(values, num_groups)
            keys_full = keys_rep.permute(0, 2, 1, 3).reshape(
                bsz, k_len, d_model
            )
            values_pre = values_rep.permute(0, 2, 1, 3).reshape(
                bsz, k_len, d_model
            )
            values_full = torch.nn.functional.linear(
                values_pre.float(), module.o_proj.weight.float(), bias=None
            )
            phi_k = self._phi(keys_full, layer_idx)
        else:
            raise ValueError(
                f"Unknown CORE memory_value_space={self.memory_value_space!r}"
            )
        w = p_bar.unsqueeze(-1)                                 # (B, L, 1)
        # M_bar: (B, d_mem, d_model) = Σ_i φ(k_i) ⊗ v_i (weighted)
        M_bar = torch.einsum("bld,bln->bdn", phi_k * w, values_full)
        # b_bar: (B, d_mem) = Σ_i φ(k_i)² (weighted)
        b_bar = ((phi_k * phi_k) * w).sum(dim=1)
        M_bar = M_bar * self.memory_write_eta
        b_bar = b_bar * self.memory_write_eta

        # Mass-aware decayed update:
        #   Z_t = gamma * Z_{t-1} + mu_E
        #   M_t = (gamma Z_{t-1} M_{t-1} + mu_E M_bar) / (Z_t + eps)
        #   b_t = (gamma Z_{t-1} b_{t-1} + mu_E b_bar) / (Z_t + eps)
        gamma = self.memory_decay
        Z_prev = self._Z.get(layer_idx)
        if (
            Z_prev is None
            or Z_prev.shape != pi_E.shape
            or Z_prev.device != pi_E.device
        ):
            Z_prev = torch.zeros_like(pi_E)

        M_prev = self._M.get(layer_idx)
        b_prev = self._b.get(layer_idx)
        if (
            M_prev is None
            or M_prev.shape != M_bar.shape
            or M_prev.device != M_bar.device
        ):
            M_prev = torch.zeros_like(M_bar)
            b_prev = torch.zeros_like(b_bar)

        gamma_Z = gamma * Z_prev
        Z_t = gamma_Z + pi_E
        denom = Z_t + eps

        old_mass_M = gamma_Z.unsqueeze(-1).unsqueeze(-1)
        new_mass_M = pi_E.unsqueeze(-1).unsqueeze(-1)
        M_t = (
            old_mass_M * M_prev + new_mass_M * M_bar
        ) / denom.unsqueeze(-1).unsqueeze(-1)

        old_mass_b = gamma_Z.unsqueeze(-1)
        new_mass_b = pi_E.unsqueeze(-1)
        b_t = (
            old_mass_b * b_prev + new_mass_b * b_bar
        ) / denom.unsqueeze(-1)

        self._M[layer_idx] = M_t
        self._b[layer_idx] = b_t
        self._Z[layer_idx] = Z_t

        # Plain gather-prune (values NOT mutated; readout applied to output).
        keep_exp_h = keep_indices.unsqueeze(1).unsqueeze(-1).expand(
            -1, num_kv_heads, -1, head_dim
        )
        keys = keys.gather(2, keep_exp_h).contiguous()
        values = values.gather(2, keep_exp_h).contiguous()
        return keys, values

    def apply_memory_readout(self, module: nn.Module, kwargs: dict, output):
        if not self.enable_memory_writing or self.memory_module is None:
            return output

        layer_idx = int(getattr(module, "layer_idx", 0))

        if (
            os.environ.get("CORE_MEMORY_DIAGNOSTICS") == "1"
            and not self._diagnostic_state_logged
        ):
            logger.warning(
                "CORE memory state before first decode read: "
                "n_layers=%d layers=%s",
                len(self._M),
                sorted(self._M.keys()),
            )
            self._diagnostic_state_logged = True

        M = self._M.get(layer_idx)
        b = self._b.get(layer_idx)

        if M is None or b is None:
            return output

        query_hidden = kwargs["hidden_states"]
        batch_size, q_len, hidden_dim = query_hidden.shape
        # Pre-W_O query width can differ from the residual-stream width.
        d_model = int(module.q_proj.out_features) if hasattr(module, "q_proj") else hidden_dim
        position_ids = kwargs.get("position_ids")
        position_embeddings = kwargs.get("position_embeddings")
        # The CORE FlashAttention adapter already captures the exact post-RoPE
        # query tensor used by this attention call. Reusing it avoids a second
        # q_proj + RoPE pass for every generated token and layer.
        cached_query_states = getattr(
            module, "_core_cached_query_states", None
        )
        residual_batches = []
        for batch_idx in range(batch_size):
            if self.memory_query_space == "hidden":
                # Exact compatibility path for policy-v2 checkpoints, whose
                # slow weights were trained with attention-input hidden states
                # on the read side. Never silently use this for new weights.
                query_full = query_hidden[batch_idx].float()
                residual_batches.append(
                    self.memory_module.readout(
                        query_full,
                        M[batch_idx].float(),
                        b[batch_idx].float(),
                        layer_idx,
                        gate_override=self.memory_gate_override,
                    )
                )
                continue
            if self.memory_query_space != "post_rope_q":
                raise ValueError(
                    f"Unknown CORE memory_query_space={self.memory_query_space!r}"
                )
            batch_position_ids = (
                position_ids[batch_idx : batch_idx + 1]
                if position_ids is not None and position_ids.shape[0] == batch_size
                else position_ids
            )
            batch_position_embeddings = position_embeddings
            if position_embeddings is not None:
                cos, sin = position_embeddings
                if cos.shape[0] == batch_size:
                    cos = cos[batch_idx : batch_idx + 1]
                    sin = sin[batch_idx : batch_idx + 1]
                batch_position_embeddings = (cos, sin)
            query_states = None
            if (
                cached_query_states is not None
                and cached_query_states.ndim == 4
                and cached_query_states.shape[0] == batch_size
                and cached_query_states.shape[-2] == q_len
            ):
                query_states = cached_query_states[batch_idx]
            if query_states is None:
                query_states = BackboneStatsCollector._recompute_post_rope_queries(
                    module,
                    query_hidden[batch_idx : batch_idx + 1],
                    batch_position_ids,
                    batch_position_embeddings,
                )
            if query_states is None:
                raise RuntimeError(
                    "CORE memory readout could not reconstruct post-RoPE "
                    "queries; refusing to mix hidden-state queries with "
                    "post-RoPE memory keys."
                )
            query_full = query_states.permute(1, 0, 2).reshape(q_len, d_model)
            residual_batches.append(
                self.memory_module.readout(
                    query_full.float(),
                    M[batch_idx].float(),
                    b[batch_idx].float(),
                    layer_idx,
                    gate_override=self.memory_gate_override,
                )
            )
        residual = torch.stack(residual_batches, dim=0).to(output[0].dtype)
        if self.memory_value_space == "pre_o_proj":
            if residual.shape[-1] == d_model:
                # Handle fast state stored with explicitly expanded GQA dimensions.
                output_weight = module.o_proj.weight.float()
            else:
                num_q_heads = int(module.config.num_attention_heads)
                num_kv_heads = int(module.config.num_key_value_heads)
                num_groups = num_q_heads // num_kv_heads
                head_dim = d_model // num_q_heads
                expected_width = num_kv_heads * head_dim
                if residual.shape[-1] != expected_width:
                    raise RuntimeError(
                        "Unexpected compact CORE memory width: "
                        f"got {residual.shape[-1]}, expected {expected_width}"
                    )
                output_weight = self._gqa_folded_o_proj(
                    module, num_kv_heads, num_groups, head_dim
                )
            residual = torch.nn.functional.linear(
                residual.float(), output_weight, bias=None
            ).to(output[0].dtype)
        elif self.memory_value_space != "post_o_proj":
            raise ValueError(
                f"Unknown CORE memory_value_space={self.memory_value_space!r}"
            )
        if os.environ.get("CORE_MEMORY_DIAGNOSTICS") == "1" and self._diagnostic_reads < 32:
            residual_rms = residual.float().pow(2).mean().sqrt()
            output_rms = output[0].float().pow(2).mean().sqrt().clamp_min(1e-12)
            logger.warning(
                "CORE memory diagnostic layer=%d q_len=%d residual_rms=%.6g "
                "attention_output_rms=%.6g ratio=%.6g M_rms=%.6g b_mean=%.6g",
                layer_idx, q_len, float(residual_rms), float(output_rms),
                float(residual_rms / output_rms),
                float(M.float().pow(2).mean().sqrt()), float(b.float().mean()),
            )
            self._diagnostic_reads += 1
        new_attn_out = output[0] + residual
        if isinstance(output, (list, tuple)):
            return type(output)([new_attn_out] + list(output[1:]))
        return new_attn_out

    def forward_hook(self, module: nn.Module, input, kwargs, output):
        """Run compression and cache write-back, then add memory readout.

        The gated residual is added once to output[0], the post-o_proj
        attention output. Cached values are unchanged.
        """
        layer_idx = int(getattr(module, "layer_idx", 0))
        if (
            os.environ.get("CORE_MEMORY_DIAGNOSTICS") == "1"
            and layer_idx == 0
            and self._diagnostic_hooks < 20
        ):
            cache_position = kwargs.get("cache_position")
            logger.warning(
                "CORE memory hook diagnostic call=%d q_len=%d cache_first=%s "
                "cache_last=%s has_memory_before=%s",
                self._diagnostic_hooks,
                int(kwargs["hidden_states"].shape[1]),
                str(int(cache_position[0])) if cache_position is not None else "none",
                str(int(cache_position[-1])) if cache_position is not None else "none",
                layer_idx in self._M,
            )
            self._diagnostic_hooks += 1

        q_len = kwargs["hidden_states"].shape[1]
        if kwargs["cache_position"][-1] <= q_len:
            # Initial attention has already seen all prompt tokens.
            return super().forward_hook(module, input, kwargs, output)
        return self.apply_memory_readout(module, kwargs, output)


@dataclass
class COREIndexerPress(COREScorerPress):
    """
    Backward-compatible press with optional absolute budget support.

    Extends :class:`COREScorerPress` to add a fixed-budget mode where exactly
    ``budget`` tokens are retained regardless of ``compression_ratio``.
    Also guards against compressing when ``seq_len <= 1`` (single token
    during initial decode steps).
    """

    budget: Optional[int] = None

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.shape[2] <= 1:
            return keys, values

        self._ensure_device(hidden_states.device)

        seq_len = keys.shape[2]

        # Determine how many tokens to keep
        if self.budget is not None:
            n_kept = max(
                1,
                min(self.n_sink_protect, seq_len),
                min(self.budget, seq_len),
            )
        else:
            n_kept = max(
                1,
                min(self.n_sink_protect, seq_len),
                int(seq_len * (1 - self.compression_ratio)),
            )

        if n_kept >= seq_len:
            return keys, values

        # Temporarily override compression_ratio to match desired n_kept
        original_ratio = self.compression_ratio
        self.compression_ratio = 1.0 - (n_kept / seq_len)
        result = super().compress(module, hidden_states, keys, values, attentions, kwargs)
        self.compression_ratio = original_ratio
        return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _build_feature_extractor(checkpoint: dict) -> COREFeatureExtractor:
    """Build a :class:`COREFeatureExtractor` from checkpoint config."""
    raw_config = dict(checkpoint["config"])
    if "key_geometry_mode" not in raw_config:
        raw_config["key_geometry_mode"] = "legacy_head_mean"
    config = COREConfig.from_dict(raw_config)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    fe = COREFeatureExtractor(config)
    if "feature_extractor_state_dict" in checkpoint:
        fe.load_geometry_state_dict(checkpoint["feature_extractor_state_dict"])
    return fe


def _check_compression_ratio(
    requested: Optional[float],
    cfg: COREConfig,
    checkpoint_path: str,
) -> float:
    """Resolve the inference-time eviction ratio.

    ``cfg.compression_ratio`` records the cutoff used by the boundary loss
    during training; it is not a deployment constraint.  The indexer emits a
    token ranking, so inference may apply any valid Top-B budget to the same
    checkpoint.  Ratio-transfer quality is an evaluation property and must not
    be represented as a requirement to train one checkpoint per ratio.
    """
    train_cr = float(cfg.compression_ratio)
    effective = train_cr if requested is None else float(requested)
    if not 0.0 < effective < 1.0:
        raise ValueError(
            f"compression_ratio must lie in (0, 1), received {effective}"
        )
    if abs(effective - train_cr) > 1e-3:
        logger.info(
            "Applying inference ratio %.4f to checkpoint %s (training cutoff "
            "metadata: %.4f); no ratio-specific retraining is required.",
            effective,
            checkpoint_path, train_cr,
        )
    return effective


# ---------------------------------------------------------------------------
# Factory / loading functions
# ---------------------------------------------------------------------------


def _load_indexer_and_config(
    checkpoint_path: str, device: str = "cuda:0"
) -> tuple:
    """Load an indexer (stack or single) plus its feature extractor + config.

    Detects the checkpoint format: if ``is_stack`` is True (new per-layer
    training), returns a :class:`COREIndexerStack` and the press should be
    built with ``use_indexer_stack=True``.  Otherwise returns a single
    :class:`COREIndexerMLP` (legacy, broadcast-equivalent).
    """
    indexer, checkpoint = load_indexer_stack(checkpoint_path, device=device)
    raw_config = checkpoint["config"]
    validate_checkpoint_interface(raw_config)
    config = COREConfig.from_dict(raw_config)
    feature_extractor = _build_feature_extractor(checkpoint)
    feature_extractor.to(torch.device(device))
    use_stack = bool(checkpoint.get("is_stack", False))
    indexer.requires_grad_(False)
    indexer.eval()
    geometry_module = (
        feature_extractor.key_geometry
        if feature_extractor.key_geometry is not None
        else feature_extractor.key_proj
    )
    geometry_module.requires_grad_(False)
    geometry_module.eval()
    return indexer, config, feature_extractor, use_stack


def _apply_protection_fields(press: COREScorerPress, config: COREConfig) -> None:
    """Propagate the checkpoint's production selection policy."""
    press.n_sink_protect = getattr(config, "n_sink_protect", 0)
    press.recency_alpha = getattr(config, "recency_alpha", 0.0)
    press.recency_window = getattr(config, "recency_window", 64)
    press.scoring_stride = getattr(config, "scoring_stride", 1)
    press.span_block_size = getattr(config, "span_block_size", 1)
    press.span_pool_beta = getattr(config, "span_pool_beta", 5.0)
    press.span_selection_mode = getattr(
        config, "span_selection_mode", "sliding_max"
    )
    press.coverage_fraction = getattr(config, "coverage_fraction", 0.25)
    press.query_feature_weight = getattr(config, "query_feature_weight", 0.0)
    press.lexical_overlap_weight = getattr(
        config, "lexical_overlap_weight", 0.0
    )
    press.identifier_lexical_weight = getattr(
        config, "identifier_lexical_weight", 0.0
    )
    press.identifier_min_match_tokens = getattr(
        config, "identifier_min_match_tokens", 20
    )
    press.identifier_context_skip = getattr(
        config, "identifier_context_skip", 64
    )
    press.identifier_span_size = getattr(
        config, "identifier_span_size", 32
    )
    press.protect_query_tokens = getattr(
        config, "protect_query_tokens", False
    )
    press.prefill_qk_scope = getattr(config, "prefill_qk_scope", "all")


def _load_required_memory_stack(
    checkpoint_path: str,
    config: COREConfig,
    device: str,
):
    """Load a valid joint memory checkpoint or fail loudly.

    ``core_memory`` and ``core_full`` are production memory paths. Silently
    falling back to hard eviction makes an evaluation look successful while
    never using the trained memory module.
    """
    from core.train import load_memory_stack

    try:
        memory_stack = load_memory_stack(checkpoint_path, device=device)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "CORE memory evaluation requires a joint checkpoint containing a "
            "compatible memory_state_dict. Use core_indexer for an intentional "
            f"hard-eviction ablation. Checkpoint: {checkpoint_path}"
        ) from exc

    if (
        config.n_layers is not None
        and int(memory_stack.n_layers) != int(config.n_layers)
    ):
        raise ValueError(
            "CORE checkpoint memory/indexer layer mismatch: memory has "
            f"{memory_stack.n_layers} layers, config has {config.n_layers}."
        )
    expected_width = (
        config.n_q_heads * config.head_dim
        if config.n_q_heads and config.head_dim else config.hidden_size
    )
    if expected_width is not None and int(memory_stack.d_model) != int(expected_width):
        raise ValueError(
            "CORE checkpoint memory query-width mismatch: memory has "
            f"{memory_stack.d_model}, expected H_q * d_head = {expected_width}."
        )
    memory_stack.requires_grad_(False)
    memory_stack.eval()
    return memory_stack


def load_core_scorer_press(
    checkpoint_path: str,
    device: str = "cuda:0",
    compression_ratio: float = 0.5,
) -> COREScorerPress:
    """
    Load a trained CORE indexer as a :class:`ScorerPress`.

    The returned press is compatible with :class:`DecodingPress` and can be
    used as its ``base_press`` argument.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved CORE checkpoint (``.pt`` file).
    device : str
        Device to load the indexer on.
    compression_ratio : float
        Fraction of KV cache to evict (0.5 = keep 50 %).

    Returns
    -------
    COREScorerPress
    """
    indexer, config, feature_extractor, use_stack = _load_indexer_and_config(
        checkpoint_path, device
    )
    ratio = _check_compression_ratio(compression_ratio, config, checkpoint_path)

    press = COREScorerPress(
        indexer=indexer,
        feature_extractor=feature_extractor,
        compression_ratio=ratio,
        student_temp=config.student_temp,
        use_indexer_stack=use_stack,
    )
    _apply_protection_fields(press, config)
    press._ensure_device(torch.device(device))
    return press


def load_core_indexer_press(
    checkpoint_path: str,
    device: str = "cuda:0",
    compression_ratio: Optional[float] = None,
) -> COREIndexerPress:
    """
    Load a trained CORE indexer checkpoint as a kvpress-compatible press.

    **Backward-compatible** function that returns a :class:`COREIndexerPress`
    supporting both prefill-only and budget-based compression.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved CORE checkpoint.
    device : str
        Device to load the indexer on.
    compression_ratio : float or None
        Override the compression ratio.  If ``None``, uses the value stored
        in the checkpoint.

    Returns
    -------
    COREIndexerPress
    """
    indexer, config, feature_extractor, use_stack = _load_indexer_and_config(
        checkpoint_path, device
    )
    ratio = _check_compression_ratio(compression_ratio, config, checkpoint_path)

    press = COREIndexerPress(
        indexer=indexer,
        feature_extractor=feature_extractor,
        compression_ratio=ratio,
        budget=config.budget,
        student_temp=config.student_temp,
        use_indexer_stack=use_stack,
    )
    _apply_protection_fields(press, config)
    press._ensure_device(torch.device(device))
    return press


def load_core_memory_press(
    checkpoint_path: str,
    device: str = "cuda:0",
    compression_ratio: Optional[float] = None,
) -> COREMemoryPress:
    """Load a trained CORE indexer + MemoryModuleStack as a
    :class:`COREMemoryPress` (memory-writing + learned readout).

    Wraps a :class:`COREScorerPress` with the trained per-layer memory stack.
    Evicted KV entries are removed after contributing to the fast state
    ``M_t``, ``b_t``, and ``Z_t``. The gated memory readout is projected into
    the output space and added by :meth:`COREMemoryPress.forward_hook`,
    implementing the correction in CORE Eq. 12.

    Honors the checkpoint config's ``enable_memory_writing`` flag - when False
    the press still scores+selects identically but performs plain eviction
    (useful for ablation against :func:`load_core_indexer_press`).

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved CORE checkpoint (must contain a
        ``memory_state_dict`` written by the joint-training path).
    device : str
        Device to load the indexer + memory stack on.
    compression_ratio : float or None
        Override the compression ratio. If ``None``, uses the checkpoint value.

    Returns
    -------
    COREMemoryPress
    """
    indexer, config, feature_extractor, use_stack = _load_indexer_and_config(
        checkpoint_path, device
    )
    ratio = _check_compression_ratio(compression_ratio, config, checkpoint_path)

    base = COREScorerPress(
        indexer=indexer,
        feature_extractor=feature_extractor,
        compression_ratio=ratio,
        student_temp=config.student_temp,
        use_indexer_stack=use_stack,
    )
    _apply_protection_fields(base, config)
    base._ensure_device(torch.device(device))

    memory_stack = _load_required_memory_stack(
        checkpoint_path, config, device
    )

    press = COREMemoryPress(
        base_press=base,
        memory_module=memory_stack,
        memory_decay=config.memory_decay,
        memory_write_eta=config.memory_write_eta,
        memory_uniform_fraction=config.memory_uniform_fraction,
        enable_memory_writing=config.enable_memory_writing,
        memory_query_space=config.memory_query_space,
        memory_value_space=config.memory_value_space,
    )
    press._ensure_device(torch.device(device))
    return press


def load_core_prefill_decoding_press(
    checkpoint_path: str,
    device: str = "cuda:0",
    compression_ratio: Optional[float] = None,
    decoding_compression_interval: int = 128,
    decoding_target_size: Optional[int] = None,
    decoding_hidden_states_buffer_size: Optional[int] = None,
) -> COREPrefillDecodingPress:
    """
    Load CORE scoring and memory for both prefill and periodic decoding.

    Both phases share one :class:`COREMemoryPress` and its memory state.
    Prefill uses ``compression_ratio``; :class:`COREDecodingPress` periodically
    prunes the growing cache to the configured decoding budget.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved CORE checkpoint.
    device : str
        Device to load the indexer on.
    compression_ratio : float or None
        Compression ratio for the prefilling phase.  If ``None``, uses the
        checkpoint value.
    decoding_compression_interval : int
        Number of decoding steps between compression during decoding.
    decoding_target_size : int or None
        Target KV cache size in tokens. None reuses the retained prefill budget.
    decoding_hidden_states_buffer_size : int or None
        Defaults to the compression interval so all interval queries are used.
        Explicit sizes must be at least the compression interval.

    Returns
    -------
    PrefillDecodingPress
    """
    indexer, config, feature_extractor, use_stack = _load_indexer_and_config(
        checkpoint_path, device
    )
    prefill_ratio = _check_compression_ratio(compression_ratio, config, checkpoint_path)

    memory_stack = _load_required_memory_stack(
        checkpoint_path, config, device
    )

    # Prefill press: COREScorerPress wrapped in COREMemoryPress (memory
    # write + learned readout).
    prefill_base = COREScorerPress(
        indexer=indexer,
        feature_extractor=feature_extractor,
        compression_ratio=prefill_ratio,
        student_temp=config.student_temp,
        use_indexer_stack=use_stack,
    )
    _apply_protection_fields(prefill_base, config)
    prefill_base._ensure_device(torch.device(device))
    # One shared press/state is used for prefill and decode. This preserves the
    # prefill M/b/Z state and lets periodic decode writes update the same memory.
    shared_memory_press = COREMemoryPress(
        base_press=prefill_base,
        memory_module=memory_stack,
        memory_decay=config.memory_decay,
        memory_write_eta=config.memory_write_eta,
        memory_uniform_fraction=config.memory_uniform_fraction,
        enable_memory_writing=config.enable_memory_writing,
        memory_query_space=config.memory_query_space,
        memory_value_space=config.memory_value_space,
    )

    # DecodingPress temporarily overrides compression_ratio on this same press.
    decoding_press = COREDecodingPress(
        base_press=shared_memory_press,
        compression_interval=decoding_compression_interval,
        target_size=decoding_target_size or 1,
        hidden_states_buffer_size=decoding_hidden_states_buffer_size,
    )

    combined = COREPrefillDecodingPress(
        match_prefill_budget=decoding_target_size is None,
        prefilling_press=shared_memory_press,
        decoding_press=decoding_press,
        query_aware_prefill=False,
    )
    return combined
