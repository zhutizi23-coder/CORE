"""
Frozen-backbone statistics for online and offline CORE training.

Runs the frozen backbone forward pass over LongAlpaca-12k samples and collects,
for each layer, the per-token statistics needed to build:
  - CORE 14-dim feature vectors (via COREFeatureExtractor)
  - Diversity-aware teacher distribution (via DiversityAwareTeacher)

Online collection retains statistics on the model device. The offline path
moves feature vectors and teacher distributions to CPU for serialization.
"""

import logging

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from core.config import COREConfig
from core.data import load_longalpaca, tokenize_sample
from core.features import COREFeatureExtractor
from core.teacher import DiversityAwareTeacher

logger = logging.getLogger(__name__)


class BackboneStatsCollector:
    """
    Collects per-layer attention statistics from the frozen backbone via
    forward hooks. Stats stay on the model device.

    For each attention layer, stores:
      - key_states:        (n_kv_heads, seq_len, head_dim)  [post-RoPE]
      - value_states:      (n_kv_heads, seq_len, head_dim)
      - query_states:     (n_q_heads, seq_len, head_dim) [post-RoPE]
      - attention_weights: (n_q_heads, seq_len, seq_len), if returned by the backend
    """

    def __init__(self, model: AutoModelForCausalLM, capture_all_layers: bool = True):
        self.model = model
        self.handles: list = []
        self._pre_o_outputs: dict = {}
        self.layer_stats: dict = {}
        self._was_training = model.training
        self.device = next(model.parameters()).device
        # Layers whose attention outputs and queries are captured for memory supervision.
        # capture_all_layers selects every layer; otherwise _capture_output_layers
        # selects the layers to retain.
        self._capture_all_layers = capture_all_layers
        self._capture_output_layers: set = set()
        # Store attention weights separately under layer_stats[layer_idx].
        self._store_per_layer_attn: bool = True

    def _make_pre_o_hook(self, layer_idx: int):
        def hook(module, args):
            if self._capture_all_layers or layer_idx in self._capture_output_layers:
                self._pre_o_outputs[layer_idx] = args[0].detach()
        return hook

    def _make_hook(self, layer_idx: int):
        """Create a forward hook for a specific attention layer."""

        def hook(module, args, kwargs, output):
            self.layer_stats[layer_idx] = {}

            # Attention weights have shape (1, n_q_heads, seq_q, seq_k).
            # Retain them per captured layer for teacher supervision.
            is_capture_layer = (
                self._capture_all_layers or layer_idx in self._capture_output_layers
            )
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                aw = output[1].detach()
                if aw.dim() == 4:
                    aw = aw.squeeze(0)
                if self._store_per_layer_attn and is_capture_layer:
                    self.layer_stats[layer_idx]["attention_weights"] = aw
                else:
                    # Release attention weights for layers that are not captured.
                    del aw

            # Capture pre-o_proj reconstruction targets for selected layers.
            # capture_all_layers includes every layer by default.
            if is_capture_layer and isinstance(output, tuple) \
                    and len(output) >= 1 and output[0] is not None:
                ao = self._pre_o_outputs.pop(layer_idx, None)
                if ao is None:
                    raise RuntimeError("Missing pre-o_proj attention output for memory supervision")
                if ao.dim() == 3:  # (B, L, d_model) - squeeze batch if 1
                    ao = ao.squeeze(0) if ao.shape[0] == 1 else ao
                self.layer_stats[layer_idx]["attn_output"] = ao
                # Cache attention inputs for query reconstruction and legacy
                # consumers; the default memory path uses post-RoPE queries.
                hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                if hs is not None:
                    hs = hs.detach()
                    if hs.dim() == 3 and hs.shape[0] == 1:
                        hs = hs.squeeze(0)
                    self.layer_stats[layer_idx]["hidden_states_in"] = hs  # (L, d_model)

            # Capture cached keys (post-RoPE) and values for each selected layer.
            if is_capture_layer:
                try:
                    past_kv = kwargs.get("past_key_values", None)
                    if past_kv is not None:
                        keys = past_kv.layers[layer_idx].keys.detach()
                        values = past_kv.layers[layer_idx].values.detach()
                        if keys.dim() == 4:
                            keys = keys.squeeze(0)
                            values = values.squeeze(0)
                        self.layer_stats[layer_idx]["key_states"] = keys
                        self.layer_stats[layer_idx]["value_states"] = values
                except Exception as e:
                    logger.debug(f"Could not extract KV from cache at layer {layer_idx}: {e}")

            # Recompute post-RoPE queries from the attention input hidden states
            # so teacher logits use the same rotary coordinates as the cached keys.
            if is_capture_layer:
                try:
                    hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                    position_ids = kwargs.get("position_ids", None)
                    position_embeddings = kwargs.get("position_embeddings", None)
                    if hidden_states is not None:
                        q_states = self._recompute_post_rope_queries(
                            module, hidden_states, position_ids, position_embeddings
                        )
                        if q_states is not None:
                            self.layer_stats[layer_idx]["query_states"] = q_states
                except Exception as e:
                    logger.debug(f"Could not recompute query states at layer {layer_idx}: {e}")

        return hook

    @staticmethod
    @torch.no_grad()
    def _recompute_post_rope_queries(
        attn_module, hidden_states, position_ids, position_embeddings=None
    ):
        """
        Recompute post-RoPE query states from the hidden states fed into an
        attention module. ``hidden_states`` is assumed already layernormed
        (as passed to ``self_attn.forward`` by the decoder layer).

        Version-robust RoPE application:
          * transformers >= 4.43 precomputes (cos, sin) once and passes them
            as ``position_embeddings`` in the attention kwargs — we reuse them
            directly (guaranteed to match the keys).
          * otherwise, fall back to recomputing via the model-level
            ``rotary_emb`` (older layout stored it on the attention module).

        Returns query states of shape (n_q_heads, seq_q, head_dim) on the
        module's device, or None if the module layout is unsupported.
        """
        bsz, q_len, _ = hidden_states.shape
        if not hasattr(attn_module, "q_proj"):
            return None

        # num_heads / head_dim are on the attention module (older) or its config.
        cfg = attn_module.config
        num_heads = getattr(attn_module, "num_heads", cfg.num_attention_heads)
        head_dim = getattr(attn_module, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

        q = attn_module.q_proj(hidden_states)  # (bsz, q_len, num_heads*head_dim)
        q = q.view(bsz, q_len, num_heads, head_dim)
        if hasattr(attn_module, "q_norm"):
            q = attn_module.q_norm(q)
        q = q.transpose(1, 2)  # (bsz, n_q, q_len, d)

        # Apply the SAME rotary embedding the attention layer used, so the
        # recomputed queries align with the cached (post-RoPE) keys.
        if position_embeddings is not None:
            # New layout (>=4.43): (cos, sin) precomputed upstream.
            cos, sin = position_embeddings
            # Pass q as the 'k' arg too (we only keep q); newer apply_rotary_pos_emb
            # signature is (q, k, cos, sin, unsqueeze_dim) and multiplies both.
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        else:
            # Recompute cos/sin for an explicit query window. This path is also
            # used during decode because the forward kwargs contain cos/sin for
            # only the current token, while hidden_states buffers the interval.
            rotary_emb = getattr(attn_module, "rotary_emb", None)
            if rotary_emb is None:
                return None
            if position_ids is None:
                position_ids = torch.arange(
                    q_len, device=q.device, dtype=torch.long
                ).unsqueeze(0).expand(bsz, -1)
            try:
                # transformers >= 4.43 model-level LlamaRotaryEmbedding.
                cos, sin = rotary_emb(q, position_ids)
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            except (TypeError, ValueError):
                # Unsupported rotary layouts return None to select the caller's fallback.
                return None

        if bsz == 1:
            q = q.squeeze(0)  # (n_q, q_len, head_dim)
        else:
            q = q.mean(dim=0)  # pool batch (prefill uses bsz=1 anyway)
        return q.detach()

    def __enter__(self):
        self.model.eval()
        language_model = (
            self.model.model.language_model
            if hasattr(self.model.model, "language_model")
            else self.model.model
        )
        for layer in language_model.layers:
            layer_idx = layer.self_attn.layer_idx
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                self._make_pre_o_hook(layer_idx)
            ))
            h = layer.self_attn.register_forward_hook(self._make_hook(layer_idx), with_kwargs=True)
            self.handles.append(h)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []
        self._pre_o_outputs.clear()
        self.model.train(self._was_training)

    def get_stats(self) -> dict:
        return self.layer_stats

    def reset(self):
        self.layer_stats = {}


def collect_training_data(
    config: COREConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    feature_extractor: COREFeatureExtractor,
    teacher: DiversityAwareTeacher,
    samples: list = None,
) -> dict:
    """
    Run Phase 1 data collection.

    For each LongAlpaca sample:
      1. Forward pass through frozen backbone (collect per-layer stats on GPU)
      2. Build 14-dim feature vector per token (on GPU)
      3. Build teacher distribution per token (on GPU)
      4. Compute boundary windows P_bd / N_bd
      5. Move final tensors to CPU for storage

    Parameters
    ----------
    config : COREConfig
    model : backbone model (frozen)
    tokenizer : matching tokenizer
    feature_extractor : COREFeatureExtractor (already moved to device)
    teacher : DiversityAwareTeacher
    samples : list, optional
        Pre-loaded samples. If None, loads via load_longalpaca.

    Returns
    -------
    dict
        Dictionary with keys 'X', 'pi_T', 'P_bd', 'N_bd', 'budgets'.
    """
    logger.info("Phase 1: collecting training data from frozen backbone")

    if samples is None:
        samples = load_longalpaca(config, tokenizer)
    logger.info(f"Using {len(samples)} samples for collection")

    if config.budget is not None:
        # Use the configured absolute budget, capped by the candidate count.
        budget_fn = lambda n: min(config.budget, n - 1)
    elif config.retention_range > 0:
        # Sample a retention fraction for each chunk
        # to supervise boundary windows at different ranks.
        center = 1.0 - config.compression_ratio  # default keep-fraction
        lo = max(0.1, center - config.retention_range / 2)
        hi = min(0.9, center + config.retention_range / 2)

        def budget_fn(n):
            frac = torch.empty(1).uniform_(lo, hi).item()
            return max(1, min(n - 1, int(n * frac)))
    else:
        # Derive the budget from the configured compression ratio.
        budget_fn = lambda n: max(1, int(n * (1 - config.compression_ratio)))

    collected = {
        "X": [], "pi_T": [], "P_bd": [], "N_bd": [], "budgets": [],
        # Memory MSE supervision (IndexMem eq.8): captured per-chunk for the
        # middle layer only. o_full = full attention output (no eviction),
        # o_compressed = output reconstructed from retained KV, query_hidden =
        # the middle layer's input hidden_states (to compute Linear_θ(q)),
        # keep_indices = teacher's Top-B indices used to reconstruct o_compressed.
        "o_full": [], "o_compressed": [], "query_hidden": [], "keep_indices": [],
    }

    device = torch.device(config.backbone_device)
    dtype = getattr(torch, config.dtype, torch.bfloat16)

    collector = BackboneStatsCollector(model)
    # Select the middle-layer reconstruction target for this offline path.
    language_model = (
        model.model.language_model if hasattr(model.model, "language_model") else model.model
    )
    _mid_layer_for_capture = len(language_model.layers) // 2
    collector._capture_output_layers = {_mid_layer_for_capture}
    with collector:
        for i, sample in enumerate(tqdm(samples, desc="Collecting")):
            try:
                tok = tokenize_sample(tokenizer, sample, config.max_seq_len, device)
                input_ids = tok["input_ids"]
                seq_len = input_ids.shape[1]

                if seq_len < config.n_sink + 2:
                    logger.debug(f"Sample {i} too short ({seq_len} tokens), skipping")
                    continue

                # Forward pass with attention weights
                collector.reset()
                with torch.no_grad():
                    # Start each sample with an empty KV cache.
                    model(input_ids=input_ids, output_attentions=True, use_cache=True,
                          past_key_values=None)
                # Drop the returned outputs (incl. the new cache) immediately so
                # the only live GPU refs are inside collector.layer_stats.
                torch.cuda.empty_cache()

                layer_stats = collector.get_stats()
                if not layer_stats:
                    logger.debug(f"No stats collected for sample {i}, skipping")
                    continue

                use_layers = sorted(layer_stats.keys())

                # ---- Build teacher distribution (on GPU) ----
                # Attention weights were accumulated incrementally across all
                # layers during the forward pass (see _make_hook) to avoid
                # storing 32 per-layer tensors (~34 GB). Finalize the mean here.
                if collector._attn_sum is None or collector._attn_count == 0:
                    logger.debug(f"No attention weights for sample {i}, skipping")
                    continue
                pooled_attn = (collector._attn_sum / collector._attn_count).to(dtype)  # (n_q, sq, sk)

                # Use the middle layer's keys for coverage and teacher utility.
                mid_idx = use_layers[len(use_layers) // 2]
                mid_keys_t = layer_stats[mid_idx].get("key_states")  # (n_kv, sk, d)

                if mid_keys_t is None:
                    logger.debug(f"Middle layer {mid_idx} has no key_states for sample {i}, skipping")
                    continue
                keys_mid_layer = mid_keys_t.to(dtype)  # (n_kv, seq_len, head_dim)
                z_norm = feature_extractor.project_keys(keys_mid_layer, mid_idx)

                # Use the middle layer's post-RoPE queries for teacher utility.
                # Use the probability surrogate when queries were not captured.
                mid_queries = layer_stats[mid_idx].get("query_states")  # may be None

                # Prefer the pooled-attention path (Path A) over the per-layer
                # logits path (Path B). Path A recovers logits via
                # log(p*|C_t|) with causal normalization, which compensates for
                # early queries seeing fewer keys; Path B computes raw <q,k>/sqrt(d)
                # without that compensation, producing a flatter teacher
                # (KL(teacher||uniform) ~0.75 vs ~0.47, verified). The flatter
                # teacher made the KL loss unable to converge (0.629 vs smoke's 0.068).
                if pooled_attn is not None:
                    pi_T = teacher.compute_teacher_distribution(
                        z=z_norm, attention_weights=pooled_attn.float()
                    )  # (N,)
                elif mid_queries is not None and mid_keys_t is not None:
                    pi_T = teacher.compute_teacher_distribution(
                        z=z_norm,
                        query_states=mid_queries.to(dtype).float(),
                        key_states=mid_keys_t.to(dtype).float(),
                    )  # (N,)
                else:
                    logger.debug(f"No attention weights or query/key states for sample {i}, skipping")
                    continue

                # ---- Build features (on GPU) ----
                # Q-K relation features: we derive them from pooled attention weights
                # directly (attention weights already encode q-k matching). We pass
                # attention weights into the feature extractor as the "query_states"
                # surrogate via a dedicated path.
                mid_keys = layer_stats[mid_idx]["key_states"].to(dtype)  # (n_kv, sk, d)
                mid_values = layer_stats[mid_idx]["value_states"].to(dtype)  # (n_kv, sk, d)
                # Use the cross-layer pooled attention (incremental mean built
                # during the forward) instead of a per-layer copy, which is no
                # longer stored (memory optimization).
                mid_attn = pooled_attn.to(dtype)  # (n_q, sq, sk)

                # Build the stats dict expected by feature extractor
                stats_for_fe = {
                    mid_idx: {
                        "query_states": None,  # will be derived from attention
                        "key_states": mid_keys,
                        "value_states": mid_values,
                        "attention_weights": mid_attn,
                    }
                }

                features = _extract_features_gpu(
                    feature_extractor, stats_for_fe, pooled_attn, keys_mid_layer, mid_values,
                    query_states=mid_queries.to(dtype) if mid_queries is not None else None,
                    key_states_mid=mid_keys_t.to(dtype) if mid_keys_t is not None else None,
                    token_ids=input_ids[0],
                    layer_idx=mid_idx,
                )  # (N, 14)

                # ---- Boundary windows ----
                B = budget_fn(seq_len)
                rank_T, P_bd, N_bd = teacher.compute_teacher_ranking(
                    pi_T, B, config.boundary_delta
                )

                collected["X"].append(features.cpu())
                collected["pi_T"].append(pi_T.cpu())
                collected["P_bd"].append(P_bd.cpu())
                collected["N_bd"].append(N_bd.cpu())
                collected["budgets"].append(B)

                # ---- Memory MSE supervision (IndexMem eq.8) ----
                # Capture o_full (pre-o_proj attention output, d_model) +
                # query_hidden for the middle layer, and reconstruct o_compressed
                # (pre-o_proj over the retained KV) from the teacher's Top-B
                # retained positions. o_compressed = flatten(A_ret @ V_ret) where
                # A_ret is re-softmaxed over the retained set, matching a compressed
                # forward. Uses teacher ranking as keep_indices since the indexer
                # isn't trained yet at Phase-1 collection time.
                o_full_t = layer_stats[mid_idx].get("attn_output")    # (L, d_model) or None
                q_hidden_t = layer_stats[mid_idx].get("hidden_states_in")  # (L, d_model) or None
                keep_idx_t = pi_T.topk(B).indices  # (B,) teacher's Top-B positions

                # Fetch the middle layer's o_proj for reconstruction.
                mid_o_proj = language_model.layers[mid_idx].self_attn.o_proj

                if o_full_t is not None and q_hidden_t is not None and mid_queries is not None:
                    o_full_cpu = o_full_t.float().cpu()
                    q_hidden_cpu = q_hidden_t.float().cpu()
                    keep_idx_cpu = keep_idx_t.cpu()
                    # Reconstruct o_compressed on CPU to avoid GPU OOM (the
                    # backbone forward already uses ~all GPU memory; the
                    # reconstruction intermediates (n_q, L, B) would push it over).
                    o_compressed_t = _reconstruct_compressed_output(
                        mid_keys.float().cpu(),    # (n_kv, L, d) post-RoPE
                        mid_values.float().cpu(),  # (n_kv, L, d)
                        mid_queries.float().cpu(),# (n_q, L, d) post-RoPE
                        keep_idx_cpu,             # (B,) retained positions
                        mid_o_proj,               # nn.Linear(d_model, d_model)
                        num_kv_groups=config.n_q_heads // max(1, config.n_kv_heads) if config.n_kv_heads else 1,
                    )
                    collected["o_full"].append(o_full_cpu)
                    collected["o_compressed"].append(o_compressed_t.float().cpu() if o_compressed_t is not None else o_full_cpu.clone())
                    collected["query_hidden"].append(q_hidden_cpu)
                    collected["keep_indices"].append(keep_idx_cpu)
                else:
                    # Fallback: no supervision data (hook didn't capture output).
                    collected["o_full"].append(None)
                    collected["o_compressed"].append(None)
                    collected["query_hidden"].append(None)
                    collected["keep_indices"].append(None)

                if (i + 1) % 50 == 0:
                    logger.info(
                        f"Collected {i + 1}/{len(samples)} samples, "
                        f"last chunk: N={seq_len}, B={B}"
                    )

                # Free per-sample GPU tensors
                del pooled_attn, keys_mid_layer, k_pooled, z, z_norm, pi_T, features
                collector.reset()
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"OOM on sample {i} (len={seq_len}): {e}; skipping")
                # Clear any partially-built GPU state. Local names below may not
                # exist yet if the OOM happened mid-forward, so guard with del +
                # try/except rather than a bare del.
                for _name in ("pooled_attn", "keys_mid_layer", "k_pooled",
                              "z", "z_norm", "pi_T", "features"):
                    try:
                        del locals()[_name]
                    except KeyError:
                        pass
                collector.reset()
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                logger.warning(f"Error on sample {i}: {e}; skipping")
                collector.reset()
                continue

    total_tokens = sum(x.shape[0] for x in collected["X"])
    logger.info(
        f"Phase 1 complete: collected {len(collected['X'])} chunks, "
        f"{total_tokens} total tokens"
    )
    return collected


def _extract_features_gpu(
    feature_extractor: COREFeatureExtractor,
    stats_for_fe: dict,
    pooled_attn: torch.Tensor,
    keys_single_layer: torch.Tensor,
    values_mid: torch.Tensor,
    query_states: torch.Tensor = None,
    key_states_mid: torch.Tensor = None,
    token_ids: torch.Tensor = None,
    is_prefill: bool = True,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """Build the 14-dimensional feature vector on GPU.

    Compute Q-K features from query/key states when available, otherwise
    use pooled attention probabilities. Coverage features use the selected
    layer's keys and the configured geometry projection."""
    if pooled_attn is not None:
        device = pooled_attn.device
        dtype = pooled_attn.dtype
        # pooled_attn: (n_q, seq_q, seq_k)
        n_q, seq_q, seq_k = pooled_attn.shape
    elif query_states is not None and key_states_mid is not None:
        device = query_states.device
        dtype = query_states.dtype
        n_q, seq_q = query_states.shape[:2]
        seq_k = key_states_mid.shape[1]
    else:
        raise ValueError("Q/K states or pooled attention are required")
    n_kv = feature_extractor.n_kv_heads
    num_groups = feature_extractor.num_kv_groups
    head_dim = feature_extractor.head_dim
    N = seq_k

    # Compute Q-K relation features with the shared training/inference extractor.
    if query_states is not None and key_states_mid is not None:
        qk_features = feature_extractor.compute_qk_relation_features(
            query_states.to(dtype), key_states_mid.to(dtype)
        )
    else:
        qk_features = feature_extractor.compute_qk_relation_features_from_attention(
            pooled_attn.float()
        )

    # Compute coverage features from the selected layer's keys on GPU.
    if layer_idx is None:
        if len(stats_for_fe) != 1:
            raise ValueError("layer_idx is required when stats_for_fe is ambiguous")
        layer_idx = int(next(iter(stats_for_fe)))
    cov_features = feature_extractor.compute_coverage_features(
        keys_single_layer, layer_idx
    )  # (N, 4)

    # ---- Auxiliary features (on GPU) ----
    aux_features = feature_extractor.compute_auxiliary_features(
        values_mid, seq_len=N, is_prefill=is_prefill, token_ids=token_ids
    )  # (N, 4)

    features = torch.cat([qk_features.float(), cov_features.float(), aux_features.float()], dim=-1)
    if not torch.isfinite(features).all():
        bad = (~torch.isfinite(features)).sum().item()
        raise FloatingPointError(
            f"CORE feature extraction produced {bad} non-finite values"
        )
    return features


@torch.no_grad()
def _reconstruct_compressed_output(
    keys: torch.Tensor,        # (n_kv, L, d) post-RoPE
    values: torch.Tensor,      # (n_kv, L, d)
    queries: torch.Tensor,     # (n_q, L, d) post-RoPE
    keep_indices: torch.Tensor,# (B,) retained positions (1D)
    o_proj,                    # backbone o_proj (d_model -> d_model)
    num_kv_groups: int,        # n_q_heads // n_kv_heads (GQA group size)
    query_tail: int | None = None,
    query_chunk_size: int = 128,
    eps: float = 1e-8,
    output_space: str = "pre_o_proj",
):
    """
    Reconstruct the pre-o_proj attention output over the RETAINED KV set,
    matching what a compressed forward (only keep_indices kept) would produce.

    For each requested query position s, the compressed attention is:
        logits_ret[s] = <q_s, K_ret> / sqrt(d)        # (n_q, B)  (per kv-head)
        apply causal mask (query s attends only to retained keys <= s)
        A_ret[s]   = softmax(logits_ret[s])           # re-normalized over retained
        o_head[s]  = A_ret[s] @ V_ret                  # (n_heads, d) after repeat_kv
        o[s]       = flatten(o_head)         # (d_model,)

    This gives o_compressed in the SAME (pre-o_proj, d_model) space as o_full
    (captured at the input to o_proj), so the MSE loss ||o_full - o_compressed - g(q)·m(q)||
    is well-defined.

    Parameters
    ----------
    keys, values : torch.Tensor
        Cached (post-RoPE) keys/values, shape (n_kv, L, d).
    queries : torch.Tensor
        Recomputed post-RoPE queries, shape (n_q, L, d).
    keep_indices : torch.Tensor
        1D tensor of retained positions, shape (B,).
    o_proj
        The attention module's output projection (nn.Linear).
    num_kv_groups : int
        GQA group size (n_q_heads // n_kv_heads).
    query_tail : int or None
        If set, reconstruct only the final ``query_tail`` query positions. The
        joint memory objective supervises only this causal tail, so computing
        all L query rows first wastes O(n_q * L * B) memory.
    query_chunk_size : int
        Maximum query rows processed at once. This also bounds memory for the
        legacy full-sequence caller.
    eps : float
        Numerical stability.

    Returns
    -------
    torch.Tensor or None
        o_compressed of shape (Q, d_model), where Q is L when ``query_tail`` is
        None and otherwise ``min(L, query_tail)``; None if infeasible.
    """
    n_kv, L, d = keys.shape
    n_q = queries.shape[0]
    B = keep_indices.shape[0]
    if B == 0:
        return None
    if n_q != n_kv * num_kv_groups:
        raise ValueError(
            f"GQA shape mismatch: n_q={n_q}, n_kv={n_kv}, "
            f"num_kv_groups={num_kv_groups}"
        )
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    # Compute attention in fp32 without materializing an (n_q, L, B) tensor.
    # Grouped GQA avoids repeating K/V across query heads.
    device = keys.device
    head_dim = d
    keep = keep_indices.to(device).long()      # (B,)
    keep_sorted, _ = keep.sort()
    if keep_sorted.min() < 0 or keep_sorted.max() >= L:
        raise IndexError("keep_indices contains a position outside the KV sequence")

    K_ret = keys.to(device).float()[:, keep_sorted, :]   # (n_kv, B, d)
    V_ret = values.to(device).float()[:, keep_sorted, :] # (n_kv, B, d)

    q_count = L if query_tail is None else max(1, min(L, int(query_tail)))
    query_start = L - q_count
    q = queries[:, query_start:, :].to(device).float()
    # repeat_kv orders heads as (kv_head, group), so this view exactly matches
    # the backbone's GQA head mapping without allocating repeated K/V tensors.
    q = q.reshape(n_kv, num_kv_groups, q_count, head_dim)

    w = o_proj.weight.data.to(device=device, dtype=torch.float32)
    b = (
        o_proj.bias.data.to(device=device, dtype=torch.float32)
        if o_proj.bias is not None else None
    )
    outputs = []
    scale = head_dim ** 0.5
    for start in range(0, q_count, query_chunk_size):
        end = min(q_count, start + query_chunk_size)
        q_chunk = q[:, :, start:end, :]  # (n_kv, groups, Qc, d)
        logits = torch.einsum("ngqd,nkd->ngqk", q_chunk, K_ret) / scale

        # Use absolute prompt positions even when only the tail is requested.
        pos = torch.arange(
            query_start + start, query_start + end, device=device
        ).view(1, 1, -1, 1)
        causal = pos >= keep_sorted.view(1, 1, 1, B)
        logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        o_grouped = torch.einsum("ngqk,nkd->ngqd", attn, V_ret)
        q_chunk_len = end - start
        o_heads = o_grouped.reshape(n_q, q_chunk_len, head_dim)
        o_flat = o_heads.permute(1, 0, 2).reshape(q_chunk_len, n_q * head_dim)
        if output_space == "pre_o_proj":
            outputs.append(o_flat)
        elif output_space == "post_o_proj":
            outputs.append(torch.nn.functional.linear(o_flat, w, bias=b))
        else:
            raise ValueError(f"Unknown attention output space: {output_space}")

    return torch.cat(outputs, dim=0).float()


@torch.no_grad()
def collect_online_batch(
    model: AutoModelForCausalLM,
    collector: "BackboneStatsCollector",
    feature_extractor: COREFeatureExtractor,
    teacher: DiversityAwareTeacher,
    config: COREConfig,
    input_ids: torch.Tensor,
    compression_ratio: float = 0.5,
    compute_teacher_compressed: bool = False,
    include_decode_features: bool = False,
    include_memory_supervision: bool = True,
    query_start: int | None = None,
    selection_end: int | None = None,
) -> dict:
    """Run one online distillation step: forward + per-layer teacher + features.

    Computes a per-layer teacher signal (utility + DRIVE D-optimal diversity)
    and the 14-dim feature vector for EVERY captured layer, returning an
    in-memory dict suitable for direct backprop through the indexer.

    Returns
    -------
    dict
        ``{layer_idx: {"X": (N,14), "pi_T": (N,), "P_bd": (n,), "N_bd": (n,), "budget": int}}``
        for each captured layer. Tensors stay on the backbone device.
    """
    device = input_ids.device
    dtype = getattr(torch, config.dtype, torch.bfloat16)
    seq_len = input_ids.shape[1]
    selection_len = (
        seq_len if selection_end is None
        else max(1, min(seq_len, int(selection_end)))
    )
    if include_memory_supervision and selection_len == seq_len:
        tail = min(int(config.memory_supervision_tail),
                   max(0, seq_len - int(config.n_sink_protect) - 1))
        selection_len -= tail
    # Skip sequences too short to be useful.
    if seq_len < config.n_sink + 2:
        return {}

    # 1) Forward pass; hooks capture per-layer attention + KV + queries.
    collector.reset()
    with collector:
        model(input_ids=input_ids, output_attentions=False, use_cache=True,
              past_key_values=None)
    layer_stats = collector.get_stats()
    captured_layers = sorted(
        idx for idx, s in layer_stats.items() if s.get("key_states") is not None
    )
    if not captured_layers:
        return {}

    B = max(
        min(config.n_sink_protect, selection_len),
        min(selection_len, int(selection_len * (1.0 - compression_ratio))),
    )
    delta = config.boundary_delta
    eps = 1e-8

    batch: dict = {}
    for layer_idx in captured_layers:
        stats = layer_stats[layer_idx]
        keys_all = stats.get("key_states")      # (n_kv, L, head_dim)
        values_all = stats.get("value_states")  # (n_kv, L, head_dim)
        queries_all = stats.get("query_states") # (n_q, L, head_dim) post-RoPE
        attn = stats.get("attention_weights")   # optional fallback only
        if keys_all is None or values_all is None or queries_all is None:
            continue
        keys = keys_all[:, :selection_len, :]
        values = values_all[:, :selection_len, :]
        scope = getattr(config, "prefill_qk_scope", "all")

        if scope == "question":
            if query_start is None:
                raise ValueError(
                    "prefill_qk_scope='question' requires query_start, "
                    "but query_start is None."
                )

            query_start = int(query_start)

            if not (0 <= query_start < selection_len):
                raise ValueError(
                    f"Invalid query_start={query_start} for "
                    f"selection_len={selection_len}"
                )

            queries = queries_all[:, query_start:selection_len, :]

        elif scope == "all":
            queries = queries_all[:, :selection_len, :]

        else:
            raise ValueError(
                f"Unknown prefill_qk_scope={scope!r}"
            )

        # Coverage representation z for this layer (from its own keys).
        keys_f = keys.to(dtype)
        z_norm = feature_extractor.project_keys(keys_f, layer_idx)

        # Per-layer teacher: utility from THIS layer's true logits, then DRIVE.
        utility = teacher.compute_attention_utility(
            queries.to(dtype).float(), keys.to(dtype).float()
        )                                                    # (L,)
        utility = teacher.propagate_local_utility(utility)
        # Apply the configured number of DRIVE-style updates, then the
        # distillation temperature (one by default) for KL and boundary ranking.
        teacher_weights = teacher.compute_d_optimal_coverage(z_norm, utility)
        pi_T = teacher.to_teacher_distribution(teacher_weights)
        pi_T = pi_T / (pi_T.sum() + 1e-12)  # numerical safety
        if not torch.isfinite(pi_T).all():
            raise FloatingPointError(
                f"Non-finite teacher distribution at layer {layer_idx}"
            )

        loss_mask = torch.ones(selection_len, dtype=torch.bool, device=device)
        n_excluded = 0
        if getattr(config, "exclude_sink_from_kl", False):
            n_excluded = min(config.n_sink, selection_len)
            loss_mask[:n_excluded] = False
        # Optional query protection also excludes question tokens from KL.
        if (
            getattr(config, "protect_query_tokens", False)
            and query_start is not None
        ):
            loss_mask[int(query_start):selection_len] = False
        boundary_budget = max(0, B - n_excluded)
        _, P_bd, N_bd = teacher.compute_teacher_ranking(
            pi_T, boundary_budget, delta, valid_mask=loss_mask
        )

        # 14-dim features for this layer (re-use _extract_features_gpu with the
        # layer's own tensors - it computes qk features from true logits when
        # query_states/key_states_mid are given, which we always provide).
        stats_for_fe = {
            layer_idx: {
                "query_states": queries,
                "key_states": keys_f,
                "value_states": values.to(dtype),
                "attention_weights": (
                    attn[..., :selection_len, :selection_len].to(dtype)
                    if attn is not None else None
                ),
            }
        }
        features = _extract_features_gpu(
            feature_extractor, stats_for_fe,
            (attn[..., :selection_len, :selection_len].to(dtype)
             if attn is not None else None),
            keys_f, values.to(dtype),
            query_states=queries.to(dtype),
            key_states_mid=keys.to(dtype),
            token_ids=input_ids[0, :selection_len],
            layer_idx=layer_idx,
        )                                                    # (L, 14)

        batch[layer_idx] = {
            "X": features,        # (L, 14)
            "pi_T": pi_T,        # (L,) temperature-scaled finite-iteration teacher distribution
            "P_bd": P_bd,        # (n_pos,) boundary retain (by pi_T rank)
            "N_bd": N_bd,         # (n_neg,) boundary evict  (by pi_T rank)
            "budget": B,
            "loss_mask": loss_mask,
            "selection_len": selection_len,
            "query_start": (
                int(query_start) if query_start is not None else None
            ),
            "token_ids": input_ids[0, :selection_len].detach(),
        }

        # Periodic decode-window supervision. The indexer sees phase=1 and the
        # same query set used during periodic decode compression: the queries
        # generated within the most recent compression interval. This prevents
        # a_phase=1 from being an unseen inference-only input.
        if include_decode_features:
            window = max(1, min(config.decode_train_window, selection_len))
            # Q must end at the candidate prefix, not at a future answer.
            decode_queries = queries_all[:, selection_len - window:selection_len, :]
            decode_utility = teacher.compute_attention_utility(
                decode_queries.to(dtype).float(), keys.to(dtype).float()
            )
            decode_utility = teacher.propagate_local_utility(decode_utility)
            decode_weights = teacher.compute_d_optimal_coverage(
                z_norm, decode_utility
            )
            decode_pi_T = teacher.to_teacher_distribution(decode_weights)
            decode_pi_T = decode_pi_T / (decode_pi_T.sum() + 1e-12)
            if not torch.isfinite(decode_pi_T).all():
                raise FloatingPointError(
                    f"Non-finite decode teacher distribution at layer {layer_idx}"
                )
            _, decode_P_bd, decode_N_bd = teacher.compute_teacher_ranking(
                decode_pi_T, boundary_budget, delta, valid_mask=loss_mask
            )
            decode_features = _extract_features_gpu(
                feature_extractor, stats_for_fe,
                attn.to(dtype) if attn is not None else None,
                keys_f, values.to(dtype),
                query_states=decode_queries.to(dtype),
                key_states_mid=keys.to(dtype),
                token_ids=input_ids[0, :selection_len],
                is_prefill=False,
                layer_idx=layer_idx,
            )
            batch[layer_idx].update({
                "decode_X": decode_features,
                "decode_pi_T": decode_pi_T,
                "decode_P_bd": decode_P_bd,
                "decode_N_bd": decode_N_bd,
                "decode_loss_mask": loss_mask,
            })

        # Release per-layer tensors to keep peak memory bounded.
        stats.pop("attention_weights", None)

    # Memory supervision tensors for each captured layer:
    #   o_full: full attention output before o_proj.
    #   o_compressed: attention reconstructed from retained KV before o_proj.
    #   memory_queries: post-RoPE query heads.
    #   keep_indices: the teacher's Top-B indices.
    if captured_layers and include_memory_supervision:
        language_model = (
            model.model.language_model
            if hasattr(model.model, "language_model") else model.model
        )
        num_kv_groups = (
            config.n_q_heads // max(1, config.n_kv_heads)
            if config.n_kv_heads else 1
        )
        for layer_idx in captured_layers:
            stats = layer_stats[layer_idx]
            o_full_t = stats.get("attn_output")          # (L, d_model) or None
            q_hidden_t = stats.get("hidden_states_in")   # (L, d_model) or None
            ly_keys = stats.get("key_states")            # (n_kv, L, head_dim)
            ly_values = stats.get("value_states")         # (n_kv, L, head_dim)
            ly_queries = stats.get("query_states")        # (n_q, L, head_dim)
            if (o_full_t is None or q_hidden_t is None
                    or ly_keys is None or ly_values is None
                    or ly_queries is None
                    or layer_idx not in batch):
                continue
            ly_o_proj = language_model.layers[layer_idx].self_attn.o_proj
            # Top-B keep indices (sorted by pi_T descending).
            keep_idx = batch[layer_idx]["pi_T"].topk(B).indices
            batch[layer_idx]["o_full"] = o_full_t.detach()
            if compute_teacher_compressed:
                # Reconstruct the teacher-selected cache for memory-only training.
                # Joint training uses the student-selected cache.
                o_compressed_t = _reconstruct_compressed_output(
                    ly_keys, ly_values, ly_queries, keep_idx, ly_o_proj,
                    num_kv_groups,
                )
                batch[layer_idx]["o_compressed"] = (
                    o_compressed_t.detach()
                    if o_compressed_t is not None
                    else o_full_t.detach().clone()
                )
            batch[layer_idx]["query_hidden"] = q_hidden_t.detach()
            batch[layer_idx]["keep_indices"] = keep_idx.detach()
            # Raw per-layer tensors retained for the joint-training step to
            # reconstruct o_compressed using the STUDENT Top-B selection.
            batch[layer_idx]["memory_keys"] = ly_keys.detach()
            batch[layer_idx]["memory_values"] = ly_values.detach()
            batch[layer_idx]["memory_queries"] = ly_queries.detach()
            batch[layer_idx]["memory_o_proj"] = ly_o_proj
            batch[layer_idx]["num_kv_groups"] = num_kv_groups

            # Do not build persistent fp32 (L, d_model) Q/K/V copies here.
            # Joint training expands only the evicted K/V rows and supervised
            # query tail. At 4K x 32 layers this removes several GiB of live
            # tensors and avoids projecting every value through o_proj.

    return batch
