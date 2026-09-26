"""
Offline and online training for CORE indexer and memory modules.

Trains a per-layer stack of independent MLPs (one per transformer layer, aligned
with IndexMem's per-layer indexer) with KL and boundary hinge losses:

    L_CORE = lambda_cal * L_cal + lambda_bd * L_bd

Each training chunk is a sequence of N tokens with:
  - features X_i (N, 14)
  - teacher distribution pi_T (N,)
  - boundary windows P_bd, N_bd
  - budget B

Offline training consumes pre-collected features and teacher distributions.
Online training computes them with the frozen backbone for each batch.
Joint training also optimizes the memory reconstruction objective.
"""

import logging
import math
import random
from pathlib import Path
from core.selection import (
    contiguous_block_scores,
    query_aware_span_selection_scores,
    region_stratified_order,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm.auto import tqdm
from transformers.models.llama.modeling_llama import repeat_kv

from core.config import COREConfig
from core.data import make_multi_evidence_retrieval_sample, tokenize_sample
from core.collector import (
    BackboneStatsCollector,
    _reconstruct_compressed_output,
    collect_online_batch,
)
from core.features import COREFeatureExtractor
from core.indexer import (
    COREIndexerMLP,
    COREIndexerStack,
    MemoryModule,
    MemoryModuleStack,
    calibrated_kl_loss,
    conditional_evicted_weights,
    core_loss,
    student_distribution,
)
from core.teacher import DiversityAwareTeacher

logger = logging.getLogger(__name__)


def _validate_training_objective(config: COREConfig) -> None:
    """Fail fast when a run would violate CORE's formal training objective."""
    if getattr(config, "memory_relative_loss", False):
        raise ValueError(
            "memory_relative_loss is a legacy experimental objective; "
            "formal CORE training uses plain reconstruction MSE"
        )
    if config.memory_value_space != "pre_o_proj":
        raise ValueError("Paper training requires pre_o_proj memory and reconstruction targets")
    n_sink = int(getattr(config, "n_sink", 0))
    n_protected = int(getattr(config, "n_sink_protect", 0))
    if n_sink != n_protected:
        raise ValueError(
            "Training and inference sink counts must match: "
            f"n_sink={n_sink}, n_sink_protect={n_protected}"
        )
    if n_protected > 0 and not getattr(config, "exclude_sink_from_kl", False):
        raise ValueError(
            "Protected sink tokens are outside the ranking candidate domain; "
            "enable exclude_sink_from_kl for KL, boundary, and memory alignment"
        )


def _paper_memory_reconstruction_loss(
    o_full: torch.Tensor,
    o_compressed: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """L_mem = E_q ||o_full - o_compressed - g(q)m(q)||_2^2."""
    return (o_full - o_compressed - residual).pow(2).sum(dim=-1).mean()


def _training_compression_ratio(config: COREConfig, global_step: int) -> float:
    """Deterministically rotate boundary supervision across formal budgets."""
    ratios = tuple(float(r) for r in getattr(config, "training_compression_ratios", ()))
    if not ratios:
        return float(config.compression_ratio)
    if any(not 0.0 < r < 1.0 for r in ratios):
        raise ValueError("training_compression_ratios must lie in (0, 1)")
    seed = int(config.seed) * 1_000_003 + int(global_step)
    return ratios[random.Random(seed).randrange(len(ratios))]


def _training_sample_for_step(
    config: COREConfig,
    sample: dict,
    global_step: int,
) -> dict:
    """Deterministically mix generic multi-evidence retrieval supervision."""
    fraction = float(getattr(config, "retrieval_augmentation_fraction", 0.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("retrieval_augmentation_fraction must be in [0, 1]")
    if fraction == 0.0:
        return sample
    decision_seed = int(config.seed) * 1_000_003 + int(global_step)
    if random.Random(decision_seed).random() >= fraction:
        return sample
    return make_multi_evidence_retrieval_sample(sample, decision_seed)


def _augment_validation_samples(config: COREConfig, samples: list) -> list:
    """Add a fixed augmented view so checkpoint selection measures both modes."""
    fraction = float(getattr(config, "retrieval_augmentation_fraction", 0.0))
    if fraction <= 0.0 or not samples:
        return samples
    count = max(1, min(len(samples), round(len(samples) * fraction)))
    augmented = [
        make_multi_evidence_retrieval_sample(
            samples[index], int(config.seed) * 2_000_003 + index
        )
        for index in range(count)
    ]
    return list(samples) + augmented


def _memory_gate_override(config: COREConfig, global_step: int) -> float | None:
    """Force memory on only during the absolute start of joint training.

    The decision is intentionally independent of ``schedule_offset`` so that
    resuming a checkpoint cannot restart gate warmup and change the trajectory.
    """
    joint_step = int(global_step) - int(
        getattr(config, "indexer_pretrain_steps", 0)
    )
    warmup_steps = int(getattr(config, "memory_gate_warmup_steps", 0))
    return 1.0 if 0 < joint_step <= warmup_steps else None


def _teacher_distribution_on_loss_axis(
    pi_teacher: torch.Tensor, loss_mask: torch.Tensor | None
) -> torch.Tensor:
    """Return the normalized teacher distribution used by the KL term."""
    if loss_mask is not None:
        pi_teacher = pi_teacher[
            loss_mask.to(device=pi_teacher.device, dtype=torch.bool)
        ]
    return pi_teacher / pi_teacher.sum().clamp_min(1e-12)


def _split_online_samples(samples: list, n_val: int) -> tuple[list, list]:
    """Deterministically reserve samples that are never used for optimization."""
    if len(samples) < 2 or n_val <= 0:
        return samples, []
    n_val = min(int(n_val), max(1, len(samples) // 20), len(samples) - 1)
    return samples[:-n_val], samples[-n_val:]


def _online_training_length(config: COREConfig, global_step: int) -> int:
    """Use a mostly-2K curriculum with periodic true-4K training examples."""
    max_length = int(config.max_seq_len)
    interval = int(getattr(config, "long_seq_every", 0))
    if interval > 0 and global_step % interval != 0:
        return min(max_length, int(getattr(config, "train_short_seq_len", 2048)))
    return max_length


def _deployment_source_layer(
    layer_idx: int, batch: dict, scoring_stride: int
) -> int:
    """Return the layer whose score is reused by deployment for layer_idx.

    A stride of one means true per-layer scoring. For stride > 1, inference
    scores at 0, stride, 2*stride, ... and reuses that score inside the group.
    Training and validation must use the same source layer.
    """
    stride = max(1, int(scoring_stride))
    source = int(layer_idx) if stride == 1 else int(layer_idx) // stride * stride
    return source if source in batch else int(layer_idx)


def _deployment_scores(
    indexer: COREIndexerStack,
    batch: dict,
    layer_idx: int,
    config: COREConfig,
    cache: dict,
    feature_key: str = "X",
) -> torch.Tensor:
    """Score with the exact cross-layer reuse policy used at inference."""
    source = _deployment_source_layer(
        layer_idx, batch, getattr(config, "scoring_stride", 1)
    )
    cache_key = (source, feature_key)
    if cache_key not in cache:
        source_data = batch[source]
        if feature_key not in source_data:
            raise KeyError(f"Layer {source} has no {feature_key} features")
        cache[cache_key] = indexer(source_data[feature_key].float(), source)
    return cache[cache_key]


def _deployment_span_scores(
    scores: torch.Tensor,
    data: dict,
    config: COREConfig,
) -> torch.Tensor:
    """Apply the exact production prefill span transform used for Top-B.

    The calibrated raw scores remain the KL and memory-writing distribution.
    Only selection-sensitive objectives and metrics use this transformed view.
    Decode scoring remains tokenwise, matching ``COREScorerPress``.
    """
    return query_aware_span_selection_scores(
        scores,
        data.get("query_start"),
        getattr(config, "span_block_size", 1),
        token_ids=data.get("token_ids"),
        identifier_lexical_weight=getattr(
            config, "identifier_lexical_weight", 0.0
        ),
        identifier_min_match_tokens=getattr(
            config, "identifier_min_match_tokens", 20
        ),
        identifier_context_skip=getattr(
            config, "identifier_context_skip", 64
        ),
        identifier_span_size=getattr(config, "identifier_span_size", 32),
        protect_query_tokens=getattr(config, "protect_query_tokens", False),
        span_pool_beta=getattr(config, "span_pool_beta", 5.0),
        span_selection_mode=getattr(
            config, "span_selection_mode", "sliding_max"
        ),
        selection_budget=data.get("budget"),
        coverage_fraction=getattr(config, "coverage_fraction", 0.25),
        n_sink_protect=getattr(config, "n_sink_protect", 0),
    )


def _deployment_teacher_selection_scores(
    pi_teacher: torch.Tensor,
    data: dict,
    config: COREConfig,
) -> torch.Tensor:
    """Map teacher utility to the exact span space used by deployment.

    Ranking raw token probabilities against block-pooled student scores creates
    contradictory labels: a teacher-negative neighbour of an important token
    is necessarily raised by the student's span policy.  Applying the shared
    transform to log teacher utility makes P_bd/N_bd, validation Top-B and
    deployed selection operate on one coordinate system.
    """
    teacher_logits = pi_teacher.float().clamp_min(1e-12).log()
    return _deployment_span_scores(teacher_logits, data, config)


def _block_selection_spec(
    data: dict,
    config: COREConfig,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, int] | None:
    """Return context-block representatives and the full-block budget.

    Query and sink tokens are reserved separately. Only blocks that fit whole
    are supervised at the cutoff; any unavoidable token-budget remainder is a
    filler region and cannot create contradictory equal-score P/N labels.
    """
    if getattr(config, "span_selection_mode", "tokenwise") != "block_lme":
        return None
    query_start = data.get("query_start")
    if query_start is None:
        return None
    query_start = int(query_start)
    length = int(loss_mask.numel())
    if query_start <= 0 or query_start >= length:
        return None
    block_size = max(1, int(getattr(config, "span_block_size", 1)))
    if block_size <= 1:
        return None

    excluded = int((~loss_mask).sum().item())
    starts = torch.arange(
        0, query_start, block_size, device=loss_mask.device, dtype=torch.long
    )
    # Pick a non-sink representative from each block and ignore a block that is
    # entirely excluded (normally impossible unless the sequence is tiny).
    ends = (starts + block_size).clamp_max(query_start)
    reps = torch.maximum(starts, starts.new_full(starts.shape, excluded))
    valid = reps < ends
    reps = reps[valid]
    if reps.numel() < 2:
        return None

    query_reserved = (
        length - query_start
        if getattr(config, "protect_query_tokens", False)
        else 0
    )
    available = max(1, int(data["budget"]) - excluded - query_reserved)
    n_full = max(1, min(int(reps.numel()) - 1, available // block_size))
    return reps, n_full



def _coverage_local_boundary_views(
    scores: torch.Tensor,
    data: dict,
    config: COREConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous score views for the two coverage-local policy channels.

    These views deliberately contain no hard Top-K membership offset.  The
    coverage channel remains tokenwise; the local channel uses differentiable
    log-mean-exp block pooling.  Both share the same standardization used by
    the deployed query-aware selector.
    """
    adjusted = scores.float()
    mean = adjusted.mean(dim=-1, keepdim=True)
    std = adjusted.std(dim=-1, keepdim=True).clamp_min(1e-6)
    coverage_scores = (adjusted - mean) / std
    local_scores = coverage_scores.clone()
    query_start = data.get("query_start")
    if query_start is not None:
        query_start = max(0, min(int(query_start), int(scores.shape[-1])))
        if query_start > 0:
            local_scores[:query_start] = contiguous_block_scores(
                coverage_scores[:query_start],
                max(1, int(getattr(config, "span_block_size", 1))),
                beta=float(getattr(config, "span_pool_beta", 5.0)),
            )
    return (
        coverage_scores.to(dtype=scores.dtype),
        local_scores.to(dtype=scores.dtype),
    )


def _coverage_local_boundary_spec(
    data: dict,
    config: COREConfig,
    loss_mask: torch.Tensor,
) -> dict | None:
    """Return context-only quotas/candidates for hybrid boundary supervision."""
    if getattr(config, "span_selection_mode", "tokenwise") not in {
        "coverage_local", "coverage_local_balanced"
    }:
        return None
    query_start = data.get("query_start")
    if query_start is None:
        return None
    length = int(loss_mask.numel())
    query_start = max(0, min(int(query_start), length))
    positions = torch.arange(length, device=loss_mask.device)
    context_mask = loss_mask.to(torch.bool) & (positions < query_start)
    context_idx = context_mask.nonzero(as_tuple=False).flatten()
    if context_idx.numel() < 2:
        return None

    # Sink and protected question tokens are structural reservations. They are
    # charged to total Top-B but excluded from both learned cutoff objectives.
    reserved = int((~loss_mask.to(torch.bool)).sum().item())
    if getattr(config, "protect_query_tokens", False):
        reserved += length - query_start
    context_budget = max(
        1, min(int(context_idx.numel()) - 1, int(data["budget"]) - reserved)
    )
    fraction = float(getattr(config, "coverage_fraction", 0.25))
    coverage_budget = max(
        0, min(context_budget, int(round(context_budget * fraction)))
    )
    local_token_budget = context_budget - coverage_budget

    block_size = max(1, int(getattr(config, "span_block_size", 1)))
    starts = torch.arange(
        0, query_start, block_size,
        device=loss_mask.device, dtype=torch.long,
    )
    ends = (starts + block_size).clamp_max(query_start)
    reps = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        valid = context_idx[(context_idx >= start) & (context_idx < end)]
        if valid.numel():
            reps.append(valid[0])
    reps = (
        torch.stack(reps)
        if reps
        else torch.empty(0, device=loss_mask.device, dtype=torch.long)
    )
    if reps.numel() >= 2 and local_token_budget > 0:
        local_blocks = max(
            1,
            min(
                int(reps.numel()) - 1,
                (local_token_budget + block_size - 1) // block_size,
            ),
        )
    else:
        local_blocks = 0
    return {
        "context_mask": context_mask,
        "context_idx": context_idx,
        "context_budget": context_budget,
        "coverage_budget": coverage_budget,
        "local_token_budget": local_token_budget,
        "block_reps": reps,
        "local_blocks": local_blocks,
        "coverage_weight": fraction,
        "local_weight": 1.0 - fraction,
    }


def _rank_boundary_window(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    budget: int,
    delta: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if budget <= 0 or budget >= candidates.numel() or candidates.numel() < 2:
        empty = candidates.new_empty(0)
        return empty, empty
    budget = max(1, min(int(budget), int(candidates.numel()) - 1))
    order = candidates[scores[candidates].argsort(descending=True, stable=True)]
    delta = max(1, int(delta))
    positive = order[max(0, budget - delta):budget]
    negative = order[budget:min(len(order), budget + delta)]
    return positive, negative


def _coverage_local_boundary_targets(
    pi_teacher: torch.Tensor,
    data: dict,
    config: COREConfig,
    loss_mask: torch.Tensor,
) -> dict | None:
    spec = _coverage_local_boundary_spec(data, config, loss_mask)
    if spec is None:
        return None
    teacher_logits = pi_teacher.float().clamp_min(1e-12).log()
    coverage_scores, local_scores = _coverage_local_boundary_views(
        teacher_logits, data, config
    )
    coverage_components = []
    if getattr(config, "span_selection_mode", "") == "coverage_local_balanced":
        coverage_order = region_stratified_order(
            coverage_scores[:int(data["query_start"])],
            spec["context_idx"],
            max(1, int(getattr(config, "span_block_size", 1))),
            block_scores=local_scores[:int(data["query_start"])],
        )
        coverage_budget = max(
            1, min(int(spec["coverage_budget"]), coverage_order.numel() - 1)
        )
        selected = coverage_order[:coverage_budget]
        delta = max(1, int(config.boundary_delta))
        block_size = max(1, int(getattr(config, "span_block_size", 1)))

        # Deployment interleaves rank-r tokens across regions. A single global
        # P/N window at the interleaved cutoff is not a learnable score
        # constraint: it can demand that a weak region's rank-r token outrank a
        # strong region's rank-(r+1) token even though round-robin, not score,
        # creates that ordering. Supervise only constraints the scorer controls:
        # the retained/evicted boundary *within each region*.
        n_blocks = (int(data["query_start"]) + block_size - 1) // block_size
        for block in range(n_blocks):
            start = block * block_size
            end = min(int(data["query_start"]), start + block_size)
            candidates = spec["context_idx"]
            candidates = candidates[(candidates >= start) & (candidates < end)]
            if candidates.numel() < 2:
                continue
            n_selected = int(torch.isin(candidates, selected).sum().item())
            if n_selected <= 0 or n_selected >= candidates.numel():
                continue
            ranked = candidates[
                coverage_scores[candidates].argsort(descending=True, stable=True)
            ]
            positive = ranked[max(0, n_selected - delta):n_selected]
            negative = ranked[
                n_selected:min(ranked.numel(), n_selected + delta)
            ]
            if positive.numel() and negative.numel():
                coverage_components.append(
                    ("coverage", positive, negative)
                )

        # If the last round gives one extra anchor to only some regions, the
        # choice is controlled by block priority (the local LME score). Add one
        # block-level cutoff constraint for exactly that partial round.
        selected_counts = []
        valid_starts = []
        for start in range(0, int(data["query_start"]), block_size):
            end = min(int(data["query_start"]), start + block_size)
            candidates = spec["context_idx"]
            candidates = candidates[(candidates >= start) & (candidates < end)]
            if candidates.numel() == 0:
                continue
            valid_starts.append(start)
            selected_counts.append(int(torch.isin(candidates, selected).sum().item()))
        if selected_counts and min(selected_counts) < max(selected_counts):
            starts_t = spec["context_idx"].new_tensor(valid_starts)
            counts_t = starts_t.new_tensor(selected_counts)
            high = counts_t.max()
            low = counts_t.min()
            positive = starts_t[counts_t == high]
            negative = starts_t[counts_t == low]
            if positive.numel() and negative.numel():
                coverage_components.append(("local", positive, negative))

        # Populate aggregate diagnostic fields. Loss and validation use
        # the decomposed regional constraints above.
        if coverage_components:
            coverage_P = torch.cat([c[1] for c in coverage_components])
            coverage_N = torch.cat([c[2] for c in coverage_components])
        else:
            coverage_P = spec["context_idx"].new_empty(0)
            coverage_N = spec["context_idx"].new_empty(0)
    else:
        coverage_P, coverage_N = _rank_boundary_window(
            coverage_scores,
            spec["context_idx"],
            spec["coverage_budget"],
            int(config.boundary_delta),
        )
    delta_blocks = max(
        1,
        (int(config.boundary_delta)
         + int(getattr(config, "span_block_size", 1)) - 1)
        // int(getattr(config, "span_block_size", 1)),
    )
    local_P, local_N = _rank_boundary_window(
        local_scores,
        spec["block_reps"],
        spec["local_blocks"],
        delta_blocks,
    )
    return {
        **spec,
        "coverage_P_bd": coverage_P,
        "coverage_N_bd": coverage_N,
        "coverage_boundary_components": coverage_components,
        "local_P_bd": local_P,
        "local_N_bd": local_N,
    }


def _coverage_local_boundary_components(
    scores: torch.Tensor,
    data: dict,
    config: COREConfig,
) -> list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] | None:
    targets = data.get("coverage_local_boundary")
    if targets is None:
        return None
    coverage_scores, local_scores = _coverage_local_boundary_views(
        scores, data, config
    )
    decomposed = targets.get("coverage_boundary_components")
    if decomposed:
        coverage_weight = float(targets["coverage_weight"])
        per_component = coverage_weight / len(decomposed)
        components = [
            (
                per_component,
                coverage_scores if view == "coverage" else local_scores,
                positive,
                negative,
            )
            for view, positive, negative in decomposed
        ]
    else:
        components = [(
            float(targets["coverage_weight"]), coverage_scores,
            targets["coverage_P_bd"], targets["coverage_N_bd"],
        )]
    components.append((
        float(targets["local_weight"]), local_scores,
        targets["local_P_bd"], targets["local_N_bd"],
    ))
    return components


def _core_loss_for_prefill(
    scores: torch.Tensor,
    pi_teacher: torch.Tensor,
    data: dict,
    config: COREConfig,
) -> tuple[torch.Tensor, dict]:
    components = _coverage_local_boundary_components(scores, data, config)
    multi_budget_targets = data.get("multi_budget_boundary_targets")
    if multi_budget_targets is not None:
        if components is not None:
            raise ValueError(
                "all-budget boundary supervision currently requires the "
                "formal tokenwise selection policy"
            )
        selection_scores = _deployment_span_scores(scores, data, config)
        components = [
            (1.0, selection_scores, positive, negative)
            for positive, negative in multi_budget_targets
        ]
    else:
        selection_scores = (
            None
            if components is not None
            else _deployment_span_scores(scores, data, config)
        )
    return core_loss(
        scores,
        pi_teacher,
        data["P_bd"],
        data["N_bd"],
        config,
        loss_mask=data.get("loss_mask"),
        selection_scores=selection_scores,
        boundary_components=components,
    )

def _deployment_boundary_targets(
    pi_teacher: torch.Tensor,
    data: dict,
    config: COREConfig,
    teacher: DiversityAwareTeacher,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cutoff labels in the exact deployed token/block space."""
    teacher_scores = _deployment_teacher_selection_scores(
        pi_teacher, data, config
    )
    block_spec = _block_selection_spec(data, config, loss_mask)
    if block_spec is not None:
        reps, n_full = block_spec
        order = reps[teacher_scores[reps].argsort(descending=True, stable=True)]
        delta_blocks = max(
            1,
            (int(config.boundary_delta)
             + int(config.span_block_size) - 1)
            // int(config.span_block_size),
        )
        positive = order[max(0, n_full - delta_blocks):n_full]
        negative = order[n_full:min(len(order), n_full + delta_blocks)]
        return positive, negative

    excluded = int((~loss_mask).sum().item())
    boundary_budget = max(0, int(data["budget"]) - excluded)
    _, positive, negative = teacher.compute_teacher_ranking(
        teacher_scores,
        boundary_budget,
        config.boundary_delta,
        valid_mask=loss_mask,
    )
    return positive, negative


def _causal_memory_partition(
    selection_scores: torch.Tensor,
    config: COREConfig,
    tail: int,
    total_length: int | None = None,
    compression_ratio: float | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Split a training sequence into compressed prefix and local suffix.

    The suffix acts as pseudo-decode queries.  Only prefix tokens may be
    evicted/written to memory; all suffix tokens stay in the reconstructed
    attention set, matching generation where the current decode token has not
    been part of the prefill memory write.
    """
    score_len = int(selection_scores.shape[0])
    if total_length is not None and int(total_length) > score_len:
        prefix_n = score_len
        N = int(total_length)
    else:
        N = score_len
        tail = max(1, min(N - 1, int(tail)))
        prefix_n = N - tail
    if config.budget is not None:
        prefix_budget = min(prefix_n, max(1, int(config.budget)))
    else:
        ratio = (
            float(config.compression_ratio)
            if compression_ratio is None
            else float(compression_ratio)
        )
        if not 0.0 < ratio < 1.0:
            raise ValueError("memory compression_ratio must lie in (0, 1)")
        prefix_budget = max(
            1, int(prefix_n * (1.0 - ratio))
        )
    n_sink = min(int(config.n_sink_protect), prefix_n)
    prefix_budget = min(prefix_n, max(prefix_budget, n_sink))

    ranked = selection_scores[:prefix_n].detach().clone()

    if n_sink > 0 and n_sink < prefix_n:
        ranked[:n_sink] = ranked.max() + 1.0

    # ``selection_scores`` has already passed through the exact deployment
    # span transform in ``_deployment_span_scores``. Expanding it again here
    # makes joint memory training select a wider cache than inference and
    # supervises memory on the wrong evicted set.
    ranked_for_selection = ranked

    # Reapply sink protection after span expansion.
    if n_sink > 0:
        ranked_for_selection[:n_sink] = (
            ranked_for_selection.max().detach() + 1.0
        )

    keep_prefix = ranked_for_selection.argsort(descending=True, stable=True)[:prefix_budget]
    local_suffix = torch.arange(
        prefix_n, N, device=selection_scores.device, dtype=torch.long
    )
    read_keep = torch.cat((keep_prefix, local_suffix), dim=0)
    evict_mask = torch.zeros(N, dtype=torch.bool, device=selection_scores.device)
    evict_mask[:prefix_n] = True
    evict_mask[keep_prefix] = False
    return prefix_n, read_keep, evict_mask


def _align_stride_boundary_targets(
    batch: dict,
    config: COREConfig,
    teacher: DiversityAwareTeacher,
) -> None:
    """Assign deployment-aligned cutoff targets to each score-reuse group.

    For ``scoring_stride > 1`` the deployment policy produces one score vector
    at the source layer and reuses it for the remaining layers in the group.
    Per-layer P/N windows can disagree, making the same score vector satisfy
    contradictory inequalities. KL over the group already behaves like a
    distribution consensus; this helper gives the boundary term the matching
    consensus ranking.
    """
    stride = max(1, int(getattr(config, "scoring_stride", 1)))
    aggregate = bool(getattr(config, "aggregate_stride_boundary", False))
    if not batch:
        return

    all_budgets = bool(
        getattr(config, "all_budget_boundary_supervision", False)
    )
    if all_budgets and (
        getattr(config, "span_selection_mode", "tokenwise") != "tokenwise"
        or int(getattr(config, "span_block_size", 1)) != 1
    ):
        raise ValueError(
            "all_budget_boundary_supervision is defined for the formal "
            "tokenwise CORE selector"
        )

    layer_indices = sorted(batch)
    if aggregate and stride > 1:
        groups = [
            [
                idx for idx in layer_indices
                if group_start <= idx < group_start + stride
            ]
            for group_start in range(0, max(layer_indices) + 1, stride)
        ]
    else:
        groups = [[idx] for idx in layer_indices]

    for group in groups:
        if not group:
            continue
        source = batch[group[0]]
        mask = source.get("loss_mask")
        if mask is None:
            mask = torch.ones_like(source["pi_T"], dtype=torch.bool)
        mask = mask.to(dtype=torch.bool, device=source["pi_T"].device)
        group_pi = torch.stack([batch[idx]["pi_T"] for idx in group]).mean(dim=0)
        group_pi = group_pi / group_pi.sum().clamp_min(1e-12)
        hybrid_targets = _coverage_local_boundary_targets(
            group_pi, source, config, mask
        )
        if hybrid_targets is None:
            positive, negative = _deployment_boundary_targets(
                group_pi, source, config, teacher, mask
            )
        else:
            # Populate aggregate P/N fields for serialization.
            # Prefill loss uses both channel-specific targets.
            positive = hybrid_targets["coverage_P_bd"]
            negative = hybrid_targets["coverage_N_bd"]
            if positive.numel() == 0 or negative.numel() == 0:
                positive = hybrid_targets["local_P_bd"]
                negative = hybrid_targets["local_N_bd"]
        for idx in group:
            batch[idx]["P_bd"] = positive
            batch[idx]["N_bd"] = negative
            if hybrid_targets is not None:
                batch[idx]["coverage_local_boundary"] = hybrid_targets

        if all_budgets:
            selection_len = int(source.get("selection_len", group_pi.numel()))
            targets = []
            for ratio in _validation_compression_ratios(config):
                ratio_budget = max(
                    min(config.n_sink_protect, selection_len),
                    min(
                        selection_len,
                        int(selection_len * (1.0 - float(ratio))),
                    ),
                )
                budget_view = dict(source)
                budget_view["budget"] = ratio_budget
                ratio_positive, ratio_negative = _deployment_boundary_targets(
                    group_pi, budget_view, config, teacher, mask
                )
                targets.append((ratio_positive, ratio_negative))
            for idx in group:
                batch[idx]["multi_budget_boundary_targets"] = targets

        if aggregate and stride > 1 and all(
            "decode_pi_T" in batch[idx] for idx in group
        ):
            decode_pi = torch.stack(
                [batch[idx]["decode_pi_T"] for idx in group]
            ).mean(dim=0)
            decode_pi = decode_pi / decode_pi.sum().clamp_min(1e-12)
            excluded = int((~mask).sum().item())
            boundary_budget = max(0, int(source["budget"]) - excluded)
            _, decode_positive, decode_negative = teacher.compute_teacher_ranking(
                decode_pi, boundary_budget, config.boundary_delta,
                valid_mask=mask,
            )
            for idx in group:
                batch[idx]["decode_P_bd"] = decode_positive
                batch[idx]["decode_N_bd"] = decode_negative


def _validation_compression_ratios(config: COREConfig) -> tuple[float, ...]:
    """Return the frozen multi-budget validation grid used during training."""
    configured = tuple(
        float(r) for r in getattr(config, "training_compression_ratios", ())
    )
    ratios = configured or (float(config.compression_ratio),)
    if any(not 0.0 < ratio < 1.0 for ratio in ratios):
        raise ValueError("validation compression ratios must lie in (0, 1)")
    # Preserve the paper/config order while avoiding duplicate validation work.
    return tuple(dict.fromkeys(ratios))


def _aggregate_multibudget_validation(
    ratio_metrics: list[tuple[float, dict]],
) -> dict:
    """Macro-average quality metrics and retain per-budget diagnostics."""
    available = [(ratio, metrics) for ratio, metrics in ratio_metrics if metrics]
    if not available:
        return {}

    aggregate: dict[str, float | int] = {
        "val_num_budgets": len(available),
    }
    keys = sorted(set().union(*(metrics.keys() for _, metrics in available)))
    for key in keys:
        values = [metrics[key] for _, metrics in available if key in metrics]
        if not values:
            continue
        if key == "val_samples":
            # Samples are shared across budgets, so do not count them four times.
            aggregate[key] = max(int(value) for value in values)
        elif key.endswith("_chunks"):
            aggregate[key] = sum(int(value) for value in values)
        else:
            aggregate[key] = sum(float(value) for value in values) / len(values)

    for ratio, metrics in available:
        label = f"cr{int(round(100 * ratio)):03d}"
        for key, value in metrics.items():
            suffix = key[4:] if key.startswith("val_") else key
            aggregate[f"val_{label}_{suffix}"] = value
    return aggregate


@torch.no_grad()
def _evaluate_online_indexer(
    config: COREConfig,
    model,
    tokenizer,
    collector: BackboneStatsCollector,
    feature_extractor: COREFeatureExtractor,
    teacher: DiversityAwareTeacher,
    indexer: COREIndexerStack,
    validation_samples: list,
    *,
    compression_ratio: float | None = None,
) -> dict:
    """Evaluate KL and deployment-set agreement on held-out samples.

    For ``coverage_local`` the hard deployment set is used only for exact
    context-set recall.  Boundary diagnostics use the same two continuous
    channels as training, so protected sinks/query tokens and hard Top-K
    offsets cannot make the metric trivial or contradictory.
    """
    if not validation_samples:
        return {}
    if compression_ratio is None:
        results = []
        for ratio in _validation_compression_ratios(config):
            metrics = _evaluate_online_indexer(
                config, model, tokenizer, collector, feature_extractor,
                teacher, indexer, validation_samples,
                compression_ratio=ratio,
            )
            results.append((ratio, metrics))
            if metrics:
                logger.info(
                    "Validation CR=%.2f: KL=%.4f Top-B=%.4f boundary=%.4f",
                    ratio, metrics["val_kl"], metrics["val_topB_recall"],
                    metrics["val_boundary_acc"],
                )
        return _aggregate_multibudget_validation(results)
    device = torch.device(config.backbone_device)
    was_training = indexer.training
    indexer.eval()
    kls: list[float] = []
    recalls: list[float] = []
    boundary_accs: list[float] = []
    boundary_margins: list[float] = []
    boundary_pair_accs: list[float] = []
    boundary_pair_gaps: list[float] = []
    coverage_recalls: list[float] = []
    local_recalls: list[float] = []
    coverage_accs: list[float] = []
    local_accs: list[float] = []
    coverage_margins: list[float] = []
    local_margins: list[float] = []
    indexer_objectives: list[float] = []

    def _boundary_stats(channel_scores, positive, negative):
        if positive.numel() == 0 or negative.numel() == 0:
            return None
        z_pos = channel_scores[positive] / config.student_temp
        z_neg = channel_scores[negative] / config.student_temp
        pair_gap = z_pos.unsqueeze(1) - z_neg.unsqueeze(0)
        margin = pair_gap.min()
        return (
            float(margin > 0), float(margin),
            float((pair_gap > 0).float().mean()), float(pair_gap.mean()),
        )

    for sample in validation_samples:
        score_cache: dict = {}
        tok = tokenize_sample(
            tokenizer, sample, config.max_seq_len, device,
            n_sink=config.n_sink, crop_seed=0,
        )
        if tok["input_ids"].shape[1] < config.n_sink + 2:
            continue
        batch = collect_online_batch(
            model, collector, feature_extractor, teacher, config,
            tok["input_ids"], compression_ratio=compression_ratio,
            include_decode_features=False,
            include_memory_supervision=False,
            query_start=tok.get("query_start"),
            selection_end=tok.get("answer_start"),
        )
        _align_stride_boundary_targets(batch, config, teacher)
        for layer_idx, data in batch.items():
            scores = _deployment_scores(indexer, batch, layer_idx, config, score_cache)
            selection_scores = _deployment_span_scores(scores, data, config)
            mask = data.get("loss_mask")
            if mask is None:
                mask = torch.ones_like(scores, dtype=torch.bool)
            else:
                mask = mask.to(device=scores.device, dtype=torch.bool)
            pi_teacher = _teacher_distribution_on_loss_axis(data["pi_T"], mask)
            pi_student = student_distribution(scores[mask], config.student_temp)
            kls.append(float(calibrated_kl_loss(pi_teacher, pi_student)))
            # Track the exact paper-defined validation objective used for
            # training (calibrated KL + pairwise boundary supervision). This
            # lets synchronized CORE-full checkpoint selection validate both
            # Stage-II objectives instead of selecting from indexer Top-B only.
            indexer_objective, _ = _core_loss_for_prefill(
                scores, data["pi_T"], data, config
            )
            indexer_objectives.append(float(indexer_objective))

            teacher_selection_scores = _deployment_teacher_selection_scores(
                data["pi_T"], data, config
            )
            hybrid = data.get("coverage_local_boundary")
            if hybrid is not None:
                # Exact deployment recall, restricted to the compressible
                # context. Sinks and protected query tokens are guaranteed by
                # policy and therefore must not inflate this measurement.
                total_budget = min(int(data["budget"]), scores.numel())
                teacher_all = teacher_selection_scores.topk(total_budget).indices
                student_all = selection_scores.topk(total_budget).indices
                context_mask = hybrid["context_mask"]
                teacher_keep = teacher_all[context_mask[teacher_all]]
                student_keep = student_all[context_mask[student_all]]
                recalls.append(float(
                    torch.isin(student_keep, teacher_keep).sum()
                    / max(1, teacher_keep.numel())
                ))

                teacher_logits = torch.log(data["pi_T"].float().clamp_min(1e-12))
                student_cov, student_local = _coverage_local_boundary_views(
                    scores, data, config
                )
                teacher_cov, teacher_local = _coverage_local_boundary_views(
                    teacher_logits, data, config
                )
                context_idx = hybrid["context_idx"]
                cov_budget = int(hybrid["coverage_budget"])
                local_budget = int(hybrid["local_blocks"])
                reps = hybrid["block_reps"]
                if cov_budget > 0:
                    if getattr(config, "span_selection_mode", "") == "coverage_local_balanced":
                        block_size = max(1, int(config.span_block_size))
                        t_cov = region_stratified_order(
                            teacher_cov[:int(data["query_start"])], context_idx,
                            block_size, block_scores=teacher_local[:int(data["query_start"])],
                        )[:cov_budget]
                        s_cov = region_stratified_order(
                            student_cov[:int(data["query_start"])], context_idx,
                            block_size, block_scores=student_local[:int(data["query_start"])],
                        )[:cov_budget]
                    else:
                        t_cov = context_idx[
                            teacher_cov[context_idx].topk(cov_budget).indices
                        ]
                        s_cov = context_idx[
                            student_cov[context_idx].topk(cov_budget).indices
                        ]
                    coverage_recalls.append(float(
                        torch.isin(s_cov, t_cov).float().mean()
                    ))
                if local_budget > 0 and reps.numel() > 0:
                    t_local = reps[teacher_local[reps].topk(local_budget).indices]
                    s_local = reps[student_local[reps].topk(local_budget).indices]
                    local_recalls.append(float(
                        torch.isin(s_local, t_local).float().mean()
                    ))

                channel_values = []
                for name, channel_scores in (
                    ("coverage", student_cov), ("local", student_local)
                ):
                    if name == "coverage" and hybrid.get(
                        "coverage_boundary_components"
                    ):
                        component_stats = []
                        for view, positive, negative in hybrid[
                            "coverage_boundary_components"
                        ]:
                            component_scores = (
                                student_cov if view == "coverage" else student_local
                            )
                            value = _boundary_stats(
                                component_scores, positive, negative
                            )
                            if value is not None:
                                component_stats.append(value)
                        if not component_stats:
                            continue
                        stats = tuple(
                            sum(value[i] for value in component_stats)
                            / len(component_stats)
                            for i in range(4)
                        )
                    else:
                        stats = _boundary_stats(
                            channel_scores,
                            hybrid[f"{name}_P_bd"], hybrid[f"{name}_N_bd"],
                        )
                        if stats is None:
                            continue
                    weight = float(hybrid[f"{name}_weight"])
                    channel_values.append((weight, stats))
                    if name == "coverage":
                        coverage_accs.append(stats[0])
                        coverage_margins.append(stats[1])
                    else:
                        local_accs.append(stats[0])
                        local_margins.append(stats[1])
                if channel_values:
                    weight_sum = sum(w for w, _ in channel_values)
                    combined = [
                        sum(w * stats[i] for w, stats in channel_values)
                        / max(weight_sum, 1e-12)
                        for i in range(4)
                    ]
                    boundary_accs.append(combined[0])
                    boundary_margins.append(combined[1])
                    boundary_pair_accs.append(combined[2])
                    boundary_pair_gaps.append(combined[3])
                continue

            valid_idx = mask.nonzero(as_tuple=False).squeeze(-1)
            n_sink = int((~mask).sum().item())
            non_sink_budget = max(1, min(int(data["budget"]) - n_sink, len(valid_idx)))
            block_spec = _block_selection_spec(data, config, mask)
            if block_spec is not None:
                reps, n_full = block_spec
                teacher_keep = reps[teacher_selection_scores[reps].topk(n_full).indices]
                student_keep = reps[selection_scores[reps].topk(n_full).indices]
            else:
                teacher_keep = valid_idx[
                    teacher_selection_scores[valid_idx].topk(non_sink_budget).indices
                ]
                student_keep = valid_idx[
                    selection_scores[valid_idx].topk(non_sink_budget).indices
                ]
            recalls.append(float(torch.isin(student_keep, teacher_keep).float().mean()))
            stats = _boundary_stats(selection_scores, data["P_bd"], data["N_bd"])
            if stats is not None:
                boundary_accs.append(stats[0])
                boundary_margins.append(stats[1])
                boundary_pair_accs.append(stats[2])
                boundary_pair_gaps.append(stats[3])

    if was_training:
        indexer.train()
    if not kls:
        return {}
    mean = lambda xs: sum(xs) / max(1, len(xs))
    result = {
        "val_kl": mean(kls),
        "val_topB_recall": mean(recalls),
        "val_boundary_acc": mean(boundary_accs),
        "val_boundary_margin": mean(boundary_margins),
        "val_boundary_pair_acc": mean(boundary_pair_accs),
        "val_boundary_pair_gap": mean(boundary_pair_gaps),
        "val_indexer_objective": mean(indexer_objectives),
        "val_samples": len(validation_samples),
        "val_layer_chunks": len(kls),
    }
    if coverage_recalls or local_recalls:
        result.update({
            "val_coverage_recall": mean(coverage_recalls),
            "val_local_recall": mean(local_recalls),
            "val_coverage_boundary_acc": mean(coverage_accs),
            "val_local_boundary_acc": mean(local_accs),
            "val_coverage_boundary_margin": mean(coverage_margins),
            "val_local_boundary_margin": mean(local_margins),
        })
    return result


def _is_better_validation(metrics: dict, best_metrics: dict | None) -> bool:
    """Prefer Top-B agreement; use lower KL as the deterministic tie-break."""
    if not metrics:
        return False
    if best_metrics is None:
        return True
    current = (metrics["val_topB_recall"], -metrics["val_kl"])
    previous = (best_metrics["val_topB_recall"], -best_metrics["val_kl"])
    return current > previous


@torch.no_grad()
def _evaluate_online_memory(
    config: COREConfig,
    model,
    tokenizer,
    collector: BackboneStatsCollector,
    feature_extractor: COREFeatureExtractor,
    teacher: DiversityAwareTeacher,
    indexer: COREIndexerStack,
    memory_stack: MemoryModuleStack,
    validation_samples: list,
    *,
    compression_ratio: float | None = None,
) -> dict:
    """Measure held-out memory recovery using the deployment eviction set.

    Indexer-only validation cannot distinguish memory checkpoints when the
    indexer is frozen.  This metric reconstructs each held-out sample with the
    exact deployment selector and reports residual error relative to the
    no-memory compression gap.  Lower ``val_memory_relative`` is better.
    """
    if not validation_samples:
        return {}
    if compression_ratio is None:
        results = []
        for ratio in _validation_compression_ratios(config):
            metrics = _evaluate_online_memory(
                config, model, tokenizer, collector, feature_extractor,
                teacher, indexer, memory_stack, validation_samples,
                compression_ratio=ratio,
            )
            results.append((ratio, metrics))
            if metrics:
                logger.info(
                    "Memory validation CR=%.2f: relative=%.4f "
                    "recovery=%.4f gate=%.4f",
                    ratio, metrics["val_memory_relative"],
                    metrics["val_memory_recovery"],
                    metrics["val_memory_gate"],
                )
        return _aggregate_multibudget_validation(results)
    device = torch.device(config.backbone_device)
    indexer_was_training = indexer.training
    memory_was_training = memory_stack.training
    indexer.eval()
    memory_stack.eval()
    raw_losses: list[float] = []
    gaps: list[float] = []
    ratios: list[float] = []
    gates: list[float] = []
    target_rms_values: list[float] = []
    residual_rms_values: list[float] = []
    residual_cosines: list[float] = []
    residual_optimal_scales: list[float] = []

    for sample in validation_samples:
        tok = tokenize_sample(
            tokenizer, sample, config.max_seq_len, device,
            include_answer=True, n_sink=config.n_sink, crop_seed=0,
        )
        input_ids = tok["input_ids"]
        if input_ids.shape[1] < config.n_sink + 2:
            continue
        batch = collect_online_batch(
            model, collector, feature_extractor, teacher, config,
            input_ids, compression_ratio=compression_ratio,
            include_decode_features=False,
            include_memory_supervision=True,
            query_start=tok.get("query_start"),
            selection_end=tok.get("answer_start"),
        )
        if not batch:
            continue
        score_cache: dict = {}
        for layer_idx, data in batch.items():
            o_full = data.get("o_full")
            raw_keys = data.get("memory_keys")
            raw_values = data.get("memory_values")
            raw_queries = data.get("memory_queries")
            o_proj = data.get("memory_o_proj")
            if (o_full is None or raw_keys is None or raw_values is None
                    or raw_queries is None or o_proj is None):
                continue
            scores = _deployment_scores(
                indexer, batch, layer_idx, config, score_cache
            )
            memory_scores = scores
            n_total = int(o_full.shape[0])
            if config.recency_alpha > 0.0 and scores.numel() > 1:
                positions = torch.arange(
                    scores.numel(), device=device, dtype=scores.dtype
                )
                memory_scores = scores + config.recency_alpha * torch.exp(
                    -(scores.numel() - 1 - positions)
                    / max(1, config.recency_window)
                )
            answer_tokens = max(0, n_total - int(memory_scores.shape[0]))
            if answer_tokens > 0:
                tail = max(1, min(
                    answer_tokens,
                    int(getattr(config, "memory_supervision_tail", 16)),
                ))
                partition_total = n_total
            else:
                tail = max(1, min(
                    n_total - 1,
                    int(getattr(config, "memory_supervision_tail", 16)),
                ))
                partition_total = None
            selection_scores = _deployment_span_scores(
                memory_scores, data, config
            )
            prefix_n, read_keep_idx, evict_mask = _causal_memory_partition(
                selection_scores, config, tail,
                total_length=partition_total,
                compression_ratio=compression_ratio,
            )
            o_comp = _reconstruct_compressed_output(
                raw_keys, raw_values, raw_queries, read_keep_idx, o_proj,
                int(data["num_kv_groups"]), query_tail=tail,
            )
            if o_comp is None:
                continue
            num_groups = int(data["num_kv_groups"])
            ev_raw_k = raw_keys[:, evict_mask, :].to(device)
            ev_raw_v = raw_values[:, evict_mask, :].to(device)
            if ev_raw_k.shape[1] == 0:
                continue
            ev_k = (
                repeat_kv(ev_raw_k.unsqueeze(0), num_groups).squeeze(0)
                .permute(1, 0, 2).reshape(ev_raw_k.shape[1], -1).float()
            )
            ev_v = (
                repeat_kv(ev_raw_v.unsqueeze(0), num_groups).squeeze(0)
                .permute(1, 0, 2).reshape(ev_raw_v.shape[1], -1).float()
            )
            prefix_evict_mask = evict_mask[:prefix_n]
            # Eq. 6/10: use the full-prefix allocation. Sink protection
            # is enforced by E; KL's eligible-axis renormalization is separate.
            p_theta = student_distribution(memory_scores[:prefix_n], config.student_temp)
            pi_e = (
                p_theta * prefix_evict_mask.to(dtype=p_theta.dtype)
            ).sum()
            write_weights = conditional_evicted_weights(
                p_theta, prefix_evict_mask,
                getattr(config, "memory_uniform_fraction", 0.0),
            )[prefix_evict_mask]
            proj_k = memory_stack.project(ev_k, layer_idx)
            m_bar = (proj_k * write_weights.unsqueeze(-1)).transpose(0, 1) @ ev_v
            b_bar = (
                (proj_k * proj_k) * write_weights.unsqueeze(-1)
            ).sum(dim=0)
            first_write_scale = pi_e / (pi_e + 1e-8)
            m_l = m_bar * config.memory_write_eta * first_write_scale
            b_l = b_bar * config.memory_write_eta * first_write_scale
            memory_queries = (
                raw_queries[:, -tail:, :].permute(1, 0, 2)
                .reshape(tail, -1).to(device).float()
            )
            residual, gate = memory_stack.readout(
                memory_queries, m_l, b_l, layer_idx,
                gate_override=None, return_gate=True,
            )
            target = o_full.to(device).float()[-tail:]
            compressed = o_comp.to(device).float()
            delta = target - compressed
            # Validate the exact paper loss. The relative ratio below is only
            # a scale-free checkpoint diagnostic; it is never optimized.
            raw = _paper_memory_reconstruction_loss(target, compressed, residual)
            gap = delta.pow(2).sum(dim=-1).mean()
            if not torch.isfinite(raw) or not torch.isfinite(gap):
                continue
            raw_value = float(raw)
            gap_value = float(gap)
            raw_losses.append(raw_value)
            gaps.append(gap_value)
            ratios.append(raw_value / max(gap_value, 1e-8))
            gates.append(float(gate.mean()))
            delta_flat = delta.reshape(-1).float()
            residual_flat = residual.reshape(-1).float()
            delta_norm = delta_flat.norm()
            residual_norm = residual_flat.norm()
            target_rms_values.append(float(delta.pow(2).mean().sqrt()))
            residual_rms_values.append(float(residual.pow(2).mean().sqrt()))
            residual_cosines.append(float(
                torch.dot(delta_flat, residual_flat)
                / (delta_norm * residual_norm).clamp_min(1e-12)
            ))
            residual_optimal_scales.append(float(
                torch.dot(delta_flat, residual_flat)
                / residual_flat.pow(2).sum().clamp_min(1e-12)
            ))

    if indexer_was_training:
        indexer.train()
    if memory_was_training:
        memory_stack.train()
    if not ratios:
        return {}
    mean = lambda values: sum(values) / len(values)
    relative = mean(ratios)
    return {
        "val_memory_relative": relative,
        "val_memory_recovery": 1.0 - relative,
        "val_memory_raw": mean(raw_losses),
        "val_memory_gap": mean(gaps),
        "val_memory_gate": mean(gates),
        "val_memory_target_rms": mean(target_rms_values),
        "val_memory_residual_rms": mean(residual_rms_values),
        "val_memory_residual_cosine": mean(residual_cosines),
        "val_memory_optimal_scale": mean(residual_optimal_scales),
        "val_memory_layer_chunks": len(ratios),
    }


def _is_better_memory_validation(
    metrics: dict, best_metrics: dict | None
) -> bool:
    """Select the checkpoint with the lowest held-out residual ratio."""
    current = metrics.get("val_memory_relative")
    if current is None or not math.isfinite(float(current)):
        return False
    if best_metrics is None:
        return True
    previous = best_metrics.get("val_memory_relative")
    return previous is None or float(current) < float(previous)


def _attach_joint_validation_objective(
    metrics: dict, config: COREConfig
) -> dict:
    """Attach the held-out objective used to select synchronized CORE-full.

    Stage II optimizes the paper indexer objective jointly with the configured
    memory objective. Checkpoint selection therefore validates both components
    on held-out LongAlpaca data; no benchmark labels are consumed.
    """
    idx = metrics.get("val_indexer_objective")
    mem = metrics.get("val_memory_relative")
    if idx is None or mem is None:
        return metrics
    joint = float(idx) + float(config.memory_train_lambda) * float(mem)
    if math.isfinite(joint):
        metrics["val_joint_objective"] = joint
    return metrics


def _is_better_joint_validation(
    metrics: dict, best_metrics: dict | None
) -> bool:
    """Select the lowest held-out Stage-II objective for CORE-full."""
    current = metrics.get("val_joint_objective")
    if current is None or not math.isfinite(float(current)):
        return False
    if best_metrics is None:
        return True
    previous = best_metrics.get("val_joint_objective")
    if previous is None:
        return True
    current_key = (
        -float(current),
        float(metrics.get("val_topB_recall", float("-inf"))),
    )
    previous_key = (
        -float(previous),
        float(best_metrics.get("val_topB_recall", float("-inf"))),
    )
    return current_key > previous_key


def train_indexer(
    config: COREConfig,
    collected: dict,
) -> COREIndexerMLP:
    """
    Train the CORE indexer MLP using collected features + teacher distributions.

    Parameters
    ----------
    config : COREConfig
    collected : dict
        Output of collect_training_data with keys:
          'X', 'pi_T', 'P_bd', 'N_bd', 'budgets'

    Returns
    -------
    COREIndexerMLP
        Trained indexer model.
    """
    _validate_training_objective(config)
    device = torch.device(config.train_device)
    n_chunks = len(collected["X"])
    total_tokens = sum(x.shape[0] for x in collected["X"])
    logger.info(
        f"Phase 2: training indexer on {n_chunks} chunks ({total_tokens} tokens)"
    )

    # Build the MLP
    input_dim = collected["X"][0].shape[-1]
    indexer = COREIndexerMLP(input_dim=input_dim, hidden_dim=config.indexer_hidden_dim)
    indexer.to(device)
    indexer.train()

    optimizer = AdamW(indexer.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    # Move all data to device once
    X_list = [x.to(device).float() for x in collected["X"]]
    pi_list = [p.to(device).float() for p in collected["pi_T"]]
    P_list = [p.to(device).long() for p in collected["P_bd"]]
    N_list = [n.to(device).long() for n in collected["N_bd"]]

    history = []

    for epoch in range(config.max_epochs):
        epoch_metrics = {"loss": 0.0, "loss_cal": 0.0, "loss_bd": 0.0, "n": 0}

        # Shuffle chunk order each epoch
        order = torch.randperm(n_chunks).tolist()

        pbar = tqdm(order, desc=f"Epoch {epoch + 1}/{config.max_epochs}")
        for chunk_idx in pbar:
            X = X_list[chunk_idx]  # (N, 14)
            pi_T = pi_list[chunk_idx]  # (N,)
            P_bd = P_list[chunk_idx]
            N_bd = N_list[chunk_idx]

            # Forward
            scores = indexer(X)  # (N,)

            # Loss
            loss, metrics = core_loss(scores, pi_T, P_bd, N_bd, config)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(indexer.parameters(), config.grad_clip)
            optimizer.step()

            # Accumulate metrics
            for k in ("loss", "loss_cal", "loss_bd"):
                epoch_metrics[k] += metrics[k]
            epoch_metrics["n"] += 1

            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                cal=f"{metrics['loss_cal']:.4f}",
                bd=f"{metrics['loss_bd']:.4f}",
                w_cal=f"{config.lambda_cal * metrics['loss_cal']:.4f}",
                w_bd=f"{config.lambda_bd * metrics['loss_bd']:.4f}",
            )

        scheduler.step()

        avg = {k: epoch_metrics[k] / max(1, epoch_metrics["n"]) for k in ("loss", "loss_cal", "loss_bd")}
        history.append(avg)
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1}: loss={avg['loss']:.4f} "
            f"(L_cal={avg['loss_cal']:.4f}×{config.lambda_cal}={config.lambda_cal * avg['loss_cal']:.4f}, "
            f"L_bd={avg['loss_bd']:.4f}×{config.lambda_bd}={config.lambda_bd * avg['loss_bd']:.4f}) "
            f"lr={cur_lr:.2e}"
        )

    indexer.eval()
    return indexer, history


def evaluate_indexer(
    config: COREConfig,
    indexer: COREIndexerMLP,
    collected: dict,
) -> dict:
    """
    Evaluate the trained indexer: report KL divergence and Top-B agreement
    with teacher ranking on held-out chunks.

    Metrics:
      - mean KL divergence per chunk
      - Top-B recall: fraction of teacher's top-B tokens correctly retained
      - Top-B precision: fraction of retained tokens that are in teacher's top-B

    Parameters
    ----------
    config : COREConfig
    indexer : COREIndexerMLP
    collected : dict

    Returns
    -------
    dict
        Evaluation metrics.
    """
    device = torch.device(config.train_device)
    indexer.to(device)
    indexer.eval()

    kls = []
    recalls = []
    precisions = []
    # Boundary discrimination: does the student separate P_bd (retain) from
    # N_bd (evict) in score space? "boundary_acc" = fraction of chunks where
    # min(score[P_bd]) > max(score[N_bd]) (i.e. the L_bd margin is satisfied
    # with m=0). "bd_margin" = mean of (min_P - max_N) over chunks.
    boundary_accs = []
    bd_margins = []

    with torch.no_grad():
        for i in range(len(collected["X"])):
            X = collected["X"][i].to(device).float()
            pi_T = collected["pi_T"][i].to(device).float()
            B = collected["budgets"][i]
            P_bd = collected["P_bd"][i].to(device).long()
            N_bd = collected["N_bd"][i].to(device).long()

            scores = indexer(X)
            pi_student = student_distribution(scores, config.student_temp)

            # KL
            kl = (pi_T * (torch.log(pi_T + 1e-8) - torch.log(pi_student + 1e-8))).sum().item()
            kls.append(kl)

            # Top-B agreement
            teacher_topB = pi_T.topk(B).indices
            student_topB = pi_student.topk(B).indices
            # Intersection
            teacher_set = set(teacher_topB.tolist())
            student_set = set(student_topB.tolist())
            inter = len(teacher_set & student_set)
            recalls.append(inter / max(1, len(teacher_set)))
            precisions.append(inter / max(1, len(student_set)))

            # Boundary discrimination (z = s / tau_m, same space as L_bd)
            if len(P_bd) > 0 and len(N_bd) > 0:
                z = scores / config.student_temp
                min_P = z[P_bd].min().item()
                max_N = z[N_bd].max().item()
                bd_margins.append(min_P - max_N)
                boundary_accs.append(1.0 if min_P > max_N else 0.0)

    import numpy as np

    metrics = {
        "mean_kl": float(np.mean(kls)),
        "mean_topB_recall": float(np.mean(recalls)),
        "mean_topB_precision": float(np.mean(precisions)),
        "mean_boundary_acc": float(np.mean(boundary_accs)) if boundary_accs else 0.0,
        "mean_bd_margin": float(np.mean(bd_margins)) if bd_margins else 0.0,
        "n_chunks": len(kls),
    }
    logger.info(
        f"Evaluation: mean KL={metrics['mean_kl']:.4f}, "
        f"Top-B recall={metrics['mean_topB_recall']:.4f}, "
        f"Top-B precision={metrics['mean_topB_precision']:.4f}, "
        f"boundary_acc={metrics['mean_boundary_acc']:.4f}, "
        f"bd_margin={metrics['mean_bd_margin']:.4f}"
    )
    return metrics


def train_memory_module(
    config: COREConfig,
    collected: dict,
    indexer: COREIndexerMLP,
) -> tuple[MemoryModule, list]:
    """
    Legacy offline memory-only training, separate from joint Stage II.

    Freezes the indexer (Phase 2 already trained it). For each chunk, simulates
    the inference-time memory write: the evicted tokens (complement of the
    teacher's Top-B) update the fast weights M, b in a no_grad block (online,
    no gradient), then the readout g(q)·m(q) is computed with gradient and the
    MSE loss ``L_mem = ||o_full - o_compressed - g(q)·m(q)||²`` (IndexMem eq.8)
    backprops only into Linear_θ and g.

    Only chunks with valid supervision data (o_full, o_compressed, query_hidden)
    are used; the rest are skipped (they were collected before this feature or
    the hook failed to capture output).

    Parameters
    ----------
    config : COREConfig
    collected : dict
        Must contain 'o_full', 'o_compressed', 'query_hidden' lists (from Phase 1).
    indexer : COREIndexerMLP
        Already-trained indexer (frozen during this phase).

    Returns
    -------
    tuple[MemoryModule, list]
        Trained memory module and per-epoch loss history.
    """
    _validate_training_objective(config)
    device = torch.device(config.train_device)
    d_model = (config.n_q_heads * config.head_dim
               if config.n_q_heads and config.head_dim else config.hidden_size or 4096)
    d_mem = config.memory_dim or max(1, d_model // 8)
    memory = MemoryModule(
        d_model,
        d_mem,
        config.memory_gate_hidden,
        projection_groups=getattr(config, "memory_projection_groups", 1),
    ).to(device)
    memory.train()

    # Freeze indexer (Phase 2 already trained it).
    for p in indexer.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(memory.parameters(), lr=config.memory_train_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.memory_train_epochs)

    # Filter to chunks with valid supervision data.
    valid = [
        i for i in range(len(collected["o_full"]))
        if collected["o_full"][i] is not None
        and collected["o_compressed"][i] is not None
        and collected["query_hidden"][i] is not None
    ]
    if not valid:
        logger.warning("No valid memory supervision data — skipping Phase 2.5 (memory untrained).")
        memory.eval()
        return memory, []

    logger.info(
        f"Phase 2.5: training memory module on {len(valid)} chunks "
        f"(d_model={d_model}, d_mem={d_mem})"
    )

    history = []
    for epoch in range(config.memory_train_epochs):
        epoch_loss = 0.0
        epoch_gap = 0.0
        n = 0
        order = valid.copy()
        # Shuffle (numpy for reproducibility with seed)
        import numpy as _np
        _np.random.RandomState(config.seed + epoch).shuffle(order)

        pbar = tqdm(order, desc=f"Memory Epoch {epoch+1}/{config.memory_train_epochs}")
        for chunk_idx in pbar:
            o_full = collected["o_full"][chunk_idx].to(device).float()        # (L, d_model)
            o_comp = collected["o_compressed"][chunk_idx].to(device).float()  # (L, d_model)
            q_hidden = collected["query_hidden"][chunk_idx].to(device).float()  # (L, d_model)
            keep_idx = collected["keep_indices"][chunk_idx].to(device).long()   # (B,)
            B = keep_idx.shape[0]
            N = o_full.shape[0]

            # --- simulate inference-time fast-weight write (no grad) ---
            # The evicted tokens are the complement of keep_idx. We need their
            # hidden states (query_hidden) to compute Linear_θ(k_i) and their
            # "values" — we use query_hidden (d_model) as the value proxy too,
            # matching the inference path (COREMemoryPress uses hidden_states).
            with torch.no_grad():
                evict_mask = torch.ones(N, dtype=torch.bool, device=device)
                evict_mask[keep_idx] = False
                ev_hs = q_hidden[evict_mask]  # (n_evict, d_model)
                if ev_hs.shape[0] == 0:
                    continue
                proj_k = memory.project(ev_hs)  # (n_evict, d_mem)
                # write: M = Σ proj_k_i ⊗ v_i, b = Σ proj_k_i ⊙ proj_k_i
                # use uniform weights (η applied via write_eta, but here we just
                # build the per-chunk M_bar, b_bar; decay handled by initing M,b=0)
                M_bar = proj_k.T @ ev_hs       # (d_mem, d_model)
                b_bar = (proj_k * proj_k).sum(dim=0)  # (d_mem,)
                M = M_bar * config.memory_write_eta
                b = b_bar * config.memory_write_eta

            # --- readout with gradient (trains Linear_θ + g) ---
            residual = memory.readout(q_hidden, M, b)  # (L, d_model)

            # CORE paper objective: ordinary elementwise reconstruction MSE.
            loss = config.memory_train_lambda * _paper_memory_reconstruction_loss(
                o_full, o_comp, residual
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(memory.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            # Also track the no-residual gap ||o_full - o_compressed||^2 to
            # confirm the supervision signal is real (not ~0, which would mean
            # reconstruction is broken and memory has nothing to learn).
            with torch.no_grad():
                gap_mse = ((o_full - o_comp) ** 2).sum(dim=-1).mean().item()
            epoch_gap += gap_mse
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", gap=f"{gap_mse:.4f}")

        scheduler.step()
        avg = epoch_loss / max(1, n)
        avg_gap = epoch_gap / max(1, n)
        history.append(avg)
        logger.info(
            f"Memory Epoch {epoch+1}: avg L_mem={avg:.6f} "
            f"(no-residual gap ||o-o_attn||^2={avg_gap:.6f})"
        )

    memory.eval()
    return memory, history


def save_indexer(
    indexer: COREIndexerMLP,
    config: COREConfig,
    history: list,
    metrics: dict,
    feature_extractor: COREFeatureExtractor | None = None,
    memory_module: MemoryModule | None = None,
    memory_history: list | None = None,
):
    """Save the trained indexer, feature extractor, and (optionally) memory module."""
    output_path = Path(config.output_dir) / "indexer"
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": indexer.state_dict(),
        "input_dim": indexer.net[0].in_features,
        "hidden_dim": indexer.net[0].out_features,
        "config": config.__dict__,
        "history": history,
        "metrics": metrics,
    }

    if feature_extractor is not None:
        checkpoint["feature_extractor_state_dict"] = feature_extractor.geometry_state_dict()

    if memory_module is not None:
        checkpoint["memory_state_dict"] = memory_module.state_dict()
        checkpoint["memory_history"] = memory_history or []
        checkpoint["memory_dims"] = {
            "d_model": memory_module.d_model,
            "d_mem": memory_module.d_mem,
            "projection_groups": getattr(
                memory_module, "projection_groups", 1
            ),
        }

    torch.save(checkpoint, output_path / "core_indexer.pt")
    logger.info(f"Saved indexer to {output_path / 'core_indexer.pt'}")


def load_indexer(path: str, device: str = "cuda:0") -> tuple[COREIndexerMLP, dict]:
    """Load a trained CORE indexer."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    indexer = COREIndexerMLP(
        input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"]
    )
    indexer.load_state_dict(ckpt["state_dict"])
    indexer.to(device)
    indexer.eval()
    return indexer, ckpt


# ---------------------------------------------------------------------------
# Online distillation training (per-layer indexer stack)
# ---------------------------------------------------------------------------


def _wsd_scheduler(optimizer, warmup: int, stable: int, decay: int,
                   peak_lr: float, final_lr: float, step_offset: int = 0):
    """Configured Warmup-Stable-Decay learning-rate schedule.

    warmup steps: linear ramp 0 -> peak_lr
    stable steps: constant peak_lr
    decay steps: cosine decay peak_lr -> final_lr
    """
    total = max(1, warmup + stable + decay)

    def lr_lambda(step):
        step = step + step_offset
        if step < warmup:
            return step / max(1, warmup)
        if step < warmup + stable:
            return 1.0
        # cosine decay to final_lr/peak_lr ratio
        # A resumed run may intentionally continue beyond the nominal WSD
        # horizon.  Clamp the decay progress so the learning rate stays at
        # ``final_lr`` instead of following the cosine into a new rising
        # half-cycle after ``warmup + stable + decay``.
        d = (step - warmup - stable) / max(1, decay)
        d = min(1.0, max(0.0, d))
        ratio = final_lr / peak_lr
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + torch.cos(torch.tensor(d * 3.14159265)).item())

    return LambdaLR(optimizer, lr_lambda)


def train_indexer_online(
    config: COREConfig,
    model,
    tokenizer,
    feature_extractor: COREFeatureExtractor,
    teacher: DiversityAwareTeacher,
    indexer: COREIndexerStack | None = None,
    max_steps: int | None = None,
    schedule_offset: int = 0,
) -> tuple[COREIndexerStack, list]:
    """Train a per-layer indexer stack via online distillation.

    Each step:
      1. Sample a LongAlpaca sequence and tokenize.
      2. Run a frozen backbone forward (hooks capture per-layer attn/K/V/Q).
      3. For every layer, compute its own teacher (DRIVE) + 14-dim features.
      4. Score with that layer's MLP and compute L_CORE (KL + boundary).
      5. Backprop through the indexer stack only (backbone stays frozen).

    Aligns with IndexMem's streaming KL training (no on-disk pre-collection).
    """
    _validate_training_objective(config)
    from core.data import load_longalpaca, tokenize_sample

    device = torch.device(config.backbone_device)
    dtype = getattr(torch, config.dtype, torch.bfloat16)

    # Build the per-layer indexer stack.
    n_layers = config.n_layers
    if n_layers is None:
        # Infer from the model if not set.
        language_model = (
            model.model.language_model if hasattr(model.model, "language_model") else model.model
        )
        n_layers = len(language_model.layers)
    if indexer is None:
        indexer = COREIndexerStack(
            n_layers=n_layers,
            input_dim=14,
            hidden_dim=config.indexer_hidden_dim,
            compact_stride=(
                config.scoring_stride
                if getattr(config, "compact_indexer_stack", False)
                else 1
            ),
        ).to(device)
    else:
        indexer = indexer.to(device)
    feature_extractor = feature_extractor.to(device)

    # Backbone frozen; only the indexer has trainable params.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    indexer.train()
    indexer.to(device)

    collector = BackboneStatsCollector(model, capture_all_layers=True)

    # Optimizer + WSD schedule.
    optimizer = AdamW(indexer.parameters(), lr=config.online_lr_peak)
    scheduler = _wsd_scheduler(
        optimizer,
        warmup=config.online_warmup,
        stable=config.online_stable,
        decay=config.online_decay,
        peak_lr=config.online_lr_peak,
        final_lr=config.online_lr_final,
        step_offset=schedule_offset,
    )

    # Data iterator (cycles through samples).
    samples = load_longalpaca(config, tokenizer)
    if not samples:
        raise RuntimeError("No training samples loaded.")
    samples, validation_samples = _split_online_samples(
        samples, getattr(config, "online_val_samples", 4)
    )
    validation_samples = _augment_validation_samples(config, validation_samples)
    if not samples:
        raise RuntimeError("No training samples remain after validation split.")
    micro_batch_size = max(1, int(getattr(config, "online_batch_size", 1)))
    sample_offset = (schedule_offset * micro_batch_size) % len(samples)
    sample_iter = iter(samples[sample_offset:] + samples[:sample_offset])
    history = []
    best_metrics = None
    early_best_topb = float("-inf")
    early_bad_validations = 0
    stop_stage1 = False
    best_path = (
        Path(config.output_dir) / "indexer" / "core_indexer_best_stage1.pt"
    )
    running = {"loss": 0.0, "loss_cal": 0.0, "loss_bd": 0.0, "n": 0}

    stage_steps = config.online_max_steps if max_steps is None else max_steps
    pbar = tqdm(total=stage_steps, desc="Indexer pretrain")
    for micro_step in range(stage_steps * micro_batch_size):
        step = micro_step // micro_batch_size
        micro_index = micro_step % micro_batch_size
        global_step = schedule_offset + step + 1
        # Cycle the data iterator.
        try:
            sample = next(sample_iter)
        except StopIteration:
            sample_iter = iter(samples)
            sample = next(sample_iter)

        training_sample = _training_sample_for_step(config, sample, global_step)
        tok = tokenize_sample(
            tokenizer, training_sample, _online_training_length(config, global_step),
            device, n_sink=config.n_sink, crop_seed=global_step,
        )
        input_ids = tok["input_ids"]
        if input_ids.shape[1] < config.n_sink + 2:
            continue

        # Online collection: forward + per-layer teacher + features.
        current_compression_ratio = _training_compression_ratio(
            config, global_step
        )
        batch = collect_online_batch(
            model, collector, feature_extractor, teacher, config,
            input_ids, compression_ratio=current_compression_ratio,
            include_decode_features=(
                config.decode_train_every > 0
                and global_step % config.decode_train_every == 0
            ),
            # Stage 1 optimizes only the selector. Building 32 layers of
            # full-size memory K/V supervision here wastes several GiB and is
            # never consumed by the loss below.
            include_memory_supervision=False,
            query_start=tok.get("query_start"),
            selection_end=tok.get("answer_start"),
        )
        if not batch:
            continue
        _align_stride_boundary_targets(batch, config, teacher)

        # Average the configured L_CORE objective uniformly over transformer
        # layers. No layer-dependent reweighting is applied.
        if micro_index == 0:
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = 0.0
            step_cal_sum = 0.0
            step_bd_sum = 0.0
        total_loss = torch.zeros((), device=device)
        total_weight = 0.0
        total_metric_cal = 0.0
        total_metric_bd = 0.0
        score_cache: dict = {}
        for layer_idx, data in batch.items():
            pi_T = data["pi_T"]
            scores = _deployment_scores(
                indexer, batch, layer_idx, config, score_cache
            )
            loss, m = _core_loss_for_prefill(
                scores, pi_T, data, config
            )
            weight = scores.new_tensor(1.0)
            total_loss = total_loss + weight * loss
            total_weight += float(weight)
            total_metric_cal += float(weight) * m["loss_cal"]
            total_metric_bd += float(weight) * m["loss_bd"]
            if "decode_X" in data:
                decode_pi = data["decode_pi_T"]
                decode_scores = _deployment_scores(
                    indexer, batch, layer_idx, config, score_cache, "decode_X"
                )
                decode_loss, m_decode = core_loss(
                    decode_scores,
                    decode_pi,
                    data["decode_P_bd"],
                    data["decode_N_bd"],
                    config,
                    loss_mask=data.get("decode_loss_mask"),
                )
                decode_weight = scores.new_tensor(config.decode_loss_weight)
                total_loss = total_loss + decode_weight * decode_loss
                total_weight += float(decode_weight)
                total_metric_cal += (
                    float(decode_weight) * m_decode["loss_cal"]
                )
                total_metric_bd += (
                    float(decode_weight) * m_decode["loss_bd"]
                )
        total_loss = total_loss / max(1.0, total_weight)
        mean_metric_cal = total_metric_cal / max(1.0, total_weight)
        mean_metric_bd = total_metric_bd / max(1.0, total_weight)

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite indexer loss at global step {global_step}; "
                "optimizer update was not applied"
            )

        step_loss_sum += float(total_loss.detach())
        step_cal_sum += mean_metric_cal
        step_bd_sum += mean_metric_bd
        (total_loss / micro_batch_size).backward()
        if micro_index + 1 < micro_batch_size:
            continue
        torch.nn.utils.clip_grad_norm_(
            indexer.parameters(), config.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        pbar.update(1)

        # Log the uniform per-layer aggregate used by the objective.
        running["loss"] += step_loss_sum / micro_batch_size
        running["loss_cal"] += step_cal_sum / micro_batch_size
        running["loss_bd"] += step_bd_sum / micro_batch_size
        running["n"] += 1
        if global_step % config.online_log_every == 0 and running["n"] > 0:
            n = running["n"]
            avg = {k: running[k] / n for k in ("loss", "loss_cal", "loss_bd")}
            lr = optimizer.param_groups[0]["lr"]
            entry = {"step": global_step, "stage": "indexer_pretrain", **avg,
                     "lr": lr, "n_layers": len(batch)}
            history.append(entry)
            pbar.set_postfix(
                loss=f"{avg['loss']:.4f}", cal=f"{avg['loss_cal']:.4f}",
                bd=f"{avg['loss_bd']:.4f}", lr=f"{lr:.2e}",
            )
            running = {"loss": 0.0, "loss_cal": 0.0, "loss_bd": 0.0, "n": 0}

        eval_every = max(1, int(config.online_eval_every))
        if validation_samples and (
            global_step % eval_every == 0 or step + 1 == stage_steps
        ):
            metrics = _evaluate_online_indexer(
                config, model, tokenizer, collector, feature_extractor,
                teacher, indexer, validation_samples,
            )
            if metrics:
                history.append({
                    "step": global_step, "stage": "validation", **metrics
                })
                logger.info(
                    "Validation step %d: KL=%.4f Top-B=%.4f boundary=%.4f",
                    global_step, metrics["val_kl"],
                    metrics["val_topB_recall"], metrics["val_boundary_acc"],
                )
                logger.info(
                    "  Boundary pairs: accuracy=%.4f mean_gap=%.4f strict_gap=%.4f",
                    metrics["val_boundary_pair_acc"],
                    metrics["val_boundary_pair_gap"],
                    metrics["val_boundary_margin"],
                )
                if "val_coverage_recall" in metrics:
                    logger.info(
                        "  Hybrid channels: coverage recall=%.4f boundary=%.4f "
                        "margin=%.4f | local recall=%.4f boundary=%.4f margin=%.4f",
                        metrics["val_coverage_recall"],
                        metrics["val_coverage_boundary_acc"],
                        metrics["val_coverage_boundary_margin"],
                        metrics["val_local_recall"],
                        metrics["val_local_boundary_acc"],
                        metrics["val_local_boundary_margin"],
                    )
                if _is_better_validation(metrics, best_metrics):
                    best_metrics = metrics
                    save_indexer_stack(
                        indexer, config, history, metrics=metrics,
                        feature_extractor=feature_extractor,
                        checkpoint_name=best_path.name,
                    )
                patience = int(getattr(config, "stage1_early_stop_patience", 0))
                min_steps = int(getattr(config, "stage1_early_stop_min_steps", 0))
                min_delta = float(getattr(config, "stage1_early_stop_min_delta", 0.0))
                topb = float(metrics["val_topB_recall"])
                if topb > early_best_topb + min_delta:
                    early_best_topb = topb
                    early_bad_validations = 0
                else:
                    early_bad_validations += 1
                if (
                    patience > 0
                    and global_step >= min_steps
                    and early_bad_validations >= patience
                ):
                    logger.info(
                        "Stage-I early stop at step %d: held-out Top-B did not "
                        "improve by %.6f for %d validations",
                        global_step, min_delta, patience,
                    )
                    stop_stage1 = True
        if stop_stage1:
            break

    if best_metrics is not None and best_path.exists():
        best_checkpoint = torch.load(
            best_path, map_location=device, weights_only=False
        )
        indexer.load_state_dict(best_checkpoint["state_dict"])
        logger.info(
            "Restored best stage-1 checkpoint: Top-B=%.4f KL=%.4f",
            best_metrics["val_topB_recall"], best_metrics["val_kl"],
        )
    indexer.eval()
    return indexer, history



def _restore_synchronized_joint_checkpoint(
    indexer: nn.Module,
    memory_stack: nn.Module,
    checkpoint: dict,
    checkpoint_path: str | Path,
) -> int:
    """Restore an indexer + memory pair from the same joint checkpoint.

    CORE's memory write distribution depends on the current indexer scores and
    Top-B eviction set.  Mixing an indexer from step ``t`` with memory slow
    weights from another step breaks that joint training/deployment contract.

    Returns
    -------
    int
        Last global step recorded in the restored checkpoint history.
    """
    checkpoint_path = Path(checkpoint_path)

    if "state_dict" not in checkpoint:
        raise RuntimeError(
            f"Joint checkpoint {checkpoint_path} has no indexer state_dict."
        )
    if "memory_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Joint checkpoint {checkpoint_path} has no memory_state_dict. "
            "A joint CORE checkpoint must contain synchronized indexer and "
            "memory weights."
        )

    indexer.load_state_dict(checkpoint["state_dict"])
    memory_stack.load_state_dict(checkpoint["memory_state_dict"])

    history = checkpoint.get("history", [])
    return max(
        (
            int(item.get("step", 0))
            for item in history
            if isinstance(item, dict)
        ),
        default=0,
    )


def train_joint_online(
    config: COREConfig,
    model,
    tokenizer,
    feature_extractor: COREFeatureExtractor,
    teacher: DiversityAwareTeacher,
    indexer: COREIndexerStack | None = None,
    memory_stack: MemoryModuleStack | None = None,
    max_steps: int | None = None,
    schedule_offset: int = 0,
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
) -> tuple[COREIndexerStack, MemoryModuleStack, list, list]:
    """Joint training of the indexer and per-layer memory module.

    Both module families use the same optimizer and configured WSD schedule.
    Each training example reuses frozen-backbone statistics for the indexer
    targets and memory reconstruction targets.

    Loss:
        total = mean_l L_CORE,l + λ_mem · mean_l L_mem,l
        where L_mem,l is the squared reconstruction loss in CORE Eq. 13.

    Top-B membership is non-differentiable. By default, scores used for memory
    weights are detached, so memory loss trains the shared projection and gate.
    ``memory_backprop_to_indexer`` optionally also differentiates through the
    soft write weights into the indexer.

    Intermediate saves: every ``config.online_save_every`` steps we persist
    both stacks via :func:`save_joint_stack`, so a mid-run crash loses at most
    one save interval rather than the whole run.

    Parameters
    ----------
    config, model, tokenizer, feature_extractor, teacher
        Same as :func:`train_indexer_online`.
    indexer, memory_stack : optional
        Pre-built / pre-loaded stacks to resume from (e.g. loaded by
        ``--skip_to_joint``). If ``None``, fresh stacks are constructed.

    Returns
    -------
    (COREIndexerStack, MemoryModuleStack, list, list)
        Trained indexer stack, trained per-layer memory stack, indexer history
        (per-log-window averages), memory history.
    """
    _validate_training_objective(config)
    import itertools
    from core.data import load_longalpaca, tokenize_sample

    device = torch.device(config.backbone_device)

    n_layers = config.n_layers
    if n_layers is None:
        language_model = (
            model.model.language_model if hasattr(model.model, "language_model") else model.model
        )
        n_layers = len(language_model.layers)
    if indexer is None:
        indexer = COREIndexerStack(
            n_layers=n_layers,
            input_dim=14,
            hidden_dim=config.indexer_hidden_dim,
            compact_stride=(
                config.scoring_stride
                if getattr(config, "compact_indexer_stack", False)
                else 1
            ),
        ).to(device)
    else:
        indexer = indexer.to(device)
    freeze_indexer = bool(
        getattr(config, "freeze_indexer_during_joint", False)
    )
    if freeze_indexer:
        indexer.eval()
        for parameter in indexer.parameters():
            parameter.requires_grad_(False)
        logger.info(
            "Frozen-indexer joint stage: optimizing memory parameters only"
        )
    else:
        indexer.train()

    d_model = (config.n_q_heads * config.head_dim
               if config.n_q_heads and config.head_dim else config.hidden_size or 4096)
    d_mem = config.memory_dim or max(1, d_model // 8)
    if memory_stack is None:
        memory_stack = MemoryModuleStack(
            n_layers=n_layers,
            d_model=d_model,
            d_mem=d_mem,
            gate_hidden=config.memory_gate_hidden,
            share_across_layers=getattr(config, "memory_share_across_layers", False),
            projection_groups=getattr(config, "memory_projection_groups", 1),
        ).to(device)
    else:
        memory_stack = memory_stack.to(device)
    memory_stack.train()

    feature_extractor = feature_extractor.to(device)

    # Backbone frozen; only indexer + memory stacks have trainable params.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    collector = BackboneStatsCollector(model, capture_all_layers=True)

    indexer_lr_scale = float(
        getattr(config, "joint_indexer_lr_scale", 1.0)
    )
    memory_lr_scale = float(
        getattr(config, "joint_memory_lr_scale", 1.0)
    )
    if indexer_lr_scale <= 0.0 or memory_lr_scale <= 0.0:
        raise ValueError("joint optimizer LR scales must be positive")
    optimizer_parameters = (
        list(memory_stack.parameters())
        if freeze_indexer
        else list(itertools.chain(indexer.parameters(), memory_stack.parameters()))
    )
    if freeze_indexer:
        optimizer_groups = [{
            "params": list(memory_stack.parameters()),
            "lr": config.online_lr_peak * memory_lr_scale,
            "name": "memory",
        }]
    else:
        optimizer_groups = [
            {
                "params": list(indexer.parameters()),
                "lr": config.online_lr_peak * indexer_lr_scale,
                "name": "indexer",
            },
            {
                "params": list(memory_stack.parameters()),
                "lr": config.online_lr_peak * memory_lr_scale,
                "name": "memory",
            },
        ]
    optimizer = AdamW(optimizer_groups, lr=config.online_lr_peak)
    logger.info(
        "Joint optimizer LR scales: indexer=%.3f memory=%.3f",
        indexer_lr_scale, memory_lr_scale,
    )
    scheduler = _wsd_scheduler(
        optimizer,
        warmup=config.online_warmup,
        stable=config.online_stable,
        decay=config.online_decay,
        peak_lr=config.online_lr_peak,
        final_lr=config.online_lr_final,
        step_offset=schedule_offset,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        logger.info("Restored joint AdamW state at global step %d", schedule_offset)
    elif schedule_offset > int(getattr(config, "indexer_pretrain_steps", 0)):
        logger.warning(
            "Joint checkpoint has no optimizer state; resuming weights at step "
            "%d with fresh AdamW moments", schedule_offset,
        )
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    samples = load_longalpaca(config, tokenizer)
    if not samples:
        raise RuntimeError("No training samples loaded for joint training.")
    samples, validation_samples = _split_online_samples(
        samples, getattr(config, "online_val_samples", 4)
    )
    validation_samples = _augment_validation_samples(config, validation_samples)
    if not samples:
        raise RuntimeError("No training samples remain after validation split.")
    micro_batch_size = max(1, int(getattr(config, "online_batch_size", 1)))
    sample_offset = (schedule_offset * micro_batch_size) % len(samples)
    sample_iter = iter(samples[sample_offset:] + samples[:sample_offset])
    history: list = []
    memory_history: list = []
    running = {
        "loss": 0.0, "loss_cal": 0.0, "loss_bd": 0.0,
        "loss_mem": 0.0, "mem_objective": 0.0, "gap": 0.0,
        "gate": 0.0, "total_loss": 0.0, "n": 0,
    }
    best_joint_metrics = None
    best_indexer_metrics = None
    last_validation_metrics = None
    stage2_initial_topb = None
    early_best_joint_objective = float("inf")
    early_bad_joint_validations = 0
    stop_stage2 = False
    best_joint_path = (
        Path(config.output_dir) / "indexer" / "core_full_best_joint.pt"
    )
    best_indexer_path = (
        Path(config.output_dir) / "indexer" / "core_indexer_best_joint.pt"
    )

    # Compare every future synchronized checkpoint against the state at the
    # start of Stage II. This prevents a run in which memory only degrades from
    # being forced to select a post-update checkpoint.
    if validation_samples:
        initial_metrics = _evaluate_online_indexer(
            config, model, tokenizer, collector, feature_extractor,
            teacher, indexer, validation_samples,
        )
        initial_metrics.update(_evaluate_online_memory(
            config, model, tokenizer, collector, feature_extractor,
            teacher, indexer, memory_stack, validation_samples,
        ))
        _attach_joint_validation_objective(initial_metrics, config)
        if "val_memory_relative" in initial_metrics:
            history.append({
                "step": schedule_offset,
                "stage": "validation_initial",
                **initial_metrics,
            })
            best_joint_metrics = dict(initial_metrics)
            early_best_joint_objective = float(
                initial_metrics["val_joint_objective"]
            )
            best_indexer_metrics = dict(initial_metrics)
            last_validation_metrics = initial_metrics
            stage2_initial_topb = float(initial_metrics["val_topB_recall"])
            save_joint_stack(
                indexer, memory_stack, config,
                history=history, memory_history=memory_history,
                feature_extractor=feature_extractor,
                metrics=initial_metrics, checkpoint_name=best_joint_path.name,
            )
            save_indexer_stack(
                indexer, config, history,
                metrics=initial_metrics,
                feature_extractor=feature_extractor,
                checkpoint_name=best_indexer_path.name,
            )
            logger.info(
                "Initial held-out joint state at step %d: objective=%.4f "
                "Top-B=%.4f memory_relative=%.4f recovery=%.4f gate=%.4f",
                schedule_offset,
                initial_metrics["val_joint_objective"],
                initial_metrics["val_topB_recall"],
                initial_metrics["val_memory_relative"],
                initial_metrics["val_memory_recovery"],
                initial_metrics["val_memory_gate"],
            )

    stage_steps = config.online_max_steps if max_steps is None else max_steps
    pbar = tqdm(total=stage_steps, desc="Joint online")
    for micro_step in range(stage_steps * micro_batch_size):
        step = micro_step // micro_batch_size
        micro_index = micro_step % micro_batch_size
        global_step = schedule_offset + step + 1
        try:
            sample = next(sample_iter)
        except StopIteration:
            sample_iter = iter(samples)
            sample = next(sample_iter)

        training_sample = _training_sample_for_step(config, sample, global_step)
        tok = tokenize_sample(
            tokenizer, training_sample, _online_training_length(config, global_step),
            device, include_answer=True, n_sink=config.n_sink,
            crop_seed=global_step,
        )
        input_ids = tok["input_ids"]
        if input_ids.shape[1] < config.n_sink + 2:
            continue

        # Single forward pass produces BOTH indexer (X/pi_T/P_bd/N_bd) and
        # memory (o_full/o_compressed/query_hidden/keep_indices) signals per
        # layer (collect_online_batch now surfaces memory supervision for every
        # captured layer, not just the middle one).
        current_compression_ratio = _training_compression_ratio(
            config, global_step
        )
        batch = collect_online_batch(
            model, collector, feature_extractor, teacher, config,
            input_ids, compression_ratio=current_compression_ratio,
            include_decode_features=(
                config.decode_train_every > 0
                and global_step % config.decode_train_every == 0
            ),
            query_start=tok.get("query_start"),
            selection_end=tok.get("answer_start"),
        )
        if not batch:
            continue
        _align_stride_boundary_targets(batch, config, teacher)

        if micro_index == 0:
            optimizer.zero_grad(set_to_none=True)
            step_metrics = {
                "loss": 0.0, "loss_cal": 0.0, "loss_bd": 0.0,
                "loss_mem": 0.0, "mem_objective": 0.0, "gap": 0.0,
                "gate": 0.0, "total_loss": 0.0,
            }

        # ---- indexer loss (uniform per-layer mean) ----
        total_loss_core = torch.zeros((), device=device)
        total_weight = 0.0
        total_metric_cal = 0.0
        total_metric_bd = 0.0
        # Accumulate per-layer memory loss in fp32 on device.
        total_loss_mem = torch.zeros((), device=device, dtype=torch.float32)
        total_raw_loss_mem = torch.zeros((), device=device, dtype=torch.float32)
        total_gap_mem = torch.zeros((), device=device, dtype=torch.float32)
        total_gate_value = 0.0
        n_mem_layers = 0
        score_cache: dict = {}

        for layer_idx, data in batch.items():
            pi_T = data["pi_T"]
            scores = _deployment_scores(
                indexer, batch, layer_idx, config, score_cache
            )
            loss_core, m_core = _core_loss_for_prefill(
                scores, pi_T, data, config
            )
            weight = scores.new_tensor(1.0)
            total_loss_core = total_loss_core + weight * loss_core
            total_weight += float(weight)
            total_metric_cal += float(weight) * m_core["loss_cal"]
            total_metric_bd += float(weight) * m_core["loss_bd"]
            if "decode_X" in data:
                decode_pi = data["decode_pi_T"]
                decode_scores = _deployment_scores(
                    indexer, batch, layer_idx, config, score_cache, "decode_X"
                )
                decode_loss, m_decode = core_loss(
                    decode_scores,
                    decode_pi,
                    data["decode_P_bd"],
                    data["decode_N_bd"],
                    config,
                    loss_mask=data.get("decode_loss_mask"),
                )
                decode_weight = scores.new_tensor(config.decode_loss_weight)
                total_loss_core = total_loss_core + decode_weight * decode_loss
                total_weight += float(decode_weight)
                total_metric_cal += float(decode_weight) * m_decode["loss_cal"]
                total_metric_bd += float(decode_weight) * m_decode["loss_bd"]

            # ---- memory loss (L_mem,l) ----
            o_full = data.get("o_full")
            raw_keys = data.get("memory_keys")
            raw_values = data.get("memory_values")
            raw_queries = data.get("memory_queries")
            o_proj = data.get("memory_o_proj")
            if (o_full is None or raw_keys is None or raw_values is None
                    or raw_queries is None or o_proj is None):
                continue
            o_full = o_full.to(device).float()
            N = o_full.shape[0]

            # Match inference selection exactly: MLP scores + sink protection,
            # then student Top-B. Selection remains non-differentiable.
            # Keep a pre-sink copy for memory weighting. Inference force-keeps
            # sink tokens for Top-B but computes p_theta from calibrated scores
            # before that overwrite; training must do exactly the same.
            memory_scores = scores
            if not getattr(config, "memory_backprop_to_indexer", False):
                memory_scores = memory_scores.detach()
            if config.recency_alpha > 0.0 and N > 1:
                positions = torch.arange(N, device=device, dtype=memory_scores.dtype)
                memory_scores = memory_scores + config.recency_alpha * torch.exp(
                    -(N - 1 - positions) / max(1, config.recency_window)
                )
            # Keep Top-B membership non-differentiable. Score gradients depend
            # on memory_backprop_to_indexer. Hold out suffix queries and write
            # only evicted prefix K/V to avoid future-token leakage.
            answer_tokens = max(0, N - int(memory_scores.shape[0]))
            if answer_tokens > 0:
                tail = max(1, min(
                    answer_tokens,
                    int(getattr(config, "memory_supervision_tail", 16)),
                ))
                partition_total = N
            else:
                tail = max(
                    1, min(N - 1,
                           int(getattr(config, "memory_supervision_tail", 16)))
                )
                partition_total = None
            # Construct the eviction set with the configured selection policy.
            # Keep raw scores for p_theta, but apply the coverage/local
            # transform for hard retained-set construction.
            memory_selection_scores = _deployment_span_scores(
                memory_scores, data, config
            )
            prefix_n, read_keep_idx, evict_mask = _causal_memory_partition(
                memory_selection_scores, config, tail,
                total_length=partition_total,
                compression_ratio=current_compression_ratio,
            )
            with torch.no_grad():
                o_comp = _reconstruct_compressed_output(
                    raw_keys, raw_values, raw_queries, read_keep_idx, o_proj,
                    int(data["num_kv_groups"]),
                    query_tail=tail,
                )
            if o_comp is None:
                continue
            o_comp = o_comp.to(device).float()
            o_full_target = o_full[-tail:]
            o_comp_target = o_comp
            # The shared Linear_theta uses concatenated post-RoPE queries and
            # GQA-expanded keys in the same query-head coordinate space.
            memory_query_target = (
                raw_queries[:, -tail:, :]
                .permute(1, 0, 2)
                .reshape(tail, -1)
                .to(device)
                .float()
            )

            # Expand only evicted GQA K/V rows into d_model space.
            num_groups = int(data["num_kv_groups"])
            ev_raw_k = raw_keys[:, evict_mask, :].to(device)
            ev_raw_v = raw_values[:, evict_mask, :].to(device)
            ev_k = (
                repeat_kv(ev_raw_k.unsqueeze(0), num_groups)
                .squeeze(0)
                .permute(1, 0, 2)
                .reshape(ev_raw_k.shape[1], -1)
                .float()
            )
            ev_v = (
                repeat_kv(ev_raw_v.unsqueeze(0), num_groups)
                .squeeze(0)
                .permute(1, 0, 2)
                .reshape(ev_raw_v.shape[1], -1)
                .float()
            )
            if ev_k.shape[0] == 0:
                continue

            prefix_evict_mask = evict_mask[:prefix_n]
            # Eq. 6/10: use the full-prefix allocation. Sink protection
            # is enforced by E; KL's eligible-axis renormalization is separate.
            p_theta = student_distribution(memory_scores[:prefix_n], config.student_temp)
            pi_E = (
                p_theta
                * prefix_evict_mask.to(dtype=p_theta.dtype)
            ).sum()
            prefix_write_weights = conditional_evicted_weights(
                p_theta,
                prefix_evict_mask,
                getattr(config, "memory_uniform_fraction", 0.0),
            )
            p_bar = prefix_write_weights[prefix_evict_mask]

            # Fast state is ephemeral, but the current-step construction must
            # remain differentiable through the shared slow weight theta on
            # both the write-key and read-query sides.
            proj_k = memory_stack.project(ev_k, layer_idx)
            # Weighted outer-product sum, written as GEMM. The equivalent
            # three-input einsum selects a contraction path that materializes
            # an (N_evict, d_mem, d_model) temporary: 2 GiB at 4K/50%
            # compression. Weighting the small projection first keeps the
            # largest intermediate at (N_evict, d_mem).
            weighted_proj_k = proj_k * p_bar.unsqueeze(-1)
            M_bar = weighted_proj_k.transpose(0, 1) @ ev_v
            b_bar = ((proj_k * proj_k) * p_bar.unsqueeze(-1)).sum(dim=0)

            # First mass-normalized memory write:
            #   Z_1 = mu_E
            #   M_1 = mu_E * M_bar / (mu_E + eps)
            #   b_1 = mu_E * b_bar / (mu_E + eps)
            memory_eps = 1e-8
            first_write_scale = pi_E / (pi_E + memory_eps)
            M_l = M_bar * config.memory_write_eta * first_write_scale
            b_l = b_bar * config.memory_write_eta * first_write_scale
            # Readout with gradient -> trains Linear_θ + gate.
            residual, gate_value = memory_stack.readout(
                memory_query_target, M_l, b_l, layer_idx,
                gate_override=_memory_gate_override(config, global_step),
                return_gate=True,
            )
            total_gate_value += float(gate_value.detach().mean())
            value_space = getattr(config, "memory_value_space", "pre_o_proj")
            if value_space != "pre_o_proj":
                raise ValueError(
                    f"Unknown memory_value_space={value_space!r}"
                )
            # CORE paper objective (Eq. memory reconstruction): no gap
            # weighting and no relative normalization.
            raw_loss_mem_l = _paper_memory_reconstruction_loss(
                o_full_target, o_comp_target, residual
            )
            gap_mem_l = ((o_full_target - o_comp_target) ** 2).sum(dim=-1).mean()
            loss_mem_l = raw_loss_mem_l
            total_loss_mem = total_loss_mem + loss_mem_l
            total_raw_loss_mem = total_raw_loss_mem + raw_loss_mem_l
            total_gap_mem = total_gap_mem + gap_mem_l
            n_mem_layers += 1

        # Normalize per-layer losses, then combine.
        total_loss_core = total_loss_core / max(1.0, total_weight)
        if n_mem_layers > 0:
            total_loss_mem = total_loss_mem / float(n_mem_layers)
            total_raw_loss_mem = total_raw_loss_mem / float(n_mem_layers)
            total_gap_mem = total_gap_mem / float(n_mem_layers)
            total = (
                config.memory_train_lambda * total_loss_mem
                if freeze_indexer
                else total_loss_core + config.memory_train_lambda * total_loss_mem
            )
        else:
            if freeze_indexer:
                # A batch without usable memory supervision cannot update any
                # trainable parameter in this mode.
                continue
            total = total_loss_core

        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite joint loss at global step {global_step}; "
                "optimizer update was not applied"
            )

        metric_denom = max(1.0, total_weight)
        step_metrics["loss"] += float(total_loss_core.detach())
        step_metrics["loss_cal"] += total_metric_cal / metric_denom
        step_metrics["loss_bd"] += total_metric_bd / metric_denom
        step_metrics["loss_mem"] += float(total_raw_loss_mem.detach().item())
        step_metrics["mem_objective"] += float(total_loss_mem.detach().item())
        step_metrics["gap"] += float(total_gap_mem.detach().item())
        step_metrics["gate"] += total_gate_value / max(1, n_mem_layers)
        step_metrics["total_loss"] += float(total.detach().item())
        (total / micro_batch_size).backward()
        if micro_index + 1 < micro_batch_size:
            continue
        torch.nn.utils.clip_grad_norm_(
            optimizer_parameters, config.grad_clip, error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        pbar.update(1)

        # ---- logging ----
        for key in step_metrics:
            running[key] += step_metrics[key] / micro_batch_size
        running["n"] += 1
        if global_step % config.online_log_every == 0 and running["n"] > 0:
            n = running["n"]
            avg = {k: running[k] / n for k in
                   ("loss", "loss_cal", "loss_bd", "loss_mem",
                    "mem_objective", "gap", "gate", "total_loss")}
            group_lrs = {
                group.get("name", f"group_{idx}"): group["lr"]
                for idx, group in enumerate(optimizer.param_groups)
            }
            lr = group_lrs.get("indexer", group_lrs.get("memory", 0.0))
            memory_lr = group_lrs.get("memory", lr)
            history.append({
                "step": global_step, "stage": "joint", "loss": avg["loss"],
                "loss_cal": avg["loss_cal"], "loss_bd": avg["loss_bd"],
                "total_loss": avg["total_loss"], "lr": lr,
                "memory_lr": memory_lr,
            })
            memory_history.append({
                "step": global_step, "stage": "joint", "loss": avg["loss_mem"],
                "objective": avg["mem_objective"], "gap": avg["gap"],
                "gate": avg["gate"], "lr": memory_lr,
            })
            pbar.set_postfix(
                loss=f"{avg['loss']:.4f}", cal=f"{avg['loss_cal']:.4f}",
                bd=f"{avg['loss_bd']:.4f}", mem=f"{avg['loss_mem']:.6f}",
                memobj=f"{avg['mem_objective']:.4f}",
                gap=f"{avg['gap']:.6f}", gate=f"{avg['gate']:.3f}",
                lr=f"{lr:.2e}/{memory_lr:.2e}",
            )
            running = {k: 0.0 for k in running}
            running["n"] = 0

        eval_every = max(1, int(config.online_eval_every))
        if validation_samples and (
            global_step % eval_every == 0 or step + 1 == stage_steps
        ):
            metrics = _evaluate_online_indexer(
                config, model, tokenizer, collector, feature_extractor,
                teacher, indexer, validation_samples,
            )
            if metrics:
                history.append({
                    "step": global_step, "stage": "validation", **metrics
                })
                logger.info(
                    "Validation step %d: KL=%.4f Top-B=%.4f boundary=%.4f",
                    global_step, metrics["val_kl"],
                    metrics["val_topB_recall"], metrics["val_boundary_acc"],
                )
                logger.info(
                    "  Boundary pairs: accuracy=%.4f mean_gap=%.4f strict_gap=%.4f",
                    metrics["val_boundary_pair_acc"],
                    metrics["val_boundary_pair_gap"],
                    metrics["val_boundary_margin"],
                )
                if "val_coverage_recall" in metrics:
                    logger.info(
                        "  Hybrid channels: coverage recall=%.4f boundary=%.4f "
                        "margin=%.4f | local recall=%.4f boundary=%.4f margin=%.4f",
                        metrics["val_coverage_recall"],
                        metrics["val_coverage_boundary_acc"],
                        metrics["val_coverage_boundary_margin"],
                        metrics["val_local_recall"],
                        metrics["val_local_boundary_acc"],
                        metrics["val_local_boundary_margin"],
                    )
                memory_metrics = _evaluate_online_memory(
                    config, model, tokenizer, collector, feature_extractor,
                    teacher, indexer, memory_stack, validation_samples,
                )
                metrics.update(memory_metrics)
                _attach_joint_validation_objective(metrics, config)
                history[-1].update(memory_metrics)
                if "val_joint_objective" in metrics:
                    history[-1]["val_joint_objective"] = metrics[
                        "val_joint_objective"
                    ]
                if memory_metrics:
                    logger.info(
                        "  Held-out memory: relative=%.4f recovery=%.4f "
                        "raw=%.6g gap=%.6g gate=%.4f chunks=%d",
                        metrics["val_memory_relative"],
                        metrics["val_memory_recovery"],
                        metrics["val_memory_raw"],
                        metrics["val_memory_gap"],
                        metrics["val_memory_gate"],
                        metrics["val_memory_layer_chunks"],
                    )
                    logger.info(
                        "  Memory alignment: target_rms=%.6g residual_rms=%.6g "
                        "cosine=%.4f optimal_scale=%.4f",
                        metrics["val_memory_target_rms"],
                        metrics["val_memory_residual_rms"],
                        metrics["val_memory_residual_cosine"],
                        metrics["val_memory_optimal_scale"],
                    )
                if "val_joint_objective" in metrics:
                    logger.info(
                        "  Held-out joint objective: %.4f = indexer %.4f + "
                        "lambda_mem %.3f * memory %.4f",
                        metrics["val_joint_objective"],
                        metrics["val_indexer_objective"],
                        config.memory_train_lambda,
                        metrics["val_memory_relative"],
                    )
                last_validation_metrics = metrics
                if _is_better_validation(metrics, best_indexer_metrics):
                    best_indexer_metrics = dict(metrics)
                    save_indexer_stack(
                        indexer, config, history,
                        metrics=metrics,
                        feature_extractor=feature_extractor,
                        checkpoint_name=best_indexer_path.name,
                    )
                joint_improved = (
                    _is_better_memory_validation(metrics, best_joint_metrics)
                    if freeze_indexer
                    else _is_better_joint_validation(
                        metrics, best_joint_metrics
                    )
                )
                max_drop = float(
                    getattr(config, "joint_max_topb_drop", 1.0)
                )
                if (
                    stage2_initial_topb is not None
                    and metrics["val_topB_recall"]
                    < stage2_initial_topb - max_drop
                ):
                    joint_improved = False
                    logger.info(
                        "  Joint checkpoint rejected: Top-B %.4f is below "
                        "Stage-II floor %.4f",
                        metrics["val_topB_recall"],
                        stage2_initial_topb - max_drop,
                    )
                if joint_improved:
                    best_joint_metrics = dict(metrics)
                    save_joint_stack(
                        indexer, memory_stack, config,
                        history=history, memory_history=memory_history,
                        feature_extractor=feature_extractor,
                        metrics=metrics,
                        checkpoint_name=best_joint_path.name,
                    )
                patience = int(getattr(config, "stage2_early_stop_patience", 0))
                min_steps = int(getattr(config, "stage2_early_stop_min_steps", 0))
                min_delta = float(getattr(config, "stage2_early_stop_min_delta", 0.0))
                objective = float(metrics["val_joint_objective"])
                if objective < early_best_joint_objective - min_delta:
                    early_best_joint_objective = objective
                    early_bad_joint_validations = 0
                else:
                    early_bad_joint_validations += 1
                joint_steps_done = global_step - schedule_offset
                if (
                    patience > 0
                    and joint_steps_done >= min_steps
                    and early_bad_joint_validations >= patience
                ):
                    logger.info(
                        "Stage-II early stop at global step %d: held-out joint "
                        "objective did not improve by %.6f for %d validations",
                        global_step, min_delta, patience,
                    )
                    stop_stage2 = True

        # ---- intermediate save (anti-crash) ----
        save_every = getattr(config, "online_save_every", 0)
        if save_every and global_step % save_every == 0:
            indexer.eval()
            memory_stack.eval()
            save_joint_stack(
                indexer, memory_stack, config,
                history=history, memory_history=memory_history,
                feature_extractor=feature_extractor,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
            )
            if not freeze_indexer:
                indexer.train()
            memory_stack.train()
            logger.info(f"Intermediate joint save at step {global_step}")
        if stop_stage2:
            break

    # Preserve the exact last-step synchronized joint state before restoring
    # the selected best joint checkpoint.
    save_joint_stack(
        indexer, memory_stack, config,
        history=history, memory_history=memory_history,
        feature_extractor=feature_extractor,
        metrics=last_validation_metrics,
        checkpoint_name="core_indexer_last_joint.pt",
    )

    # Restore indexer and memory weights from the same best joint checkpoint.
    if best_joint_metrics is not None and best_joint_path.exists():
        best_checkpoint = torch.load(
            best_joint_path, map_location=device, weights_only=False
        )
        best_step = _restore_synchronized_joint_checkpoint(
            indexer=indexer,
            memory_stack=memory_stack,
            checkpoint=best_checkpoint,
            checkpoint_path=best_joint_path,
        )
        if freeze_indexer:
            logger.info(
                "Restored best held-out MEMORY checkpoint at step %d: "
                "relative=%.4f recovery=%.4f gate=%.4f",
                best_step,
                best_joint_metrics["val_memory_relative"],
                best_joint_metrics["val_memory_recovery"],
                best_joint_metrics["val_memory_gate"],
            )
        else:
            logger.info(
                "Restored synchronized best JOINT checkpoint at step %d: "
                "objective=%.4f Top-B=%.4f memory_recovery=%.4f "
                "(indexer + memory from the same step)",
                best_step,
                best_joint_metrics["val_joint_objective"],
                best_joint_metrics["val_topB_recall"],
                best_joint_metrics["val_memory_recovery"],
            )
    else:
        logger.info(
            "No best joint checkpoint available; retaining synchronized "
            "last-step indexer + memory."
        )

    indexer.eval()
    memory_stack.eval()
    return indexer, memory_stack, history, memory_history


def save_indexer_stack(
    indexer: COREIndexerStack,
    config: COREConfig,
    history: list,
    metrics: dict | None = None,
    feature_extractor: COREFeatureExtractor | None = None,
    memory_module: MemoryModule | MemoryModuleStack | None = None,
    memory_history: list | None = None,
    checkpoint_name: str = "core_indexer.pt",
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
):
    """Save a per-layer indexer stack plus optional feature extractor and memory module.

    Accepts either a single :class:`MemoryModule` (legacy, single-layer) or a
    :class:`MemoryModuleStack` (per-layer, IndexMem §3.2). The two are
    distinguished by the ``memory_dims.is_stack`` flag written to the checkpoint
    so :func:`load_memory_stack` can rebuild the correct container.
    """
    output_path = Path(config.output_dir) / "indexer"
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": indexer.state_dict(),
        "n_layers": indexer.n_layers,
        "input_dim": indexer.input_dim,
        "hidden_dim": indexer.hidden_dim,
        "indexer_compact_stride": indexer.compact_stride,
        "config": config.__dict__,
        "history": history,
        "metrics": metrics or {},
        "is_stack": True,
    }
    if feature_extractor is not None:
        checkpoint["feature_extractor_state_dict"] = feature_extractor.geometry_state_dict()
    if optimizer_state_dict is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state_dict
    if memory_module is not None:
        if isinstance(memory_module, MemoryModuleStack):
            checkpoint["memory_state_dict"] = memory_module.state_dict()
            checkpoint["memory_dims"] = {
                "d_model": memory_module.d_model,
                "d_mem": memory_module.d_mem,
                "gate_hidden": memory_module.gate_hidden,
                "projection_groups": memory_module.projection_groups,
                "share_across_layers": memory_module.share_across_layers,
                "n_layers": memory_module.n_layers,
                "is_stack": True,
            }
        else:
            # Serialize a single memory module.
            checkpoint["memory_state_dict"] = memory_module.state_dict()
            checkpoint["memory_dims"] = {
                "d_model": memory_module.d_model,
                "d_mem": memory_module.d_mem,
                "gate_hidden": getattr(memory_module, "gate_hidden", 128),
                "projection_groups": getattr(memory_module, "projection_groups", 1),
                "n_layers": 1,
                "is_stack": False,
            }
        checkpoint["memory_history"] = memory_history or []
    checkpoint_path = output_path / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved indexer stack to {checkpoint_path}")


def save_joint_stack(
    indexer: COREIndexerStack,
    memory_stack: MemoryModuleStack,
    config: COREConfig,
    history: list,
    memory_history: list | None = None,
    feature_extractor: COREFeatureExtractor | None = None,
    metrics: dict | None = None,
    checkpoint_name: str = "core_indexer.pt",
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
):
    """Joint-train-friendly alias for :func:`save_indexer_stack`.

    Persists both the indexer stack and the per-layer MemoryModuleStack in a
    single checkpoint. Used for the final save and for intermediate saves
    during :func:`train_joint_online` (every ``config.online_save_every`` steps)
    so a mid-run crash does not lose the joint weights.
    """
    save_indexer_stack(
        indexer=indexer,
        config=config,
        history=history,
        feature_extractor=feature_extractor,
        memory_module=memory_stack,
        memory_history=memory_history,
        metrics=metrics,
        checkpoint_name=checkpoint_name,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
    )


def train_memory_module_online(
    config: COREConfig,
    model,
    tokenizer,
    feature_extractor: COREFeatureExtractor,
    teacher: DiversityAwareTeacher,
) -> tuple[MemoryModule, list]:
    """
    Legacy online memory-only training, separate from joint Stage II.

    Phase 2.5 (IndexMem §3.2). Each step:
      1. Sample a LongAlpaca sequence and run a frozen backbone forward.
      2. collect_online_batch surfaces o_full / o_compressed / query_hidden /
         keep_indices for the middle layer (same forward as Phase 2, no extra
         cost beyond the o_compressed reconstruction).
      3. Simulate the inference-time fast-weight write (no grad): build M, b
         from full-width GQA-expanded keys and post-o_proj values.  This legacy
         standalone path uses the teacher distribution for write weights; the
         default joint trainer uses the live student distribution exactly.
      4. Compute the MSE residual loss L_mem = ||o_full - o_compressed - g(q)·m(q)||²
         and backprop into Linear_θ + gate only (backbone + indexer frozen).

    The default inference loader expects the per-layer MemoryModuleStack
    produced by train_joint_online, rather than this standalone module.
    """
    _validate_training_objective(config)
    from core.data import load_longalpaca, tokenize_sample
    from core.collector import BackboneStatsCollector, collect_online_batch

    device = torch.device(config.backbone_device)
    dtype = getattr(torch, config.dtype, torch.bfloat16)

    d_model = (config.n_q_heads * config.head_dim
               if config.n_q_heads and config.head_dim else config.hidden_size or 4096)
    d_mem = config.memory_dim or max(1, d_model // 8)
    memory = MemoryModule(
        d_model,
        d_mem,
        config.memory_gate_hidden,
        projection_groups=getattr(config, "memory_projection_groups", 1),
    ).to(device)
    memory.train()

    # Freeze backbone + indexer (Phase 2 already trained the indexer).
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(memory.parameters(), lr=config.memory_train_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.memory_train_epochs)

    samples = load_longalpaca(config, tokenizer)
    if not samples:
        logger.warning("No training samples for memory module; returning untrained.")
        memory.eval()
        return memory, []

    collector = BackboneStatsCollector(model, capture_all_layers=True)
    sample_iter = iter(samples)
    history = []
    running = {"loss": 0.0, "gap": 0.0, "n": 0}

    total_steps = config.memory_train_epochs * max(1, len(samples))
    pbar = tqdm(range(total_steps), desc="Memory online")
    for step in pbar:
        try:
            sample = next(sample_iter)
        except StopIteration:
            sample_iter = iter(samples)
            sample = next(sample_iter)

        tok = tokenize_sample(tokenizer, sample, config.max_seq_len, device)
        input_ids = tok["input_ids"]
        if input_ids.shape[1] < config.n_sink + 2:
            continue

        batch = collect_online_batch(
            model, collector, feature_extractor, teacher, config,
            input_ids, compression_ratio=_training_compression_ratio(config, step + 1),
            compute_teacher_compressed=True,
        )
        if not batch:
            continue

        # Pick the middle layer's memory supervision data.
        mid_idx = sorted(batch.keys())[len(batch) // 2]
        data = batch[mid_idx]
        o_full = data.get("o_full")
        o_comp = data.get("o_compressed")
        q_hidden = data.get("query_hidden")
        keep_idx = data.get("keep_indices")
        memory_keys = data.get("memory_keys_full")
        memory_values = data.get("memory_values_full")
        pi_t = data.get("pi_T")
        if (o_full is None or o_comp is None or q_hidden is None
                or keep_idx is None or memory_keys is None
                or memory_values is None or pi_t is None):
            continue

        o_full = o_full.to(device).float()
        o_comp = o_comp.to(device).float()
        q_hidden = q_hidden.to(device).float()
        keep_idx = keep_idx.to(device).long()
        memory_keys = memory_keys.to(device).float()
        memory_values = memory_values.to(device).float()
        pi_t = pi_t.to(device).float()
        N = o_full.shape[0]

        # --- simulate inference-time fast-weight write ---
        evict_mask = torch.ones(N, dtype=torch.bool, device=device)
        evict_mask[keep_idx] = False
        ev_k = memory_keys[evict_mask]                    # (n_evict, d_model)
        ev_v = memory_values[evict_mask]                  # (n_evict, d_model)
        if ev_k.shape[0] == 0:
            continue
        with torch.no_grad():
            # Teacher weights are only for this compatibility trainer.  The
            # joint trainer above uses softmax(student_score / tau_S).
            p_evict = pi_t[evict_mask]
            pi_e = p_evict.sum().clamp_min(1e-8)
            p_bar = p_evict / pi_e
        proj_k = memory.project(ev_k)                      # (n_evict, d_mem)
        M_bar = proj_k.T @ (ev_v * p_bar.unsqueeze(-1))
        b_bar = (proj_k.square() * p_bar.unsqueeze(-1)).sum(dim=0)
        M = M_bar * config.memory_write_eta
        b = b_bar * config.memory_write_eta

        # --- readout with gradient (trains Linear_θ + gate) ---
        residual = memory.readout(q_hidden, M, b)         # (L, d_model)

        # CORE paper objective: ordinary elementwise reconstruction MSE.
        loss = _paper_memory_reconstruction_loss(
            o_full, o_comp, residual
        ) * config.memory_train_lambda

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(memory.parameters(), config.grad_clip)
        optimizer.step()

        with torch.no_grad():
            gap_mse = ((o_full - o_comp) ** 2).sum(dim=-1).mean().item()

        running["loss"] += loss.item()
        running["gap"] += gap_mse
        running["n"] += 1
        if (step + 1) % config.online_log_every == 0 and running["n"] > 0:
            n = running["n"]
            avg_loss = running["loss"] / n
            avg_gap = running["gap"] / n
            lr = optimizer.param_groups[0]["lr"]
            history.append({
                "step": step + 1, "loss": avg_loss, "gap": avg_gap, "lr": lr,
            })
            pbar.set_postfix(
                loss=f"{avg_loss:.6f}", gap=f"{avg_gap:.6f}", lr=f"{lr:.2e}",
            )
            running = {"loss": 0.0, "gap": 0.0, "n": 0}

    scheduler.step()
    memory.eval()
    return memory, history


def load_indexer_stack(path: str, device: str = "cuda:0") -> tuple[COREIndexerStack, dict]:
    """Load a per-layer indexer stack.

    Backward-compat: if the checkpoint is a single-MLP legacy format (no
    ``is_stack`` flag), the MLP weights are broadcast to every layer.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    is_stack = ckpt.get("is_stack", False)

    if is_stack:
        n_layers = ckpt["n_layers"]
        input_dim = ckpt["input_dim"]
        hidden_dim = ckpt["hidden_dim"]
        indexer = COREIndexerStack(
            n_layers=n_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            compact_stride=int(ckpt.get("indexer_compact_stride", 1)),
        )
        indexer.load_state_dict(ckpt["state_dict"])
    else:
        # Wrap single-MLP checkpoints in the stack interface.
        input_dim = ckpt.get("input_dim", 14)
        hidden_dim = ckpt.get("hidden_dim", 128)
        n_layers = ckpt.get("n_layers", 1)
        indexer = COREIndexerStack(
            n_layers=max(1, n_layers), input_dim=input_dim, hidden_dim=hidden_dim
        )
        # Load the single MLP into every layer (weight-shared fallback).
        single_state = ckpt["state_dict"]
        for layer_mlp in indexer.layers:
            layer_mlp.load_state_dict(single_state)

    indexer.to(device)
    indexer.eval()
    return indexer, ckpt


def load_memory_stack(
    path: str, device: str = "cuda:0"
) -> MemoryModuleStack:
    """Load a (per-layer) MemoryModuleStack from a joint checkpoint.

    Backward-compat: legacy single-MemoryModule checkpoints (``is_stack=False``
    or missing) are wrapped into a 1-layer stack so callers always receive a
    :class:`MemoryModuleStack`. Raises ``KeyError`` if the checkpoint has no
    ``memory_state_dict`` (i.e. it was saved without training the memory
    module - e.g. an indexer-only ``core_indexer.pt``).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "memory_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {path} has no 'memory_state_dict' - the saved run "
            "did not train the MemoryModule (joint training was not enabled)."
        )
    dims = ckpt["memory_dims"]
    n_layers = dims.get("n_layers", 1)
    d_model = dims["d_model"]
    d_mem = dims["d_mem"]
    gate_hidden = dims.get("gate_hidden", 128)
    projection_groups = dims.get("projection_groups", 1)

    memory_stack = MemoryModuleStack(
        n_layers=max(1, n_layers),
        d_model=d_model,
        d_mem=d_mem,
        gate_hidden=gate_hidden,
        share_across_layers=bool(dims.get("share_across_layers", False)),
        projection_groups=projection_groups,
    )
    state = ckpt["memory_state_dict"]
    if dims.get("is_stack", False):
        memory_stack.load_state_dict(state)
    else:
        # Broadcast a single memory module's weights to every layer.
        for layer_mem in memory_stack.layers:
            layer_mem.load_state_dict(state)

    memory_stack.to(device)
    memory_stack.eval()
    return memory_stack
