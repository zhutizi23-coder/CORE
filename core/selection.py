from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def long_query_match_mask(
    token_ids: torch.Tensor,
    query_start: int,
    min_match_tokens: int = 20,
    context_skip: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find long contiguous query token sequences repeated in the context.

    The initial context prefix is ignored because RULER QA repeats a fixed
    instruction header in both context and question.  Matching that boilerplate
    is not evidence.  Long matches later in the context are characteristic of
    identifiers such as UUID keys and provide a high-precision lexical anchor.

    Returns a context-position mask and one activation flag per batch row.
    """
    ids = token_ids
    squeeze = ids.ndim == 1
    if squeeze:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2:
        raise ValueError("token_ids must have shape (seq_len,) or (batch, seq_len)")

    batch, seq_len = ids.shape
    query_start = max(0, min(int(query_start), seq_len))
    width = max(1, int(min_match_tokens))
    skip = max(0, min(int(context_skip), query_start))
    masks = torch.zeros_like(ids, dtype=torch.bool)
    active = torch.zeros(batch, device=ids.device, dtype=torch.bool)
    if query_start - skip < width or seq_len - query_start < width:
        return (masks[0] if squeeze else masks), active

    for batch_idx in range(batch):
        context = ids[batch_idx, skip:query_start]
        query = ids[batch_idx, query_start:]
        context_windows = context.unfold(0, width, 1)
        query_windows = query.unfold(0, width, 1)
        matches = (
            context_windows[:, None, :] == query_windows[None, :, :]
        ).all(dim=-1)
        starts = matches.any(dim=1).nonzero(as_tuple=False).flatten()
        if starts.numel() == 0:
            continue
        active[batch_idx] = True
        for start in starts.tolist():
            absolute = skip + int(start)
            masks[batch_idx, absolute:absolute + width] = True
    return (masks[0] if squeeze else masks), active


def expand_span_scores(
    scores: torch.Tensor,
    span_size: int,
) -> torch.Tensor:
    """
    Propagate each high token score to its local contiguous neighborhood.

    Parameters
    ----------
    scores:
        Tensor with shape (..., seq_len).
    span_size:
        Size of the local contiguous span. 1 disables span expansion.

    Returns
    -------
    Tensor with the same shape as scores.
    """
    span_size = int(span_size)
    if span_size <= 1:
        return scores

    if scores.ndim < 1:
        raise ValueError("scores must have at least one dimension")

    seq_len = scores.shape[-1]
    if seq_len == 0:
        return scores

    span_size = min(span_size, seq_len)

    # Asymmetric padding also supports even span sizes such as 4 or 8.
    left = (span_size - 1) // 2
    right = span_size - 1 - left

    original_shape = scores.shape
    x = scores.float().reshape(-1, 1, seq_len)

    x = F.pad(
        x,
        (left, right),
        mode="constant",
        value=torch.finfo(x.dtype).min,
    )

    x = F.max_pool1d(
        x,
        kernel_size=span_size,
        stride=1,
    )

    return x.reshape(original_shape).to(dtype=scores.dtype)


def contiguous_block_scores(
    scores: torch.Tensor,
    block_size: int,
    *,
    beta: float = 5.0,
) -> torch.Tensor:
    """Give every token in a fixed contiguous block one smooth-max score.

    Unlike sliding max propagation, this makes global token Top-K operate on
    coherent blocks: all tokens in a block have the same selection score, so
    every block except at most the final budget remainder is retained whole.
    ``logmeanexp`` keeps gradients for every token while remaining close to the
    high-utility token that should nominate the surrounding evidence span.
    """
    block_size = max(1, int(block_size))
    if block_size <= 1 or scores.shape[-1] == 0:
        return scores
    if scores.ndim < 1:
        raise ValueError("scores must have at least one dimension")
    if beta <= 0:
        raise ValueError("beta must be positive")

    seq_len = int(scores.shape[-1])
    block_size = min(block_size, seq_len)
    pad = (-seq_len) % block_size
    original_shape = scores.shape
    flat = scores.float().reshape(-1, seq_len)
    if pad:
        flat = F.pad(
            flat,
            (0, pad),
            mode="constant",
            value=torch.finfo(flat.dtype).min,
        )
    blocks = flat.reshape(flat.shape[0], -1, block_size)
    valid_counts = torch.full(
        (blocks.shape[1],),
        block_size,
        device=blocks.device,
        dtype=blocks.dtype,
    )
    if pad:
        valid_counts[-1] = block_size - pad
    pooled = (
        torch.logsumexp(blocks * float(beta), dim=-1)
        - valid_counts.log().unsqueeze(0)
    ) / float(beta)
    expanded = pooled.unsqueeze(-1).expand_as(blocks).reshape(flat.shape)
    expanded = expanded[:, :seq_len]
    return expanded.reshape(original_shape).to(dtype=scores.dtype)



def region_stratified_order(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    block_size: int,
    *,
    block_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Order tokens with diminishing returns per fixed context region.

    Blocks are ordered by ``block_scores`` (or their maximum raw score), while
    token rank is interleaved across blocks: the best token from every block
    precedes the second-best token from any block. Invalid/padded positions and
    protected tokens absent from ``candidates`` never enter the result.
    """
    if candidates.numel() == 0:
        return candidates
    block_size = max(1, int(block_size))
    seq_len = int(scores.numel())
    n_blocks = (seq_len + block_size - 1) // block_size
    padded_len = n_blocks * block_size
    positions = torch.arange(
        padded_len, device=scores.device, dtype=torch.long
    ).reshape(n_blocks, block_size)
    valid = positions < seq_len
    allowed = torch.zeros(seq_len, device=scores.device, dtype=torch.bool)
    allowed[candidates] = True
    valid &= torch.where(
        positions < seq_len, allowed[positions.clamp_max(seq_len - 1)], False
    )
    safe_pos = positions.clamp_max(seq_len - 1)
    token_scores = scores[safe_pos].masked_fill(~valid, float("-inf"))
    within = token_scores.argsort(dim=1, descending=True)
    ranked = positions.gather(1, within)
    ranked_valid = valid.gather(1, within)

    if block_scores is None:
        priority = token_scores.max(dim=1).values
    else:
        priority = block_scores[positions[:, 0].clamp_max(seq_len - 1)]
    block_order = priority.argsort(descending=True)
    ranked = ranked[block_order]
    ranked_valid = ranked_valid[block_order]
    # transpose: rank-0 across all regions, then rank-1, etc.
    interleaved = ranked.transpose(0, 1).reshape(-1)
    interleaved_valid = ranked_valid.transpose(0, 1).reshape(-1)
    return interleaved[interleaved_valid]

def coverage_first_local_completion_scores(
    scores: torch.Tensor,
    query_start: int,
    budget: int,
    block_size: int,
    *,
    beta: float = 5.0,
    coverage_fraction: float = 0.25,
    n_sink_protect: int = 0,
    protect_query_tokens: bool = False,
    balanced_local_completion: bool = False,
) -> torch.Tensor:
    """Encode an exact coverage-first/local-completion set as Top-K scores.

    The total token budget is split into two complementary channels. First,
    ``coverage_fraction`` of the context budget keeps globally high raw-score
    tokens. The learned score already contains the teacher's D-optimal coverage
    signal, so this channel can retain evidence from separated regions. Second,
    the remaining budget completes high log-mean-exp blocks, preferentially
    filling blocks that contain coverage tokens. Reserved sink/question tokens
    are charged to the same total budget.

    With ``balanced_local_completion=False`` the local stage fills whole
    blocks in score order (the v9/v10 baseline). With it enabled, the coverage quota is selected
    region-stratified: at most one raw-score token is taken from each block per
    round. The local quota then completes the highest-value blocks normally.
    This prevents a few peaks from consuming the nominal coverage budget while
    preserving coherent local structures.

    The returned tensor is rank-equivalent to the constructed set: exactly
    ``budget`` entries receive a detached offset larger than the score range.
    Raw-score entries keep tokenwise gradients; local-completion entries use the
    differentiable block score. Thus BasePress can keep using its normal Top-K
    implementation while training, validation and inference share one policy.
    """
    if scores.ndim < 1:
        raise ValueError("scores must have at least one dimension")
    if not 0.0 <= float(coverage_fraction) <= 1.0:
        raise ValueError("coverage_fraction must be in [0, 1]")
    if beta <= 0:
        raise ValueError("beta must be positive")

    seq_len = int(scores.shape[-1])
    if seq_len == 0:
        return scores
    query_start = max(0, min(int(query_start), seq_len))
    budget = max(1, min(int(budget), seq_len))
    block_size = max(1, min(int(block_size), max(1, query_start)))
    sink = max(0, min(int(n_sink_protect), query_start))

    original_shape = scores.shape
    flat = scores.float().reshape(-1, seq_len)
    outputs = []
    for row in flat:
        keep = torch.zeros(seq_len, device=row.device, dtype=torch.bool)
        local = torch.zeros_like(keep)

        # Structural reservations are part of, not additional to, Top-B.
        if sink:
            keep[:sink] = True
        if protect_query_tokens and query_start < seq_len:
            keep[query_start:] = True
        if int(keep.sum().item()) > budget:
            reserved = keep.nonzero(as_tuple=False).flatten()
            keep.zero_()
            keep[reserved[-budget:]] = True

        remaining = budget - int(keep.sum().item())
        candidates = torch.arange(sink, query_start, device=row.device)
        if remaining > 0 and candidates.numel() > 0:
            coverage_budget = min(
                remaining,
                max(0, int(round(remaining * float(coverage_fraction)))),
            )
            coverage_anchors = candidates.new_empty(0)
            starts = torch.arange(
                0, query_start, block_size,
                device=row.device, dtype=torch.long,
            )
            context_block_scores = contiguous_block_scores(
                row[:query_start], block_size, beta=beta
            )
            block_order = starts[
                context_block_scores[starts].argsort(descending=True)
            ]
            if coverage_budget:
                if balanced_local_completion:
                    # Region-stratified coverage: take at most one raw-score
                    # token from every block per round.  A global raw Top-K can
                    # spend the entire "coverage" quota inside a few peaks,
                    # which is not coverage at all.  Ordering blocks by their
                    # learned utility preserves relevance while round-robin
                    # ranks enforce diminishing returns within each region.
                    coverage_anchors = region_stratified_order(
                        row[:query_start], candidates, block_size,
                        block_scores=context_block_scores,
                    )[:coverage_budget]
                else:
                    coverage_anchors = candidates[
                        row[candidates].topk(coverage_budget).indices
                    ]
                keep[coverage_anchors] = True
                remaining -= int(coverage_anchors.numel())

            if remaining > 0:
                # Local completion deliberately remains concentrated: after
                # region-stratified coverage, fill the highest-value blocks in
                # order.  This retains complete key/value structures without
                # allowing them to erase global coverage.
                quantum = block_size

                # Round-robin block completion gives each nominated region a
                # bounded first allocation before any region receives more.
                while remaining > 0:
                    made_progress = False
                    for start_tensor in block_order:
                        if remaining <= 0:
                            break
                        start = int(start_tensor.item())
                        end = min(query_start, start + block_size)
                        block_idx = torch.arange(start, end, device=row.device)
                        block_idx = block_idx[block_idx >= sink]
                        missing = block_idx[~keep[block_idx]]
                        if missing.numel() == 0:
                            continue
                        take_n = min(
                            remaining, quantum, int(missing.numel())
                        )
                        if take_n < missing.numel():
                            anchors_here = block_idx[keep[block_idx]]
                            if anchors_here.numel():
                                center = anchors_here[
                                    row[anchors_here].argmax()
                                ]
                            else:
                                center = block_idx[row[block_idx].argmax()]
                            # Prefer a contiguous neighbourhood around the
                            # anchor; raw score only breaks equal-distance ties.
                            distance = (missing - center).abs().float()
                            tie = (row[missing] - row[missing].min())
                            tie = tie / tie.max().clamp_min(1e-6)
                            priority = -distance + tie * 1e-3
                            missing = missing[priority.topk(take_n).indices]
                        keep[missing] = True
                        local[missing] = True
                        remaining -= take_n
                        made_progress = True
                    if not made_progress:
                        break

            # Degenerate/rounding fallback: preserve the exact budget.
            if remaining > 0:
                available = candidates[~keep[candidates]]
                take_n = min(remaining, int(available.numel()))
                if take_n:
                    chosen = available[row[available].topk(take_n).indices]
                    keep[chosen] = True

        block_view = contiguous_block_scores(
            row[:query_start], block_size, beta=beta
        )
        base = row.clone()
        if query_start:
            base[:query_start] = torch.where(
                local[:query_start], block_view, row[:query_start]
            )
        score_range = (base.max() - base.min()).detach()
        offset = score_range + 1.0
        outputs.append(base + keep.to(base.dtype) * offset)

    return torch.stack(outputs).reshape(original_shape).to(dtype=scores.dtype)


def query_aware_span_selection_scores(
    scores: torch.Tensor,
    query_start: int | None,
    span_size: int,
    *,
    standardize: bool = True,
    token_ids: torch.Tensor | None = None,
    identifier_lexical_weight: float = 0.0,
    identifier_min_match_tokens: int = 20,
    identifier_context_skip: int = 64,
    identifier_span_size: int = 32,
    protect_query_tokens: bool = False,
    span_pool_beta: float = 5.0,
    span_selection_mode: str = "tokenwise",
    selection_budget: int | None = None,
    coverage_fraction: float = 0.25,
    n_sink_protect: int = 0,
) -> torch.Tensor:
    """Apply the production prefill span policy to calibrated token scores.

    This helper is shared by deployment, online training, held-out validation,
    and best-checkpoint selection.  Keeping the transform in one place prevents
    validation from measuring raw tokenwise Top-B while deployment selects from
    span-expanded scores.

    The context prefix is partitioned into contiguous blocks. Every token in a
    block receives the same differentiable smooth-max score, so the downstream
    fixed-budget Top-K keeps coherent evidence instead of isolated subwords.
    The appended question is left tokenwise unless it is explicitly protected.
    """
    if query_start is None:
        return scores
    query_start = int(query_start)
    span_size = max(1, int(span_size))
    identifier_enabled = (
        float(identifier_lexical_weight) != 0.0 and token_ids is not None
    )
    if query_start <= 0 or query_start >= scores.shape[-1]:
        return scores
    if span_selection_mode == "tokenwise":
        return scores
    if span_size <= 1 and not identifier_enabled and not protect_query_tokens:
        return scores

    adjusted = scores.float()
    if standardize:
        raw_mean = adjusted.mean(dim=-1, keepdim=True)
        raw_std = adjusted.std(dim=-1, keepdim=True).clamp_min(1e-6)
        adjusted = (adjusted - raw_mean) / raw_std
    leading_shape = adjusted.shape[:-1]
    flat = adjusted.reshape(-1, adjusted.shape[-1])
    row_spans = torch.full(
        (flat.shape[0],), span_size, device=flat.device, dtype=torch.long
    )

    weight = float(identifier_lexical_weight)
    if weight != 0.0 and token_ids is not None:
        ids = token_ids.to(device=flat.device)
        match_mask, active = long_query_match_mask(
            ids,
            query_start,
            min_match_tokens=identifier_min_match_tokens,
            context_skip=identifier_context_skip,
        )
        if match_mask.ndim == 1:
            match_mask = match_mask.unsqueeze(0)
        batch = match_mask.shape[0]
        rows_per_batch = max(1, flat.shape[0] // batch)
        match_mask = match_mask.repeat_interleave(rows_per_batch, dim=0)
        active = active.repeat_interleave(rows_per_batch)
        match_mask = match_mask[:flat.shape[0], :flat.shape[-1]]
        active = active[:flat.shape[0]]
        if active.any():
            lexical = match_mask.to(torch.float32)
            context_lexical = lexical[:, :query_start]
            lex_mean = context_lexical.mean(dim=-1, keepdim=True)
            lex_std = context_lexical.std(dim=-1, keepdim=True).clamp_min(1e-6)
            lexical = (lexical - lex_mean) / lex_std
            flat[active] = flat[active] + weight * lexical[active]
            row_spans[active] = max(span_size, int(identifier_span_size))

    transformed = []
    for row_idx, row in enumerate(flat):
        row_span = int(row_spans[row_idx].item())
        if span_selection_mode == "block_lme":
            context_scores = contiguous_block_scores(
                row[:query_start],
                block_size=row_span,
                beta=span_pool_beta,
            )
        elif span_selection_mode in {
            "coverage_local", "coverage_local_balanced"
        }:
            if selection_budget is None:
                raise ValueError(
                    "coverage_local requires the exact selection_budget"
                )
            transformed.append(
                coverage_first_local_completion_scores(
                    row,
                    query_start,
                    selection_budget,
                    row_span,
                    beta=span_pool_beta,
                    coverage_fraction=coverage_fraction,
                    n_sink_protect=n_sink_protect,
                    protect_query_tokens=protect_query_tokens,
                    balanced_local_completion=(
                        span_selection_mode == "coverage_local_balanced"
                    ),
                )
            )
            continue
        elif span_selection_mode == "sliding_max":
            context_scores = expand_span_scores(
                row[:query_start], span_size=row_span
            )
        else:
            raise ValueError(
                f"Unknown span_selection_mode={span_selection_mode!r}"
            )
        transformed.append(torch.cat((context_scores, row[query_start:])))
    result = torch.stack(transformed).reshape(
        *leading_shape, adjusted.shape[-1]
    )
    if protect_query_tokens:
        result = result.clone()
        query_floor = result.amax(dim=-1, keepdim=True).detach() + 1.0
        result[..., query_start:] = query_floor
    return result.to(dtype=scores.dtype)
