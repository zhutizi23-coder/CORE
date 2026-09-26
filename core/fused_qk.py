"""Fused CUDA primitives for CORE QK feature reductions."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as libdevice
except ImportError:  # CPU-only development and unit-test environments
    triton = None
    tl = None
    libdevice = None


def supports_strict_head_topk(num_kv_groups: int, topk_qk: int) -> bool:
    """Tiled exact reductions: all heads, or all except the smallest head."""
    return num_kv_groups > 0 and topk_qk > 0 and topk_qk >= num_kv_groups - 1


if triton is not None:

    @triton.jit
    def _causal_row_lse_kernel(
        query_ptr,
        key_ptr,
        lse_ptr,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kh,
        stride_kn,
        stride_kd,
        seq_q,
        seq_k,
        q_start,
        scale,
        num_groups: tl.constexpr,
        head_dim: tl.constexpr,
        ROUND_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Online causal log-sum-exp without materialising a QK matrix."""
        block_m = tl.program_id(0)
        query_head = tl.program_id(1)
        key_head = query_head // num_groups
        offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_d = tl.arange(0, BLOCK_D)
        query = tl.load(
            query_ptr
            + query_head * stride_qh
            + offsets_m[:, None] * stride_qm
            + offsets_d[None, :] * stride_qd,
            mask=(offsets_m[:, None] < seq_q)
            & (offsets_d[None, :] < head_dim),
            other=0.0,
        )

        running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        query_position = q_start + offsets_m

        max_visible_key = tl.minimum(
            seq_k, q_start + (block_m + 1) * BLOCK_M
        )
        for key_start in tl.range(0, max_visible_key, BLOCK_N):
            offsets_n = key_start + tl.arange(0, BLOCK_N)
            key = tl.load(
                key_ptr
                + key_head * stride_kh
                + offsets_n[None, :] * stride_kn
                + offsets_d[:, None] * stride_kd,
                mask=(offsets_d[:, None] < head_dim)
                & (offsets_n[None, :] < seq_k),
                other=0.0,
            )
            logits = tl.dot(query, key, input_precision="ieee")
            if ROUND_MODE == 1:
                logits = logits.to(tl.float16).to(tl.float32)
            elif ROUND_MODE == 2:
                logits = logits.to(tl.bfloat16).to(tl.float32)
            logits *= scale
            visible = (
                (offsets_m[:, None] < seq_q)
                & (offsets_n[None, :] < seq_k)
                & (offsets_n[None, :] <= query_position[:, None])
            )
            logits = tl.where(visible, logits, -float("inf"))
            tile_max = tl.max(logits, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            running_sum = (
                running_sum * libdevice.exp(running_max - new_max)
                + tl.sum(libdevice.exp(logits - new_max[:, None]), axis=1)
            )
            running_max = new_max

        lse = running_max + libdevice.log(running_sum)
        tl.store(
            lse_ptr + query_head * seq_q + offsets_m,
            lse,
            mask=offsets_m < seq_q,
        )


    @triton.jit
    def _grouped_relation_tile_kernel(
        query_ptr,
        key_ptr,
        lse_ptr,
        output_ptr,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kh,
        stride_kn,
        stride_kd,
        seq_q,
        seq_k,
        key_start,
        tile_len,
        q_start,
        scale,
        log_floor,
        num_groups: tl.constexpr,
        head_topk: tl.constexpr,
        head_dim: tl.constexpr,
        ROUND_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Compute one reduced (KV-head, key-tile, query) relation tile."""
        block_m = tl.program_id(0)
        block_n = tl.program_id(1)
        key_head = tl.program_id(2)
        offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n_local = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_n = key_start + offsets_n_local
        offsets_d = tl.arange(0, BLOCK_D)
        # Skip blocks wholly above the causal diagonal before loading Q/K or
        # issuing the tensor-core dot product. The caller initializes those
        # output locations to log_floor, exactly matching the masked result.
        block_max_query = q_start + (block_m + 1) * BLOCK_M - 1
        block_min_key = key_start + block_n * BLOCK_N
        if block_min_key > block_max_query:
            return
        key = tl.load(
            key_ptr
            + key_head * stride_kh
            + offsets_n[None, :] * stride_kn
            + offsets_d[:, None] * stride_kd,
            mask=(offsets_d[:, None] < head_dim)
            & (offsets_n_local[None, :] < tile_len)
            & (offsets_n[None, :] < seq_k),
            other=0.0,
        )
        relation_sum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        if head_topk < num_groups:
            relation_min = tl.full((BLOCK_M, BLOCK_N), float("inf"), tl.float32)
        query_position = q_start + offsets_m
        candidate_count = tl.minimum(query_position + 1, seq_k).to(tl.float32)

        for group_offset in tl.static_range(0, num_groups):
            query_head = key_head * num_groups + group_offset
            query = tl.load(
                query_ptr
                + query_head * stride_qh
                + offsets_m[:, None] * stride_qm
                + offsets_d[None, :] * stride_qd,
                mask=(offsets_m[:, None] < seq_q)
                & (offsets_d[None, :] < head_dim),
                other=0.0,
            )
            logits = tl.dot(query, key, input_precision="ieee")
            if ROUND_MODE == 1:
                logits = logits.to(tl.float16).to(tl.float32)
            elif ROUND_MODE == 2:
                logits = logits.to(tl.bfloat16).to(tl.float32)
            row_lse = tl.load(
                lse_ptr + query_head * seq_q + offsets_m,
                mask=offsets_m < seq_q,
                other=0.0,
            )
            relation = (
                logits * scale
                - row_lse[:, None]
                + libdevice.log(candidate_count)[:, None]
            )
            visible = (
                (offsets_m[:, None] < seq_q)
                & (offsets_n_local[None, :] < tile_len)
                & (offsets_n[None, :] < seq_k)
                & (offsets_n[None, :] <= query_position[:, None])
            )
            relation = tl.where(
                visible, tl.maximum(relation, log_floor) + libdevice.log1p(libdevice.exp(-tl.abs(relation - log_floor))), log_floor
            )
            # A single visible key has probability exactly one. Avoid FMA/LSE
            # cancellation changing the sign of log(1 + eps), which would alter
            # f3's positive count even though the mathematical value is positive.
            relation = tl.where(
                visible & (candidate_count[:, None] == 1),
                libdevice.log1p(libdevice.exp(log_floor)),
                relation,
            )
            relation_sum += relation
            if head_topk < num_groups:
                relation_min = tl.minimum(relation_min, relation)

        # For Qwen3-14B GQA=5, TopMean-4 is exactly (sum - min) / 4.
        # Keep only tile-sized accumulators; never expand a head/query/key tensor.
        if head_topk < num_groups:
            relation_sum -= relation_min
        output = relation_sum / head_topk
        output_offset = (
            key_head * tile_len * seq_q
            + offsets_n_local[None, :] * seq_q
            + offsets_m[:, None]
        )
        tl.store(
            output_ptr + output_offset,
            output,
            mask=(offsets_m[:, None] < seq_q)
            & (offsets_n_local[None, :] < tile_len),
        )


    @triton.jit
    def _basic_relation_stats_kernel(
        relation_ptr,
        f1_ptr,
        f3_ptr,
        f5_ptr,
        seq_q,
        tile_len,
        key_start,
        q_start,
        num_kv_heads: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        """Reduce f1/f3/f5 in one streaming read of a relation tile."""
        key_local = tl.program_id(0)
        key_position = key_start + key_local
        running_max = tl.full((), -float("inf"), tl.float32)
        positive_sum = tl.zeros((), tl.float32)
        positive_count = tl.zeros((), tl.int32)
        positive_query_count = tl.zeros((), tl.int32)

        for query_block_start in tl.range(0, seq_q, BLOCK_Q):
            query_offset = query_block_start + tl.arange(0, BLOCK_Q)
            valid_query = query_offset < seq_q
            head_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
            for key_head in tl.static_range(0, num_kv_heads):
                offset = (
                    key_head * tile_len * seq_q
                    + key_local * seq_q
                    + query_offset
                )
                value = tl.load(
                    relation_ptr + offset,
                    mask=valid_query,
                    other=-float("inf"),
                )
                running_max = tl.maximum(
                    running_max, tl.max(value, axis=0)
                )
                positive = valid_query & (value > 0.0)
                positive_sum += tl.sum(
                    tl.where(positive, value, 0.0), axis=0
                )
                positive_count += tl.sum(positive.to(tl.int32), axis=0)
                head_max = tl.maximum(head_max, value)
            positive_query_count += tl.sum(
                (valid_query & (head_max > 0.0)).to(tl.int32), axis=0
            )

        valid_count = seq_q - tl.maximum(key_position - q_start, 0)
        valid_count = tl.maximum(1, tl.minimum(valid_count, seq_q))
        tl.store(f1_ptr + key_local, running_max)
        tl.store(
            f3_ptr + key_local,
            positive_sum / tl.maximum(positive_count, 1).to(tl.float32),
        )
        tl.store(
            f5_ptr + key_local,
            positive_query_count.to(tl.float32) / seq_q,
        )


    @triton.jit
    def _causal_advantage_kernel(
        logits_ptr,
        output_ptr,
        seq_q: tl.constexpr,
        seq_k: tl.constexpr,
        q_start,
        scale,
        log_floor,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        query_idx = row % seq_q
        query_pos = q_start + query_idx
        offsets = tl.arange(0, BLOCK)
        valid_key = offsets < seq_k
        visible = valid_key & (offsets <= query_pos)
        values = tl.load(
            logits_ptr + row * seq_k + offsets,
            mask=valid_key,
            other=-float("inf"),
        ).to(tl.float32)
        values = tl.where(visible, values * scale, -float("inf"))
        row_max = tl.max(values, axis=0)
        normalizer = tl.sum(libdevice.exp(values - row_max), axis=0)
        relation = (
            values
            - row_max
            - libdevice.log(normalizer)
            + libdevice.log(
                tl.minimum(query_pos + 1, seq_k).to(tl.float32)
            )
        )
        relation = tl.maximum(relation, log_floor) + libdevice.log1p(libdevice.exp(-tl.abs(relation - log_floor)))
        tl.store(
            output_ptr + row * seq_k + offsets,
            relation,
            mask=valid_key,
        )


def causal_attention_advantage(
    logits: torch.Tensor,
    *,
    q_start: int,
    scale: float,
    log_floor: float = math.log(1e-8),
) -> torch.Tensor:
    """Fuse causal masking, FP32 log-softmax, clamp and advantage shift."""
    if triton is None or not logits.is_cuda:
        raise RuntimeError("The fused CORE QK kernel requires CUDA Triton")
    if logits.ndim != 3 or not logits.is_contiguous():
        raise ValueError("Expected contiguous QK logits with shape (heads, q, k)")
    _, seq_q, seq_k = logits.shape
    block = triton.next_power_of_2(seq_k)
    output = torch.empty(logits.shape, device=logits.device, dtype=torch.float32)
    _causal_advantage_kernel[(logits.shape[0] * seq_q,)](
        logits,
        output,
        seq_q=seq_q,
        seq_k=seq_k,
        q_start=int(q_start),
        scale=float(scale),
        log_floor=float(log_floor),
        BLOCK=block,
        num_warps=8,
    )
    return output


def _round_mode(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0


def strict_causal_qk_features(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    num_kv_groups: int,
    topk_qk: int,
    n_recent_queries: int,
    row_lse: torch.Tensor | None = None,
    key_tile_size: int = 512,
    log_floor: float = math.log(1e-8),
) -> torch.Tensor:
    """Compute the six CORE QK features without a full relation tensor.

    When ``row_lse`` is absent, a first fused pass obtains causal row
    log-normalizers directly from Q/K. When FlashAttention already exposed its
    row LSE, that tensor is reused and the first QK pass is skipped. The
    remaining fused pass emits one GQA-reduced key tile at a time; the six
    statistics are finalized immediately and the tile is discarded.
    Supports averaging all mapped query heads and exact leave-one-out TopMean
    (including Qwen3-14B's Top-4 of 5). Scratch storage is bounded by the key
    tile width rather than the complete query-by-key relation matrix.
    Reusing external LSE can differ numerically from local normalization:
    the fused logits are rounded to the input dtype before normalization.
    """
    if triton is None or not query_states.is_cuda or not key_states.is_cuda:
        raise RuntimeError("Strict fused CORE QK reduction requires CUDA Triton")
    if query_states.ndim != 3 or key_states.ndim != 3:
        raise ValueError("Expected Q=(Hq,Q,D) and K=(Hkv,K,D)")
    n_q, seq_q, head_dim = query_states.shape
    n_kv, seq_k, key_dim = key_states.shape
    if key_dim != head_dim or n_q != n_kv * int(num_kv_groups):
        raise ValueError("Incompatible Q/K head geometry")
    if not supports_strict_head_topk(int(num_kv_groups), int(topk_qk)):
        raise ValueError(
            "Strict fused reduction requires positive Top-K >= GQA group size - 1"
        )
    if query_states.dtype != key_states.dtype:
        raise ValueError("Q and K must use the same dtype")

    q = query_states.contiguous()
    k = key_states.contiguous()
    q_start = max(0, seq_k - seq_q)
    scale = head_dim**-0.5
    round_mode = _round_mode(q.dtype)
    block_d = triton.next_power_of_2(head_dim)
    block_m = 32
    block_n = 64
    if row_lse is None:
        lse = torch.empty((n_q, seq_q), device=q.device, dtype=torch.float32)
        _causal_row_lse_kernel[(triton.cdiv(seq_q, block_m), n_q)](
            q,
            k,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            seq_q,
            seq_k,
            q_start,
            scale,
            num_groups=int(num_kv_groups),
            head_dim=head_dim,
            ROUND_MODE=round_mode,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )
    else:
        if row_lse.ndim == 3:
            if row_lse.shape[0] != 1:
                raise ValueError("Batched row_lse must have batch size one")
            row_lse = row_lse[0]
        if tuple(row_lse.shape) != (n_q, seq_q):
            raise ValueError(
                "row_lse must have shape "
                f"({n_q}, {seq_q}), got {tuple(row_lse.shape)}"
            )
        if row_lse.device != q.device:
            raise ValueError("row_lse must be on the same device as Q/K")
        lse = row_lse.to(dtype=torch.float32).contiguous()

    features = torch.empty((seq_k, 6), device=q.device, dtype=torch.float32)
    tile_capacity = max(1, int(key_tile_size))
    n_recent = min(int(n_recent_queries), seq_q)
    neg = torch.finfo(torch.float32).min

    for key_start in range(0, seq_k, tile_capacity):
        tile_len = min(tile_capacity, seq_k - key_start)
        relation = torch.full(
            (n_kv, tile_len, seq_q),
            float(log_floor),
            device=q.device,
            dtype=torch.float32,
        )
        _grouped_relation_tile_kernel[
            (
                triton.cdiv(seq_q, block_m),
                triton.cdiv(tile_len, block_n),
                n_kv,
            )
        ](
            q,
            k,
            lse,
            relation,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            seq_q,
            seq_k,
            key_start,
            tile_len,
            q_start,
            scale,
            float(log_floor),
            num_groups=int(num_kv_groups),
            head_topk=min(int(topk_qk), int(num_kv_groups)),
            head_dim=head_dim,
            ROUND_MODE=round_mode,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )

        key_pos = torch.arange(
            key_start, key_start + tile_len, device=q.device
        )
        valid_per_token = (
            seq_q - (key_pos - q_start).clamp_min(0)
        ).clamp(min=1, max=seq_q)
        f1 = torch.empty(tile_len, device=q.device, dtype=torch.float32)
        f3 = torch.empty_like(f1)
        f5 = torch.empty_like(f1)
        _basic_relation_stats_kernel[(tile_len,)](
            relation,
            f1,
            f3,
            f5,
            seq_q,
            tile_len,
            key_start,
            q_start,
            num_kv_heads=n_kv,
            BLOCK_Q=1024,
            num_warps=4,
        )
        k2_per_token = (valid_per_token * n_kv).clamp(
            min=1, max=min(int(topk_qk), seq_q * n_kv)
        )
        k2_max = min(int(topk_qk), seq_q * n_kv)
        k2_per_head = min(k2_max, seq_q)
        f2_candidates = relation.topk(
            k2_per_head, dim=2, sorted=False
        ).values
        f2_top = (
            f2_candidates.permute(1, 0, 2)
            .reshape(tile_len, n_kv * k2_per_head)
            .topk(k2_max, dim=1)
            .values
        )
        f2 = f2_top.cumsum(dim=1).gather(
            1, (k2_per_token - 1).unsqueeze(1)
        ).squeeze(1) / k2_per_token

        recent = relation[:, :, -n_recent:]
        recent_start = q_start + seq_q - n_recent
        recent_valid = (
            q_start + seq_q
            - torch.maximum(
                key_pos, torch.full_like(key_pos, recent_start)
            )
        ).clamp(min=1, max=n_recent)
        k4_per_token = (recent_valid * n_kv).clamp(
            min=1, max=min(int(topk_qk), n_recent * n_kv)
        )
        k4_max = min(int(topk_qk), n_recent * n_kv)
        k4_per_head = min(k4_max, n_recent)
        f4_candidates = recent.topk(
            k4_per_head, dim=2, sorted=False
        ).values
        f4_top = (
            f4_candidates.permute(1, 0, 2)
            .reshape(tile_len, n_kv * k4_per_head)
            .topk(k4_max, dim=1)
            .values
        )
        f4 = f4_top.cumsum(dim=1).gather(
            1, (k4_per_token - 1).unsqueeze(1)
        ).squeeze(1) / k4_per_token

        k_frac = torch.ceil(valid_per_token.float() * 0.25).long().clamp_min(1)
        k_frac_max = int(k_frac.max().item())
        f6_top = relation.topk(k_frac_max, dim=2).values
        f6_mean = f6_top.cumsum(dim=2).gather(
            2,
            (k_frac - 1).view(1, tile_len, 1).expand(n_kv, tile_len, 1),
        ).squeeze(2) / k_frac.view(1, tile_len)
        f6 = (f6_mean > 0).float().mean(dim=0)

        features[key_start : key_start + tile_len] = torch.stack(
            (f1, f2, f3, f4, f5, f6), dim=-1
        )
        del relation, recent

    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite strict fused CORE QK features")
    return features
