"""
Token-level feature extraction for CORE indexer.

Constructs 14-dimensional feature vectors per token:
  - 6 query-key relation features (f1-f6)
  - 4 coverage/redundancy features (o, d_proto, delta_proto, g)
  - 4 auxiliary features (age, sink, value_energy, phase)

All features are computed from frozen backbone forward pass statistics.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from core.config import COREConfig
from core.fused_qk import (
    causal_attention_advantage,
    strict_causal_qk_features,
    supports_strict_head_topk,
)

logger = logging.getLogger(__name__)


class LayerwiseKeyGeometryProjector(torch.nn.Module):
    """Fixed layer-specific Gaussian projection of head-preserving keys."""

    def __init__(self, n_layers, n_kv_heads, head_dim, output_dim, seed, eps=1e-8):
        super().__init__()
        if min(n_layers, n_kv_heads, head_dim, output_dim) <= 0:
            raise ValueError("Key-geometry dimensions must all be positive")
        self.n_layers = int(n_layers)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.output_dim = int(output_dim)
        self.eps = float(eps)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        weight = torch.randn(
            self.n_layers,
            self.output_dim,
            self.n_kv_heads * self.head_dim,
            generator=generator,
            dtype=torch.float32,
        ) / np.sqrt(self.output_dim)
        self.register_buffer("weight", weight, persistent=True)

    @torch.no_grad()
    def forward(self, key_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if key_states.ndim != 3:
            raise ValueError(
                "key_states must have shape (n_kv_heads, seq_len, head_dim), "
                f"got {tuple(key_states.shape)}"
            )
        if key_states.shape[0] != self.n_kv_heads or key_states.shape[2] != self.head_dim:
            raise ValueError(
                "Key geometry shape mismatch: expected "
                f"({self.n_kv_heads}, N, {self.head_dim}), got {tuple(key_states.shape)}"
            )
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx={layer_idx} outside [0, {self.n_layers})")
        keys_fp32 = key_states.float()
        per_head = keys_fp32 / (
            torch.linalg.vector_norm(keys_fp32, dim=-1, keepdim=True) + self.eps
        )
        k_bar = per_head.permute(1, 0, 2).reshape(
            key_states.shape[1], self.n_kv_heads * self.head_dim
        )
        projection = self.weight[layer_idx]
        z = F.linear(k_bar.to(projection.dtype), projection)
        z = z.float()
        return z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + self.eps)


class PrototypeBank:
    """
    Fixed prototype bank for coverage/redundancy features.

    Prototypes are initialized from random unit vectors and fixed throughout
    training (not learned).
    """

    def __init__(
        self,
        n_prototypes: int,
        dim: int,
        seed: int = 42,
        prototypes: torch.Tensor | None = None,
    ):
        if prototypes is None:
            rng = np.random.RandomState(seed)
            raw = rng.randn(n_prototypes, dim).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            prototypes = torch.from_numpy(raw / norms)  # (M, dim)
        else:
            prototypes = torch.as_tensor(prototypes, dtype=torch.float32).clone()
            if prototypes.ndim not in (2, 3):
                raise ValueError(
                    "prototypes must have shape (M, d) or (L, M, d), got "
                    f"{tuple(prototypes.shape)}"
                )
            if tuple(prototypes.shape[-2:]) != (n_prototypes, dim):
                raise ValueError(
                    "Prototype shape mismatch: expected trailing dimensions "
                    f"({n_prototypes}, {dim}), got {tuple(prototypes.shape)}"
                )
            prototypes = F.normalize(prototypes, dim=-1, eps=1e-8)
        self.prototypes = prototypes

    @property
    def is_per_layer(self) -> bool:
        return self.prototypes.ndim == 3

    def state_dict(self) -> dict:
        return {"prototypes": self.prototypes.detach().cpu()}

    def load_state_dict(self, state_dict: dict) -> None:
        prototypes = torch.as_tensor(state_dict["prototypes"], dtype=torch.float32)
        if prototypes.ndim not in (2, 3):
            raise ValueError("Invalid serialized prototype bank rank")
        self.prototypes = F.normalize(prototypes, dim=-1, eps=1e-8)

    def _for_layer(self, layer_idx: int | None) -> torch.Tensor:
        if not self.is_per_layer:
            return self.prototypes
        if layer_idx is None:
            raise ValueError("layer_idx is required for a per-layer prototype bank")
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < self.prototypes.shape[0]:
            raise IndexError(
                f"layer_idx={layer_idx} outside [0, {self.prototypes.shape[0]})"
            )
        return self.prototypes[layer_idx]

    def to(self, device: torch.device, dtype: torch.dtype) -> "PrototypeBank":
        self.prototypes = self.prototypes.to(device=device, dtype=dtype)
        return self

    @torch.no_grad()
    def get_coverage_features(
        self,
        z: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute coverage/redundancy features from token representations.

        Parameters
        ----------
        z : torch.Tensor
            Normalized token representations of shape (N, d_z).
        valid_mask : torch.Tensor, optional
            Tokens whose centered representation has a reliable direction.
            Invalid (near-centroid) tokens are excluded from occupancy counts
            and receive neutral prototype features.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - o: occupancy feature, shape (N,)
            - d_proto: distance to nearest prototype, shape (N,)
            - delta_proto: margin between nearest and second-nearest, shape (N,)
            - Prototype assignment counts n_m (useful externally)
        """
        prototypes = self._for_layer(layer_idx)
        # Cosine similarity: (N, M) — z and prototypes are both normalized
        cos_sim = F.linear(z, prototypes)  # (N, M)

        # Nearest prototype
        m1_idx = cos_sim.argmax(dim=-1)  # (N,)
        cos_m1 = cos_sim.gather(-1, m1_idx.unsqueeze(-1)).squeeze(-1)  # (N,)

        # Second nearest prototype
        cos_copy = cos_sim.clone()
        cos_copy.scatter_(-1, m1_idx.unsqueeze(-1), -1e9)
        cos_m2 = cos_copy.max(dim=-1).values  # (N,)

        if valid_mask is None:
            valid_mask = torch.ones(z.shape[0], device=z.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=z.device, dtype=torch.bool)
            if valid_mask.shape != (z.shape[0],):
                raise ValueError(
                    f"valid_mask must have shape ({z.shape[0]},), "
                    f"got {tuple(valid_mask.shape)}"
                )

        # Prototype occupancy counts. Near-centroid tokens have negligible
        # centered covariance and no stable direction, so they must not create
        # an artificial pile-up in prototype zero.
        n_m = torch.zeros(prototypes.shape[0], device=z.device, dtype=torch.long)
        n_m.scatter_add_(0, m1_idx, valid_mask.long())

        # Features
        o_i = torch.log1p(n_m[m1_idx].float())  # occupancy
        d_proto = 1.0 - cos_m1  # distance to nearest prototype
        delta_proto = cos_m1 - cos_m2  # margin between nearest and second-nearest

        # Neutral, deterministic values for representations at the centroid.
        o_i = torch.where(valid_mask, o_i, torch.zeros_like(o_i))
        d_proto = torch.where(valid_mask, d_proto, torch.ones_like(d_proto))
        delta_proto = torch.where(
            valid_mask, delta_proto, torch.zeros_like(delta_proto)
        )

        return o_i, d_proto, delta_proto, n_m


class HadamardProjector:
    """
    Fixed random Hadamard rotation for efficient distribution statistics.

    Generates R = (1/sqrt(d)) * H * D where H is Hadamard matrix, D is
    random sign diagonal. Used once and fixed.
    """

    def __init__(self, dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Random sign diagonal
        signs = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
        self.D = torch.diag(torch.from_numpy(signs))
        # Construct the Walsh-Hadamard matrix recursively.
        # For dimensions that aren't power of 2, we use a subset
        h_dim = self._next_power_of_2(dim)
        H = torch.tensor(self._hadamard(h_dim), dtype=torch.float32)
        H = H[:dim, :dim]  # Truncate to dim
        # R = H D / sqrt(d); apply the normalization once.
        self.H = H
        self.R = (1.0 / np.sqrt(dim)) * self.H @ self.D

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        p = 1
        while p < n:
            p *= 2
        return p

    @staticmethod
    def _hadamard(n: int) -> np.ndarray:
        """Generate Hadamard matrix of order n (n must be power of 2)."""
        H = np.array([[1]], dtype=np.float64)
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H

    def to(self, device: torch.device, dtype: torch.dtype) -> "HadamardProjector":
        self.R = self.R.to(device=device, dtype=dtype)
        self.D = self.D.to(device=device, dtype=dtype)
        return self

    @torch.no_grad()
    def rotate(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard rotation: z_tilde = R @ z. Shape: (N, d_z) -> (N, d_z)."""
        return z.float() @ self.R.float().T


class COREFeatureExtractor:
    """
    Extracts 14-dimensional token features for CORE indexer.

    Requires collected per-layer statistics from backbone forward pass:
      - query_states: (n_q_heads, seq_len, head_dim) per layer
      - key_states: (n_kv_heads, seq_len, head_dim) per layer
      - value_states: (n_kv_heads, seq_len, head_dim) per layer
      - attention_weights: optional legacy fallback for relation features

    The default path extracts features independently for each layer.
    The legacy extract_features method also supports pooled statistics.
    """

    def __init__(self, config: COREConfig):
        self.config = config
        self.n_q_heads = config.n_q_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.n_q_heads // self.n_kv_heads
        self.n_sink = config.n_sink
        self.topk_qk = config.topk_qk
        self.qk_query_block_size = max(1, int(config.qk_query_block_size))
        self.qk_primary_mode = getattr(
            config, "qk_primary_mode", "attention_advantage"
        )
        if self.qk_primary_mode not in (
            "attention_advantage", "raw_logit_max"
        ):
            raise ValueError(
                f"Unknown qk_primary_mode={self.qk_primary_mode!r}"
            )
        self.n_recent = config.n_recent_queries
        self.shrinkage_rho = config.shrinkage_rho
        self.dtype = getattr(torch, config.dtype, torch.bfloat16)

        self.key_geometry_mode = getattr(
            config, "key_geometry_mode", "head_concat_layer_gaussian"
        )
        self.key_geometry_eps = float(getattr(config, "key_geometry_eps", 1e-8))
        if self.key_geometry_mode == "head_concat_layer_gaussian":
            self.key_geometry = LayerwiseKeyGeometryProjector(
                int(config.n_layers), int(self.n_kv_heads), int(self.head_dim),
                int(config.proto_dim), int(config.geometry_seed), self.key_geometry_eps,
            )
            self.key_proj = None
        elif self.key_geometry_mode == "legacy_head_mean":
            self.key_proj = torch.nn.Linear(self.head_dim, config.proto_dim, bias=False)
            self.key_proj.requires_grad_(False)
            self.key_geometry = None
        else:
            raise ValueError(f"Unknown key_geometry_mode={self.key_geometry_mode!r}")

        # Fixed data-independent directional sketch used by CORE. Its state is
        # saved with the checkpoint so training and inference use the same bank.
        self.prototype_bank = PrototypeBank(
            config.n_prototypes, config.proto_dim,
            seed=config.geometry_seed,
        )

        # Hadamard projector
        self.hadamard = HadamardProjector(config.proto_dim, seed=config.geometry_seed)

    def to(self, device: torch.device) -> "COREFeatureExtractor":
        if self.key_geometry is not None:
            # Geometry is tiny (~8 MiB for 32x64x1024) relative to the model;
            # retain the sampled fp32 matrix rather than quantizing the fixed
            # space differently on CPU and GPU.
            self.prototype_bank.to(device, torch.float32)
            self.hadamard.to(device, torch.float32)
            self.key_geometry.to(device=device)
        else:
            # Preserve checkpoint prototype dtype.
            self.prototype_bank.to(device, self.dtype)
            self.hadamard.to(device, self.dtype)
            self.key_proj.to(device=device, dtype=self.dtype)
        return self

    @property
    def geometry_device(self) -> torch.device:
        if self.key_geometry is not None:
            return self.key_geometry.weight.device
        return self.key_proj.weight.device

    def geometry_state_dict(self) -> dict:
        module = self.key_geometry if self.key_geometry is not None else self.key_proj
        return {
            "format_version": 3,
            "hadamard_rotation": self.hadamard.R.detach().clone(),
            "geometry_state_dict": module.state_dict(),
            "prototype_bank_state_dict": self.prototype_bank.state_dict(),
        }

    def load_geometry_state_dict(self, state_dict: dict) -> None:
        module = self.key_geometry if self.key_geometry is not None else self.key_proj
        if "geometry_state_dict" in state_dict:
            module.load_state_dict(state_dict["geometry_state_dict"])
            if "hadamard_rotation" in state_dict:
                self.hadamard.R = state_dict["hadamard_rotation"].to(
                    device=self.geometry_device, dtype=torch.float32
                ).clone()
            if "prototype_bank_state_dict" in state_dict:
                self.prototype_bank.load_state_dict(
                    state_dict["prototype_bank_state_dict"]
                )
                self.prototype_bank.to(self.geometry_device, torch.float32)
        else:
            # Load a geometry-only checkpoint.
            module.load_state_dict(state_dict)

    @torch.no_grad()
    def project_keys(self, key_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Canonical normalized z shared by teacher, prototypes and inference."""
        if self.key_geometry is not None:
            return self.key_geometry(key_states, layer_idx)
        k_pooled = key_states.mean(dim=0).to(dtype=self.key_proj.weight.dtype)
        return F.normalize(
            self.key_proj(k_pooled), dim=-1, eps=self.key_geometry_eps
        )

    @torch.no_grad()
    def compute_qk_relation_features(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        softmax_lse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the six causal QK features with query-block streaming.

        Reduces query blocks without retaining logits for all queries.
        Online Top-K state supplies f2/f4/f6 in the 14-D feature schema.
        Floating-point results can differ across dense, streaming, and fused
        paths, including when externally computed softmax_lse is reused.
        """
        n_q, seq_q, d = query_states.shape
        n_kv, seq_k, _ = key_states.shape
        if n_q != n_kv * self.num_kv_groups:
            raise ValueError(
                f"Q/K head mismatch: {n_q} query heads, {n_kv} KV heads, "
                f"group size {self.num_kv_groups}"
            )

        # Formal CUDA path: obtain causal row normalizers directly from Q/K,
        # then reduce one key tile at a time. This never materializes the
        # complete FP32 [H_q, Q, K] relation tensor. The CPU and non-standard
        # head-TopK cases retain the streaming reference below for portability.
        if (
            query_states.is_cuda
            and key_states.is_cuda
            and self.qk_primary_mode == "attention_advantage"
            and supports_strict_head_topk(self.num_kv_groups, self.topk_qk)
        ):
            return strict_causal_qk_features(
                query_states,
                key_states,
                num_kv_groups=self.num_kv_groups,
                topk_qk=self.topk_qk,
                n_recent_queries=self.n_recent,
                row_lse=softmax_lse,
            )

        query_fp32 = query_states
        keys_expanded = repeat_kv(
            key_states.unsqueeze(0), self.num_kv_groups
        ).squeeze(0)
        keys_t_fp32 = keys_expanded.transpose(-1, -2)
        q_start = max(0, seq_k - seq_q)
        device = query_states.device
        key_pos = torch.arange(seq_k, device=device)
        neg = torch.finfo(torch.float32).min
        n_candidates = seq_k

        # Number of causal observations per key without a persistent QxK mask.
        valid_per_token = (
            seq_q - (key_pos - q_start).clamp_min(0)
        ).clamp(min=1, max=seq_q)
        n_recent = min(self.n_recent, seq_q)
        recent_start = q_start + seq_q - n_recent
        recent_valid = (
            q_start + seq_q
            - torch.maximum(key_pos, torch.full_like(key_pos, recent_start))
        ).clamp(min=1, max=n_recent)

        k2_per_token = (valid_per_token * n_kv).clamp(
            min=1, max=min(self.topk_qk, seq_q * n_kv)
        )
        k2_max = min(self.topk_qk, seq_q * n_kv)
        k4_per_token = (recent_valid * n_kv).clamp(
            min=1, max=min(self.topk_qk, n_recent * n_kv)
        )
        k4_max = min(self.topk_qk, n_recent * n_kv)
        k_frac = torch.ceil(valid_per_token.float() * 0.25).long().clamp_min(1)
        k_frac_max = (seq_q + 3) // 4

        f1 = torch.full(
            (n_candidates,), neg, device=device, dtype=torch.float32
        )
        raw_f1 = torch.full_like(f1, neg)
        f2_top = torch.full(
            (k2_max, n_candidates), neg, device=device, dtype=torch.float32
        )
        f4_top = torch.full(
            (k4_max, n_candidates), neg, device=device, dtype=torch.float32
        )
        f6_top = torch.full(
            (n_kv, k_frac_max, n_candidates),
            neg,
            device=device,
            dtype=torch.float32,
        )
        positive_sum = torch.zeros(
            n_candidates, device=device, dtype=torch.float32
        )
        positive_count = torch.zeros(
            n_candidates, device=device, dtype=torch.long
        )
        f5_count = torch.zeros(n_candidates, device=device, dtype=torch.long)

        head_topk = min(self.topk_qk, self.num_kv_groups)
        block_size = max(1, self.qk_query_block_size)
        scale = d**-0.5

        for block_start in range(0, seq_q, block_size):
            block_end = min(seq_q, block_start + block_size)
            block_len = block_end - block_start
            query_pos = torch.arange(
                q_start + block_start, q_start + block_end, device=device
            )
            visible = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)

            # Only this query block's logits are live. Softmax still spans all
            # keys and therefore preserves the exact causal probabilities.
            logits = torch.matmul(
                query_fp32[:, block_start:block_end], keys_t_fp32
            )
            if logits.is_cuda and self.qk_primary_mode == "attention_advantage":
                relation = causal_attention_advantage(
                    logits.contiguous(),
                    q_start=q_start + block_start,
                    scale=scale,
                    log_floor=float(np.log(1e-8)),
                )
            else:
                logits = logits.float() * scale
                logits.masked_fill_(~visible.unsqueeze(0), neg)
                log_probabilities = F.log_softmax(logits, dim=-1)
                candidate_count = (query_pos + 1).clamp(max=seq_k).to(
                    log_probabilities.dtype
                )
                relation = log_probabilities
                relation = torch.logaddexp(
                    relation + torch.log(candidate_count).view(1, block_len, 1),
                    relation.new_tensor(float(np.log(1e-8))),
                )
            if self.qk_primary_mode == "raw_logit_max":
                raw_grouped = logits.view(
                    n_kv, self.num_kv_groups, block_len, n_candidates
                )
                if head_topk == self.num_kv_groups:
                    raw_reduced = raw_grouped.mean(dim=1)
                else:
                    raw_reduced = raw_grouped.topk(
                        head_topk, dim=1
                    ).values.mean(dim=1)
                raw_reduced.masked_fill_(~visible.unsqueeze(0), neg)
                raw_f1 = torch.maximum(
                    raw_f1, raw_reduced.amax(dim=(0, 1))
                )
            grouped = relation.view(
                n_kv, self.num_kv_groups, block_len, n_candidates
            )
            if head_topk == self.num_kv_groups:
                reduced = grouped.mean(dim=1)
            else:
                reduced = grouped.topk(head_topk, dim=1).values.mean(dim=1)
            # The clamped log-probability of an invisible entry is finite and
            # no greater than a valid entry at the clamp floor. Every token has
            # enough valid KV-group observations for the configured Top-K, so
            # keeping this finite floor avoids two full-tensor mask passes
            # without changing any of the six reductions.
            max_over_m = reduced.max(dim=0).values
            f1 = torch.maximum(f1, max_over_m.amax(dim=0))

            # Reduction order is irrelevant for Top-K. Keeping the native
            # (KV-group, query, key) layout avoids a 512 MiB permute copy at 4K.
            flattened = reduced.reshape(-1, n_candidates)
            block_f2 = flattened.topk(
                min(k2_max, flattened.shape[0]), dim=0
            ).values
            if block_start == 0 and block_end == seq_q:
                f2_top = block_f2
            else:
                f2_top = torch.cat((f2_top, block_f2), dim=0).topk(
                    k2_max, dim=0
                ).values

            positive = flattened > 0
            positive_sum += torch.where(
                positive, flattened, torch.zeros_like(flattened)
            ).sum(dim=0)
            positive_count += positive.sum(dim=0)

            recent_offset = max(0, seq_q - n_recent - block_start)
            if recent_offset < block_len:
                recent_flat = reduced[:, recent_offset:, :].reshape(
                    -1, n_candidates
                )
                block_f4 = recent_flat.topk(
                    min(k4_max, recent_flat.shape[0]), dim=0
                ).values
                if block_start == 0 and block_end == seq_q:
                    f4_top = block_f4
                else:
                    f4_top = torch.cat((f4_top, block_f4), dim=0).topk(
                        k4_max, dim=0
                    ).values

            f5_count += (
                (max_over_m > 0) & visible
            ).sum(dim=0)

            # A final query block may be shorter than the global fractional
            # Top-K (for example, a 4K block plus a short remainder).  Keep all
            # available observations from that block, then merge and truncate
            # to the global K.  The union of per-block Top-K sets contains the
            # exact global Top-K, so this is mathematically identical to
            # materialising every query observation at once.
            block_f6 = reduced.topk(
                min(k_frac_max, reduced.shape[1]), dim=1
            ).values
            if block_start == 0 and block_end == seq_q:
                f6_top = block_f6
            else:
                f6_top = torch.cat((f6_top, block_f6), dim=1).topk(
                    k_frac_max, dim=1
                ).values

            del logits, relation, grouped, reduced, flattened

        f2_cumsum = f2_top.cumsum(dim=0)
        f2 = f2_cumsum.gather(
            0, (k2_per_token - 1).unsqueeze(0)
        ).squeeze(0) / k2_per_token
        f3 = positive_sum / positive_count.clamp_min(1)
        f4_cumsum = f4_top.cumsum(dim=0)
        f4 = f4_cumsum.gather(
            0, (k4_per_token - 1).unsqueeze(0)
        ).squeeze(0) / k4_per_token
        f5 = f5_count.float() / seq_q
        f6_cumsum = f6_top.cumsum(dim=1)
        gather_idx = (k_frac - 1).view(1, 1, n_candidates).expand(
            n_kv, 1, n_candidates
        )
        top_frac_mean = (
            f6_cumsum.gather(1, gather_idx).squeeze(1)
            / k_frac.view(1, n_candidates)
        )
        f6 = (top_frac_mean > 0).float().mean(dim=0)

        if self.qk_primary_mode == "raw_logit_max":
            f1 = raw_f1
        features = torch.stack([f1, f2, f3, f4, f5, f6], dim=-1)
        if not torch.isfinite(features).all():
            bad_by_column = (~torch.isfinite(features)).sum(dim=0).tolist()
            raise FloatingPointError(
                "Non-finite streaming causal QK features by column "
                f"[f1..f6]: {bad_by_column}"
            )
        return features

    @torch.no_grad()
    def _compute_qk_relation_features_dense_reference(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 6 query-key relation features per candidate token.

        The relation is the attention advantage,
        ``r(t,i)=log p(i|t)+log|C_t|``.  Candidate sets are causal and query
        heads are grouped by their shared KV head before the six statistics are
        reduced.  Training and inference both call this implementation.

        Parameters
        ----------
        query_states : torch.Tensor
            Shape (n_q_heads, seq_len, head_dim) — RoPE-applied queries from prefill.
        key_states : torch.Tensor
            Shape (n_kv_heads, seq_len, head_dim) — RoPE-applied keys.

        Returns
        -------
        torch.Tensor
            Shape (N_candidates, 6) — q-k relation features.
        """
        n_q, seq_q, d = query_states.shape
        n_kv, seq_k, _ = key_states.shape

        # Repeat KV to match query heads and compute true logits.
        keys_expanded = repeat_kv(key_states.unsqueeze(0), self.num_kv_groups).squeeze(0)
        logits = torch.matmul(query_states.float(), keys_expanded.float().transpose(-1, -2)) / (d**0.5)

        # Causal candidate set C_t. For decode buffers (seq_q < seq_k), queries
        # correspond to the final seq_q positions in the current cache.
        q_start = max(0, seq_k - seq_q)
        query_pos = torch.arange(q_start, q_start + seq_q, device=logits.device)
        key_pos = torch.arange(seq_k, device=logits.device)
        visible = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)  # (seq_q, seq_k)
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~visible.unsqueeze(0), neg)

        raw_f1 = None
        if self.qk_primary_mode == "raw_logit_max":
            raw_grouped = logits.view(
                n_kv, self.num_kv_groups, seq_q, seq_k
            )
            raw_topk = min(self.topk_qk, self.num_kv_groups)
            raw_reduced = (
                raw_grouped.mean(dim=1)
                if raw_topk == self.num_kv_groups
                else raw_grouped.topk(raw_topk, dim=1).values.mean(dim=1)
            )
            raw_f1 = raw_reduced.masked_fill(
                ~visible.unsqueeze(0), neg
            ).amax(dim=(0, 1))

        # Attention advantage: r(t,h,i) = log p(t,h,i) + log|C_t|.
        p_attn = F.softmax(logits, dim=-1)
        candidate_count = visible.sum(dim=-1).clamp_min(1).to(p_attn.dtype)
        r = torch.logaddexp(
            F.log_softmax(logits, dim=-1) + torch.log(candidate_count).view(1, seq_q, 1),
            logits.new_tensor(float(np.log(1e-8))),
        )
        r = r.masked_fill(~visible.unsqueeze(0), neg)

        # Aggregate to KV groups: TopMean over query heads in each group
        # Group mapping: heads 0..num_kv_groups-1 -> kv_group 0, etc.
        r_grouped = r.view(n_kv, self.num_kv_groups, seq_q, seq_k)
        # TopMean: mean of top-K values along dim=1 (query heads within group)
        topk_vals = r_grouped.topk(
            min(self.topk_qk, self.num_kv_groups), dim=1
        ).values
        R_group = topk_vals.mean(dim=1)  # (n_kv, seq_q, seq_k)
        # For causally invisible positions every selected head contains
        # float32.min. torch.mean sums before dividing, so four such values
        # overflow to -inf. Later, f3 multiplies them by a zero mask and
        # -inf*0 becomes NaN (exactly N-1 bad rows for a length-N prefill).
        # Restore the intended finite sentinel after head aggregation.
        R_group = R_group.masked_fill(~visible.unsqueeze(0), neg)
        # R_{t,m,i}: first dim is kv_group (m), second is query token (t), third is candidate token (i)

        # Candidate tokens: all positions (seq_len)
        # Now compute 6 features per candidate token i
        N = seq_k

        # f1: max over (t, m) of R_{t,m,i}
        f1 = R_group.max(dim=0).values.max(dim=0).values  # (N,)
        if raw_f1 is not None:
            f1 = raw_f1

        # f2: MeanTopK over (t, m) of R_{t,m,i}
        # Flatten (t, m) to single dim
        R_flat = R_group.permute(1, 0, 2).reshape(-1, N)
        # Near the causal diagonal a candidate can have fewer than K valid
        # (query, KV-group) pairs. A fixed Top-K then pulls in mask sentinels;
        # for the last token in a 2-KV-head test, Top-4 contained two
        # float32.min values and overflowed during mean. Use a per-token K.
        valid_per_token = visible.sum(dim=0).clamp_min(1)
        k2_per_token = (valid_per_token * n_kv).clamp(
            min=1, max=min(self.topk_qk, R_flat.shape[0])
        )
        k2_max = int(k2_per_token.max().item())
        f2_top = R_flat.topk(k2_max, dim=0).values
        f2_cumsum = f2_top.cumsum(dim=0)
        f2 = f2_cumsum.gather(
            0, (k2_per_token - 1).unsqueeze(0)
        ).squeeze(0) / k2_per_token

        # f3: Mean over (t, m) where R_{t,m,i} > 0
        positive_mask = (R_flat > 0).float()
        positive_values = torch.where(
            positive_mask.bool(), R_flat, torch.zeros_like(R_flat)
        )
        f3 = positive_values.sum(dim=0) / positive_mask.sum(dim=0).clamp(min=1)

        # f4: MeanTopK over recent queries only
        n_recent = min(self.n_recent, seq_q)
        R_recent = R_group[:, -n_recent:, :]  # (n_kv, n_recent, N)
        R_recent_flat = R_recent.permute(1, 0, 2).reshape(-1, N)
        recent_valid = visible[-n_recent:].sum(dim=0).clamp_min(1)
        k4_per_token = (recent_valid * n_kv).clamp(
            min=1, max=min(self.topk_qk, R_recent_flat.shape[0])
        )
        k4_max = int(k4_per_token.max().item())
        f4_top = R_recent_flat.topk(k4_max, dim=0).values
        f4_cumsum = f4_top.cumsum(dim=0)
        f4 = f4_cumsum.gather(
            0, (k4_per_token - 1).unsqueeze(0)
        ).squeeze(0) / k4_per_token

        # f5: fraction of query tokens t where max_m R_{t,m,i} > 0
        max_over_m = R_group.max(dim=0).values  # (seq_q, N)
        f5 = ((max_over_m > 0) & visible).float().sum(dim=0) / seq_q

        # f6: fraction of KV groups m where TopMean_t R_{t,m,i} > 0
        # For each KV group, TopFracMean over the visible queries (top 25%).
        # A variable k per token is implemented with one topk + cumulative sum.
        k_frac = torch.ceil(valid_per_token.float() * 0.25).long().clamp_min(1)
        k_max = int(k_frac.max().item())
        top_vals = R_group.masked_fill(~visible.unsqueeze(0), neg).topk(k_max, dim=1).values
        top_cumsum = top_vals.cumsum(dim=1)
        gather_idx = (k_frac - 1).view(1, 1, N).expand(n_kv, 1, N)
        top_frac_mean = top_cumsum.gather(1, gather_idx).squeeze(1) / k_frac.view(1, N)
        f6 = (top_frac_mean > 0).float().mean(dim=0)

        features = torch.stack([f1, f2, f3, f4, f5, f6], dim=-1)  # (N, 6)
        if not torch.isfinite(features).all():
            bad_by_column = (~torch.isfinite(features)).sum(dim=0).tolist()
            raise FloatingPointError(
                "Non-finite causal QK features by column "
                f"[f1..f6]: {bad_by_column}"
            )
        return features

    @torch.no_grad()
    def compute_coverage_features(
        self,
        key_states: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Compute 4 coverage/redundancy features per candidate token.

        Parameters
        ----------
        key_states : torch.Tensor
            Shape (n_kv_heads, seq_len, head_dim).

        Returns
        -------
        torch.Tensor
            Shape (N_candidates, 4) — coverage features.
        """
        z_norm = self.project_keys(key_states, layer_idx)

        if getattr(self.config, "center_prototype_features", True):
            # Coverage-teacher coordinate system used by the centered variant.
            z_centered = z_norm - z_norm.mean(dim=0, keepdim=True)
            centered_norm = torch.linalg.vector_norm(
                z_centered, dim=-1, keepdim=True
            )
            reliable_direction = centered_norm.squeeze(-1) > 1e-6
            z_proto = z_centered / centered_norm.clamp_min(1e-6)
        else:
            # Use uncentered prototype coordinates when centering is disabled.
            z_proto = z_norm
            reliable_direction = None

        # Prototype features
        o_i, d_proto, delta_proto, n_m = self.prototype_bank.get_coverage_features(
            z_proto, valid_mask=reliable_direction, layer_idx=layer_idx
        )
        # o_i: (N,), d_proto: (N,), delta_proto: (N,)

        # Global deviation via Hadamard rotation
        z_tilde = self.hadamard.rotate(z_norm.float())  # (N, d_z)
        d_z = z_tilde.shape[-1]

        # First moment
        mu = z_tilde.mean(dim=0)  # (d_z,)
        # Variance per dimension
        # E[z^2] - E[z]^2 can be a tiny negative number from fp32
        # cancellation, especially when many projected keys are nearly
        # identical. Feeding that value to the standardized deviation below
        # makes log1p receive an argument below -1 and poisons the indexer with
        # NaNs. Variance is non-negative by definition.
        sigma2 = ((z_tilde**2).mean(dim=0) - mu**2).clamp_min(0.0)  # (d_z,)
        # Shrinkage toward global average variance
        avg_var = sigma2.mean()
        sigma2_eff = (
            (1 - self.shrinkage_rho) * sigma2
            + self.shrinkage_rho * avg_var
        ) + 1e-8

        # Mahalanobis-like deviation
        deviations = (z_tilde - mu.unsqueeze(0))**2  # (N, d_z)
        g = torch.log1p(
            (deviations / sigma2_eff.unsqueeze(0)).mean(dim=-1)
        )  # (N,)

        features = torch.stack([o_i, d_proto, delta_proto, g], dim=-1)  # (N, 4)
        if not torch.isfinite(features).all():
            raise FloatingPointError("Non-finite CORE coverage features")
        return features

    @torch.no_grad()
    def compute_auxiliary_features(
        self,
        value_states: torch.Tensor,
        seq_len: int,
        is_prefill: bool = True,
        token_ids: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute 4 auxiliary features per candidate token.

        Parameters
        ----------
        value_states : torch.Tensor
            Shape (n_kv_heads, seq_len, head_dim).
        seq_len : int
            Total sequence length (for computing relative age).
        is_prefill : bool
            Whether we are in prefill or decode phase.

        Returns
        -------
        torch.Tensor
            Shape (N_candidates, 4) — auxiliary features.
        """
        N = value_states.shape[1]
        positions = (torch.arange(N, device=value_states.device)
                     if token_positions is None else token_positions.to(value_states.device))
        if positions.shape != (N,):
            raise ValueError("token_positions must match the retained KV axis")
        if (positions < 0).any() or (positions >= seq_len).any():
            raise ValueError("Token positions must be within the original sequence length")

        # a_age: relative position (log-scaled)
        age_denom = torch.log1p(
            torch.tensor(float(max(seq_len - 1, 1)), device=value_states.device)
        )
        a_age = torch.log1p(seq_len - 1 - positions.float()) / age_denom

        # a_sink: sink token OR tokenizer special token indicator.
        a_sink_mask = positions < self.n_sink
        if token_ids is not None and self.config.special_token_ids:
            ids = token_ids.reshape(-1).to(value_states.device)
            if ids.numel() < N:
                ids = F.pad(ids, (0, N - ids.numel()), value=-1)
            ids = ids[:N]
            special = torch.zeros(N, dtype=torch.bool, device=value_states.device)
            for special_id in self.config.special_token_ids:
                special |= ids == int(special_id)
            a_sink_mask = a_sink_mask | special
        a_sink = a_sink_mask.float()

        # a_value: log(1 + ||v||^2 / d)
        v_norms = (value_states**2).sum(dim=-1).mean(dim=0)  # mean over kv_heads -> (N,)
        a_value = torch.log1p(v_norms / self.head_dim)

        # a_phase is one during decoding and zero during prefill.
        a_phase = torch.full((N,), 0.0 if is_prefill else 1.0, device=value_states.device)

        features = torch.stack([a_age, a_sink, a_value, a_phase], dim=-1)  # (N, 4)
        return features

    @torch.no_grad()
    def compute_qk_relation_features_from_attention(
        self,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 6 query-key relation features from **observed** attention weights.

        This avoids the expensive Q@K^T matmul by reusing attention weights
        already computed by the model's attention layer.  The result is
        equivalent to ``compute_qk_relation_features`` up to numerical
        precision (the model applies the same softmax, and the log-ratio
        transformation follows identically).

        This is the inference fallback when post-RoPE Q/K tensors cannot be
        recovered.  It applies the same causal candidate sets and attention-
        advantage reduction as ``compute_qk_relation_features``.

        Parameters
        ----------
        attention_weights : torch.Tensor
            Observed attention weights of shape ``(n_q_heads, seq_len, seq_len)``.
            Typically ``output[1]`` from the forward hook.

        Returns
        -------
        torch.Tensor
            Shape ``(N_candidates, 6)`` — q-k relation features.
        """
        if self.qk_primary_mode == "raw_logit_max":
            raise RuntimeError(
                "raw_logit_max cannot be reconstructed from probabilities; "
                "post-RoPE query/key states are required"
            )
        # p_attn is already softmax-applied attention. Re-apply the causal
        # candidate mask because some eager backends return tiny non-zero values
        # outside the valid set after dtype conversion.
        p_attn = attention_weights.float()
        n_q, seq_q, seq_k = p_attn.shape
        n_kv = n_q // self.num_kv_groups
        q_start = max(0, seq_k - seq_q)
        query_pos = torch.arange(q_start, q_start + seq_q, device=p_attn.device)
        key_pos = torch.arange(seq_k, device=p_attn.device)
        visible = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        p_attn = p_attn.masked_fill(~visible.unsqueeze(0), 0.0)
        p_attn = p_attn / p_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        candidate_count = visible.sum(dim=-1).clamp_min(1).to(p_attn.dtype)
        neg = torch.finfo(p_attn.dtype).min
        r = torch.logaddexp(
            torch.log(p_attn) + torch.log(candidate_count).view(1, seq_q, 1),
            p_attn.new_tensor(float(np.log(1e-8))),
        )
        r = r.masked_fill(~visible.unsqueeze(0), neg)

        # Aggregate to KV groups: TopMean over query heads in each group
        r_grouped = r.view(n_kv, self.num_kv_groups, seq_q, seq_k)
        topk_vals = r_grouped.topk(
            min(self.topk_qk, self.num_kv_groups), dim=1
        ).values
        R_group = topk_vals.mean(dim=1)
        R_group = R_group.masked_fill(~visible.unsqueeze(0), neg)

        N = seq_k

        # f1: max over (t, m) of R_{t,m,i}
        f1 = R_group.max(dim=0).values.max(dim=0).values  # (N,)

        # f2: MeanTopK over (t, m) of R_{t,m,i}
        R_flat = R_group.permute(1, 0, 2).reshape(-1, N)
        valid_per_token = visible.sum(dim=0).clamp_min(1)
        k2_per_token = (valid_per_token * n_kv).clamp(
            min=1, max=min(self.topk_qk, R_flat.shape[0])
        )
        k2_max = int(k2_per_token.max().item())
        f2_top = R_flat.topk(k2_max, dim=0).values
        f2_cumsum = f2_top.cumsum(dim=0)
        f2 = f2_cumsum.gather(
            0, (k2_per_token - 1).unsqueeze(0)
        ).squeeze(0) / k2_per_token

        # f3: Mean over (t, m) where R_{t,m,i} > 0
        positive_mask = (R_flat > 0).float()
        positive_values = torch.where(
            positive_mask.bool(), R_flat, torch.zeros_like(R_flat)
        )
        f3 = positive_values.sum(dim=0) / positive_mask.sum(dim=0).clamp(min=1)

        # f4: MeanTopK over recent queries only
        n_recent = min(self.n_recent, seq_q)
        R_recent = R_group[:, -n_recent:, :]
        R_recent_flat = R_recent.permute(1, 0, 2).reshape(-1, N)
        recent_valid = visible[-n_recent:].sum(dim=0).clamp_min(1)
        k4_per_token = (recent_valid * n_kv).clamp(
            min=1, max=min(self.topk_qk, R_recent_flat.shape[0])
        )
        k4_max = int(k4_per_token.max().item())
        f4_top = R_recent_flat.topk(k4_max, dim=0).values
        f4_cumsum = f4_top.cumsum(dim=0)
        f4 = f4_cumsum.gather(
            0, (k4_per_token - 1).unsqueeze(0)
        ).squeeze(0) / k4_per_token

        # f5: fraction of query tokens t where max_m R_{t,m,i} > 0
        max_over_m = R_group.max(dim=0).values
        f5 = ((max_over_m > 0) & visible).float().sum(dim=0) / seq_q

        # f6: fraction of KV groups m where TopMean_t R_{t,m,i} > 0
        k_frac = torch.ceil(valid_per_token.float() * 0.25).long().clamp_min(1)
        k_max = int(k_frac.max().item())
        top_vals = R_group.masked_fill(~visible.unsqueeze(0), neg).topk(k_max, dim=1).values
        top_cumsum = top_vals.cumsum(dim=1)
        gather_idx = (k_frac - 1).view(1, 1, N).expand(n_kv, 1, N)
        top_frac_mean = top_cumsum.gather(1, gather_idx).squeeze(1) / k_frac.view(1, N)
        f6 = (top_frac_mean > 0).float().mean(dim=0)

        features = torch.stack([f1, f2, f3, f4, f5, f6], dim=-1)  # (N, 6)
        if not torch.isfinite(features).all():
            bad_by_column = (~torch.isfinite(features)).sum(dim=0).tolist()
            raise FloatingPointError(
                "Non-finite attention-fallback QK features by column "
                f"[f1..f6]: {bad_by_column}"
            )
        return features

    @torch.no_grad()
    def extract_features(
        self,
        layer_stats: dict,
    ) -> torch.Tensor:
        """
        Extract full 14-dim feature vector for each candidate token.

        .. deprecated::
            This helper pools statistics across layers. Active training and
            inference compute features with the per-layer helpers instead.

        This method aggregates per-layer statistics into a single set of
        per-token features. The aggregation strategy:
          - Q-K relation: compute from a middle layer (layer 16 for 32-layer model)
          - Coverage: compute from key states pooled across layers
          - Auxiliary: compute from value states pooled across layers

        Parameters
        ----------
        layer_stats : dict
            Dictionary keyed by layer_idx, each containing:
              - 'query_states': (n_q_heads, seq_len, head_dim)
              - 'key_states': (n_kv_heads, seq_len, head_dim)
              - 'value_states': (n_kv_heads, seq_len, head_dim)
              - 'attention_weights': (n_q_heads, seq_len, seq_len)

        Returns
        -------
        torch.Tensor
            Shape (N_candidates, 14) — feature vectors.
        """
        layer_indices = sorted(layer_stats.keys())
        n_layers = len(layer_indices)

        # Pick middle layer for q-k features (or average a few)
        mid_idx = layer_indices[n_layers // 2]
        mid = layer_stats[mid_idx]
        logger.debug(f"Using layer {mid_idx} for q-k relation features")

        # Q-K relation features from middle layer
        qk_features = self.compute_qk_relation_features(
            query_states=mid["query_states"].to(self.dtype),
            key_states=mid["key_states"].to(self.dtype),
        )

        # Pool key/value across layers for coverage and auxiliary features
        # Use every 4th layer to reduce compute
        sampled_layers = layer_indices[::4] if n_layers > 8 else layer_indices
        keys_list = [layer_stats[l]["key_states"].to(self.dtype) for l in sampled_layers]
        values_list = [layer_stats[l]["value_states"].to(self.dtype) for l in sampled_layers]

        # Average across sampled layers
        keys_pooled = torch.stack(keys_list, dim=0).mean(dim=0)  # (n_kv, seq_len, d)
        values_pooled = torch.stack(values_list, dim=0).mean(dim=0)  # (n_kv, seq_len, d)

        # Coverage features
        cov_features = self.compute_coverage_features(keys_pooled, mid_idx)

        # Auxiliary features
        aux_features = self.compute_auxiliary_features(
            values_pooled, seq_len=keys_pooled.shape[1], is_prefill=True
        )

        # Concatenate: (N, 6) + (N, 4) + (N, 4) = (N, 14)
        features = torch.cat([qk_features, cov_features, aux_features], dim=-1)


        return features
