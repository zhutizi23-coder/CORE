"""
Diversity-aware teacher distribution for CORE.

Constructs teacher signal combining:
  1. Backbone attention utility (query demand)
  2. D-optimal coverage (non-redundant coverage in representation space)
  3. Entropy regularization

Uses a fixed number of DRIVE-style updates to approximate the
entropy-regularized continuous weighting objective on the simplex.

DRIVE (Sec. 3) objective on the simplex:

    w* = argmax_w [ sum_i w_i c_i
                    + eta * log |Sigma_w|
                    - tau * KL(w || p) ]

with the per-iteration closed-form update (DRIVE Eq. 5):

    w_i^(t+1) ∝ p_i * exp( (c_i + eta * h_i^(t)) / tau )

where h_i = x_i^T Sigma_w^{-1} x_i is the statistical leverage (marginal
diversity gain). For CORE, the contribution c_i is replaced by the backbone
attention utility.
"""

import logging

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from core.config import COREConfig

logger = logging.getLogger(__name__)


class DiversityAwareTeacher:
    """
    Computes diversity-aware teacher distribution.

    Approximates the following optimization on the simplex using fixed,
    unweightedly centered x_bar_i = z_i - mean_j(z_j):
        w* = argmax_w [ sum_i w_i u_i
                        + beta * log det(I + alpha * sum_i w_i x_bar_i x_bar_i^T)
                        - tau_iter * sum_i w_i log w_i ]

    via fixed-point iteration (closed-form per step):
        w_i^(t+1) ∝ exp( (u_i + beta * d_i^(t)) / tau_iter )
    """

    def __init__(self, config: COREConfig):
        self.config = config
        self.beta = config.beta_coverage
        self.alpha = config.alpha_coverage
        # drive_temp controls the fixed-point update; teacher_temp controls
        # the distillation distribution derived from its output.
        self.tau = config.drive_temp
        self.teacher_temp = config.teacher_temp
        self.n_iter = config.teacher_n_iter
        self.eps = 1e-8

    @torch.no_grad()
    def compute_attention_utility(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        query_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute token utility from the **true** backbone attention logits.

        Aggregate query heads within each KV group using TopMean, then take
        the maximum over query positions and KV groups.

        u_i = max over (query token t, KV group m) of TopMean_h a(t, h, i)

        Parameters
        ----------
        query_states : torch.Tensor
            Post-RoPE query states, shape (n_q_heads, seq_q, head_dim).
        key_states : torch.Tensor
            Post-RoPE key states, shape (n_kv_heads, seq_k, head_dim).
        query_mask : torch.Tensor, optional
            Boolean mask of shape (seq_q,) selecting query tokens in Q.

        Returns
        -------
        torch.Tensor
            Token utility of shape (seq_k,).
        """
        q = query_states.float()  # (n_q, seq_q, d)
        k = key_states.float()    # (n_kv, seq_k, d)
        n_q, seq_q, d = q.shape
        n_kv, seq_k, _ = k.shape
        num_groups = n_q // n_kv
        topk_h = min(self.config.topk_qk, num_groups)

        # Expand KV heads -> query heads (GQA). Query blocks below avoid the
        # dense (n_q_heads, seq_q, seq_k) tensor, which is the main obstacle to
        # training on 4K sequences.
        if n_kv != n_q:
            k_expanded = repeat_kv(k.unsqueeze(0), num_groups).squeeze(0)  # (n_q, seq_k, d)
        else:
            k_expanded = k
        keys_t = k_expanded.transpose(-1, -2)

        # For a decode buffer, its queries occupy the final seq_q positions of
        # the current seq_k cache.
        q_start = max(0, seq_k - seq_q)
        if query_mask is not None:
            if query_mask.shape != (seq_q,):
                raise ValueError(
                    f"query_mask shape {tuple(query_mask.shape)} != ({seq_q},)"
                )
            query_ids = query_mask.to(device=q.device, dtype=torch.bool).nonzero(
                as_tuple=False
            ).squeeze(-1)
        else:
            query_ids = torch.arange(seq_q, device=q.device)
        if query_ids.numel() == 0:
            raise ValueError("query_mask selects no queries")

        key_pos = torch.arange(seq_k, device=q.device)
        neg = torch.finfo(torch.float32).min
        utility = torch.full((seq_k,), neg, device=q.device, dtype=torch.float32)
        block_size = max(1, int(self.config.qk_query_block_size))
        scale = d**-0.5
        for start in range(0, query_ids.numel(), block_size):
            ids = query_ids[start : start + block_size]
            logits = torch.matmul(q[:, ids, :], keys_t) * scale
            query_pos = q_start + ids
            visible = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
            logits = logits.masked_fill(~visible.unsqueeze(0), neg)

            # Aggregate within each KV group via TopMean, then take the exact
            # running max over query positions and KV groups.
            grouped = logits.view(n_kv, num_groups, -1, seq_k)
            # Llama-3.1 GQA has four query heads per KV head and CORE's
            # default topk_qk is also four. Sorting all four elements merely
            # to average all of them is exactly equivalent to mean(), but is
            # substantially slower on the streamed QxK blocks.
            if topk_h == num_groups:
                group_score = grouped.mean(dim=1)
            else:
                group_score = grouped.topk(topk_h, dim=1).values.mean(dim=1)
            utility = torch.maximum(
                utility, group_score.amax(dim=1).amax(dim=0)
            )
        return utility

    @torch.no_grad()
    def compute_d_optimal_coverage(
        self,
        z: torch.Tensor,
        utility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate teacher weights with a fixed number of DRIVE-style updates.

        The objective combines utility and entropy with the log-determinant
        of the centered feature covariance

            Sigma_w = I + alpha * sum_i w_i x_bar_i x_bar_i^T

        Features use fixed, unweighted centering: x_bar_i = z_i - mean(z).

        Per-step update (CORE Eq. 5):
            w_i ∝ exp( (u_i + beta * h_i) / tau )
        with marginal diversity gain  h_i = alpha * x_bar_i^T Sigma_w^{-1} x_bar_i.

        Parameters
        ----------
        z : torch.Tensor
            Normalized token representations of shape (N, d_z).
        utility : torch.Tensor
            Attention utility of shape (N,).
        Returns
        -------
        torch.Tensor
            Teacher weights w of shape (N,).
        """
        N, d_z = z.shape

        # Use raw attention utility with the configured coverage weight and temperature.

        # Center features about the candidate-set mean (DRIVE's x̄_i).
        z_c = z - z.mean(dim=0, keepdim=True)  # fixed unweighted centering

        # Initialize with uniform weights
        w = torch.full((N,), 1.0 / N, device=z.device, dtype=z.dtype)

        I = torch.eye(d_z, device=z.device, dtype=z.dtype)

        for it in range(self.n_iter):
            # Sigma = I + alpha * z_c^T diag(w) z_c   (d_z x d_z, small & invertible)
            zw = z_c * w.unsqueeze(-1)            # (N, d_z)
            Sigma = I + self.alpha * (z_c.T @ zw)  # (d_z, d_z)

            # Marginal coverage gain: h_i = alpha * x_bar_i^T Sigma^-1 x_bar_i
            # Solve Sigma X = z_c^T  =>  X = Sigma^-1 z_c^T  (d_z, N)
            Sigma_inv_zT = torch.linalg.solve(Sigma, z_c.T)        # (d_z, N)
            d = self.alpha * (z_c * Sigma_inv_zT.T).sum(dim=-1)    # (N,)

            # CORE Eq. 5: w_i ∝ exp((u_i + beta*h_i) / tau).
            logits = (utility + self.beta * d) / self.tau
            logits = logits - logits.max()  # numerical stability
            w = F.softmax(logits, dim=-1)

        return w

    @torch.no_grad()
    def propagate_local_utility(self, utility: torch.Tensor) -> torch.Tensor:
        """Mix token demand with a smooth local max on the teacher side."""
        window = max(1, int(getattr(self.config, "teacher_local_window", 1)))
        mix = float(getattr(self.config, "teacher_local_mix", 0.0))
        if window <= 1 or mix <= 0.0 or utility.numel() == 0:
            return utility
        if not 0.0 <= mix <= 1.0:
            raise ValueError("teacher_local_mix must be in [0, 1]")
        window = min(window, int(utility.numel()))
        left = (window - 1) // 2
        right = window - 1 - left
        local = F.max_pool1d(
            F.pad(
                utility.float().view(1, 1, -1),
                (left, right),
                value=torch.finfo(torch.float32).min,
            ),
            kernel_size=window,
            stride=1,
        ).view_as(utility)
        return utility.float().lerp(local, mix).to(dtype=utility.dtype)

    @torch.no_grad()
    def to_teacher_distribution(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply the distillation temperature to finite-iteration teacher weights.

        Normalize w_i ** (1 / teacher_temp), preserving zero support.
        The computation uses log space when teacher_temp differs from one.
        """
        if self.teacher_temp <= 0:
            raise ValueError("teacher_temp must be positive")
        if self.teacher_temp == 1.0:
            return weights / weights.sum(dim=-1, keepdim=True)
        # Explicit temperature ablation: preserve zero support and tiny mass.
        log_pi = torch.log(weights) / self.teacher_temp
        return F.softmax(log_pi, dim=-1)

    @torch.no_grad()
    def compute_teacher_distribution(
        self,
        z: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        query_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the full diversity-aware teacher distribution.

        Utility uses backbone attention logits:
        a(t,h,i) = <q_t^h, k_i^g(h)> / sqrt(d).

        Parameters
        ----------
        z : torch.Tensor
            Normalized token representations for coverage, shape (N, d_z).
        query_states : torch.Tensor
            Post-RoPE query states, shape (n_q_heads, seq_q, head_dim).
        key_states : torch.Tensor
            Post-RoPE key states, shape (n_kv_heads, seq_k, head_dim).
        query_mask : torch.Tensor, optional
            Boolean mask selecting query tokens.

        Returns
        -------
        torch.Tensor
            Teacher distribution pi_T of shape (N,).
        """
        # Compute utility by aggregating attention logits within KV groups.
        utility = self.compute_attention_utility(
            query_states, key_states, query_mask
        )

        # Compute DRIVE weights, then apply the distillation temperature.
        weights = self.compute_d_optimal_coverage(z, utility)
        pi_T = self.to_teacher_distribution(weights)

        # Renormalize the final distribution with an epsilon guard.
        pi_T = pi_T / (pi_T.sum() + self.eps)
        return pi_T

    @torch.no_grad()
    def compute_teacher_ranking(
        self,
        pi_T: torch.Tensor,
        budget: int,
        delta: int,
        valid_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute teacher ranking and boundary windows for boundary margin loss.

        Parameters
        ----------
        pi_T : torch.Tensor
            Teacher distribution of shape (N,).
        budget : int
            Number of tokens to retain (Top-B).
        delta : int
            Boundary window size.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - rank_T: ranking of tokens (1=best), shape (N,)
            - P_bd: indices of boundary-retain window, shape (<=delta,)
            - N_bd: indices of boundary-evict window, shape (<=delta,)
        """
        if valid_mask is None:
            candidate_idx = torch.arange(len(pi_T), device=pi_T.device)
        else:
            valid_mask = valid_mask.to(device=pi_T.device, dtype=torch.bool)
            if valid_mask.shape != pi_T.shape:
                raise ValueError("valid_mask must have the same shape as pi_T")
            candidate_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)

        # Excluded tokens receive rank 0 and never enter boundary windows.
        local_order = pi_T[candidate_idx].argsort(descending=True, stable=True)
        sort_idx = candidate_idx[local_order]
        rank_T = torch.zeros_like(pi_T, dtype=torch.long)
        rank_T[sort_idx] = torch.arange(
            1, len(sort_idx) + 1, device=pi_T.device
        )
        budget = max(0, min(int(budget), len(sort_idx)))
        if budget == 0 or budget == len(sort_idx):
            return rank_T, sort_idx[:0], sort_idx[:0]

        # Boundary retain window: ranks [B - delta + 1, B]
        p_start = max(0, budget - delta)
        p_end = budget
        P_bd = sort_idx[p_start:p_end]

        # Boundary evict window: ranks [B + 1, B + delta]
        n_start = budget
        n_end = min(len(sort_idx), budget + delta)
        N_bd = sort_idx[n_start:n_end]

        return rank_T, P_bd, N_bd
