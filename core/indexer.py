"""
CORE Indexer: lightweight per-layer MLP that predicts token retention scores
from 14-dim feature vectors, with KL and boundary hinge losses.

Trained to fit the diversity-aware teacher distribution (L_cal) while
maintaining a margin at the Top-B decision boundary (L_bd).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import COREConfig

logger = logging.getLogger(__name__)


class GroupedLinear(nn.Module):
    """Block-diagonal linear map with independent input/output groups."""

    def __init__(self, in_features: int, out_features: int, groups: int):
        super().__init__()
        if groups <= 1:
            raise ValueError("GroupedLinear requires groups > 1")
        if in_features % groups != 0 or out_features % groups != 0:
            raise ValueError(
                "GroupedLinear dimensions must be divisible by groups: "
                f"in={in_features}, out={out_features}, groups={groups}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.groups = int(groups)
        self.in_per_group = self.in_features // self.groups
        self.out_per_group = self.out_features // self.groups
        self.weight = nn.Parameter(torch.empty(
            self.groups, self.out_per_group, self.in_per_group
        ))
        for group in range(self.groups):
            nn.init.kaiming_uniform_(self.weight[group], a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grouped = x.reshape(*x.shape[:-1], self.groups, self.in_per_group)
        projected = torch.einsum("...gi,goi->...go", grouped, self.weight)
        return projected.reshape(*x.shape[:-1], self.out_features)

    def dense_weight(self) -> torch.Tensor:
        """Return the equivalent block-diagonal two-dimensional weight."""
        return torch.block_diag(
            *[self.weight[group] for group in range(self.groups)]
        )


def conditional_evicted_weights(
    probabilities: torch.Tensor,
    evict_mask: torch.Tensor,
    uniform_fraction: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize evicted-token write weights, with optional uniform mixing."""
    mask = evict_mask.to(device=probabilities.device, dtype=probabilities.dtype)
    conditional = probabilities * mask
    mass = conditional.sum(dim=-1, keepdim=True)
    conditional = conditional / torch.where(mass > 0, mass, torch.ones_like(mass))
    fraction = float(uniform_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("memory_uniform_fraction must be in [0, 1]")
    if fraction == 0.0:
        return conditional
    uniform = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (1.0 - fraction) * conditional + fraction * uniform


class COREIndexerMLP(nn.Module):
    """
    CORE indexer with two hidden layers and a scalar output layer.

    Input:  14-dim feature vector xi_i per token
    Output: scalar score s_i per token

    Same MLP is applied independently to each token (weight sharing across
    tokens within a layer).
    """

    def __init__(self, input_dim: int = 14, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xi : torch.Tensor
            Feature vectors of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Scores of shape (...) — one scalar per token.
        """
        return self.net(xi).squeeze(-1)


class MemoryModule(nn.Module):
    """Trainable projection and gate for CORE latent memory (Eqs. 11–13).

    The shared, bias-free projection maps query and key vectors from d_model
    into d_mem. In the default path, d_model is the concatenated query-head
    width; queries and GQA-expanded keys use post-RoPE coordinates.

    Fast state M, b, Z is managed by the caller. The readout is
        m(q) = phi(q) @ M / sqrt(phi(q)^2 @ b + eps)
    and the correction is g(q) * m(q). M has shape (d_mem, d_value), so the
    returned correction has width d_value. Training uses pre-o_proj values;
    inference can store compact GQA values and fold expansion into o_proj.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        gate_hidden: int = 128,
        projection_groups: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.projection_groups = int(projection_groups)
        # Linear_θ: shared projection for query (read) and keys (write), no bias.
        if self.projection_groups == 1:
            self.theta = nn.Linear(d_model, d_mem, bias=False)
        else:
            self.theta = GroupedLinear(
                d_model, d_mem, groups=self.projection_groups
            )
        # Scalar gate g(q) ∈ [0,1]: MLP(d_model → gate_hidden → 1) → sigmoid.
        self.gate = nn.Sequential(
            nn.Linear(d_model, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the shared Linear_θ to query or key vectors.

        Shape: (..., d_model) → (..., d_mem).
        """
        return self.theta(x)

    def theta_dense_weight(self) -> torch.Tensor:
        """Return Linear_theta as a 2-D matrix for exact GQA folding."""
        if isinstance(self.theta, GroupedLinear):
            return self.theta.dense_weight()
        return self.theta.weight

    def readout(
        self,
        query_hidden: torch.Tensor,
        M: torch.Tensor,
        b: torch.Tensor,
        eps: float = 1e-8,
        gate_override: float | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gated memory readout residual g(q)·m(q) (CORE Eq. 12).

        Parameters
        ----------
        query_hidden : torch.Tensor
            Query vectors, shape (..., d_model). The default path uses
            concatenated post-RoPE query heads.
        M : torch.Tensor
            Per-layer fast-weight memory matrix, shape (d_mem, d_value).
            Constructed and updated by the caller.
        b : torch.Tensor
            Per-layer fast-weight stabilizer, shape (d_mem,).
        eps : float
            Numerical stability for the denominator.

        Returns
        -------
        torch.Tensor
            Residual of shape (..., d_value), in M's value coordinates.
            With return_gate=True, also returns the scalar gate per query.
        """
        proj_q = self.project(query_hidden)               # (..., d_mem)
        # numerator: proj_q @ M  → (..., d_model)
        num = proj_q @ M                                  # (..., d_model)
        # denominator: (proj_q ⊙ proj_q) · b  → (...,) scalar per token
        denom = (proj_q * proj_q) @ b + eps               # (...,)
        m_q = num / denom.sqrt().unsqueeze(-1)                   # (..., d_model)
        if gate_override is None:
            g = self.gate(query_hidden).squeeze(-1)       # (...,) scalar in [0,1]
        else:
            g = torch.full_like(denom, float(gate_override))
        residual = g.unsqueeze(-1) * m_q                  # (..., d_model)
        return (residual, g) if return_gate else residual

    def compute_mse_loss(
        self,
        o_full: torch.Tensor,
        o_compressed: torch.Tensor,
        query_hidden: torch.Tensor,
        M: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruction loss L_mem = ||o - o_attn - g(q)·m(q)||² (CORE Eq. 13).

        ``o_full`` is the full attention output (no eviction), ``o_compressed``
        is the output computed over the retained KV cache only, and the memory
        readout g(q)·m(q) should close the gap. M, b are ephemeral fast
        weights rather than optimizer parameters. Their current-step
        construction remains differentiable through the write-side
        Linear_theta; the caller controls state lifetime and detachment.

        Parameters
        ----------
        o_full, o_compressed : torch.Tensor
            Both shape (..., d_model). Pre-o_proj attention outputs.
        query_hidden : torch.Tensor
            Query vectors in the shared query/key coordinates, shape (..., d_model).
        M, b : torch.Tensor
            Fast weights (d_mem, d_model) and (d_mem,). Current-step gradients
            may flow through their construction into Linear_theta.

        Returns
        -------
        torch.Tensor
            Scalar MSE loss.
        """
        residual = self.readout(query_hidden, M, b)
        diff = o_full - o_compressed - residual
        return (diff * diff).sum(dim=-1).mean()


def student_distribution(
    scores: torch.Tensor,
    student_temp: float = 1.0,
    eps: float = 1e-8,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the calibrated student distribution on an eligible axis.

    By default this is the original definition

        pi_theta(i) = softmax(s_i / tau_M).

    ``valid_mask`` optionally restricts a supervised candidate axis. The
    non-sink restriction is used for KL calibration. Memory event mass instead
    uses the full-cache distribution (Eq. 6/10); protected sinks cannot be
    written because they are excluded from the eviction set.

    Excluded positions receive exactly zero probability; eligible positions
    are renormalized to sum to one.  A one-dimensional mask over the final
    token axis is broadcast over leading batch dimensions.
    """
    if student_temp <= 0:
        raise ValueError("student_temp must be > 0")

    logits = scores / student_temp
    if valid_mask is None:
        return F.softmax(logits, dim=-1)

    mask = valid_mask.to(device=scores.device, dtype=torch.bool)
    if mask.shape != scores.shape:
        if mask.ndim == 1 and mask.shape[0] == scores.shape[-1]:
            view_shape = (1,) * (scores.ndim - 1) + (mask.shape[0],)
            mask = mask.view(view_shape).expand_as(scores)
        else:
            raise ValueError(
                f"valid_mask shape {tuple(mask.shape)} is incompatible with "
                f"scores shape {tuple(scores.shape)}"
            )

    if not mask.any(dim=-1).all():
        raise ValueError("valid_mask excludes every token for at least one row")

    masked_logits = logits.masked_fill(~mask, float("-inf"))
    probabilities = F.softmax(masked_logits, dim=-1)
    # Make the contract explicit even for unusual low-precision softmax paths.
    return probabilities.masked_fill(~mask, 0.0)


def calibrated_kl_loss(
    pi_teacher: torch.Tensor,
    pi_student: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calibrated KL divergence: D_KL(pi_T || pi_theta).

    L_cal = sum_i pi_T(i) * (log pi_T(i) - log pi_theta(i))

    This makes the student fit the teacher's full diversity-aware probability
    mass, not just the token ranking.

    Parameters
    ----------
    pi_teacher : torch.Tensor
        Teacher distribution of shape (N,).
    pi_student : torch.Tensor
        Student distribution of shape (N,).
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    torch.Tensor
        Scalar KL divergence loss.
    """
    log_ratio = torch.log(pi_student + eps) - torch.log(pi_teacher + eps)
    kl = (pi_teacher * (-log_ratio)).sum()
    return kl


def boundary_margin_loss(
    scores: torch.Tensor,
    student_temp: float,
    P_bd: torch.Tensor,
    N_bd: torch.Tensor,
    margin: float = 0.5,
    beta: float = 5.0,
    loss_type: str = "lse_hinge",
) -> torch.Tensor:
    """
    Boundary loss on temperature-normalized student scores z_i = s_i / tau_m.

    Supports hard hinge, pairwise hinge, all-pairs softplus, and LogSumExp
    hinge objectives. The margin uses the same temperature-normalized space
    as inference memory writing. For the LogSumExp objective:

        L_bd = relu(soft_max(z_N) - soft_min(z_P) + margin)
        soft_max(z_N) = logsumexp(beta * z_N) / beta
        soft_min(z_P) = -logsumexp(-beta * z_P) / beta

    Parameters
    ----------
    scores : torch.Tensor
        Student scores of shape (N,).
    student_temp : float
        Student temperature tau_M.
    P_bd : torch.Tensor
        Indices of the boundary-retain window (ranks [B-delta+1, B]).
    N_bd : torch.Tensor
        Indices of the boundary-evict window (ranks [B+1, B+delta]).
    margin : float
        Required margin m (in z-space).
    beta : float
        Fixed LogSumExp sharpness (higher = closer to hard max/min).

    Returns
    -------
    torch.Tensor
        Scalar boundary loss for the selected formulation.
    """
    if len(P_bd) == 0 or len(N_bd) == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    # Temperature-normalized student scores: z = s / tau_m.
    z_P = scores[P_bd] / student_temp
    z_N = scores[N_bd] / student_temp

    if loss_type == "hard_hinge":
        # Hard hinge at the boundary between retained and evicted tokens.
        return F.relu(z_N.max() - z_P.min() + margin)

    if loss_type == "pairwise_hinge":
        pairwise_gap = z_P.unsqueeze(1) - z_N.unsqueeze(0)
        return F.relu(margin - pairwise_gap).mean()

    if loss_type == "all_pairs_softplus":
        # Average over the complete cutoff window, not only one soft extreme.
        # This is robust to a small number of noisy boundary labels and keeps
        # the scale independent of boundary-window width.
        pairwise_gap = z_P.unsqueeze(1) - z_N.unsqueeze(0)
        return F.softplus(margin - pairwise_gap).mean()
    if loss_type != "lse_hinge":
        raise ValueError(f"Unknown boundary_loss_type: {loss_type}")

    # LogSumExp smooth approximation: soft_max_N ≈ max(z_N), soft_min_P ≈ min(z_P)
    # Use log-MEAN-exp rather than raw logsumexp. Without the normalization,
    # tied scores create an artificial 2*log(window_size)/beta gap (1.386 for
    # delta=32, beta=5), so L_bd starts at ~1.886 instead of the configured
    # margin 0.5 and changes merely when the boundary-window width changes.
    log_n = z_N.new_tensor(float(z_N.numel())).log()
    log_p = z_P.new_tensor(float(z_P.numel())).log()
    soft_max_N = (torch.logsumexp(z_N * beta, dim=0) - log_n) / beta
    soft_min_P = -(
        torch.logsumexp(-z_P * beta, dim=0) - log_p
    ) / beta

    loss = F.relu(soft_max_N - soft_min_P + margin)
    return loss


def core_loss(
    scores: torch.Tensor,
    pi_teacher: torch.Tensor,
    P_bd: torch.Tensor,
    N_bd: torch.Tensor,
    config: COREConfig,
    loss_mask: torch.Tensor | None = None,
    selection_scores: torch.Tensor | None = None,
    boundary_components: list[tuple[
        float, torch.Tensor, torch.Tensor, torch.Tensor
    ]] | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Combined CORE training loss.

    L_CORE = lambda_cal * L_cal + lambda_bd * L_bd

    Parameters
    ----------
    scores : torch.Tensor
        Student scores of shape (N,).
    pi_teacher : torch.Tensor
        Teacher distribution of shape (N,).
    P_bd : torch.Tensor
        Boundary-retain window indices.
    N_bd : torch.Tensor
        Boundary-evict window indices.
    config : COREConfig
        Configuration with loss hyperparameters.

    Returns
    -------
    tuple[torch.Tensor, dict]
        - Total loss scalar
        - Dict with loss components for logging
    """
    if loss_mask is not None:
        loss_mask = loss_mask.to(device=scores.device, dtype=torch.bool)
        if loss_mask.shape != scores.shape:
            raise ValueError(
                f"loss_mask shape {tuple(loss_mask.shape)} does not match "
                f"scores shape {tuple(scores.shape)}"
            )
        if not loss_mask.any():
            raise ValueError("loss_mask excludes every token")
        scores_for_kl = scores[loss_mask]
        teacher_for_kl = pi_teacher[loss_mask]
    else:
        scores_for_kl = scores
        teacher_for_kl = pi_teacher

    # Both distributions must be normalized on the same (possibly sink-free)
    # key axis. The teacher was originally normalized over all tokens.
    teacher_for_kl = teacher_for_kl / teacher_for_kl.sum().clamp_min(1e-12)
    pi_student = student_distribution(scores_for_kl, config.student_temp)

    l_cal = calibrated_kl_loss(teacher_for_kl, pi_student)
    boundary_scores = scores if selection_scores is None else selection_scores
    if boundary_scores.shape != scores.shape:
        raise ValueError(
            "selection_scores shape does not match calibrated scores: "
            f"{tuple(boundary_scores.shape)} != {tuple(scores.shape)}"
        )
    if boundary_components is None:
        l_bd = boundary_margin_loss(
            boundary_scores,
            config.student_temp,
            P_bd,
            N_bd,
            config.boundary_margin,
            config.bd_beta,
            getattr(config, "boundary_loss_type", "lse_hinge"),
        )
    else:
        # A hard set constructor cannot be differentiated by adding a large
        # keep-set offset and applying one margin loss.  Each continuous policy
        # channel supplies its own scores and cutoff windows; their weights sum
        # to one so lambda_bd keeps the same interpretation and scale.
        active = [
            c for c in boundary_components
            if c[0] > 0 and c[2].numel() > 0 and c[3].numel() > 0
        ]
        weight_sum = sum(float(c[0]) for c in active)
        if not active or weight_sum <= 0:
            l_bd = scores.new_zeros(())
        else:
            l_bd = scores.new_zeros(())
            for weight, component_scores, positive, negative in active:
                if component_scores.shape != scores.shape:
                    raise ValueError(
                        "boundary component score shape does not match "
                        f"calibrated scores: {tuple(component_scores.shape)} "
                        f"!= {tuple(scores.shape)}"
                    )
                component_loss = boundary_margin_loss(
                    component_scores,
                    config.student_temp,
                    positive,
                    negative,
                    config.boundary_margin,
                    config.bd_beta,
                    getattr(config, "boundary_loss_type", "lse_hinge"),
                )
                l_bd = l_bd + (float(weight) / weight_sum) * component_loss

    total = config.lambda_cal * l_cal + config.lambda_bd * l_bd

    return total, {
        "loss": total.item(),
        "loss_cal": l_cal.item(),
        "loss_bd": l_bd.item(),
    }


class COREIndexerStack(nn.Module):
    """Logical per-layer stack of COREIndexerMLPs.

    With ``compact_stride=1`` every transformer layer owns an MLP. With a
    larger stride, one MLP is allocated per deployed IndexCache group and
    logical layers in that group route to the same source module. This avoids
    storing optimizer state for MLPs that deployment can never call.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers (one MLP per layer).
    input_dim : int
        Feature dimension (default 14).
    hidden_dim : int
        MLP hidden width (default 128).
    compact_stride : int
        Number of logical layers routed to each allocated source MLP.
    """

    def __init__(
        self,
        n_layers: int,
        input_dim: int = 14,
        hidden_dim: int = 128,
        compact_stride: int = 1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.compact_stride = max(1, int(compact_stride))
        n_modules = (n_layers + self.compact_stride - 1) // self.compact_stride
        self.layers = nn.ModuleList(
            [
                COREIndexerMLP(input_dim=input_dim, hidden_dim=hidden_dim)
                for _ in range(n_modules)
            ]
        )

    def forward(self, features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Score tokens using the MLP for layer ``layer_idx``.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor of shape (..., input_dim).
        layer_idx : int
            Layer index selecting which MLP to apply. Clamped to [0, n_layers-1].

        Returns
        -------
        torch.Tensor
            Scores of shape (...) - one scalar per token.
        """
        layer_idx = max(0, min(layer_idx, self.n_layers - 1))
        module_idx = layer_idx // self.compact_stride
        return self.layers[module_idx](features)

    def __getitem__(self, idx: int) -> COREIndexerMLP:
        logical_idx = max(0, min(int(idx), self.n_layers - 1))
        return self.layers[logical_idx // self.compact_stride]

    def __len__(self) -> int:
        return self.n_layers

    def __iter__(self):
        return iter(self.layers)


class MemoryModuleStack(nn.Module):
    """Per-layer stack of independent MemoryModules.

    As in IndexMem, each layer's module is shared across attention heads.
    By default, layers have independent ``Linear_θ`` and ``gate`` parameters;
    ``share_across_layers`` optionally shares these slow weights.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers (one MemoryModule per layer).
    d_model : int
        Query/key vector width (query-head count times head dimension by default).
    d_mem : int
        Latent memory dimension (512 per layer in the paper).
    gate_hidden : int
        Hidden width of the gate MLP (constructor default 128; paper config 64).
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_mem: int,
        gate_hidden: int = 128,
        share_across_layers: bool = False,
        projection_groups: int = 1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_mem = d_mem
        self.share_across_layers = bool(share_across_layers)
        self.projection_groups = int(projection_groups)
        n_slow_modules = 1 if self.share_across_layers else n_layers
        self.gate_hidden = gate_hidden
        self.layers = nn.ModuleList(
            [
                MemoryModule(
                    d_model,
                    d_mem,
                    gate_hidden,
                    projection_groups=self.projection_groups,
                )
                for _ in range(n_slow_modules)
            ]
        )

    def _module_idx(self, layer_idx: int) -> int:
        if self.share_across_layers:
            return 0
        return max(0, min(layer_idx, self.n_layers - 1))

    def project(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply layer ``layer_idx``'s Linear_θ to ``x`` (used for both the
        query at read-time and the keys at write-time)."""
        module_idx = self._module_idx(layer_idx)
        return self.layers[module_idx].project(x)

    def readout(
        self,
        query_hidden: torch.Tensor,
        M: torch.Tensor,
        b: torch.Tensor,
        layer_idx: int,
        eps: float = 1e-8,
        gate_override: float | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the gated memory readout residual for layer ``layer_idx``.

        Mirrors :meth:`MemoryModule.readout` but dispatches to the per-layer
        module. ``M`` / ``b`` are this layer's fast weights.
        """
        module_idx = self._module_idx(layer_idx)
        return self.layers[module_idx].readout(
            query_hidden, M, b, eps, gate_override=gate_override,
            return_gate=return_gate,
        )

    def __getitem__(self, idx: int) -> MemoryModule:
        return self.layers[self._module_idx(idx)]

    def __len__(self) -> int:
        return self.n_layers

    def __iter__(self):
        return iter(self.layers)
