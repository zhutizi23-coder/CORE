"""
Configuration for CORE indexer training and inference.
"""

from dataclasses import dataclass, field, fields
from typing import Optional
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class COREConfig:
    """CORE indexer configuration."""

    @classmethod
    def from_dict(cls, d: dict) -> "COREConfig":
        """Create from a dict (e.g. checkpoint config), ignoring unknown keys
        for backward-compat with old checkpoints that have removed fields."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    # ---- Model ----
    model_name: str = str(_PROJECT_ROOT / "model")
    # Number of layers (auto-detected from config if None)
    n_layers: Optional[int] = None
    # Number of KV heads (auto-detected from config if None)
    n_kv_heads: Optional[int] = None
    # Number of query heads
    n_q_heads: Optional[int] = None
    # Head dimension
    head_dim: Optional[int] = None
    # Hidden size
    hidden_size: Optional[int] = None
    # Attention backend for the frozen training backbone. CORE recomputes its
    # teacher/features from captured Q/K, so returned attention matrices are
    # unnecessary and FlashAttention is safe.
    training_attn_implementation: str = "flash_attention_2"

    # ---- Dataset ----
    data_path: str = str(_PROJECT_ROOT / "data" / "LongAlpaca-12k_.csv")
    # Maximum tokenized training length. QK statistics are streamed in query
    # blocks without retaining a full attention matrix for every layer.
    max_seq_len: int = 4096
    # Most optimizer steps stay at 2K for throughput; every
    # ``long_seq_every`` steps uses max_seq_len to expose the indexer to the
    # actual 4K evaluation regime. Held-out validation always uses max_seq_len.
    train_short_seq_len: int = 2048
    long_seq_every: int = 8
    max_samples: Optional[int] = None  # None or 0: complete LongAlpaca dataset
    # Fraction of optimizer steps replaced by a deterministic, corpus-derived
    # multi-evidence associative-retrieval example.  This teaches one generic
    # ranking model to cover several separated query-relevant records; it does
    # not use RULER prompts, labels, or compression-ratio-specific targets.
    retrieval_augmentation_fraction: float = 0.0

    # ---- KV Eviction Budget ----
    # Default inference eviction ratio. Boundary supervision uses the
    # configured training budgets; KL is independent of the Top-B cutoff.
    compression_ratio: float = 0.5  # Inference-time default evict fraction
    budget: Optional[int] = None  # If set, overrides compression_ratio at inference
    # Width of the retention-fraction sampling interval around
    # 1 - compression_ratio. Zero disables budget sampling.
    retention_range: float = 0.0
    # Formal multi-budget boundary supervision for one reusable checkpoint.
    training_compression_ratios: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)
    # Use the same teacher distribution to supervise every formal Top-B
    # cutoff on each prefill step.  This removes the avoidable variance from
    # showing the boundary objective only one randomly selected budget while
    # leaving the budget-independent KL term and inference unchanged.
    all_budget_boundary_supervision: bool = True

    # ---- Feature Extraction ----
    # Prototype bank size for coverage features
    n_prototypes: int = 64
    # Center projected keys before computing prototype features.
    center_prototype_features: bool = True

    proto_dim: int = 64
    # Normalize each KV head, concatenate heads, then use one fixed Gaussian
    # projection per transformer layer. The same z-space is consumed by the
    # teacher, prototype features, and online inference.
    key_geometry_mode: str = "head_concat_layer_gaussian"
    key_geometry_eps: float = 1e-8
    # Shrinkage factor for variance estimation
    shrinkage_rho: float = 0.1
    # Number of sink tokens (never evict)
    n_sink: int = 4
    # Fixed sink tokens are force-kept within the KV budget and therefore are
    # not ranking candidates.  Exclude them from the teacher/student KL,
    # and boundary loss. Memory uses the full-C allocation of Eq. 6/10.
    exclude_sink_from_kl: bool = True
    # K for TopMean aggregation in q-k relation features
    topk_qk: int = 4
    # Query-block size for streaming Q-K feature computation.
    qk_query_block_size: int = 128
    # QK statistic: attention_advantage uses log-normalized attention;
    # raw_logit_max uses unnormalized attention logits.
    qk_primary_mode: str = "attention_advantage"
    # Prefill queries used for QK relevance. Question scope requires
    # a reliable appended-question boundary.
    prefill_qk_scope: str = "all"
    # Number of recent queries for f_4
    n_recent_queries: int = 16
    # Token ids treated as structural/special by a_sink. Populated from the
    # tokenizer at training setup and persisted in the checkpoint.
    special_token_ids: tuple[int, ...] = field(default_factory=tuple)

    # ---- Teacher Distribution (DRIVE fixed-point iteration) ----
    # DRIVE prior/KL temperature tau used inside the fixed-point update (Eq.5):
    #   w_i ∝ exp((c_i + beta*h_i) / tau)
    drive_temp: float = 8.0
    # Distillation temperature tau_T, applied after DRIVE
    # and configured independently of drive_temp.
    teacher_temp: float = 1.0
    # Local utility smoothing window and mixing fraction.
    teacher_local_window: int = 1
    teacher_local_mix: float = 0.0
    # Number of DRIVE fixed-point iterations.
    teacher_n_iter: int = 8
    # D-optimal coverage weight beta.
    beta_coverage: float = 128
    # Scale alpha for the weighted covariance and its marginal coverage gain.
    alpha_coverage: float = 1.0

    # ---- Indexer (Student) ----
    # Width of each per-layer indexer MLP. Active and allocated parameter
    # counts also depend on scoring_stride and compact_indexer_stack.
    indexer_hidden_dim: int = 128
    # Allocate indexer MLPs only for the layers that compute scores.
    compact_indexer_stack: bool = True
    # Student temperature tau_m: z_i = s_i / tau_m.
    # Training and inference memory writing must use the same value.
    student_temp: float = 1.0

    # ---- Learned memory writing/readout ----
    # Evicted tokens update per-layer fast weights M_t/b_t.
    # The learned gated readout is added to subsequent attention outputs.
    # False disables memory writing and readout.
    enable_memory_writing: bool = True
    # γ - memory decay across compression batches. Larger keeps older evicted
    # batches alive longer (slower forgetting).
    memory_decay: float = 0.9
    # Deprecated fixed value-folding knobs, retained only for loading older
    # configs/checkpoints. The learned MemoryModule path does not use them.
    memory_lambda: float = 0.1
    memory_clamp_ratio: float = 0.5

    # ---- Memory Module Training ----
    # Skip the memory reconstruction training stage when enabled.
    skip_memory_training: bool = False
    # Memory-only training settings. Joint training uses
    # online_lr_peak and online_max_steps.
    memory_train_lr: float = 1e-2          # Joint training uses online_lr_peak
    memory_train_epochs: int = 15          # Joint training uses online_max_steps
    memory_train_lambda: float = 1.0     # λ_mem: weight of the paper MSE
    # Kept only so historical checkpoints remain loadable. New training rejects
    # True: CORE uses the unweighted, unnormalized reconstruction MSE.
    memory_relative_loss: bool = False
    # Number of initial steps with the memory gate fixed at one.
    # Zero keeps the gate learned throughout training.
    memory_gate_warmup_steps: int = 0
    # Allow the memory loss to update conditional student write weights.
    # Hard Top-B membership remains non-differentiable.
    memory_backprop_to_indexer: bool = False
    # Freeze indexer parameters during joint training.
    freeze_indexer_during_joint: bool = False
    # Number of trailing queries used for memory reconstruction supervision.
    # Memory is written from the preceding prefix; the suffix stays in attention.
    memory_supervision_tail: int = 16
    # Memory projection width, gate width, and cross-layer parameter sharing.
    memory_dim: int = 512
    memory_gate_hidden: int = 64
    memory_share_across_layers: bool = False
    # Block-diagonal groups in each layer's Linear_theta. One is the original
    # dense projection; four retains d_mem=512 with one quarter of its weights.
    memory_projection_groups: int = 1
    memory_write_eta: float = 1.0  # η: fast-weight write scaling
    # Fraction of uniform mass mixed into score-conditioned memory writes.
    # Zero uses conditional student weights; one uses uniform weights.
    memory_uniform_fraction: float = 0.0
    # Query coordinates used by the memory projection on reads.
    # The adapter also supports checkpoints trained in hidden-state coordinates.
    memory_query_space: str = "post_rope_q"
    # Store fast-memory values before the attention output projection. Because
    # o_proj is linear, projecting the single query readout is exactly
    # equivalent to projecting every evicted value before the outer products,
    # while removing the dominant 4K memory-training/inference GEMMs.
    memory_value_space: str = "pre_o_proj"


    # Weight of the calibrated KL loss.
    lambda_cal: float = 1.0
    # Weight of the boundary loss.
    lambda_bd: float = 4.0
    # Boundary loss formulation.
    boundary_loss_type: str = "pairwise_hinge"
    # With IndexCache-style score reuse, one student score vector serves an
    # entire stride group. Build one consensus boundary target per group instead
    # of imposing potentially contradictory per-layer P/N windows on it.
    aggregate_stride_boundary: bool = True
    # Number of tokens on each side of the Top-B boundary window.
    boundary_delta: int = 32
    # Boundary margin in temperature-normalized score space z = s / tau_m.
    boundary_margin: float = 0.5
    # Fixed LogSumExp sharpness used only when boundary_loss_type="lse_hinge".
    # The primary pairwise_hinge objective does not read this value.
    bd_beta: float = 5.0
    # Learning rate
    lr: float = 1e-3
    # Batch size (number of tokens per batch)
    batch_size: int = 2048
    # Max training epochs
    max_epochs: int = 10
    # Gradient clipping
    grad_clip: float = 1.0

    # ---- Sink token protection (inference) ----
    # Force-keep the first N tokens (attention sinks + BOS/system markers).
    # Prevents attention destabilisation from dropping sink tokens.
    n_sink_protect: int = 4

    # ---- Recency prior (inference) ----
    # Add a recency bonus to the indexer score for the most recent tokens:
    #   final_score = score + recency_alpha * exp(-(T - pos) / recency_window)
    # Default alpha=0 keeps this optional policy disabled.
    recency_alpha: float = 0.0
    recency_window: int = 64

    # ---- Selection policy ----
    # Training and inference must use the same selection mode.
    span_block_size: int = 1
    span_selection_mode: str = "tokenwise"
    # Fraction of the non-reserved context budget assigned to tokenwise global
    # coverage before the remainder performs contiguous local completion.
    # Used only by span_selection_mode="coverage_local".
    coverage_fraction: float = 0.25
    # Sharpness of differentiable log-mean-exp block pooling.
    span_pool_beta: float = 5.0
    query_feature_weight: float = 0.0
    lexical_overlap_weight: float = 0.0
    # Optional lexical adjustment; zero disables it.
    identifier_lexical_weight: float = 0.0
    identifier_min_match_tokens: int = 20
    identifier_context_skip: int = 64
    identifier_span_size: int = 32
    # If enabled, force-keeps appended query tokens outside learned selection.
    protect_query_tokens: bool = False

    # ---- Online distillation training ----
    # Frozen-backbone forward passes per online batch.
    online_batch_size: int = 4
    # Total optimizer steps in the configured WSD schedule.
    online_max_steps: int = 5400
    # Indexer-only steps before joint indexer and memory training.
    indexer_pretrain_steps: int = 1000
    # Add a decode-window supervision batch every N global steps. 0 disables.
    decode_train_every: int = 4
    decode_train_window: int = 128
    decode_loss_weight: float = 1.0
    # WSD schedule: warmup steps -> stable steps -> decay steps.
    online_warmup: int = 100
    online_stable: int = 3100
    online_decay: int = 2200
    # Peak and final learning rates for the WSD schedule.
    online_lr_peak: float = 5e-4
    online_lr_final: float = 7.5e-6
    # Learning-rate multipliers for the joint-training optimizer groups.
    joint_indexer_lr_scale: float = 0.1
    joint_memory_lr_scale: float = 1.0
    # A joint checkpoint may improve memory but is rejected if its held-out
    # Top-B recall falls farther than this below the Stage-II starting state.
    joint_max_topb_drop: float = 1.0
    # How often (in steps) to log training metrics.
    online_log_every: int = 25
    # How often (in steps) to run evaluation on a held-out chunk.
    online_eval_every: int = 200
    # Held-out validation sample count. Zero uses all samples for training.
    online_val_samples: int = 0
    # Early-stopping patience. Zero uses the fixed-step schedule.
    # Stage I monitors Top-B recall; Stage II monitors joint validation loss.
    stage1_early_stop_patience: int = 0
    stage1_early_stop_min_steps: int = 0
    stage1_early_stop_min_delta: float = 0.0
    stage2_early_stop_patience: int = 0
    stage2_early_stop_min_steps: int = 0
    stage2_early_stop_min_delta: float = 0.0
    # How often (in steps) to write an intermediate joint checkpoint
    # (indexer + memory stack) during train_joint_online. 0 disables
    # intermediate saves.
    online_save_every: int = 500

    # ---- Inference: per-layer scoring ----
    # Compute scores every N layers and reuse within each group.
    # One computes scores independently for every layer.
    scoring_stride: int = 1

    # ---- Paths ----
    output_dir: str = str(_PROJECT_ROOT / "experiments" / "paper")
    # Device for backbone inference (feature collection)
    backbone_device: str = "cuda:0"
    # Device for indexer training
    train_device: str = "cuda:0"

    # ---- Misc ----
    seed: int = 42
    geometry_seed: int = 42  # Fixed across training-seed comparisons (Appendix E.5).
    dtype: str = "bfloat16"


def validate_checkpoint_interface(config: dict) -> None:
    """Check actual tensor coordinate contracts, independently of version labels."""
    required = {
        "memory_query_space": "post_rope_q",
        "memory_value_space": "pre_o_proj",
        "key_geometry_mode": "head_concat_layer_gaussian",
    }
    for field, expected in required.items():
        if config.get(field) != expected:
            raise ValueError(
                f"Checkpoint interface mismatch: {field}={config.get(field)!r}; "
                f"the current method requires {expected!r}"
            )
