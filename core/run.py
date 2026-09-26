"""
CORE indexer training - main entry point.

Pipeline (online distillation, per-layer indexer + memory stacks):
  Stage 1: Pretrain a per-layer COREIndexerStack via online distillation.
           Each step runs a frozen backbone forward, computes per-layer
           teacher (DRIVE) + 14-dim features in-loop, and backprops through
           the indexer stack only, without on-disk Phase-1 pre-collection.
  Stage 2: Jointly train the indexer and per-layer MemoryModuleStack for the
           remainder of the same 5400-step WSD schedule.

Usage:
    python -m core.run                     # default config
    python -m core.run --max_samples 100   # quick test
    python -m core.run --online_max_steps 50 --max_seq_len 1024   # smoke test

    # Resume a staged/joint checkpoint from its last logged global step.
    python -m core.run --skip_to_joint
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import COREConfig, validate_checkpoint_interface
from core.features import COREFeatureExtractor
from core.teacher import DiversityAwareTeacher
from core.train import (
    load_indexer_stack,
    load_memory_stack,
    save_joint_stack,
    save_indexer_stack,
    train_indexer_online,
    train_joint_online,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("core")


def _summarize_training_history(history: list, memory_history: list) -> dict:
    """Persist final losses together with the best held-out validation."""
    train_entries = [item for item in history if "loss" in item]
    validation_entries = [
        item for item in history if "val_topB_recall" in item
    ]
    best_validation = max(
        validation_entries,
        key=lambda item: (item["val_topB_recall"], -item["val_kl"]),
        default={},
    )
    metrics = {
        "final_loss": train_entries[-1].get("loss") if train_entries else None,
        "final_mem_loss": (
            memory_history[-1].get("loss") if memory_history else None
        ),
        "n_steps": max(
            (int(item.get("step", 0)) for item in history), default=0
        ),
    }
    metrics.update({
        key: value for key, value in best_validation.items()
        if key.startswith("val_")
    })
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORE indexer training (online distillation)")
    # Common overrides
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--geometry_seed", type=int, default=None)
    p.add_argument("--beta_coverage", type=float, default=None)
    p.add_argument("--alpha_coverage", type=float, default=None)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional sample cap for smoke tests; 0/default uses all rows.")
    p.add_argument(
        "--retrieval_augmentation_fraction", type=float, default=None,
        help=(
            "Fraction of training steps using corpus-derived multi-evidence "
            "retrieval examples (0 disables; no RULER data is used)."
        ),
    )
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--train_short_seq_len", type=int, default=None)
    p.add_argument(
        "--long_seq_every", type=int, default=None,
        help="Use a max_seq_len example every N global steps (1 = always).",
    )
    p.add_argument("--compression_ratio", type=float, default=None)
    p.add_argument("--qk_query_block_size", type=int, default=None)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--backbone_device", type=str, default=None)
    p.add_argument("--train_device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    # Online training overrides
    p.add_argument("--online_max_steps", type=int, default=None,
                   help="Total optimizer steps in the configured WSD schedule.")
    p.add_argument(
        "--online_batch_size", type=int, default=None,
        help="Sequential microbatches averaged before each optimizer step.",
    )
    p.add_argument("--online_lr_peak", type=float, default=None)
    p.add_argument("--joint_indexer_lr_scale", type=float, default=None)
    p.add_argument("--joint_memory_lr_scale", type=float, default=None)
    p.add_argument("--joint_max_topb_drop", type=float, default=None)
    p.add_argument(
        "--indexer_hidden_dim", type=int, default=None,
        help="Width of each per-layer CORE MLP; active count also depends on scoring_stride.",
    )
    p.add_argument(
        "--compact_indexer_stack",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--memory_dim", type=int, default=None)
    p.add_argument("--memory_gate_hidden", type=int, default=None)
    p.add_argument(
        "--memory_projection_groups", type=int, default=None,
        help=(
            "Block-diagonal groups in every layer's memory projection "
            "(1=dense; 4 uses one quarter of the projection parameters)."
        ),
    )
    p.add_argument("--teacher_n_iter", type=int, default=None)
    p.add_argument("--memory_train_lambda", type=float, default=None)
    p.add_argument(
        "--freeze_indexer_during_joint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze a validated stage-1 selector and train only memory in stage 2.",
    )
    p.add_argument("--memory_gate_warmup_steps", type=int, default=None)
    p.add_argument("--memory_uniform_fraction", type=float, default=None)
    p.add_argument("--memory_supervision_tail", type=int, default=None)
    p.add_argument(
        "--skip_memory_training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For selector-only ablations on the legacy --no-joint_training path.",
    )
    p.add_argument(
        "--memory_share_across_layers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Share memory slow weights across layers (disabled in the primary configuration).",
    )
    p.add_argument("--lambda_bd", type=float, default=None)
    p.add_argument("--boundary_margin", type=float, default=None)
    p.add_argument(
        "--boundary_loss_type",
        choices=("pairwise_hinge", "hard_hinge", "lse_hinge", "all_pairs_softplus"),
        default=None,
    )
    p.add_argument(
        "--aggregate_stride_boundary", action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--all_budget_boundary_supervision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Average boundary supervision over every configured training CR "
            "on each prefill forward; KL and inference are unchanged."
        ),
    )
    p.add_argument("--online_log_every", type=int, default=None,
                   help="Log training metrics every N steps (default 25).")
    p.add_argument("--online_val_samples", type=int, default=None)
    p.add_argument(
        "--online_eval_every", type=int, default=None,
        help="Run held-out training validation every N optimizer steps.",
    )
    p.add_argument("--stage1_early_stop_patience", type=int, default=None)
    p.add_argument("--stage1_early_stop_min_steps", type=int, default=None)
    p.add_argument("--stage1_early_stop_min_delta", type=float, default=None)
    p.add_argument("--stage2_early_stop_patience", type=int, default=None)
    p.add_argument("--stage2_early_stop_min_steps", type=int, default=None)
    p.add_argument("--stage2_early_stop_min_delta", type=float, default=None)
    p.add_argument(
        "--online_save_every", type=int, default=None,
        help="Save a recoverable joint checkpoint every N global steps (0 disables).",
    )
    p.add_argument("--indexer_pretrain_steps", type=int, default=None,
                   help="Indexer-only steps within the total WSD schedule (default 1000).")
    p.add_argument("--online_warmup", type=int, default=None)
    p.add_argument("--online_stable", type=int, default=None)
    p.add_argument("--online_decay", type=int, default=None)
    p.add_argument("--decode_train_every", type=int, default=None,
                   help="Add decode-window supervision every N global steps (0 disables).")
    p.add_argument("--decode_train_window", type=int, default=None)
    p.add_argument("--decode_loss_weight", type=float, default=None)
    p.add_argument("--drive_temp", type=float, default=None)
    p.add_argument("--teacher_temp", type=float, default=None)
    p.add_argument("--teacher_local_window", type=int, default=None)
    p.add_argument("--teacher_local_mix", type=float, default=None)
    p.add_argument(
        "--exclude_protected_query_from_kl",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--qk_primary_mode",
        choices=("attention_advantage", "raw_logit_max"),
        default=None,
        help="Feature-0 QK statistic; raw_logit_max aligns with teacher utility.",
    )
    p.add_argument(
        "--prefill_qk_scope", choices=("all", "question"), default=None,
        help="Use all prompt queries (legacy) or only the appended question for QK features.",
    )
    p.add_argument("--recency_alpha", type=float, default=None,
                   help="Recency prior strength (0 disables, try 0.1-1.0).")
    p.add_argument("--n_sink_protect", type=int, default=None)
    p.add_argument(
        "--exclude_sink_from_kl",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument("--scoring_stride", type=int, default=None,
                   help="Inference scoring stride (1 = per-layer, 4 = IndexCache reuse).")
    p.add_argument(
        "--span_block_size",
        type=int,
        default=None,
        help=(
            "Contiguous evidence-block width. Production default is 64; use 1 "
            "for a tokenwise ablation."
        ),
    )
    p.add_argument("--span_pool_beta", type=float, default=None)
    p.add_argument(
        "--coverage_fraction", type=float, default=None,
        help="Global token-coverage share for coverage_local selection.",
    )
    p.add_argument(
        "--span_selection_mode",
        choices=(
            "tokenwise", "block_lme", "sliding_max", "coverage_local",
            "coverage_local_balanced",
        ),
        default=None,
    )
    p.add_argument(
        "--protect_query_tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--center_prototype_features",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--skip_to_memory", action="store_true",
        help="Skip Phase 2; load a trained indexer stack from checkpoint. "
             "(Legacy: pre-joint-training Phase 2.5 path. Prefer --skip_to_joint.)",
    )
    p.add_argument(
        "--skip_to_joint", action="store_true",
        help="Skip fresh training; load a trained indexer + memory stack from "
             "checkpoint and resume from the checkpoint's last global step "
             "(useful to keep "
             "training a previously-saved joint checkpoint).",
    )
    p.add_argument(
        "--resume_indexer_only", action="store_true",
        help="Continue selector-only training from --resume_checkpoint (or the "
             "output directory checkpoint) without allocating memory supervision.",
    )
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument(
        "--joint_training", action="store_true", default=True,
        help="Use joint training of indexer + per-layer memory module "
             "(IndexMem §3.2). On by default; pass --no-joint_training to "
             "fall back to the legacy two-phase (Phase 2 indexer then Phase 2.5 "
             "memory) path for backward compatibility.",
    )
    p.add_argument(
        "--no-joint_training", dest="joint_training", action="store_false",
        help="Disable joint training and use the legacy two-phase path.",
    )
    return p.parse_args()


def _restore_checkpoint_config(
    config: COREConfig, checkpoint_config: dict
) -> COREConfig:
    """Restore known persisted fields while ignoring obsolete metadata."""
    for key, value in checkpoint_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def build_config(args: argparse.Namespace) -> COREConfig:
    """Build COREConfig from CLI overrides."""
    config = COREConfig()
    # Restore checkpoint configuration, then apply explicit CLI overrides.
    resume_path = None
    if args.skip_to_joint:
        resume_dir = Path(args.output_dir or config.output_dir)
        resume_path = resume_dir / "indexer" / "core_indexer.pt"
    elif args.resume_indexer_only:
        resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else (
            Path(args.output_dir or config.output_dir) / "indexer" / "core_indexer.pt"
        )
    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        validate_checkpoint_interface(checkpoint_config)
        _restore_checkpoint_config(config, checkpoint_config)
        logger.info("Restored training config from %s before CLI overrides", resume_path)

    overrides = {
        "seed": args.seed,
        "geometry_seed": args.geometry_seed,
        "beta_coverage": args.beta_coverage,
        "alpha_coverage": args.alpha_coverage,
        "model_name": args.model_name,
        "data_path": args.data_path,
        "max_samples": args.max_samples,
        "retrieval_augmentation_fraction": args.retrieval_augmentation_fraction,
        "max_seq_len": args.max_seq_len,
        "train_short_seq_len": args.train_short_seq_len,
        "long_seq_every": args.long_seq_every,
        "compression_ratio": args.compression_ratio,
        "qk_query_block_size": args.qk_query_block_size,
        "budget": args.budget,
        "backbone_device": args.backbone_device,
        "train_device": args.train_device,
        "output_dir": args.output_dir,
        "online_max_steps": args.online_max_steps,
        "online_batch_size": args.online_batch_size,
        "online_lr_peak": args.online_lr_peak,
        "joint_indexer_lr_scale": args.joint_indexer_lr_scale,
        "joint_memory_lr_scale": args.joint_memory_lr_scale,
        "joint_max_topb_drop": args.joint_max_topb_drop,
        "indexer_hidden_dim": args.indexer_hidden_dim,
        "compact_indexer_stack": args.compact_indexer_stack,
        "memory_dim": args.memory_dim,
        "memory_gate_hidden": args.memory_gate_hidden,
        "memory_projection_groups": args.memory_projection_groups,
        "memory_train_lambda": args.memory_train_lambda,
        "teacher_n_iter": args.teacher_n_iter,
        "freeze_indexer_during_joint": args.freeze_indexer_during_joint,
        "memory_gate_warmup_steps": args.memory_gate_warmup_steps,
        "memory_uniform_fraction": args.memory_uniform_fraction,
        "memory_supervision_tail": args.memory_supervision_tail,
        "skip_memory_training": args.skip_memory_training,
        "memory_share_across_layers": args.memory_share_across_layers,
        "lambda_bd": args.lambda_bd,
        "boundary_margin": args.boundary_margin,
        "boundary_loss_type": args.boundary_loss_type,
        "aggregate_stride_boundary": args.aggregate_stride_boundary,
        "all_budget_boundary_supervision": args.all_budget_boundary_supervision,
        "online_log_every": args.online_log_every,
        "online_val_samples": args.online_val_samples,
        "online_eval_every": args.online_eval_every,
        "stage1_early_stop_patience": args.stage1_early_stop_patience,
        "stage1_early_stop_min_steps": args.stage1_early_stop_min_steps,
        "stage1_early_stop_min_delta": args.stage1_early_stop_min_delta,
        "stage2_early_stop_patience": args.stage2_early_stop_patience,
        "stage2_early_stop_min_steps": args.stage2_early_stop_min_steps,
        "stage2_early_stop_min_delta": args.stage2_early_stop_min_delta,
        "online_save_every": args.online_save_every,
        "indexer_pretrain_steps": args.indexer_pretrain_steps,
        "online_warmup": args.online_warmup,
        "online_stable": args.online_stable,
        "online_decay": args.online_decay,
        "decode_train_every": args.decode_train_every,
        "decode_train_window": args.decode_train_window,
        "decode_loss_weight": args.decode_loss_weight,
        "drive_temp": args.drive_temp,
        "teacher_temp": args.teacher_temp,
        "teacher_local_window": args.teacher_local_window,
        "teacher_local_mix": args.teacher_local_mix,
        "exclude_protected_query_from_kl": args.exclude_protected_query_from_kl,
        "qk_primary_mode": args.qk_primary_mode,
        "prefill_qk_scope": args.prefill_qk_scope,
        "recency_alpha": args.recency_alpha,
        "n_sink_protect": args.n_sink_protect,
        "exclude_sink_from_kl": args.exclude_sink_from_kl,
        "scoring_stride": args.scoring_stride,
        "span_block_size": args.span_block_size,
        "span_pool_beta": args.span_pool_beta,
        "coverage_fraction": args.coverage_fraction,
        "span_selection_mode": args.span_selection_mode,
        "protect_query_tokens": args.protect_query_tokens,
        "center_prototype_features": args.center_prototype_features,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(config, k, v)
    return config


def setup_model_and_tokenizer(config: COREConfig):
    """Load backbone model and tokenizer; auto-detect dims."""
    logger.info(f"Loading model from {config.model_name}")
    dtype = getattr(torch, config.dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Teacher utility and all 14 CORE features are recomputed from captured
    # Q/K/V, so training does not need returned attention matrices. Use the
    # same efficient backend as evaluation, especially for 4K sequences.
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.backbone_device,
        attn_implementation=config.training_attn_implementation,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Auto-detect dims
    cfg = model.config
    config.n_layers = cfg.num_hidden_layers
    config.n_q_heads = cfg.num_attention_heads
    config.n_kv_heads = cfg.num_key_value_heads
    config.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    config.hidden_size = cfg.hidden_size
    config.special_token_ids = tuple(int(x) for x in tokenizer.all_special_ids)

    logger.info(
        f"Model: {cfg.architectures}, layers={config.n_layers}, "
        f"q_heads={config.n_q_heads}, kv_heads={config.n_kv_heads}, "
        f"head_dim={config.head_dim}"
    )
    return model, tokenizer


def main():
    args = parse_args()
    config = build_config(args)
    if not args.joint_training:
        raise ValueError("The legacy two-phase memory path is not paper-aligned; use default joint_training")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("=" * 60)
    logger.info("CORE Indexer Online Distillation Training")
    logger.info("=" * 60)
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(
        f"compression_ratio={config.compression_ratio}, max_seq_len={config.max_seq_len}, "
        f"online_max_steps={config.online_max_steps}, max_samples={config.max_samples}, "
        f"effective_batch={config.online_batch_size}, boundary_margin={config.boundary_margin}"
    )
    logger.info(f"Protection: n_sink_protect={config.n_sink_protect}, "
                f"recency_alpha={config.recency_alpha}, scoring_stride={config.scoring_stride}")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Setup backbone (needed for online distillation) ----
    model, tokenizer = setup_model_and_tokenizer(config)

    # Estimate trainable slow-weight parameters.
    # Fast memory state is not an optimizer parameter.
    h = int(config.indexer_hidden_dim)
    n_layers = int(config.n_layers)
    indexer_params = n_layers * (h * h + 17 * h + 1)
    active_indexer_modules = (
        n_layers + max(1, int(config.scoring_stride)) - 1
    ) // max(1, int(config.scoring_stride))
    active_indexer_params = active_indexer_modules * (h * h + 17 * h + 1)
    allocated_indexer_params = (
        active_indexer_params
        if config.compact_indexer_stack else indexer_params
    )
    gate_h = int(config.memory_gate_hidden)
    if args.memory_projection_groups is None and not (args.skip_to_joint or args.resume_indexer_only):
        config.memory_projection_groups = 4 if getattr(model.config, "model_type", "") == "qwen3" else 1
    projection_groups = int(config.memory_projection_groups)
    memory_width = int(config.n_q_heads) * int(config.head_dim)
    if projection_groups < 1:
        raise ValueError("memory_projection_groups must be positive")
    if (
        memory_width % projection_groups != 0
        or int(config.memory_dim) % projection_groups != 0
    ):
        raise ValueError(
            "H_q * head_dim and memory_dim must both be divisible by "
            "memory_projection_groups"
        )
    memory_per_module = (
        memory_width * int(config.memory_dim) // projection_groups
        + memory_width * gate_h + 2 * gate_h + 1
    )
    memory_modules = 1 if config.memory_share_across_layers else n_layers
    memory_params = memory_modules * memory_per_module
    logger.info(
        "Capacity plan: indexer logical=%.2fM, active=%.2fM, allocated=%.2fM "
        "parameters (hidden=%d, stride=%d, compact=%s), "
        "memory slow weights=%.2fM parameters "
        "(d_mem=%d, projection_groups=%d, shared=%s)",
        indexer_params / 1e6, active_indexer_params / 1e6,
        allocated_indexer_params / 1e6, h, config.scoring_stride,
        config.compact_indexer_stack, memory_params / 1e6,
        config.memory_dim, projection_groups, config.memory_share_across_layers,
    )
    feature_extractor = COREFeatureExtractor(config)
    feature_extractor.to(torch.device(config.backbone_device))
    teacher = DiversityAwareTeacher(config)

    if args.resume_indexer_only:
        ckpt_path = Path(args.resume_checkpoint) if args.resume_checkpoint else (
            Path(config.output_dir) / "indexer" / "core_indexer.pt"
        )
        indexer, checkpoint = load_indexer_stack(
            str(ckpt_path), device=config.train_device
        )
        if "feature_extractor_state_dict" in checkpoint:
            feature_extractor.load_geometry_state_dict(
                checkpoint["feature_extractor_state_dict"]
            )
        old_history = checkpoint.get("history", [])
        completed_step = max(
            (int(item.get("step", 0)) for item in old_history), default=0
        )
        remaining_steps = max(0, config.online_max_steps - completed_step)
        if remaining_steps <= 0:
            raise ValueError(
                f"Checkpoint is already at step {completed_step}; set "
                "--online_max_steps to a larger target."
            )
        indexer, new_history = train_indexer_online(
            config, model, tokenizer, feature_extractor, teacher,
            indexer=indexer, max_steps=remaining_steps,
            schedule_offset=completed_step,
        )
        save_indexer_stack(
            indexer, config, old_history + new_history,
            feature_extractor=feature_extractor,
        )
        logger.info(
            "Selector-only resume complete: step %d -> %d, saved to %s",
            completed_step, config.online_max_steps,
            Path(config.output_dir) / "indexer" / "core_indexer.pt",
        )
        return

    # ================================================================
    # Joint training path (default, IndexMem §3.2)
    # ================================================================
    # indexer + per-layer MemoryModuleStack are trained together over a single
    # WSD 5400-step schedule. A single forward per step feeds both losses.
    if getattr(args, "joint_training", True):
        if args.skip_to_joint:
            ckpt_path = Path(config.output_dir) / "indexer" / "core_indexer.pt"
            indexer, checkpoint = load_indexer_stack(
                str(ckpt_path), device=config.train_device
            )
            if "feature_extractor_state_dict" in checkpoint:
                feature_extractor.load_geometry_state_dict(
                    checkpoint["feature_extractor_state_dict"]
                )
            validate_checkpoint_interface(checkpoint["config"])
            history = checkpoint.get("history", [])
            metrics = checkpoint.get("metrics", {})
            try:
                memory_stack = load_memory_stack(str(ckpt_path), device=config.train_device)
                memory_history = checkpoint.get("memory_history", [])
                logger.info(
                    f"Loaded indexer stack + memory stack from {ckpt_path} "
                    "(skip_to_joint). Resuming joint training."
                )
            except KeyError:
                memory_stack = None
                memory_history = []
                logger.warning(
                    f"Checkpoint at {ckpt_path} has no memory_state_dict; "
                    "skip_to_joint will start memory stack from scratch."
                )
            completed_step = max(
                (int(item.get("step", 0)) for item in history), default=0
            )
            remaining_steps = max(0, config.online_max_steps - completed_step)
            if remaining_steps > 0:
                indexer, memory_stack, resumed_history, resumed_memory_history = (
                    train_joint_online(
                        config, model, tokenizer, feature_extractor, teacher,
                        indexer=indexer, memory_stack=memory_stack,
                        max_steps=remaining_steps,
                        schedule_offset=completed_step,
                        optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
                        scheduler_state_dict=checkpoint.get("scheduler_state_dict"),
                    )
                )
                history = history + resumed_history
                memory_history = memory_history + resumed_memory_history
            else:
                logger.info(
                    "Checkpoint already reached global step %d; nothing to resume.",
                    completed_step,
                )
            metrics = _summarize_training_history(history, memory_history)
        else:
            total_steps = max(1, config.online_max_steps)
            pretrain_steps = min(
                max(0, config.indexer_pretrain_steps), max(0, total_steps - 1)
            )
            pretrain_history = []
            indexer = None
            if pretrain_steps > 0:
                logger.info("=" * 60)
                logger.info(
                    "Stage 1/2: indexer-only pretraining (%d global steps)",
                    pretrain_steps,
                )
                logger.info("=" * 60)
                indexer, pretrain_history = train_indexer_online(
                    config, model, tokenizer, feature_extractor, teacher,
                    max_steps=pretrain_steps, schedule_offset=0,
                )
                # Save a recoverable stage-1 checkpoint before allocating the
                # memory stack. --skip_to_joint can continue from this file.
                save_indexer_stack(
                    indexer, config, pretrain_history,
                    feature_extractor=feature_extractor,
                )
                # Start joint memory training from the selector chosen by
                # held-out Top-B, rather than the final stage-1 optimizer step.
                best_stage1_path = (
                    Path(config.output_dir) / "indexer"
                    / "core_indexer_best_stage1.pt"
                )
                if config.online_val_samples > 0 and best_stage1_path.exists():
                    indexer, best_stage1 = load_indexer_stack(
                        str(best_stage1_path), device=config.train_device
                    )
                    if "feature_extractor_state_dict" in best_stage1:
                        feature_extractor.load_geometry_state_dict(
                            best_stage1["feature_extractor_state_dict"]
                        )
                    logger.info(
                        "Restored best stage-1 selector from %s",
                        best_stage1_path,
                    )

            completed_pretrain_steps = max(
                (int(item.get("step", 0)) for item in pretrain_history),
                default=pretrain_steps,
            )
            if completed_pretrain_steps != pretrain_steps:
                logger.info(
                    "Stage I ended at global step %d (configured cap %d)",
                    completed_pretrain_steps, pretrain_steps,
                )
                config.indexer_pretrain_steps = completed_pretrain_steps
            joint_steps = total_steps - completed_pretrain_steps
            logger.info("=" * 60)
            logger.info(
                "Stage 2/2: joint indexer + memory training (%d global steps)",
                joint_steps,
            )
            logger.info("=" * 60)
            indexer, memory_stack, joint_history, memory_history = train_joint_online(
                config, model, tokenizer, feature_extractor, teacher,
                indexer=indexer, max_steps=joint_steps,
                schedule_offset=completed_pretrain_steps,
            )
            history = pretrain_history + joint_history
            metrics = _summarize_training_history(history, memory_history)

        # Final save (intermediate saves already happened during training).
        save_joint_stack(
            indexer, memory_stack, config,
            history=history, memory_history=memory_history,
            feature_extractor=feature_extractor,
            metrics=metrics,
        )

        logger.info("=" * 60)
        logger.info("Joint training complete!")
        train_entries = [item for item in history if "loss" in item]
        if train_entries:
            last_train = train_entries[-1]
            logger.info(f"  Final loss:     {last_train['loss']:.4f}")
            logger.info(f"  Final KL:       {last_train['loss_cal']:.4f}")
            logger.info(f"  Final L_bd:     {last_train['loss_bd']:.4f}")
        if "val_topB_recall" in metrics:
            logger.info(
                "  Best held-out:   Top-B=%.4f KL=%.4f",
                metrics["val_topB_recall"], metrics["val_kl"],
            )
        if memory_history:
            logger.info(f"  Final L_mem:    {memory_history[-1].get('loss', '?'):.6f}")
            logger.info(f"  Final gap:      {memory_history[-1].get('gap', '?'):.6f}")
        logger.info(
            f"  Saved to: {Path(config.output_dir) / 'indexer' / 'core_indexer.pt'}"
        )
        logger.info("=" * 60)
        return

    # Sequential indexer-only and memory-only training (--no-joint_training).
    if args.skip_to_memory:
        ckpt_path = Path(config.output_dir) / "indexer" / "core_indexer.pt"
        indexer, checkpoint = load_indexer_stack(str(ckpt_path), device=config.train_device)
        history = checkpoint.get("history", [])
        metrics = checkpoint.get("metrics", {})
        logger.info(f"Loaded indexer stack from {ckpt_path} (skip_to_memory).")
    else:
        indexer, history = train_indexer_online(
            config, model, tokenizer, feature_extractor, teacher
        )
        metrics = _summarize_training_history(history, [])

    # ---- Phase 2.5: MemoryModule online training (IndexMem §3.2) ----
    memory_module = None
    memory_history: list = []
    if not config.skip_memory_training:
        from core.train import train_memory_module_online
        logger.info("=" * 60)
        logger.info("Phase 2.5: training MemoryModule (Linear_θ + gate)")
        logger.info("=" * 60)
        memory_module, memory_history = train_memory_module_online(
            config, model, tokenizer, feature_extractor, teacher
        )
        if memory_history:
            logger.info(f"  Final L_mem: {memory_history[-1].get('loss', '?'):.6f}")
            logger.info(f"  Final gap:   {memory_history[-1].get('gap', '?'):.6f}")
    else:
        logger.info("Phase 2.5 skipped (skip_memory_training=True).")

    # ---- Save ----
    save_indexer_stack(
        indexer, config, history, metrics=metrics,
        feature_extractor=feature_extractor,
        memory_module=memory_module,
        memory_history=memory_history,
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    # Report the most recent row containing each metric;
    # validation rows may omit training-loss fields.
    def _last_metric(rows, key):
        return next(
            (row[key] for row in reversed(rows) if isinstance(row.get(key), (int, float))),
            None,
        )

    for label, key, digits in (
        ("Final loss", "loss", 4),
        ("Final KL", "loss_cal", 4),
        ("Final L_bd", "loss_bd", 4),
    ):
        value = _last_metric(history, key)
        if value is not None:
            logger.info(f"  {label}: {value:.{digits}f}")
    value = _last_metric(memory_history, "loss")
    if value is not None:
        logger.info(f"  Final L_mem: {value:.6f}")
    logger.info(f"  Saved to: {Path(config.output_dir) / 'indexer' / 'core_indexer.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
