"""
Minimal inference script that plugs a trained CORE indexer into KVPress.

Supports two modes:

1. **Prefill + memory readout** (``--no-enable_decoding``): Compresses the context during
   prefill, writes evicted tokens into memory, and reads that memory while
   generating the answer.

2. **Prefill + Decode** (default): Uses a
   :class:`PrefillDecodingPress` that compresses during both prefilling and
   decoding phases.  During decoding the KV cache is periodically compressed
   down to a target size.

Usage examples:

    # Prefill compression + memory readout
    python -m core.infer_with_core_indexer \\
        --checkpoint_path ./experiments/paper/indexer/core_indexer.pt \\
        --context_file ./my_long_context.txt \\
        --question "What is the main conclusion?" \\
        --no-enable_decoding

    # Prefill + Decode compression
    python -m core.infer_with_core_indexer \\
        --checkpoint_path ./experiments/paper/indexer/core_indexer.pt \\
        --context_file ./my_long_context.txt \\
        --question "What is the main conclusion?" \\
        --decoding_compression_interval 128
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def _bootstrap_imports():
    repo_root = Path(__file__).resolve().parents[1]
    kvpress_root = repo_root / "kvpress"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(kvpress_root) not in sys.path:
        sys.path.insert(0, str(kvpress_root))


_bootstrap_imports()

import kvpress  # noqa: F401  # registers the kv-press-text-generation pipeline
import core.kvpress_pipeline  # noqa: F401  # registers CORE metadata hooks and pipeline
from core.config import COREConfig
from core.kvpress_adapter import (
    load_core_memory_press,
    load_core_prefill_decoding_press,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("core.infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KV compression with a trained CORE indexer"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./experiments/paper/indexer/core_indexer.pt",
        help="Path to the saved CORE checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=COREConfig().model_name,
        help="Backbone model to run inference with",
    )
    parser.add_argument(
        "--context_file",
        type=str,
        default=None,
        help="Optional text file containing a long context",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional inline context string; ignored if --context_file is set",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Please summarize the important details from the context.",
        help="Question to ask about the compressed context",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--compression_ratio", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Backbone attention implementation (CORE recomputes its own Q/K features)",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--output_answer_file",
        type=str,
        default=None,
        help="Optional file to save the answer",
    )

    # --- Decode-phase compression arguments ---
    parser.add_argument(
        "--enable_decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable decode-phase KV cache compression in addition to prefill. "
            "Uses PrefillDecodingPress internally."
        ),
    )
    parser.add_argument(
        "--decoding_compression_interval",
        type=int,
        default=128,
        help="Number of decoding steps between compression (only with --enable_decoding)",
    )
    parser.add_argument(
        "--decoding_target_size",
        type=int,
        default=None,
        help="Decode KV budget; by default uses the retained prefill budget",
    )
    parser.add_argument(
        "--decoding_hidden_states_buffer_size",
        type=int,
        default=None,
        help="Query buffer size; defaults to the compression interval and cannot be smaller",
    )
    return parser.parse_args()


def load_context(args: argparse.Namespace) -> str:
    if args.context_file:
        return Path(args.context_file).read_text(encoding="utf-8")
    if args.context:
        return args.context

    raise ValueError("Please provide --context_file or --context for a real long-context inference run.")


def main() -> None:
    args = parse_args()
    context = load_context(args)

    # --- Load press ---
    if args.enable_decoding:
        logger.info("Loading CORE PrefillDecodingPress from %s", args.checkpoint_path)
        logger.info(
            "Decode settings: compression_interval=%d, target_size=%s, buffer_size=%s",
            args.decoding_compression_interval,
            args.decoding_target_size,
            args.decoding_hidden_states_buffer_size,
        )
        press = load_core_prefill_decoding_press(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            compression_ratio=args.compression_ratio,
            decoding_compression_interval=args.decoding_compression_interval,
            decoding_target_size=args.decoding_target_size,
            decoding_hidden_states_buffer_size=args.decoding_hidden_states_buffer_size,
        )
    else:
        logger.info("Loading CORE prefill press (indexer + memory) from %s", args.checkpoint_path)
        press = load_core_memory_press(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            compression_ratio=args.compression_ratio,
        )

    # --- Load model ---
    logger.info("Loading model %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    first_param_device = next(model.parameters()).device
    logger.info("Model first parameter device: %s", first_param_device)
    if hasattr(model, "hf_device_map"):
        logger.info("Model device map: %s", model.hf_device_map)

    text_pipe = pipeline("core-kv-press-text-generation", model=model, tokenizer=tokenizer)

    # --- Run inference ---
    logger.info(
        "Running KV compression inference (mode=%s)",
        "prefill+decode-compression+memory"
        if args.enable_decoding
        else "prefill-compression+memory-readout",
    )
    with torch.inference_mode():
        result = text_pipe(
            context,
            question=args.question,
            press=press,
            max_new_tokens=args.max_new_tokens,
        )

    answer = result["answer"]
    print("\n=== ANSWER ===\n")
    print(answer)

    if args.output_answer_file:
        Path(args.output_answer_file).write_text(answer, encoding="utf-8")
        logger.info("Saved answer to %s", args.output_answer_file)


if __name__ == "__main__":
    main()
