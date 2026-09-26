# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import time
from copy import deepcopy

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from datasets import load_dataset
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import FineGrainedFP8Config, Pipeline, pipeline

from core.kvpress_adapter import COREScorerPress

from kvpress import (
    ComposedPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    FinchPress,
    ObservedAttentionPress,
    PrefillDecodingPress,
    ScorerPress,
    ThinKPress,
)

logger = logging.getLogger(__name__)


CORE_PRESS_NAMES = ("core_indexer", "core_memory", "core_full")

LONGBENCH_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)

LONGBENCH_E_TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)


def _extract_literal_dict_from_python(file_path: Path, variable_name: str) -> dict:
    """Read a literal dictionary without executing the source module."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == variable_name:
                value = ast.literal_eval(value_node)
                if not isinstance(value, dict):
                    raise TypeError(f"{variable_name} in {file_path} is not a dict")
                return value
    raise KeyError(f"Could not find {variable_name!r} in {file_path}")


def _load_local_longbench(data_dir: str, task: str, use_longbench_e: bool) -> pd.DataFrame:
    """Load official LongBench JSONL and apply KVPress prompt formatting."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local LongBench directory does not exist: {root}")

    valid_tasks = LONGBENCH_E_TASKS if use_longbench_e else LONGBENCH_TASKS
    benchmark_name = "LongBench-E" if use_longbench_e else "LongBench"
    if task not in valid_tasks:
        raise ValueError(f"Invalid {benchmark_name} task {task!r}; expected one of {valid_tasks}")

    suffix = "_e" if use_longbench_e else ""
    jsonl_path = root / f"{task}{suffix}.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Local {benchmark_name} file not found: {jsonl_path}")

    preprocessing_file = (
        Path(__file__).resolve().parent
        / "benchmarks"
        / "longbench"
        / "create_huggingface_dataset.py"
    )
    if not preprocessing_file.is_file():
        raise FileNotFoundError(
            f"KVPress LongBench preprocessing file not found: {preprocessing_file}"
        )

    context_prefix = _extract_literal_dict_from_python(preprocessing_file, "context_prefix")
    question_template = _extract_literal_dict_from_python(preprocessing_file, "question_template")
    answer_prefix = _extract_literal_dict_from_python(preprocessing_file, "answer_prefix")
    max_new_tokens_map = _extract_literal_dict_from_python(
        preprocessing_file, "DATA_NAME_TO_MAX_NEW_TOKENS"
    )

    df = pd.read_json(jsonl_path, lines=True)
    required = {"input", "context", "answers", "all_classes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{jsonl_path} is missing LongBench columns: {sorted(missing)}")

    original_inputs = df["input"].astype(str).copy()
    df["context"] = df.apply(
        lambda row: context_prefix[task].format(**row.to_dict()), axis=1
    )

    def build_question(raw_input: str) -> str:
        if task == "trec":
            raw_input = raw_input.removesuffix("Type:")
        elif task == "triviaqa":
            raw_input = raw_input.removesuffix("Answer:")
        elif task == "samsum":
            raw_input = raw_input.removesuffix("Summary:")
        return question_template[task].format(input=raw_input)

    df["question"] = original_inputs.map(build_question)
    df["answer_prefix"] = answer_prefix.get(task, "")
    df["task"] = task
    df["max_new_tokens"] = int(max_new_tokens_map[task]) + 20
    logger.info("Loaded local %s task '%s' with %d samples", benchmark_name, task, len(df))
    return df


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "ruler"
    task: Optional[str] = None
    longbench_task: Optional[str] = None
    paper_protocol: bool = False
    data_dir: Optional[str] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[str] = None
    press_name: str = "knorm"
    compression_ratio: float = 1.0
    key_channel_compression_ratio: Optional[float] = None
    head_compression_ratio: Optional[float] = None
    threshold: Optional[float] = None
    core_indexer_checkpoint_path: Optional[str] = None
    core_indexer_compression_ratio: Optional[float] = None
    core_enable_memory: bool = True
    core_memory_gate_override: Optional[float] = None
    # Evaluation-only selector overrides.  None means: use the exact policy
    # persisted in the checkpoint.  These are useful for controlled span/query
    # protection ablations without editing or rewriting the checkpoint.
    core_span_block_size: Optional[int] = None
    core_span_pool_beta: Optional[float] = None
    core_span_selection_mode: Optional[str] = None
    core_coverage_fraction: Optional[float] = None
    core_protect_query_tokens: Optional[bool] = None
    core_query_feature_weight: Optional[float] = None
    core_lexical_overlap_weight: Optional[float] = None
    core_identifier_lexical_weight: Optional[float] = None
    core_identifier_min_match_tokens: Optional[int] = None
    core_identifier_context_skip: Optional[int] = None
    core_identifier_span_size: Optional[int] = None

    # Dataset and generation parameters
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    query_aware: bool = False
    needle_depth: Optional[int] = None

    # Decoding parameters
    compression_interval: Optional[int] = None
    target_size: Optional[int] = None
    hidden_states_buffer_size: Optional[int] = None

    # Output and logging
    output_dir: str = "./results"
    log_level: str = "INFO"

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42

    # Quantization
    fp8: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        if self.paper_protocol:
            if self.target_size is not None:
                raise ValueError("paper_protocol derives the decode budget from prefill; omit target_size")
            if self.query_aware:
                raise ValueError("paper_protocol uses the shared KVPress context/question split")
            incompatible = {
                "core_span_block_size": (None, 1),
                "core_span_selection_mode": (None, "tokenwise"),
                "core_protect_query_tokens": (None, False),
                "core_query_feature_weight": (None, 0.0),
                "core_lexical_overlap_weight": (None, 0.0),
                "core_identifier_lexical_weight": (None, 0.0),
                "core_memory_gate_override": (None,),
            }
            for field, allowed in incompatible.items():
                if getattr(self, field) not in allowed:
                    raise ValueError(f"{field} conflicts with paper_protocol; use paper_protocol=False for ablations")
            if (self.core_indexer_compression_ratio is not None
                    and self.core_indexer_compression_ratio != self.compression_ratio):
                raise ValueError("paper_protocol uses compression_ratio as the shared budget")
        if (self.dataset in ("longbench", "longbench-e") and self.data_dir
                and Path(self.data_dir).is_dir() and not self.longbench_task):
            raise ValueError("Local LongBench JSONL requires longbench_task")

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"

        if self.press_name == "no_press":
            # override compression_ratio to 0.0
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0

        if self.press_name in ("core_indexer", "core_memory", "core_full"):
            assert (
                self.core_indexer_checkpoint_path is not None
            ), f"{self.press_name} requires core_indexer_checkpoint_path"

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"

        # Initialize model_kwargs if None
        if self.model_kwargs is None:
            self.model_kwargs = {}

        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert self.max_context_length is not None, "max_context_length must be set for needle_in_haystack"

    def get_results_dir(self, output_dir: Path) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path

        Returns
        -------
        Path
            The path to the results directory
        """
        # Build directory name components
        components = [
            self.dataset,
            f"task-{self.task}" if self.task else "",
            Path(self.data_dir).name if self.data_dir else "",
            self.longbench_task or "",
            self.model.replace("/", "--"),
            self.press_name,
            f"{self.compression_ratio:.2f}",
        ]

        if self.threshold is not None:
            components[-1] = f"{self.threshold:.2f}"
        elif self.head_compression_ratio is not None:
            components[-1] = f"{self.head_compression_ratio:.2f}"
        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.paper_protocol:
            components.append("paper_v15_fixed_budget")
        if self.query_aware:
            components.append("query_aware")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")
        if self.press_name in ("core_indexer", "core_memory", "core_full") and self.core_indexer_checkpoint_path is not None:
            components.append(Path(self.core_indexer_checkpoint_path).stem)
        if self.press_name in ("core_indexer", "core_memory", "core_full") and self.core_indexer_compression_ratio is not None:
            components.append(f"core_indexer_cr{self.core_indexer_compression_ratio:.2f}")
        if self.press_name == "core_full" and not self.core_enable_memory:
            components.append("memory_off")
        if (
            self.press_name in ("core_memory", "core_full")
            and self.core_memory_gate_override is not None
        ):
            components.append(f"memorygate{self.core_memory_gate_override:g}")
        if (
            self.press_name in ("core_indexer", "core_memory", "core_full")
            and self.core_span_block_size is not None
        ):
            components.append(f"span{int(self.core_span_block_size)}")
        if (
            self.press_name in ("core_indexer", "core_memory", "core_full")
            and self.core_span_pool_beta is not None
        ):
            components.append(f"spanbeta{self.core_span_pool_beta:g}")
        if (
            self.press_name in ("core_indexer", "core_memory", "core_full")
            and self.core_span_selection_mode is not None
        ):
            components.append(f"spanmode-{self.core_span_selection_mode}")
        if (
            self.press_name in ("core_indexer", "core_memory", "core_full")
            and self.core_coverage_fraction is not None
        ):
            components.append(f"coverage{self.core_coverage_fraction:g}")
        if (
            self.press_name in ("core_indexer", "core_memory", "core_full")
            and self.core_protect_query_tokens is not None
        ):
            components.append(
                "queryprotect1" if self.core_protect_query_tokens else "queryprotect0"
            )
        if (
            self.press_name in ("core_indexer", "core_memory", "core_full")
            and self.core_query_feature_weight is not None
        ):
            components.append(f"qfeature{self.core_query_feature_weight:g}")
        if (
            self.press_name in ("core_indexer", "core_memory", "core_full")
            and self.core_lexical_overlap_weight is not None
        ):
            components.append(f"lexical{self.core_lexical_overlap_weight:g}")
        if (
            self.press_name in ("core_indexer", "core_memory", "core_full")
            and self.core_identifier_lexical_weight is not None
        ):
            components.append(
                f"identifier{self.core_identifier_lexical_weight:g}"
            )
        if (
            self.press_name in ("core_indexer", "core_memory", "core_full")
            and self.core_identifier_span_size is not None
        ):
            components.append(f"identifierspan{self.core_identifier_span_size}")

        dir_name = "__".join(filter(None, components))  # Filter None/empty strings
        config_dir = output_dir / dir_name

        # Make sure the directory does not exist, if it does, add a number to the end
        # This is to avoid overwriting results
        if config_dir.exists():
            i = 1
            while (config_dir / f"{i}").exists():
                i += 1
            config_dir = config_dir / f"{i}"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def save_config(self, config_filename: Path):
        """
        Saves the evaluation configuration to a YAML file.
        """
        config_dict = asdict(self)
        if self.threshold is not None or self.head_compression_ratio is not None:
            config_dict.pop("compression_ratio", None)
        if self.threshold is None:
            config_dict.pop("threshold", None)
        if self.head_compression_ratio is None:
            config_dict.pop("head_compression_ratio", None)
        with open(str(config_filename), "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path) -> dict:
    """Loads a YAML file. Returns an empty dict if it doesn't exist."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using only command-line arguments and defaults.")
        return {}


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : EvaluationConfig
        The configuration for the evaluation run.

    The final output will be predictions_<config>.csv and metrics_<config>.json in the output_dir.
    If the evaluation files already exist, evaluation will be skipped.

    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the EvaluationRunner with a given configuration.

        Parameters
        ----------
        config : EvaluationConfig
            The configuration for the evaluation run.
        """
        self.config = config
        self.pipeline: Optional[Pipeline] = None  # Will be set by _setup_model_pipeline()
        self.press: None | ScorerPress = None  # Will be set by _setup_press()
        self.df: Optional[pd.DataFrame] = None  # Will be set by _load_dataset()
        self._setup_logging()
        self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        """Set deterministic seeds for reproducible results."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic seeds to {self.config.seed}")

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.log_level.upper()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.

        Returns
        -------
        Path
            The path to the output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = deepcopy(PRESS_REGISTRY[press_name])

        if press_name in ("core_indexer", "core_memory", "core_full"):
            checkpoint_path = Path(
                str(self.config.core_indexer_checkpoint_path)
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"CORE checkpoint does not exist: {checkpoint_path}"
                )
            press_device = self.config.device
            if press_device is None or press_device == "auto":
                press_device = "cuda:0" if torch.cuda.is_available() else "cpu"

            kwargs = {
                "checkpoint_path": self.config.core_indexer_checkpoint_path,
                "device": press_device,
                # Use compression_ratio unless an explicit CORE-specific override is supplied.
                "compression_ratio": (
                    self.config.core_indexer_compression_ratio
                    if self.config.core_indexer_compression_ratio is not None
                    else compression_ratio
                ),
            }
            self.config.core_indexer_compression_ratio = float(
                kwargs["compression_ratio"]
            )
            if press_name == "core_full":
                kwargs.update(
                    decoding_compression_interval=self.config.compression_interval or 128,
                    decoding_target_size=self.config.target_size,
                    decoding_hidden_states_buffer_size=self.config.hidden_states_buffer_size,
                )
            press = press(**kwargs)

            # Resolve the underlying COREScorerPress.  core_indexer is already
            # a scorer; core_memory wraps it in COREMemoryPress.base_press; and
            # core_full wraps COREMemoryPress as PrefillDecodingPress.prefilling_press.
            if press_name == "core_full":
                core_wrapper = getattr(press, "prefilling_press", None)
            else:
                core_wrapper = press
            if core_wrapper is None:
                raise RuntimeError(
                    f"Could not resolve CORE prefill press for {press_name}."
                )
            core_base = getattr(core_wrapper, "base_press", core_wrapper)
            if not isinstance(core_base, COREScorerPress):
                raise TypeError(
                    "Resolved CORE base press is not a COREScorerPress: "
                    f"{type(core_base).__name__}"
                )

            # Explicit evaluation-only policy overrides.  These do not change
            # the learned indexer weights.  In particular, do NOT add an
            # evaluation override for prefill_qk_scope: that field is part of
            # the trained feature contract and should come from the checkpoint.
            if self.config.core_span_block_size is not None:
                span = int(self.config.core_span_block_size)
                if span < 1:
                    raise ValueError("core_span_block_size must be >= 1")
                core_base.span_block_size = span
            if self.config.core_span_pool_beta is not None:
                beta = float(self.config.core_span_pool_beta)
                if beta <= 0.0:
                    raise ValueError("core_span_pool_beta must be > 0")
                core_base.span_pool_beta = beta
            if self.config.core_span_selection_mode is not None:
                mode = str(self.config.core_span_selection_mode)
                if mode not in {
                    "tokenwise", "block_lme", "sliding_max", "coverage_local",
                    "coverage_local_balanced",
                }:
                    raise ValueError(
                        "core_span_selection_mode must be block_lme, "
                        "sliding_max, coverage_local, or "
                        "coverage_local_balanced"
                    )
                core_base.span_selection_mode = mode
            if self.config.core_coverage_fraction is not None:
                fraction = float(self.config.core_coverage_fraction)
                if not 0.0 <= fraction <= 1.0:
                    raise ValueError("core_coverage_fraction must be in [0, 1]")
                core_base.coverage_fraction = fraction

            if self.config.core_protect_query_tokens is not None:
                value = self.config.core_protect_query_tokens
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
                        raise ValueError(
                            "core_protect_query_tokens must be a boolean"
                        )
                    value = normalized in {"true", "1", "yes"}
                core_base.protect_query_tokens = bool(value)

            if self.config.core_query_feature_weight is not None:
                core_base.query_feature_weight = float(
                    self.config.core_query_feature_weight
                )

            if self.config.core_lexical_overlap_weight is not None:
                weight = float(self.config.core_lexical_overlap_weight)
                if weight < 0.0:
                    raise ValueError(
                        "core_lexical_overlap_weight must be >= 0"
                    )
                core_base.lexical_overlap_weight = weight

            if self.config.core_identifier_lexical_weight is not None:
                weight = float(self.config.core_identifier_lexical_weight)
                if weight < 0.0:
                    raise ValueError(
                        "core_identifier_lexical_weight must be >= 0"
                    )
                core_base.identifier_lexical_weight = weight
            if self.config.core_identifier_min_match_tokens is not None:
                core_base.identifier_min_match_tokens = max(
                    1, int(self.config.core_identifier_min_match_tokens)
                )
            if self.config.core_identifier_context_skip is not None:
                core_base.identifier_context_skip = max(
                    0, int(self.config.core_identifier_context_skip)
                )
            if self.config.core_identifier_span_size is not None:
                core_base.identifier_span_size = max(
                    1, int(self.config.core_identifier_span_size)
                )

            # Persist the resolved policy, including values inherited from the
            # checkpoint. Without this, a completed run misleadingly saves
            # these fields as null even though CORE used non-null values.
            self.config.core_span_block_size = int(core_base.span_block_size)
            self.config.core_span_pool_beta = float(
                getattr(core_base, "span_pool_beta", 5.0)
            )
            self.config.core_span_selection_mode = str(
                getattr(core_base, "span_selection_mode", "sliding_max")
            )
            self.config.core_coverage_fraction = float(
                getattr(core_base, "coverage_fraction", 0.25)
            )
            self.config.core_protect_query_tokens = bool(
                core_base.protect_query_tokens
            )
            self.config.core_query_feature_weight = float(
                core_base.query_feature_weight
            )
            self.config.core_lexical_overlap_weight = float(
                core_base.lexical_overlap_weight
            )
            self.config.core_identifier_lexical_weight = float(
                core_base.identifier_lexical_weight
            )
            self.config.core_identifier_min_match_tokens = int(
                core_base.identifier_min_match_tokens
            )
            self.config.core_identifier_context_skip = int(
                core_base.identifier_context_skip
            )
            self.config.core_identifier_span_size = int(
                core_base.identifier_span_size
            )

            logger.info(
                "CORE eval policy: scoring_stride=%s, span_block_size=%s, "
                "span_pool_beta=%s, span_selection_mode=%s, "
                "coverage_fraction=%s, protect_query_tokens=%s, "
                "query_feature_weight=%s, "
                "lexical_overlap_weight=%s, identifier_weight=%s, "
                "identifier_min_match=%s, identifier_span=%s, "
                "prefill_qk_scope=%s",
                getattr(core_base, "scoring_stride", None),
                getattr(core_base, "span_block_size", None),
                getattr(core_base, "span_pool_beta", None),
                getattr(core_base, "span_selection_mode", None),
                getattr(core_base, "coverage_fraction", None),
                getattr(core_base, "protect_query_tokens", None),
                getattr(core_base, "query_feature_weight", None),
                getattr(core_base, "lexical_overlap_weight", None),
                getattr(core_base, "identifier_lexical_weight", None),
                getattr(core_base, "identifier_min_match_tokens", None),
                getattr(core_base, "identifier_span_size", None),
                getattr(core_base, "prefill_qk_scope", None),
            )

            if press_name in ("core_memory", "core_full"):
                shared_memory_press = (
                    press
                    if press_name == "core_memory"
                    else getattr(press, "prefilling_press", None)
                )
                if (
                    shared_memory_press is None
                    or not hasattr(shared_memory_press, "memory_module")
                ):
                    raise TypeError(
                        f"{press_name} did not construct a COREMemoryPress"
                    )
                if self.config.core_enable_memory:
                    if shared_memory_press.memory_module is None:
                        raise RuntimeError(
                            f"{press_name} requested trained memory, but no "
                            "memory module was loaded from the checkpoint"
                        )
                    if not shared_memory_press.enable_memory_writing:
                        raise RuntimeError(
                            f"{press_name} requested trained memory, but the "
                            "checkpoint has enable_memory_writing=False"
                        )
                    gate_override = self.config.core_memory_gate_override
                    if gate_override is not None:
                        if not 0.0 <= float(gate_override) <= 1.0:
                            raise ValueError("core_memory_gate_override must be in [0, 1]")
                        shared_memory_press.memory_gate_override = float(gate_override)
                        logger.info(
                            "CORE memory gate overridden to %.4f for controlled evaluation.",
                            float(gate_override),
                        )
                else:
                    shared_memory_press.enable_memory_writing = False
                    shared_memory_press.reset_memory()
                    logger.info(
                        "CORE memory writing/readout disabled; running the "
                        "query-aware indexer-only ablation."
                    )

            logger.info(
                "Loaded %s press from %s on device %s",
                press_name,
                self.config.core_indexer_checkpoint_path,
                press_device,
            )
            if self.config.paper_protocol:
                policy = core_base.feature_extractor.config
                expected = {
                    "scoring_stride": 1, "span_selection_mode": "tokenwise",
                    "span_block_size": 1, "protect_query_tokens": False,
                    "recency_alpha": 0.0, "query_feature_weight": 0.0,
                    "lexical_overlap_weight": 0.0, "identifier_lexical_weight": 0.0,
                    "n_sink_protect": 4, "prefill_qk_scope": "all",
                }
                for name, value in expected.items():
                    if getattr(core_base, name, getattr(policy, name, None)) != value:
                        raise ValueError(f"Checkpoint policy {name} is not paper-aligned; retrain or use paper_protocol=False")
                if press_name != "core_full":
                    from core.kvpress_adapter import COREDecodingPress, COREPrefillDecodingPress
                    press = COREPrefillDecodingPress(
                        prefilling_press=press,
                        decoding_press=COREDecodingPress(
                            base_press=press, target_size=1,
                            compression_interval=self.config.compression_interval or 128,
                            hidden_states_buffer_size=self.config.hidden_states_buffer_size,
                        ),
                    )
            self.press = press
            self.config.press_init_command = str(press)
            logger.info(f"KV Press '{press_name}' setup.")
            return

        # Apply compression ratios based on press type
        if isinstance(press, DuoAttentionPress):
            assert (
                self.config.head_compression_ratio is not None
            ), "head_compression_ratio must be set for DuoAttentionPress"
            press.head_compression_ratio = self.config.head_compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {press.head_compression_ratio}")
        elif isinstance(press, DMSPress):
            assert self.config.threshold is not None, "threshold must be set for DMSPress"
            press.threshold = self.config.threshold
            logger.info(f"Set DMSPress threshold to {press.threshold}")
        elif isinstance(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    assert (
                        key_channel_compression_ratio is not None
                    ), "key_channel_compression_ratio must be set for ThinKPress in ComposedPress"
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                    logger.info(f"Set ComposedPress key_channel_compression_ratio to {key_channel_compression_ratio}")
                else:
                    # Check if compression_ratio attribute exists before setting
                    if hasattr(ps, "compression_ratio"):
                        ps.compression_ratio = compression_ratio
                        logger.info(f"Set ComposedPress compression_ratio to {compression_ratio}")
                    else:
                        logger.warning(
                            f"ComposedPress component {ps.__class__.__name__} has no 'compression_ratio' attribute."
                        )
        elif isinstance(press, ThinKPress):
            assert key_channel_compression_ratio is not None, "key_channel_compression_ratio must be set for ThinKPress"
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")
        elif isinstance(press, DecodingPress):
            press.compression_interval = self.config.compression_interval or press.compression_interval
            press.target_size = self.config.target_size or press.target_size
            press.hidden_states_buffer_size = self.config.hidden_states_buffer_size or press.hidden_states_buffer_size
            logger.info(
                f"Set DecodingPress compression_interval to {self.config.compression_interval}, target_size to {self.config.target_size}, hidden_states_buffer_size to {self.config.hidden_states_buffer_size}"
            )
        else:
            if hasattr(press, "compression_ratio"):
                press.compression_ratio = compression_ratio
                logger.info(f"Set {press.__class__.__name__} compression_ratio to {compression_ratio}")
            else:
                logger.warning(
                    f"Press {press.__class__.__name__} has no 'compression_ratio' attribute. This is expected is you set `no_press`."
                )

        if self.config.paper_protocol:
            from core.paper_protocol import wrap_paper_baseline
            press = wrap_paper_baseline(press, self.config.compression_interval or 128)
        self.press = press
        # Set the press info in the config for saving to YAML
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")

    def _load_and_prepare_dataset(self):
        """
        Loads the dataset specified in the config and applies sampling/filtering.
        """
        dataset_name = self.config.dataset
        data_dir = str(self.config.data_dir) if self.config.data_dir else None
        fraction = self.config.fraction

        if dataset_name == "ruler":
            if data_dir is None:
                raise ValueError(
                    "RULER evaluation now loads only from a local directory. Set data_dir to the local 4096 path."
                )

            local_data_dir = Path(data_dir).expanduser()
            if not local_data_dir.is_absolute():
                repo_local_dir = REPO_ROOT / "data" / "ruler" / local_data_dir
                if repo_local_dir.exists():
                    local_data_dir = repo_local_dir
                else:
                    local_data_dir = (Path.cwd() / local_data_dir).resolve()

            if not local_data_dir.exists():
                raise FileNotFoundError(
                    f"Local RULER data directory does not exist: {local_data_dir}. "
                    "Place the 4096 folder there or update data_dir to the full local path."
                )
            if not local_data_dir.is_dir():
                raise ValueError(f"RULER data_dir must be a directory, got: {local_data_dir}")

            logger.info(f"Loading local RULER dataset from: {local_data_dir}")
            parquet_files = sorted(local_data_dir.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files found in local RULER data directory: {local_data_dir}. "
                    "Expected files like test-00000-of-00001.parquet."
                )

            df = pd.concat([pd.read_parquet(parquet_file) for parquet_file in parquet_files], ignore_index=True)
        elif (dataset_name in ("longbench", "longbench-e") and data_dir
                and Path(data_dir).is_dir()):
            df = _load_local_longbench(data_dir, self.config.longbench_task,
                                       dataset_name == "longbench-e")
        else:
            logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")
            df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()

        if self.config.task is not None:
            if "task" not in df.columns:
                raise ValueError(
                    f"Dataset {dataset_name} has no task column; "
                    f"cannot select task={self.config.task}"
                )
            original_len = len(df)
            df = df[df["task"] == self.config.task].copy()
            if df.empty:
                available = sorted(map(str, pd.unique(
                    pd.concat(
                        [pd.read_parquet(path) for path in parquet_files],
                        ignore_index=True,
                    )["task"]
                ))) if dataset_name == "ruler" else []
                raise ValueError(
                    f"No rows found for task={self.config.task}; "
                    f"available tasks={available}"
                )
            logger.info(
                "Selected task %s: %d/%d entries.",
                self.config.task, len(df), original_len,
            )

        if fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        # if we have needle in a haystack, we need to insert it in the context
        if self.config.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df, self.pipeline.tokenizer, self.config.max_context_length, self.config.needle_depth
            )

        if isinstance(self.press, FinchPress):
            if not self.config.query_aware:
                logger.error("FinchPress requires 'query_aware' to be set to True.")
                raise ValueError("FinchPress requires query_aware to be set to True")
            # FinchPress uses a delimiter token to separate context and question
            # So we need to update the tokenizer and the model embeddings.
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(self.pipeline.model, self.pipeline.tokenizer)  # type: ignore[attr-defined]
            df["context"] = df["context"] + self.press.delimiter_token  # type: ignore[attr-defined, index]

        if self.config.query_aware:
            logger.info("Query-aware compression: including question in context for compression.")
            if self.config.press_name in CORE_PRESS_NAMES:
                df["_query_start_char"] = df["context"].str.len()
            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")

        model_kwargs = self.config.model_kwargs or {}

        if self.config.fp8:
            model_kwargs["quantization_config"] = FineGrainedFP8Config()
            logger.info("FP8 quantization enabled.")

        # ObservedAttentionPress requires eager attention to reuse the model's
        # attention weights.  COREScorerPress/COREIndexerPress no longer force
        # eager attention — they have a fallback that recomputes a small Q@K^T
        # on the scoring layer, allowing FlashAttention for the model itself.
        backend_press = getattr(self.press, "prefilling_press", self.press)
        while getattr(backend_press, "base_press", None) is not None:
            backend_press = backend_press.base_press
        attention_reusing_presses = (ObservedAttentionPress,)
        attention_reusing_presses = tuple(
            p for p in attention_reusing_presses if isinstance(p, type)
        )
        if isinstance(backend_press, attention_reusing_presses):
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["output_attentions"] = True
            logger.info(
                "Attention-reusing press detected (%s), setting attn_implementation to 'eager'.",
                type(self.press).__name__,
            )
        else:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")
                pass

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")
        pipeline_kwargs = {
            "model": model_name,
            "model_kwargs": model_kwargs,
            "trust_remote_code": True,
        }
        if device == "auto":
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device
        task_name = "kv-press-text-generation"
        if self.config.press_name in CORE_PRESS_NAMES or self.config.paper_protocol:
            import core.kvpress_pipeline  # noqa: F401
            task_name = "core-kv-press-text-generation"
        self.pipeline = pipeline(task_name, **pipeline_kwargs)

        self.pipeline.model.eval()
        logger.info("Model pipeline loaded.")

    @torch.inference_mode()
    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """

        self.df["predicted_answer"] = None  # type: ignore[index]

        if self.config.press_name in CORE_PRESS_NAMES or self.config.paper_protocol:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            inference_start = time.perf_counter()
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running paper-protocol inference"):
                query_start_char = (
                    int(row["_query_start_char"])
                    if self.config.query_aware and "_query_start_char" in row else None
                )
                output = self.pipeline(
                    row["context"],
                    question=row["question"],
                    query_start_char=query_start_char,
                    answer_prefix=row["answer_prefix"],
                    press=self.press,
                    max_new_tokens=self.config.max_new_tokens or row["max_new_tokens"],
                    max_context_length=self.config.max_context_length,
                )
                self.df.loc[index, "predicted_answer"] = output["answer"]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_start
            logger.info(
                "Paper-protocol synchronized inference: %.3fs, %.3f samples/s, peak_cuda=%.3f GiB",
                inference_seconds,
                len(self.df) / max(inference_seconds, 1e-9),
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available() else 0.0,
            )
            return


        if (
            isinstance(self.press, (DecodingPress, PrefillDecodingPress))
            or getattr(self.press, "query_aware_prefill", False)
        ):
            logger.info(
                "Decode-capable/query-aware press detected, running inference "
                "for each context-question pair."
            )
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
                context = row["context"]
                question = row["question"]
                answer_prefix = row["answer_prefix"]
                max_new_tokens = self.config.max_new_tokens or row["max_new_tokens"]
                output = self.pipeline(
                    context,
                    question=question,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                self.df.loc[index, "predicted_answer"] = output["answer"]  # type: ignore[union-attr]

        else:
            df_context_grouped = self.df.groupby("context")  # type: ignore[union-attr]
            assert all(
                df_context_grouped["answer_prefix"].nunique() == 1
            ), "Inconsistent 'answer_prefix' within the same context group detected."

            logger.info("Starting inference...")
            for context, df_group in tqdm(
                df_context_grouped, total=self.df["context"].nunique(), desc="Running Inference"
            ):  # type: ignore[union-attr]
                questions = df_group["question"].to_list()
                # Use max_new_tokens from config, or fallback to dataset's default for the task
                max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
                answer_prefix = df_group["answer_prefix"].iloc[0]

                output = self.pipeline(  # type: ignore[misc]
                    context,
                    questions=questions,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                self.df.loc[df_group.index, "predicted_answer"] = output["answers"]  # type: ignore[union-attr]
                # Store the actual compression ratio used (if the press has one)
                self.df.loc[df_group.index, "compression_ratio"] = (
                    self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                )  # type: ignore[union-attr, attr-defined]

        logger.info("Inference completed.")

    def _save_results(self, save_filename: Path):
        """
        Saves the predicted answers and compression ratios to a CSV file.

        Parameters
        ----------
        save_filename : Path
            The full path including filename to save the CSV.
        """
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        self.df[list(set(self.df.columns) - set(["context"]))].to_csv(
            str(save_filename), index=False
        )  # type: ignore[index]
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        """
        Calculates evaluation metrics and saves them to a JSON file.

        Parameters
        ----------
        save_filename : Path
            The base filename (e.g., CSV path) to derive the JSON path from.
        """
        dataset_name = self.config.dataset
        scorer = SCORER_REGISTRY[dataset_name]

        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        metrics = scorer(self.df)  # type: ignore[call-arg]

        with open(str(save_filename), "w") as f:
            json.dump(metrics, f, indent=4)  # Pretty print JSON

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process.
        """
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()

        results_dir = self.config.get_results_dir(output_dir)
        predictions_filename = results_dir / "predictions.csv"
        metrics_filename = results_dir / "metrics.json"
        config_filename = results_dir / "config.yaml"

        if predictions_filename.exists() and metrics_filename.exists():
            logger.info(
                f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
            )
            return

        self._setup_press()
        self._setup_model_pipeline()
        self._load_and_prepare_dataset()

        self._run_inference()
        self._save_results(predictions_filename)
        self._calculate_and_save_metrics(metrics_filename)
        self.config.save_config(config_filename)
        logger.info("Evaluation run completed successfully.")


# --- Command-Line Interface ---
class CliEntryPoint:
    """
    CLI entry point for building configuration and running the evaluation.

    This class provides a command-line interface for running KVPress evaluations.
    Configuration can be specified via:
    1. YAML config file (default: "./evaluate_config.yaml")
    2. Command-line arguments (highest priority)
    """

    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        """
        Builds the configuration and runs the evaluation.

        Configuration is built by layering:
        1. Default values from EvaluationConfig
        2. Values from YAML config file
        3. Command-line arguments (highest priority)
        """
        # 1. Start with dataclass defaults.
        final_args = asdict(EvaluationConfig())

        # 2. Layer YAML values on top.
        yaml_config = _load_yaml_config(config_file)
        final_args.update(yaml_config)

        # 3. Layer CLI arguments on top (highest priority).
        # Filter out None values from CLI overrides
        cli_args = {k: v for k, v in cli_overrides.items() if v is not None}
        final_args.update(cli_args)

        # 4. Create and validate the final config object.
        try:
            config = EvaluationConfig(**final_args)
        except TypeError as e:
            # Provide a user-friendly error for bad arguments.
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)
