"""LongAlpaca loading and boundary-preserving tokenization for CORE."""

import logging
import random

import pandas as pd
import torch
from transformers import AutoTokenizer

from core.config import COREConfig

logger = logging.getLogger(__name__)

_END_MARKERS = (
    "now the second paper ends.",
    "now the review guideline ends.",
    "now the material ends.",
    "now the paper ends.",
)


def _split_longalpaca_row(instruction: str, inp: str) -> tuple[str, str]:
    """Recover context/question from this repository's mixed CSV schema."""
    lowered = instruction.lower()
    candidates = []
    for marker in _END_MARKERS:
        pos = lowered.rfind(marker)
        if pos >= 0:
            candidates.append((pos + len(marker), marker))
    if candidates:
        boundary, _ = max(candidates)
        return instruction[:boundary], instruction[boundary:].strip()
    # Standard Alpaca fallback: instruction is the question, input is context.
    if inp and inp != "nan":
        return inp, instruction
    return "", instruction


def make_multi_evidence_retrieval_sample(
    sample: dict,
    seed: int,
    *,
    n_records: int = 8,
    n_targets: int = 3,
) -> dict:
    """Build a deterministic multi-evidence example from training text.

    Opaque record identifiers and their values are inserted across excerpts of
    the source context.  The appended query requests several separated records,
    so the frozen backbone's question-to-context attention provides genuine
    multi-evidence teacher supervision.  No benchmark examples or labels are
    consumed, and the returned object follows the ordinary LongAlpaca schema.
    """
    context = " ".join(str(sample.get("context", "")).split())
    if not context:
        return dict(sample)
    n_records = max(4, int(n_records))
    n_targets = max(2, min(int(n_targets), n_records))
    rng = random.Random(int(seed))

    # Roughly 10--12K characters maps to a 3--4K-token prompt for the current
    # tokenizer.  Evenly spaced excerpts keep requested records far apart.
    excerpt_width = max(256, min(1400, len(context) // n_records))
    max_start = max(0, len(context) - excerpt_width)
    starts = [
        round(i * max_start / max(1, n_records - 1))
        for i in range(n_records)
    ]
    records = []
    rendered = []
    for index, start in enumerate(starts):
        excerpt = context[start:start + excerpt_width].strip()
        words = excerpt.split()
        if words:
            value_start = rng.randrange(max(1, len(words) - 7))
            value = " ".join(words[value_start:value_start + 8])
        else:
            value = f"source fragment {index}"
        key_bits = rng.getrandbits(48)
        key = f"rec_{key_bits:012x}_{index:02d}"
        records.append((key, value))
        rendered.append(
            f"{excerpt}\n\n[Record {key}]\nAssociated value: {value}\n"
        )

    # Choose targets from different parts of the document rather than allowing
    # all requested evidence to collapse into one local region.
    bins = [round(i * (n_records - 1) / max(1, n_targets - 1))
            for i in range(n_targets)]
    jittered = []
    for index in bins:
        lo = max(0, index - 1)
        hi = min(n_records - 1, index + 1)
        choice = rng.randint(lo, hi)
        if choice not in jittered:
            jittered.append(choice)
    for index in range(n_records):
        if len(jittered) >= n_targets:
            break
        if index not in jittered:
            jittered.append(index)
    target_records = [records[index] for index in jittered[:n_targets]]
    requested = ", ".join(key for key, _ in target_records)
    query = (
        "Retrieve the associated value for each of these record identifiers "
        f"from the document: {requested}. Return every requested identifier."
    )
    output = "\n".join(f"{key}: {value}" for key, value in target_records)
    augmented_context = "\n\n".join(rendered)
    return {
        "context": augmented_context,
        "query": query,
        "output": output,
        "text": augmented_context + "\n\n" + query,
        "retrieval_augmented": True,
        "retrieval_target_keys": tuple(key for key, _ in target_records),
    }


def load_longalpaca(config: COREConfig, tokenizer: AutoTokenizer) -> list[dict]:
    """Load LongAlpaca as ``long context -> question -> answer`` samples.

    Supports separate ``input``/``instruction`` fields and documents embedded
    in ``instruction``. Keeps question and answer boundaries for configurable
    query scopes and held-out memory reconstruction targets.
    """
    logger.info("Loading dataset from %s", config.data_path)
    df = pd.read_csv(config.data_path)
    samples = []
    for row_index, row in df.iterrows():
        def text_field(name):
            value = row.get(name, "")
            return "" if pd.isna(value) else str(value)

        instruction = text_field("instruction")
        inp = text_field("input")
        answer = text_field("output")
        context, query = _split_longalpaca_row(instruction, inp)
        if not query.strip():
            # Keep context-only examples as ordinary user prompts too.
            query, context = context or inp or instruction, ""
        if not query.strip():
            raise ValueError(f"LongAlpaca row {row_index} has no usable prompt")
        samples.append({
            "context": context,
            "query": query,
            "output": answer,
            "text": context + "\n\n" + query if context else query,
            "source_row": int(row_index),
        })

    # Short instruction-only rows are part of LongAlpaca-12k. Boundary
    # windows shrink to the available candidates; they are not a data filter.
    random.Random(config.seed).shuffle(samples)
    limit = config.max_samples
    if limit is not None and limit < 0:
        raise ValueError("max_samples must be nonnegative, or None for all rows")
    if limit:
        samples = samples[:limit]
    logger.info("Loaded %d/%d LongAlpaca rows (max_samples=%s)",
                len(samples), len(df), limit)
    return samples


def _encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _token_boundary(tokenizer, rendered: str, char_boundary: int) -> int:
    """Map a character boundary in rendered chat text to a token boundary."""
    try:
        encoded = tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        for idx, (_, end) in enumerate(offsets):
            if end > char_boundary:
                return idx
        return len(offsets)
    except (TypeError, NotImplementedError, KeyError):
        # Slow-tokenizer fallback. At worst a BPE merge moves the boundary by
        # one token; the structural separator makes that unlikely.
        return len(_encode_text(tokenizer, rendered[:char_boundary]))


def _truncate_keep_sink_and_tail(
    ids: list[int], query_start: int, answer_start: int, max_length: int,
    n_sink: int, crop_seed: int | None = None,
) -> tuple[list[int], int, int]:
    """Retain sink + a context window + the complete question/answer suffix."""
    if len(ids) <= max_length:
        return ids, query_start, answer_start
    sink = min(max(0, int(n_sink)), max_length, len(ids))
    # Question/answer are the deployment-critical suffix. Use the remaining
    # budget for a contiguous context crop whose position changes by training
    # step, exposing the selector to evidence at the beginning, middle, and end
    # instead of always training on the final context window.
    suffix_start = max(sink, query_start)
    suffix = ids[suffix_start:]
    if (len(suffix) >= max_length - sink
            and sink < answer_start < len(ids)):
        # An overlength answer must not erase every non-sink prompt token.
        # Keep the trailing question and trailing response in disjoint spans;
        # the returned boundary still separates selection from reconstruction.
        room = max_length - sink
        if room < 2:
            raise ValueError("Training crop needs room for prompt and response")
        prompt_start = min(max(sink, query_start), answer_start - 1)
        prompt_count = min(answer_start - prompt_start, max(1, room // 2))
        answer_count = min(len(ids) - answer_start, room - prompt_count)
        prompt_count = min(answer_start - sink, room - answer_count)
        prompt_start = answer_start - prompt_count
        kept = ids[:sink] + ids[prompt_start:answer_start] + ids[-answer_count:]
        return kept, sink + max(0, query_start - prompt_start), sink + prompt_count
    if len(suffix) >= max_length - sink:
        tail_start = len(ids) - (max_length - sink)
        kept = ids[:sink] + ids[tail_start:]
        context_start = tail_start
        context = []
        suffix_start = tail_start
    else:
        context_budget = max_length - sink - len(suffix)
        context_end = suffix_start
        max_start = max(sink, context_end - context_budget)
        if context_end - sink <= context_budget:
            context_start = sink
        elif crop_seed is None:
            context_start = max_start
        else:
            context_start = random.Random(int(crop_seed)).randint(sink, max_start)
        context = ids[context_start:min(context_end, context_start + context_budget)]
        kept = ids[:sink] + context + suffix
        tail_start = suffix_start

    def remap(boundary: int) -> int:
        if boundary <= sink:
            return boundary
        if boundary < suffix_start:
            # Boundaries inside the cropped context map into the retained context span.
            return min(len(kept), sink + max(0, boundary - context_start))
        return sink + len(context) + boundary - suffix_start

    return kept, remap(query_start), remap(answer_start)


def tokenize_sample(
    tokenizer: AutoTokenizer,
    sample: dict,
    max_length: int,
    device: torch.device,
    include_answer: bool = False,
    n_sink: int = 4,
    crop_seed: int | None = None,
) -> dict:
    """Tokenize a sample and locate question/answer token boundaries."""
    context = str(sample.get("context", ""))
    query = str(sample.get("query", ""))
    # Handle samples without an explicit question.
    if not query:
        query = str(sample.get("text", ""))
        context = ""
    content = (context + "\n\n" + query) if context else query
    answer = str(sample.get("output", ""))

    if tokenizer.chat_template is not None:
        prompt_messages = [{"role": "user", "content": content}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        query_char = prompt_text.rfind(query)
        if query_char < 0:
            raise ValueError("Could not locate question in rendered chat prompt")
        query_start = _token_boundary(tokenizer, prompt_text, query_char)
        prompt_ids = _encode_text(tokenizer, prompt_text)
        if include_answer and answer:
            full_text = tokenizer.apply_chat_template(
                prompt_messages + [{"role": "assistant", "content": answer}],
                tokenize=False, add_generation_prompt=False,
            )
            full_ids = _encode_text(tokenizer, full_text)
            # The generation prompt is designed to be the exact prefix up to
            # assistant content. Use the longest common prefix defensively.
            answer_start = 0
            for left, right in zip(prompt_ids, full_ids):
                if left != right:
                    break
                answer_start += 1
            ids = full_ids
        else:
            ids = prompt_ids
            answer_start = len(ids)
    else:
        prompt_text = content
        query_char = len(context) + 2 if context else 0
        query_start = _token_boundary(tokenizer, prompt_text, query_char)
        prompt_ids = _encode_text(tokenizer, prompt_text)
        answer_start = len(prompt_ids)
        ids = (
            _encode_text(tokenizer, prompt_text + "\n\n" + answer)
            if include_answer and answer else prompt_ids
        )

    ids, query_start, answer_start = _truncate_keep_sink_and_tail(
        ids, query_start, answer_start, max_length, n_sink, crop_seed,
    )
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "length": input_ids.shape[1],
        "query_start": min(query_start, input_ids.shape[1]),
        "answer_start": min(answer_start, input_ids.shape[1]),
        "has_answer": bool(include_answer and answer),
        "output": answer,
    }
