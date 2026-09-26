[![Hugging Face Leaderboard](https://img.shields.io/badge/🤗%20HuggingFace-Leaderboard-orange)](https://huggingface.co/spaces/nvidia/kvpress-leaderboard)

# Evaluation

We support evaluation for all the presses implemented in the library, on a variety of popular benchmarks.

### Quick Start 🚀
> Evaluation requires some additional packages. You can install them with `uv sync --extra eval`

Run evaluation from the CORE repository root (the directory containing `core/`,
`kvpress/`, and `RULER/`). Relative paths in the supplied configuration are
resolved from this working directory. Activate your Python environment, then:

1. **Configure your evaluation** - Edit `kvpress/evaluation/evaluate_config.yaml` to specify your *method*, *press*, and *dataset*
2. **Run the evaluation** - Execute the script: ```bash kvpress/evaluation/evaluate.sh```

The script will read from `evaluate_config.yaml` and run inference accordingly. 
If you want, you can override the settings via command line, for instance:

```bash
bash kvpress/evaluation/evaluate.sh --dataset loogle --data_dir shortdep_qa --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name expected_attention --compression_ratio 0.5
```

or pass a custom configuration file:

```bash
bash kvpress/evaluation/evaluate.sh --config_file <your_config.yaml>
```

💡 Results (predictions & metrics) are automatically saved to the `output_dir` directory .


### Configuration 

Customize your evaluation by editing `evaluate_config.yaml`. This allows you to flexibly configure a variety of settings, like the `fraction` of dataset to use (for quick testing) and the model arguments (e.g. for scaling RoPE). For complete parameter details, see the `evaluation_config.yaml`

💡 Set `query_aware: true` to include the question in the context during compression. This enables query-aware compression as used in methods like SnapKV and FinchPress.


### Available Presses and Datasets 
We support evaluation with all the presses implemented in the library (and possible combinations). 

- All implemented presses are listed in the `PRESS_REGISTRY` variable in `evaluate_registry.py`.
- All implemented dataset are listed in `DATASET_REGISTRY` variable in `evaluate_registry.py`. 

At the moment, we support the following standard popular benchmarks:

- [Loogle](benchmarks/loogle/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/loogle))
- [RULER](benchmarks/ruler/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/ruler))
- [Zero Scrolls](benchmarks/zero_scrolls/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/zero_scrolls))
- [Infinitebench](benchmarks/infinite_bench/README.md) ([hf link](https://huggingface.co/datasets/MaxJeblick/InfiniteBench))
- [longbench](benchmarks/longbench/README.md)([hf link](https://huggingface.co/datasets/Xnhyacinth/LongBench))
- [longbench-v2](benchmarks/longbenchv2/README.md)([hf link](https://huggingface.co/datasets/simonjegou/LongBench-v2))
- [Needle in a Haystack](benchmarks/needle_in_haystack/README.md)([hf link][Paul Graham's essays](https://huggingface.co/datasets/alessiodevoto/paul_graham_essays))

Each dataset directory is structured as follows:

```bash
$dataset
├── README.md
├── calculate_metrics.py
├── create_huggingface_dataset.py
```

Where:
- `create_huggingface_dataset.py` is a script that generates the Hugging Face dataset from the original dataset. Each dataset is associated with a set of parquet files with the following structure:
  - `context`: ... 
  - `question`: ...
  - `answer_prefix`: ...
  - `answer`:  ...
  - `max_new_tokens`:  ...
- `calculate_metrics.py` is a script that calculates the metrics based on the output of `evaluate.py`


### Multi GPU Evaluation
Use `evaluate.sh` once per configuration. To run on different GPUs, launch separate
commands with `CUDA_VISIBLE_DEVICES` set explicitly for each process.

### Leaderboard 🥇
After evaluating your model, you can easily submit it to the [KVPress Leaderboard](https://huggingface.co/spaces/nvidia/kvpress-leaderboard) on Hugging Face! Just copy the output directory in the huggingface space, and your method will soon be displayed in the leaderboard.

## CORE and the shared paper protocol

Use this directory's `evaluate.py` / `evaluate.sh` for CORE and baselines.
The previous separate `evaluate_paper.py` and paper shell wrapper have been removed.
`evaluate_config.yaml` enables `paper_protocol: true`: protect four sinks inside
Top-B, use tokenwise CORE retention, and recompress every 128 tokens to each
sample's retained prefill budget. Conflicting legacy selection overrides are
rejected. Use `--paper_protocol False` for explicit legacy/baseline ablations.
CORE always uses the metadata-aware pipeline, including per-sequence memory reset.
Methods with custom merging or variable-head budgets still need a verified adapter;
the shared protocol does not silently replace their algorithm.

From the CORE root:

```bash
bash kvpress/evaluation/evaluate.sh \
  --press_name core_full --compression_ratio 0.75 \
  --core_indexer_checkpoint_path ./experiments/paper/indexer/core_indexer.pt

bash kvpress/evaluation/evaluate.sh \
  --press_name keydiff --compression_ratio 0.75
```

For local LongBench JSONL, set `--dataset longbench --data_dir ./data/longbench
--longbench_task qasper`; repeat over the 16 English/code tasks for the paper
aggregate. RULER loads every parquet shard. Keep all examples (`fraction=1`) for
reported results. A configured checkpoint path does not create trained weights.

Training uses the existing `core.run` entry point from the CORE root:

```bash
python -m core.run
```

Defaults resolve from the installed CORE source directory: `model/`,
`data/LongAlpaca-12k_.csv`, and `experiments/paper/` for output. Override them
with `--model_name`, `--data_path`, and `--output_dir` when using another backbone
or dataset. For example, a local Qwen3-14B model can be selected with
`--model_name ./models/Qwen3-14B`.
When resuming an older checkpoint, explicitly override any obsolete paths saved
in its configuration. The scripts directory is not required.

Default `max_samples=None` (CLI `--max_samples 0`) retains all 12,000
CSV rows, including short and instruction-only examples. No internal validation
rows are removed (`online_val_samples=0`). Positive caps or validation sizes are
explicit experiments and no longer describe full-data training. The two-stage
schedule remains 1,000 + 4,400 optimizer updates, with effective batch size 4.

### Current CORE implementation

The corrected method runs directly through `core.run` and the existing evaluation
entry point. The default output/checkpoint directory is `experiments/paper/`.
There is no numbered policy selector or version-label gate. Checkpoint loading
checks actual coordinate spaces, geometry and tensor dimensions. Existing
checkpoint paths can still be passed explicitly. Loading weights does not
establish that they were trained or evaluated with the corrected method.

The memory allocation is normalized over the full candidate cache (paper
Eq. 6/10); sinks are absent from the eviction set and from KL/boundary
supervision. Held-out response/prompt queries cannot influence preceding-prefix
features. Geometry uses `geometry_seed=42` independently of the optimizer/data
`seed`, and its Hadamard rotation is stored in the checkpoint. Run seeds 42/43/44
with `--seed`; keep `--geometry_seed 42` for Appendix E.5. Teacher coverage
sweeps can use `--beta_coverage` and `--alpha_coverage` explicitly.

Mixed-task LongBench input is scored per task and macro-averaged. A subset's
macro-average is not the 16-task paper aggregate: check `task_count` and retain
all 16 task scores. RULER's existing metrics remain per-subtask; average all 13
subtask scores with equal weight. Short smoke runs are not paper results.
