# RLHF Course: An End-to-End RLHF Pipeline

[![CI](https://github.com/nhan-dam/rlhf-course/actions/workflows/ci.yml/badge.svg)](https://github.com/nhan-dam/rlhf-course/actions/workflows/ci.yml)

This repository implements the classical reinforcement learning from human feedback (RLHF) pipeline end to end, from a pre-trained base model to a policy aligned with human preferences. It uses a small base model so that every stage, including the multi-model proximal policy optimisation (PPO) stage, fits on a single workstation.

This README covers what is here and how to run it. Design choices, diagnostics, and results for each stage are in that stage's report under `reports/`.

## 1. Overview

RLHF aligns a language model with human preferences in three stages, each consuming the artefact produced by the previous one: supervised fine-tuning (SFT), reward modelling (RM), then PPO against the learned reward. A fourth stage, direct preference optimisation (DPO), is the modern alternative that collapses RM and PPO into a single supervised loss. It sits alongside PPO here rather than replacing it, configured for a controlled comparison.

Every stage is low-rank adaptation (LoRA) based: only adapters, plus the RM's scalar head, are ever trained, and each stage's artefact is an adapter the next stage merges. The shared base model is Qwen2.5-0.5B, defined once in `src/common/config.py` together with the on-disk artefacts exchanged between stages.

| Stage | Script | Data | Output | Report |
|---|---|---|---|---|
| SFT | `src/pipeline/sft_lora_dolly.py` | `databricks/databricks-dolly-15k` | SFT adapter (`sft-model`) | `reports/report_sft_lora_dolly.md` |
| RM | `src/pipeline/reward_model_hh.py` | `Anthropic/hh-rlhf` | RM adapter (`rm-model`) | `reports/report_reward_model_hh.md` |
| PPO | `src/pipeline/ppo_rlhf_loop.py` | `Anthropic/hh-rlhf` prompts | policy adapter (`ppo-model`) | `reports/report_ppo_rlhf_loop.md` |
| DPO | `src/pipeline/dpo_lora_hh.py` | `Anthropic/hh-rlhf` pairs | policy adapter (`dpo-model`) | `reports/report_dpo_lora_hh.md` |

### Where to Start Reading

The reports are the substance of this project, and three findings are the ones worth reading first:

- **Several of TRL's logged PPO metrics are corrupted by post-EOS padding.** PPO report, Section 8, audits all fifteen logged scalars against the trainer source and classifies each as clean or affected, with the deflation factor derived per update.
- **The first held-out evaluation was underpowered, and the reported result was corrected.** PPO report, Section 6, reports the four-sample paired comparison that replaced it, and explains why the single-draw win rate could not support the claim originally made from it.
- **Negative and null results are reported rather than omitted**, e.g. the reward model's adversarial blind spots (RM report, Sections 6.3 and 7), the higher-capacity configuration that bought no accuracy (RM report, Section 8), and the checkpoint sweep that refuses to be used for selection (PPO report, Section 6.2).

## 2. Repository Layout

- `src/`: the pipeline code, run as modules, listed in workflow order.
  - `common/`: `config.py` (base model and canonical artefact paths) and `model_utils.py` (adapter merging, memory-management callback).
  - `eda/`: one exploratory-data-analysis script per stage, plus shared helpers in `eda_utils.py`.
  - `pipeline/`: the four training stages, plus `rm_adversarial_probes.py` (RM probe fixtures) and `promote_run.py`.
  - `analysis/`: post-run tooling that reads artefacts only, namely `aggregate_metrics.py`, `plot_loss_curves.py`, and `plot_ppo_curves.py`.
  - `diagnostics/`: post-run tooling that loads models, namely `generate_ppo_completions.py`, `sweep_ppo_checkpoints.py`, and `compare_policies.py` (the PPO-versus-DPO evaluation).
  - `export/`: `ppo_to_ollama.py`, packaging a completed PPO run for Ollama.
- `configs/`: JSON configuration files. Each `<stage>_default.json` lists every hyperparameter at its default, so it doubles as a reference when writing a custom config.
- `reports/`: a technical note for each stage, covering design choices, diagnostics, and results.
- `reports/assets/images/`: figures embedded in the reports.
- `reports/data/`: a committed snapshot of the small artefacts the reports cite, so a number can be checked without rerunning a stage. See its own README for provenance.
- `results/`: training outputs, checkpoints, and per-run metrics. Not committed; regenerate with the stage commands below.
- `sft-model`, `rm-model`, `ppo-model`, `dpo-model`: the canonical artefacts, each holding the promoted run of its stage. The `-merged` siblings are merge caches, created automatically on first use. Not committed.
- `tests/`: pytest suite for repository invariants. Run with `uv run pytest`.
- `OLLAMA.md`: walkthrough for testing a completed PPO policy locally in Ollama.

## 3. Setup

The project targets Python 3.12 or later and uses `uv`. To create the environment from the lockfile:

```bash
uv sync
```

This also installs the package itself, exposing the `rlhf-sft`, `rlhf-rm`, `rlhf-ppo`, `rlhf-dpo`, `rlhf-promote`, and `rlhf-ppo-ollama` commands.

## 4. Running the Pipeline

### 4.1. Data Understanding

Each stage has an EDA script to run before training. Each prints to the screen and writes a text report under that stage's `results/` directory. Pass `--sample N` for a faster partial run.

```bash
uv run python -m src.eda.eda_sft_dataset
uv run python -m src.eda.eda_reward_dataset
uv run python -m src.eda.eda_ppo_dataset
uv run python -m src.eda.eda_dpo_dataset
```

### 4.2. Training

Run the stages in order, since each depends on the previous artefact.

```bash
EXPORT_CANONICAL=1 uv run rlhf-sft   # 1. supervised fine-tuning
EXPORT_CANONICAL=1 uv run rlhf-rm    # 2. reward modelling
uv run rlhf-ppo                      # 3. PPO alignment
uv run rlhf-dpo                      # 3'. DPO alignment (alternative arm; needs only step 1)
```

Each command is a shortcut for `python -m src.pipeline.<stage>`. Either form works.

`EXPORT_CANONICAL=1` publishes a run to the shared `sft-model` / `rm-model` paths that the next stage reads. Without it a run writes only to its own labelled directory, so candidate configs can be compared before one is promoted (see [Section 6](#6-promoting-a-run-to-the-shared-artefact)).

Hyperparameters can be overridden on the command line or supplied as a JSON file:

```bash
uv run rlhf-sft --n_epochs 1
uv run rlhf-rm configs/rm_capacity.json
uv run rlhf-rm --lora_r 32 --n_epochs 2
```

### 4.3. Analysis and Diagnostics

`aggregate_metrics.py` collects per-run metric files and prints one ranked table per stage, also writing `summary_<stage>.tsv` and `summary_<stage>.md`. `plot_loss_curves.py` and `plot_ppo_curves.py` regenerate a run's figures from its saved trainer state.

```bash
uv run python -m src.analysis.aggregate_metrics             # all stages
uv run python -m src.analysis.aggregate_metrics --stage rm  # one stage
```

After a PPO run, two diagnostics support the post-run review. `generate_ppo_completions` writes a side-by-side review file of held-out completions from the trained policy, the SFT reference, and the base model, scored by the frozen RM. `sweep_ppo_checkpoints` evaluates every saved checkpoint on the same prompts, writing a readable table and a JSON record of every per-prompt and per-sample score. Both filenames carry the prompt and sample counts, so sweeps run at different settings never overwrite one another.

```bash
uv run python -m src.diagnostics.generate_ppo_completions --label <label>   # --skip-base omits the base column
uv run python -m src.diagnostics.sweep_ppo_checkpoints --label <label> --num-prompts 100 --samples-per-prompt 4
```

Both default to 20 prompts and one sample per prompt, which is fast but too noisy to report. Use the flags shown above for any comparison that will be written up. The prompt count is capped by the run's held-out split (`eval_examples`, 100 by default). The findings for the first full run are in `reports/report_ppo_rlhf_loop.md`, Section 7 for the completions and Section 6.2 for the sweep.

## 5. Outputs and Tracking

Every run writes its resolved configuration (`config_<label>.json`) and a metrics summary (`metrics_<label>.json`) under `results/`, both keyed by a hash of the full configuration, so distinct experiments never overwrite one another. Training curves go to TensorBoard:

```bash
uv run tensorboard --logdir results
```

Every stage prints a one-line memory report separating the live working set from reclaimable cache. In-step peak memory is opt-in:

```bash
TRACK_PEAK_MEM=1 uv run rlhf-rm configs/rm_capacity.json
```

## 6. Promoting a Run to the Shared Artefact

Runs write to labelled directories and never overwrite the shared paths automatically. Promotion is explicit, either by re-running the chosen config with `EXPORT_CANONICAL=1` (which resumes the completed run rather than retraining) or by copying it:

```bash
uv run rlhf-promote ppo <label>      # also accepts sft, rm, and dpo
```

Promotion clears the stale `<path>-merged` cache and records the source run in `promoted_from.json`. PPO must use the copy route, since `PPOTrainer` cannot resume.

## 7. Testing the PPO Policy in Ollama

Once a PPO run has been promoted, the policy can be packaged for interactive testing in [Ollama](https://ollama.com):

```bash
uv sync --extra ollama                                  # one-time: installs the gguf package
uv run rlhf-ppo-ollama                                  # exports the promoted policy
ollama run qwen2.5-0.5b-hh-rlhf-ppo-<label>
```

Pass `--label <label>` to export a run that has not been promoted, and `--ollama-name` to override the registered name. Full setup, prompt format, and troubleshooting are in **[OLLAMA.md](OLLAMA.md)**.

## 8. Licence and Attribution

The code, reports, and figures in this repository are released under the MIT Licence (see [LICENSE](LICENSE)).

The datasets and base model the pipeline builds on carry their own licences, which this repository neither alters nor supersedes:

| Source | Used for | Licence |
|---|---|---|
| [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | SFT training data | CC BY-SA 3.0 |
| [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) | RM, PPO, and DPO training data | MIT |
| [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) | base model for every stage | Apache-2.0 |

`reports/data/` redistributes a small number of verbatim excerpts from those datasets: six Dolly records in the SFT exploratory-data-analysis output, and HH-RLHF dialogues in the RM and PPO EDA outputs and in the PPO completion review. Those excerpts stay under their upstream licences, CC BY-SA 3.0 and MIT respectively, and the share-alike term on the Dolly excerpts applies to them rather than to this repository's own code or prose. Model completions in the completion review are generated output rather than dataset text.

Trained adapters are not distributed here (see [Section 2](#2-repository-layout)). Anyone redistributing a model trained with this pipeline should carry the base model's Apache-2.0 terms forward.
