"""
Exploratory Data Analysis - PPO Prompt Dataset
==============================================
Inspect the prompts the PPO stage trains on before running the loop, so
configuration choices rest on the data. PPO consumes only the prompt side of
Anthropic/hh-rlhf: the prompt is extracted from each 'chosen' dialogue (text up
to and including the final '\\n\\nAssistant:' marker) and the policy generates
the response itself. The aim is enough of a picture to run PPO effectively: the
prompt-length distribution that fixes max_prompt_tokens (long prompts are
filtered, not truncated), the conversation depth of the prompts, and basic
data-quality checks. Output goes to the screen and to a text file.

Inputs
------
Command-line flags (see parse_args): which split to analyse, how many prompts to
sample, candidate max_prompt_tokens caps, sample-preview size, and the output
path. Defaults are pulled from PPORunConfig so the tokenisation matches training.

Outputs
-------
A printed report (schema, splits, prompt sample preview, prompt-length
distribution and the max_prompt_tokens filtering trade-off, conversation depth,
and data quality), also written verbatim to
results/ppo_rlhf_loop/eda_ppo_dataset.txt.

Public API
----------
main()                                       - run the full EDA and dump it.
prompt_token_lengths(dataset, tokenizer, num_proc) - per-prompt token lengths, as PPOTrainer sees them.
"""

# stdlib
import argparse
import sys
from collections import Counter

# third-party
import numpy as np
from datasets import load_dataset
from rich.table import Table

# local
from ..common.config import PROJECT_ROOT
from .eda_utils import (
    cap_tradeoff,
    dump,
    length_percentiles,
    make_console,
    preview_samples,
    report_schema,
    report_splits,
)
from ..common.model_utils import resolve_model_path
from ..pipeline.ppo_rlhf_loop import PPORunConfig, _load_tokenizer, extract_prompt

RESULT_PATH = f"{PROJECT_ROOT}/results/ppo_rlhf_loop"

# HH-RLHF dialogues are turn-delimited by these markers.
HUMAN_MARKER = "\n\nHuman:"
ASSISTANT_MARKER = "\n\nAssistant:"


def main() -> None:
    args = parse_args()
    candidates = sorted(int(c) for c in args.candidates.split(","))
    output_path = args.output or f"{RESULT_PATH}/eda_ppo_dataset.txt"

    config = PPORunConfig()
    tokenizer = _load_tokenizer(resolve_model_path(config.sft_model_path, "causal-lm"))
    console = make_console()
    console.rule(f"EDA - {config.dataset_name} (PPO prompts)")

    # 1-2. Schema, field types, and splits.
    probe = load_dataset(config.dataset_name, split=f"{args.split}[:1]")[0]
    report_schema(console, config.dataset_name, probe)
    console.print(
        "PPO ignores the preference labels: it takes only the prompt, extracted "
        "from each 'chosen' dialogue up to the final '\\n\\nAssistant:' marker, and "
        "the policy generates the response. So the units analysed below are prompts, "
        "not preference pairs.\n"
    )
    report_splits(console, config.dataset_name, {
        "train": f"PPO training + {config.eval_examples:,} eval prompts (eval carved from train)",
        "test":  "shipped test split (unused by the PPO stage)",
    })

    # Load, sample, and extract the prompt text into its own column.
    dataset = _load_prompt_split(config, args.split, args.sample, console)

    # 3. Qualitative look at the extracted prompts.
    preview_samples(console, dataset, ["prompt"], args.num_samples, args.sample_chars, config.seed)

    # 4. Prompt-length distribution and the max_prompt_tokens filtering trade-off.
    prompt_len = prompt_token_lengths(dataset, tokenizer, args.num_proc)
    _report_lengths(console, prompt_len, candidates, config.max_prompt_tokens)

    # 5-6. Conversation depth and data-quality checks.
    _report_structure_and_quality(console, dataset["prompt"])

    dump(console, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train",
                        help="Split to analyse (default: train).")
    parser.add_argument("--sample", type=int, default=0,
                        help="Prompts to sample; 0 = whole split (default: 0, the full split). "
                             "Pass e.g. --sample 20000 for a faster partial run.")
    parser.add_argument("--candidates", default="128,192,256,320,384",
                        help="Comma-separated max_prompt_tokens values to evaluate.")
    parser.add_argument("--num-samples", type=int, default=6,
                        help="Random prompts to print (default: 6).")
    parser.add_argument("--sample-chars", type=int, default=700,
                        help="Character budget when printing prompts (default: 700).")
    parser.add_argument("--num-proc", type=int, default=4,
                        help="Processes for the tokenisation map (default: 4).")
    parser.add_argument("--output", default=None,
                        help="Text-dump path (default: results/ppo_rlhf_loop/eda_ppo_dataset.txt).")
    return parser.parse_args(sys.argv[1:])


def _load_prompt_split(config: PPORunConfig, split: str, sample: int, console):
    """Load the split, subsample for speed, and add a 'prompt' column via extract_prompt."""
    dataset = load_dataset(config.dataset_name, split=split)
    if sample and sample < len(dataset):
        dataset = dataset.shuffle(seed=config.seed).select(range(sample))
        console.print(f"[cyan]Analysing[/cyan] {sample:,} sampled prompts from "
                      f"'{split}' (seed {config.seed}).\n")
    else:
        console.print(f"[cyan]Analysing[/cyan] all {len(dataset):,} prompts from '{split}'.\n")
    return dataset.map(lambda ex: {"prompt": extract_prompt(ex["chosen"])},
                       remove_columns=dataset.column_names, desc="Extracting prompts")


def prompt_token_lengths(dataset, tokenizer, num_proc: int) -> np.ndarray:
    """Return per-prompt token lengths, tokenised exactly as PPOTrainer's prompt map does."""
    def _lengths(batch: dict) -> dict:
        return {"prompt_len": [len(ids) for ids in tokenizer(batch["prompt"])["input_ids"]]}

    measured = dataset.map(_lengths, batched=True, num_proc=num_proc,
                           remove_columns=dataset.column_names, desc="Tokenising prompts")
    return np.asarray(measured["prompt_len"])


def _report_lengths(console, prompt_len: np.ndarray, candidates: list[int], current: int) -> None:
    """Report the prompt-length distribution and the filtering effect of each candidate cap.

    PPO filters out over-long prompts rather than truncating them, since a
    truncated dialogue can lose the actual question and leave the policy
    optimising reward on nonsense; the cap is therefore applied with filter
    semantics on the prompt length.
    """
    length_percentiles(console, "Prompt token-length distribution", {"prompt": prompt_len})
    cap_tradeoff(console, prompt_len, candidates, current, semantics="filter")


def _report_structure_and_quality(console, prompts: list[str]) -> None:
    """Report conversation depth of the prompts and basic data-quality checks."""
    n = len(prompts)
    human_turns = np.array([p.count(HUMAN_MARKER) for p in prompts])
    asst_turns = np.array([p.count(ASSISTANT_MARKER) for p in prompts])

    structure = Table(title="Prompt conversation depth")
    structure.add_column("statistic")
    structure.add_column("value", justify="right", style="bold")
    structure.add_row("median Human turns", f"{np.median(human_turns):.0f}")
    structure.add_row("median Assistant turns (context)", f"{np.median(asst_turns):.0f}")
    structure.add_row("max Human turns", f"{human_turns.max()}")
    structure.add_row("single-turn prompts (1 Human turn)", f"{100 * np.mean(human_turns == 1):.1f}%")
    structure.add_row("multi-turn prompts (>1 Human turn)", f"{100 * np.mean(human_turns > 1):.1f}%")
    console.print(structure)
    turn_hist = ", ".join(f"{k}:{v}" for k, v in sorted(Counter(human_turns.tolist()).items())[:8])
    console.print(f"Human-turn counts (turns:prompts): {turn_hist}\n")

    n_empty = sum(1 for p in prompts if not p.strip())
    n_unique = len(set(prompts))
    n_duplicate = n - n_unique

    quality = Table(title="Data quality checks")
    quality.add_column("check")
    quality.add_column("count", justify="right")
    quality.add_column("% of sample", justify="right", style="bold")
    quality.add_row("empty / whitespace prompt", f"{n_empty:,}", f"{100 * n_empty / n:.2f}%")
    quality.add_row("duplicate prompts", f"{n_duplicate:,}", f"{100 * n_duplicate / n:.2f}%")
    console.print(quality)
    if n_duplicate:
        console.print(
            "[yellow]Note:[/yellow] HH-RLHF reuses each prompt across several preference "
            "pairs, so duplicate prompts are expected here; PPO only needs the distinct "
            f"prompts ({n_unique:,} unique in this sample).\n"
        )
    else:
        console.print("No duplicate prompts in this sample.\n")


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - main: reads as the EDA outline - schema and splits, a prompt sample preview,
#   the prompt-length distribution and the max_prompt_tokens trade-off, then
#   conversation depth and data quality - before dumping the report to a file.
# - Prompt-only view: PPO discards the preference labels, so _load_prompt_split
#   reuses extract_prompt to turn each 'chosen' dialogue into the prompt the
#   policy will actually condition on, and everything downstream analyses prompts.
# - Shared rendering: schema/splits tables, the sample preview, the percentile
#   table, the cap trade-off, and the file dump come from eda_utils, matching the
#   SFT and RM EDA output format.
# - Filter, not truncate: over-long prompts are dropped rather than trimmed, so
#   the cap trade-off uses filter semantics; the cost of a low cap is fewer
#   prompts, never a prompt with its question cut off.
# - Duplicates expected: HH-RLHF pairs many chosen/rejected responses to the same
#   prompt, so duplicate prompts are normal and are reported with the distinct
#   count rather than flagged as an error.
# - Design choice: prompt extraction and tokenisation run in a parallel map while
#   the turn and duplicate counts run in plain Python over the prompt column,
#   keeping the heavy work where it belongs.
# =============================================================================
