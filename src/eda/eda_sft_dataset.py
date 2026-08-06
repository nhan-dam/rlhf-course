"""
Exploratory Data Analysis - Supervised Fine-Tuning Dataset
==========================================================
Inspect the databricks/databricks-dolly-15k instruction dataset before the SFT
stage, so configuration choices rest on the data. The aim is enough of a picture
to fine-tune effectively: the format and size, how the examples split into
prompt and completion (the boundary that enables completion-only loss), the
token-length distribution that informs max_length (SFT truncates rather than
filters), the mix of task categories, and basic data-quality checks. Output goes
to the screen and to a text file.

Inputs
------
Command-line flags (see parse_args): how many examples to sample, candidate
max_length caps, sample-preview size, and the output path. Defaults are pulled
from SFTTrainingConfig and the shared base model so the tokenisation matches
training.

Outputs
-------
A printed report (schema, splits, sample preview, prompt/completion token
lengths and the max_length truncation trade-off, category mix and context
coverage, and data quality), also written verbatim to
results/sft_lora_dolly/eda_sft_dataset.txt.

Public API
----------
main()                                            - run the full EDA and dump it.
prompt_completion_lengths(dataset, tokenizer, num_proc) - prompt/completion/total token lengths.
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
from ..common.config import BASE_MODEL, PROJECT_ROOT
from .eda_utils import (
    cap_tradeoff,
    dump,
    length_percentiles,
    make_console,
    preview_samples,
    report_schema,
    report_splits,
)
from ..pipeline.sft_lora_dolly import SFTTrainingConfig, _load_tokenizer, format_prompt

RESULT_PATH = f"{PROJECT_ROOT}/results/sft_lora_dolly"


def main() -> None:
    args = parse_args()
    candidates = sorted(int(c) for c in args.candidates.split(","))
    output_path = args.output or f"{RESULT_PATH}/eda_sft_dataset.txt"

    config = SFTTrainingConfig()
    tokenizer = _load_tokenizer(config.model_name)
    console = make_console()
    console.rule(f"EDA - {config.dataset_name}")

    # 1-2. Schema, field types, and splits.
    probe = load_dataset(config.dataset_name, split="train[:1]")[0]
    report_schema(console, config.dataset_name, probe)
    console.print(
        "Each example is an (instruction, context, response, category) record; the "
        "'context' field is an optional reference passage and is empty for most "
        "examples. Training maps it to a prompt/completion pair, and the column "
        "boundary is what lets SFTTrainer apply the completion-only loss.\n"
    )
    report_splits(console, config.dataset_name, {
        "train": f"SFT training ({config.val_fraction:.0%} held out for eval at train time)",
    })
    console.print(
        "Dolly ships a single 'train' split, so the SFT stage carves its own "
        f"{config.val_fraction:.0%} validation slice for overfitting detection.\n"
    )

    # Load and (optionally) subsample, then map to prompt/completion.
    dataset = _load_analysis_split(config, args.sample, console)

    # 3. Qualitative look at raw examples.
    preview_samples(console, dataset, ["category", "instruction", "context", "response"],
                    args.num_samples, args.sample_chars, config.seed)

    # 4. Prompt/completion token lengths and the max_length truncation trade-off.
    prompt_len, completion_len, total_len = prompt_completion_lengths(
        dataset, tokenizer, args.num_proc
    )
    _report_lengths(console, prompt_len, completion_len, total_len, candidates, config.max_length)

    # 5. Task-category mix and context coverage.
    _report_composition(console, dataset)

    # 6. Data-quality checks.
    _report_quality(console, dataset)

    dump(console, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0,
                        help="Examples to sample; 0 = whole dataset (default: 0, Dolly is small).")
    parser.add_argument("--candidates", default="256,384,512,640,768",
                        help="Comma-separated max_length values to evaluate.")
    parser.add_argument("--num-samples", type=int, default=6,
                        help="Random raw examples to print (default: 6).")
    parser.add_argument("--sample-chars", type=int, default=600,
                        help="Per-field character budget when printing samples (default: 600).")
    parser.add_argument("--num-proc", type=int, default=4,
                        help="Processes for the tokenisation map (default: 4).")
    parser.add_argument("--output", default=None,
                        help="Text-dump path (default: results/sft_lora_dolly/eda_sft_dataset.txt).")
    return parser.parse_args(sys.argv[1:])


def _load_analysis_split(config: SFTTrainingConfig, sample: int, console):
    """Load Dolly's train split, optionally subsampling it for speed."""
    dataset = load_dataset(config.dataset_name, split="train")
    if sample and sample < len(dataset):
        dataset = dataset.shuffle(seed=config.seed).select(range(sample))
        console.print(f"[cyan]Analysing[/cyan] {sample:,} sampled examples (seed {config.seed}).\n")
    else:
        console.print(f"[cyan]Analysing[/cyan] all {len(dataset):,} examples.\n")
    return dataset


def prompt_completion_lengths(
    dataset, tokenizer, num_proc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (prompt, completion, total) token lengths, as SFTTrainer sees them.

    The prompt is rendered with the training template via format_prompt and the
    completion is the raw response; total is their concatenation, which is the
    sequence SFTConfig.max_length truncates. EOS is appended to the completion to
    mirror the trainer's packing, so the totals match training within a token.
    """
    eos = tokenizer.eos_token

    def _lengths(batch: dict) -> dict:
        prompts = [format_prompt(instr, ctx)
                   for instr, ctx in zip(batch["instruction"], batch["context"])]
        completions = [resp + eos for resp in batch["response"]]
        prompt_ids = tokenizer(prompts)["input_ids"]
        completion_ids = tokenizer(completions, add_special_tokens=False)["input_ids"]
        return {
            "prompt_len":     [len(ids) for ids in prompt_ids],
            "completion_len": [len(ids) for ids in completion_ids],
            "total_len":      [len(p) + len(c) for p, c in zip(prompt_ids, completion_ids)],
        }

    measured = dataset.map(_lengths, batched=True, num_proc=num_proc,
                           remove_columns=dataset.column_names, desc="Tokenising examples")
    return (np.asarray(measured["prompt_len"]),
            np.asarray(measured["completion_len"]),
            np.asarray(measured["total_len"]))


def _report_lengths(console, prompt: np.ndarray, completion: np.ndarray, total: np.ndarray,
                    candidates: list[int], current: int) -> None:
    """Report the length distribution and the truncation effect of each candidate cap.

    SFT truncates the (prompt + completion) sequence rather than dropping rows, so
    the binding quantity is the total length and the cap is applied with truncate
    semantics: a row over the cap keeps its prompt but loses trailing completion
    tokens, which is precisely the supervision signal, so over-truncation quietly
    weakens training.
    """
    length_percentiles(
        console,
        "Token-length distribution (prompt / completion / total)",
        {"prompt": prompt, "completion": completion, "total": total},
    )
    cap_tradeoff(console, total, candidates, current, semantics="truncate")


def _report_composition(console, dataset) -> None:
    """Report the task-category mix and how often a context passage is present."""
    categories = Counter(dataset["category"])
    n = len(dataset)
    table = Table(title="Task-category mix")
    table.add_column("category")
    table.add_column("count", justify="right")
    table.add_column("% of data", justify="right", style="bold")
    for category, count in categories.most_common():
        table.add_row(category, f"{count:,}", f"{100 * count / n:.1f}%")
    console.print(table)

    has_context = sum(1 for ctx in dataset["context"] if ctx and ctx.strip())
    console.print(
        f"Context present in {has_context:,} / {n:,} examples "
        f"({100 * has_context / n:.1f}%); the rest are instruction-only, so the "
        "prompt template drops the context block for them.\n"
    )


def _report_quality(console, dataset) -> None:
    """Report empty fields and exact-duplicate (instruction, context, response) records."""
    n = len(dataset)
    instructions, contexts, responses = dataset["instruction"], dataset["context"], dataset["response"]
    n_empty_instruction = sum(1 for x in instructions if not x or not x.strip())
    n_empty_response = sum(1 for x in responses if not x or not x.strip())
    n_duplicate = n - len({hash((i, c, r)) for i, c, r in zip(instructions, contexts, responses)})

    table = Table(title="Data quality checks")
    table.add_column("check")
    table.add_column("count", justify="right")
    table.add_column("% of data", justify="right", style="bold")
    table.add_row("empty / whitespace instruction", f"{n_empty_instruction:,}",
                  f"{100 * n_empty_instruction / n:.2f}%")
    table.add_row("empty / whitespace response", f"{n_empty_response:,}",
                  f"{100 * n_empty_response / n:.2f}%")
    table.add_row("duplicate records", f"{n_duplicate:,}", f"{100 * n_duplicate / n:.2f}%")
    console.print(table)
    if n_empty_response:
        console.print("[yellow]Note:[/yellow] examples with an empty response give the "
                      "completion-only loss nothing to learn from.\n")
    else:
        console.print("No empty-response examples found.\n")


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - main: reads as the EDA outline - schema and splits, a random sample preview,
#   prompt/completion lengths and the max_length trade-off, the category mix and
#   context coverage, then data quality - before dumping the report to a file.
# - Shared rendering: the schema/splits tables, sample preview, percentile table,
#   cap trade-off, and file dump come from eda_utils, so this matches the RM and
#   PPO EDA output format exactly.
# - Faithful lengths: prompt_completion_lengths renders the prompt with the same
#   format_prompt template the trainer uses and appends EOS to the completion, so
#   the total length is the sequence SFTConfig.max_length actually truncates.
# - Truncate, not filter: SFT keeps every row and trims overflow tokens, so the
#   cap trade-off uses truncate semantics - the cost of a low cap is lost
#   completion supervision, not dropped examples.
# - Composition: Dolly is multi-task, so the category mix and the fraction of
#   examples carrying a context passage are reported - both shape what the SFT
#   model learns to expect at inference.
# - Design choice: the heavy tokenisation runs in a parallel map while the light
#   category/context/quality counts run in plain Python over the columns, which
#   is trivial at Dolly's 15k scale.
# =============================================================================
