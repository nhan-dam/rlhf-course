"""
Exploratory Data Analysis - DPO Preference Dataset
==================================================
Inspect the preference pairs the DPO stage trains on before running it, so the
configuration rests on the data. DPO consumes Anthropic/hh-rlhf whole: the
shared prompt (implicit in the two dialogues) and both full sides, so THREE
lengths matter, each with filter semantics -- the prompt against
max_prompt_tokens (PPO-cap parity) and each side with EOS against
max_pair_tokens (RM-cap parity). The aim is enough of a picture to run DPO
effectively: the per-cap and compound retention that fix the two caps, the
chosen-versus-rejected length bias that predicts length-gamed implicit
rewards, and basic data-quality checks. Output goes to the screen and to a
text file.

Inputs
------
Command-line flags (see parse_args): which split to analyse, how many pairs to
sample, candidate caps for both filters, sample-preview size, and the output
path. Defaults are pulled from DPOTrainingConfig so tokenisation matches
training.

Outputs
-------
A printed report (schema, splits, pair sample preview, the three length
distributions, per-cap and compound filtering trade-offs, length-bias check,
and data quality), also written verbatim to
results/dpo_lora_hh/eda_dpo_dataset.txt.

Public API
----------
main()                                   - run the full EDA and dump it.
pair_token_lengths(dataset, tokenizer, num_proc) - prompt/chosen/rejected token lengths, as DPOTrainer sees them.
"""

# stdlib
import argparse
import sys

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
from ..pipeline.dpo_lora_hh import DPOTrainingConfig, _load_tokenizer
from ..pipeline.ppo_rlhf_loop import extract_prompt

RESULT_PATH = f"{PROJECT_ROOT}/results/dpo_lora_hh"


def main() -> None:
    args = parse_args()
    prompt_candidates = sorted(int(c) for c in args.prompt_candidates.split(","))
    pair_candidates = sorted(int(c) for c in args.pair_candidates.split(","))
    output_path = args.output or f"{RESULT_PATH}/eda_dpo_dataset.txt"

    config = DPOTrainingConfig()
    tokenizer = _load_tokenizer(resolve_model_path(config.sft_model_path, "causal-lm"))
    console = make_console()
    console.rule(f"EDA - {config.dataset_name} (DPO preference pairs)")

    # 1-2. Schema, field types, and splits.
    probe = load_dataset(config.dataset_name, split=f"{args.split}[:1]")[0]
    report_schema(console, config.dataset_name, probe)
    console.print(
        "DPO consumes both columns whole: DPOTrainer extracts the shared prompt "
        "internally and appends EOS to both sides, so the units analysed below are "
        "preference pairs, with three lengths each (prompt, chosen, rejected).\n"
    )
    report_splits(console, config.dataset_name, {
        "train": "DPO training pairs (after the two-cap filter)",
        "test":  f"in-training eval ({config.eval_examples:,}-pair subsample) + "
                 "post-training gate + PPO-vs-DPO comparison prompts",
    })

    # Load and subsample.
    dataset = _load_split(config, args.split, args.sample, console)

    # 3. Qualitative look at the pairs.
    preview_samples(console, dataset, ["chosen", "rejected"],
                    args.num_samples, args.sample_chars, config.seed)

    # 4. The three length distributions, tokenised as training will see them.
    prompt_len, chosen_len, rejected_len = pair_token_lengths(dataset, tokenizer, args.num_proc)
    pair_len = np.maximum(chosen_len, rejected_len)   # the binding pair length
    length_percentiles(console, "Token-length distributions", {
        "prompt": prompt_len,
        "chosen (+EOS)": chosen_len,
        "rejected (+EOS)": rejected_len,
        "pair max": pair_len,
    })

    # 5. Per-cap trade-offs, then the compound filter both caps apply together.
    console.print("[bold]Prompt cap[/bold] (parity with the PPO prompt filter):")
    cap_tradeoff(console, prompt_len, prompt_candidates, config.max_prompt_tokens,
                 semantics="filter")
    console.print("[bold]Pair cap[/bold] on max(chosen, rejected), both with EOS "
                  "(parity with the RM filter):")
    cap_tradeoff(console, pair_len, pair_candidates, config.max_pair_tokens,
                 semantics="filter")
    _report_compound_filter(console, prompt_len, pair_len,
                            config.max_prompt_tokens, config.max_pair_tokens)

    # 6. DPO-specific bias check: chosen-vs-rejected length asymmetry.
    _report_length_bias(console, chosen_len, rejected_len)

    # 7. Data-quality checks.
    _report_quality(console, dataset)

    dump(console, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train",
                        help="Split to analyse (default: train).")
    parser.add_argument("--sample", type=int, default=0,
                        help="Pairs to sample; 0 = whole split (default: 0, the full split). "
                             "Pass e.g. --sample 20000 for a faster partial run.")
    parser.add_argument("--prompt-candidates", default="128,192,256,320,384",
                        help="Comma-separated max_prompt_tokens values to evaluate.")
    parser.add_argument("--pair-candidates", default="384,448,512,640,768",
                        help="Comma-separated max_pair_tokens values to evaluate.")
    parser.add_argument("--num-samples", type=int, default=6,
                        help="Random pairs to print (default: 6).")
    parser.add_argument("--sample-chars", type=int, default=700,
                        help="Character budget when printing pair sides (default: 700).")
    parser.add_argument("--num-proc", type=int, default=4,
                        help="Processes for the tokenisation map (default: 4).")
    parser.add_argument("--output", default=None,
                        help="Text-dump path (default: results/dpo_lora_hh/eda_dpo_dataset.txt).")
    return parser.parse_args(sys.argv[1:])


def _load_split(config: DPOTrainingConfig, split: str, sample: int, console):
    """Load the split and subsample for speed (seeded), keeping both raw columns."""
    dataset = load_dataset(config.dataset_name, split=split)
    if sample and sample < len(dataset):
        dataset = dataset.shuffle(seed=config.seed).select(range(sample))
        console.print(f"[cyan]Analysing[/cyan] {sample:,} sampled pairs from "
                      f"'{split}' (seed {config.seed}).\n")
    else:
        console.print(f"[cyan]Analysing[/cyan] all {len(dataset):,} pairs from '{split}'.\n")
    return dataset


def pair_token_lengths(dataset, tokenizer, num_proc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (prompt, chosen, rejected) token lengths, tokenised as training does.

    The prompt is extracted with the PPO stage's extract_prompt (the same cut
    DPOTrainer applies internally), and EOS is appended to both sides before
    measuring, exactly as DPOTrainer's add_eos map and the training-time
    filter_pairs do -- so every length here matches the training decision
    within a token.
    """
    eos = tokenizer.eos_token

    def _lengths(batch: dict) -> dict:
        prompts  = [extract_prompt(text) for text in batch["chosen"]]
        chosen   = [t if t.endswith(eos) else t + eos for t in batch["chosen"]]
        rejected = [t if t.endswith(eos) else t + eos for t in batch["rejected"]]
        return {
            "prompt_len":   [len(ids) for ids in tokenizer(prompts)["input_ids"]],
            "chosen_len":   [len(ids) for ids in tokenizer(chosen)["input_ids"]],
            "rejected_len": [len(ids) for ids in tokenizer(rejected)["input_ids"]],
        }

    measured = dataset.map(_lengths, batched=True, num_proc=num_proc,
                           remove_columns=dataset.column_names, desc="Tokenising pairs")
    return (np.asarray(measured["prompt_len"]),
            np.asarray(measured["chosen_len"]),
            np.asarray(measured["rejected_len"]))


def _report_compound_filter(
    console, prompt_len: np.ndarray, pair_len: np.ndarray,
    prompt_cap: int, pair_cap: int,
) -> None:
    """Decompose the joint retention of the two caps at their current values.

    The two caps overlap (a long pair usually has a long prompt), so the
    compound retention is not the product of the marginals; this table shows
    what each cap uniquely costs, which is the number to weigh when
    considering moving either cap alone.
    """
    n = len(prompt_len)
    prompt_ok = prompt_len <= prompt_cap
    pair_ok = pair_len <= pair_cap
    kept = prompt_ok & pair_ok

    table = Table(title=f"Compound filter at the current caps "
                        f"(prompt<={prompt_cap}, pair<={pair_cap})")
    table.add_column("outcome")
    table.add_column("pairs", justify="right")
    table.add_column("% of split", justify="right", style="bold")
    table.add_row("kept (both caps pass)", f"{int(kept.sum()):,}", f"{100 * kept.mean():.2f}%")
    table.add_row("dropped by prompt cap only",
                  f"{int((~prompt_ok & pair_ok).sum()):,}",
                  f"{100 * (~prompt_ok & pair_ok).mean():.2f}%")
    table.add_row("dropped by pair cap only",
                  f"{int((prompt_ok & ~pair_ok).sum()):,}",
                  f"{100 * (prompt_ok & ~pair_ok).mean():.2f}%")
    table.add_row("dropped by both",
                  f"{int((~prompt_ok & ~pair_ok).sum()):,}",
                  f"{100 * (~prompt_ok & ~pair_ok).mean():.2f}%")
    console.print(table)
    console.print(
        f"The kept fraction is what DPO trains on; PPO kept "
        f"{100 * prompt_ok.mean():.2f}% under its prompt cap alone, so the pair cap "
        f"uniquely costs {100 * (prompt_ok & ~pair_ok).mean():.2f} percentage points "
        f"of {n:,} pairs -- the price of RM parity.\n"
    )


def _report_length_bias(console, chosen_len: np.ndarray, rejected_len: np.ndarray) -> None:
    """Report chosen-vs-rejected length asymmetry.

    DPO's implicit reward is a sum of per-token log-probability ratios, so a
    systematic length gap between the two sides lets the policy buy margin
    with verbosity instead of quality -- the DPO analogue of the RM's
    length-bias trap. A near-symmetric split means implicit-reward margins
    reflect content; a clear majority means watch output lengths downstream.
    """
    diff = chosen_len - rejected_len
    table = Table(title="Chosen-vs-rejected length bias")
    table.add_column("statistic")
    table.add_column("value", justify="right", style="bold")
    table.add_row("chosen longer", f"{100 * np.mean(diff > 0):.1f}%")
    table.add_row("rejected longer", f"{100 * np.mean(diff < 0):.1f}%")
    table.add_row("equal length", f"{100 * np.mean(diff == 0):.1f}%")
    table.add_row("mean(chosen - rejected) tokens", f"{diff.mean():+.1f}")
    table.add_row("median(chosen - rejected) tokens", f"{np.median(diff):+.0f}")
    console.print(table)
    if abs(np.mean(diff > 0) - np.mean(diff < 0)) < 0.10:
        console.print("[green]Near-symmetric[/green]: length is weakly informative of "
                      "preference, so implicit-reward margins should reflect content.\n")
    else:
        longer = "chosen" if np.mean(diff > 0) > np.mean(diff < 0) else "rejected"
        console.print(f"[yellow]Asymmetric[/yellow]: '{longer}' is longer in a clear "
                      f"majority of pairs -- watch generated response lengths during the "
                      f"PPO-vs-DPO comparison for length inflation.\n")


def _report_quality(console, dataset) -> None:
    """Report empty sides, duplicate pairs, and degenerate identical pairs."""
    chosen = dataset["chosen"]
    rejected = dataset["rejected"]
    n = len(chosen)

    n_empty = sum(1 for c, r in zip(chosen, rejected) if not c.strip() or not r.strip())
    n_identical = sum(1 for c, r in zip(chosen, rejected) if c == r)
    n_duplicate = n - len(set(zip(chosen, rejected)))

    table = Table(title="Data quality checks")
    table.add_column("check")
    table.add_column("count", justify="right")
    table.add_column("% of sample", justify="right", style="bold")
    table.add_row("empty / whitespace side", f"{n_empty:,}", f"{100 * n_empty / n:.2f}%")
    table.add_row("identical chosen == rejected", f"{n_identical:,}", f"{100 * n_identical / n:.2f}%")
    table.add_row("duplicate pairs", f"{n_duplicate:,}", f"{100 * n_duplicate / n:.2f}%")
    console.print(table)
    console.print(
        "Identical pairs carry no preference signal (the DPO loss sees a zero "
        "margin by construction) and exact duplicates re-weight their pair; both "
        "are defects here, unlike the expected prompt reuse across different "
        "pairs, which is the dataset's design.\n"
    )


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - main: reads as the EDA outline - schema and splits, a pair preview, the
#   three length distributions, the two per-cap trade-offs plus their compound
#   decomposition, the length-bias check, and data quality - then dumps the
#   report to a file.
# - Three binding lengths: DPO filters on the prompt (PPO-cap parity) and on
#   max(chosen, rejected) with EOS (RM-cap parity), so both are measured with
#   the exact tokenisation training applies (extract_prompt + EOS append,
#   mirroring filter_pairs and DPOTrainer's internal maps).
# - Compound decomposition: the caps overlap, so the joint retention is shown
#   split into 'prompt cap only', 'pair cap only', and 'both', making the
#   unique cost of each cap visible before either is moved.
# - Length bias: chosen-vs-rejected length asymmetry is the DPO analogue of
#   the RM's length-gaming trap (the implicit reward sums per-token ratios),
#   reported with a conditional interpretation line.
# - Quality: identical sides and exact duplicate pairs are defects (zero
#   margin / re-weighting); prompt reuse across pairs is the dataset's design
#   and is not flagged.
# - Shared rendering: tables, preview, percentiles, cap trade-off, and the
#   file dump come from eda_utils, matching the other stages' EDA output.
# =============================================================================
