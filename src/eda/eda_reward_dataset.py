"""
Exploratory Data Analysis - Reward-Model Preference Dataset
===========================================================
Inspect the Anthropic/hh-rlhf preference pairs before training the reward model
(RM), so configuration choices rest on the data rather than on defaults. The aim
is not an exhaustive audit but enough of a picture to train effectively: the
format and size, the token-length distribution that fixes max_length, and the
traps the data hides - chief among them a chosen-vs-rejected length bias that an
RM could learn instead of quality (the seed of length-based reward hacking in
the PPO stage). Output goes to the screen and to a text file.

Inputs
------
Command-line flags (see parse_args): how many training pairs to sample,
candidate max_length caps, sample-preview size, and the output path. Defaults
are pulled from RMTrainingConfig so the tokenisation matches training. Both
splits are always analysed: train fixes the training configuration, and test
fixes the acceptance-gate population (the gate filters it on max_length), so
their cap retentions belong side by side in one table.

Outputs
-------
A printed report (schema, splits, sample preview, per-split token lengths and
a combined max_length retention trade-off, length bias, conversation
structure, and data quality), also written verbatim to
results/reward_model_hh/eda_reward_dataset.txt.

Public API
----------
main()                                        - run the full EDA and dump it.
token_lengths(dataset, tokenizer, num_proc)   - per-side token lengths, as RewardTrainer sees them.
"""

# stdlib
import argparse
import os
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
from ..pipeline.reward_model_hh import RMTrainingConfig, _load_tokenizer

RESULT_PATH = f"{PROJECT_ROOT}/results/reward_model_hh"

# HH-RLHF dialogues are turn-delimited by these markers.
HUMAN_MARKER = "\n\nHuman:"
ASSISTANT_MARKER = "\n\nAssistant:"


def main() -> None:
    args = parse_args()
    candidates = sorted(int(c) for c in args.candidates.split(","))
    output_path = args.output or f"{RESULT_PATH}/eda_reward_dataset.txt"

    config = RMTrainingConfig()
    tokenizer = _load_tokenizer(resolve_model_path(config.sft_model_path, "causal-lm"))
    console = make_console()
    console.rule(f"EDA - {config.dataset_name}")

    # 1-2. Schema, field types, and the splits the dataset ships with.
    probe = load_dataset(config.dataset_name, split="train[:1]")[0]
    report_schema(console, config.dataset_name, probe)
    console.print(
        "Both 'chosen' and 'rejected' are full-dialogue strings in HH-RLHF's "
        "implicit-prompt format (the prompt is shared, the two sides differ in the "
        "final assistant turn); no field is a vector, and RewardTrainer consumes "
        "these columns directly.\n"
    )
    report_splits(console, config.dataset_name, {
        "train": "RM training pairs",
        "test":  f"held-out eval ({config.eval_examples:,}-pair subsample during training; "
                 f"full split, length-filtered, for the acceptance gate)",
    })
    console.print(
        "A dedicated test split ships with the dataset, so the RM evaluates on the "
        "authors' held-out pairs rather than a self-carved validation set.\n"
    )

    # Load both splits: train drives the training configuration, and test is
    # the acceptance-gate population, so both cap retentions are decisions.
    # The test split is small and is always analysed in full.
    train_split = _load_analysis_split(config, "train", args.sample, console)
    test_split  = _load_analysis_split(config, "test", 0, console)

    # 3. Qualitative look at raw examples (train side; both splits share the format).
    preview_samples(console, train_split, ["chosen", "rejected"], args.num_samples,
                    args.sample_chars, config.seed)

    # 4. Per-split token lengths and one combined max_length retention table.
    train_lengths = token_lengths(train_split, tokenizer, args.num_proc)
    test_lengths  = token_lengths(test_split, tokenizer, args.num_proc)
    _report_lengths(console, {"train": train_lengths, "test": test_lengths},
                    candidates, config.max_length)

    # 5. Length bias: does 'chosen' tend to be longer than 'rejected'?
    # Train split: this is the shortcut the RM could learn from.
    _report_length_bias(console, *train_lengths)

    # 6-7. Conversation structure and data-quality checks (train split, raw text).
    _report_structure_and_quality(console, train_split["chosen"], train_split["rejected"])

    dump(console, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0,
                        help="Training pairs to sample; 0 = whole split (default: 0). "
                             "Pass e.g. --sample 20000 for a faster partial run. The test "
                             "split is always analysed in full (it is small and fixes the "
                             "acceptance-gate population).")
    parser.add_argument("--candidates", default="256,320,384,448,512",
                        help="Comma-separated max_length values to evaluate.")
    parser.add_argument("--num-samples", type=int, default=6,
                        help="Random raw examples to print (default: 6).")
    parser.add_argument("--sample-chars", type=int, default=700,
                        help="Per-side character budget when printing samples (default: 700).")
    parser.add_argument("--num-proc", type=int, default=4,
                        help="Processes for the tokenisation map (default: 4).")
    parser.add_argument("--output", default=None,
                        help="Text-dump path (default: results/reward_model_hh/eda_reward_dataset.txt).")
    return parser.parse_args(sys.argv[1:])


def _load_analysis_split(config: RMTrainingConfig, split: str, sample: int, console):
    """Load the split, optionally shuffling and subsampling it for speed."""
    dataset = load_dataset(config.dataset_name, split=split)
    if sample and sample < len(dataset):
        dataset = dataset.shuffle(seed=config.seed).select(range(sample))
        console.print(f"[cyan]Analysing[/cyan] {sample:,} sampled pairs from "
                      f"'{split}' (seed {config.seed}).\n")
    else:
        console.print(f"[cyan]Analysing[/cyan] all {len(dataset):,} pairs from '{split}'.\n")
    return dataset


def token_lengths(dataset, tokenizer, num_proc: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (chosen_lengths, rejected_lengths), replicating RewardTrainer.

    EOS is appended exactly as TRL does, then each side is tokenised with the
    tokeniser's default settings - the same call RewardTrainer makes - so the
    measured lengths are identical to those training will see.
    """
    eos = tokenizer.eos_token

    def _lengths(batch: dict) -> dict:
        chosen = [c if c.endswith(eos) else c + eos for c in batch["chosen"]]
        rejected = [r if r.endswith(eos) else r + eos for r in batch["rejected"]]
        return {
            "chosen_len":   [len(ids) for ids in tokenizer(chosen)["input_ids"]],
            "rejected_len": [len(ids) for ids in tokenizer(rejected)["input_ids"]],
        }

    measured = dataset.map(_lengths, batched=True, num_proc=num_proc,
                           remove_columns=dataset.column_names, desc="Tokenising pairs")
    return np.asarray(measured["chosen_len"]), np.asarray(measured["rejected_len"])


def _report_lengths(console, lengths_by_split: dict[str, tuple[np.ndarray, np.ndarray]],
                    candidates: list[int], current: int) -> None:
    """Report per-split length distributions and one combined cap table.

    A pair survives only when BOTH sides fit, so the binding quantity per pair
    is the longer of the two sides; the cap is applied to that with filter
    semantics. Both splits face the same cap but in different roles: the train
    column shows what RewardTrainer's filter keeps for training, and the test
    column shows the acceptance-gate population retained at the same cap, so
    the two retentions are read together off one table. The train split comes
    first, since the cap is chosen for training and the gate inherits it.
    """
    pairs: dict[str, np.ndarray] = {}
    for split, (chosen, rejected) in lengths_by_split.items():
        pair = np.maximum(chosen, rejected)
        pairs[split] = pair
        length_percentiles(
            console,
            f"Token-length distribution, {split} split (with EOS, as RewardTrainer sees it)",
            {"chosen": chosen, "rejected": rejected, "max(pair)": pair},
        )
    cap_tradeoff(console, pairs, candidates, current, semantics="filter")


def _report_length_bias(console, chosen: np.ndarray, rejected: np.ndarray) -> None:
    """Report whether 'chosen' is systematically longer than 'rejected'."""
    diff = chosen - rejected
    longer = float(np.mean(chosen > rejected))
    table = Table(title="Length bias - does 'chosen' tend to be longer?")
    table.add_column("statistic")
    table.add_column("value", justify="right", style="bold")
    table.add_row("mean chosen length (tokens)", f"{chosen.mean():.1f}")
    table.add_row("mean rejected length (tokens)", f"{rejected.mean():.1f}")
    table.add_row("mean (chosen - rejected)", f"{diff.mean():+.1f}")
    table.add_row("median (chosen - rejected)", f"{np.median(diff):+.0f}")
    table.add_row("% pairs where chosen is longer", f"{100 * longer:.1f}%")
    console.print(table)
    if longer >= 0.55:
        console.print(
            "[yellow]Heads-up:[/yellow] 'chosen' is longer in a clear majority of pairs, "
            "so an RM can pick up length as a shortcut for quality; watch for length "
            "inflation (reward hacking) in the PPO stage.\n"
        )
    else:
        console.print("Chosen and rejected lengths are roughly balanced, so raw length "
                      "is a weak shortcut here.\n")


def _report_structure_and_quality(console, chosen_texts: list[str], rejected_texts: list[str]) -> None:
    """Report conversation structure (turns, shared prompt prefix) and data quality."""
    n = len(chosen_texts)
    human_turns = np.array([c.count(HUMAN_MARKER) for c in chosen_texts])
    asst_turns = np.array([c.count(ASSISTANT_MARKER) for c in chosen_texts])
    # Chosen and rejected agree up to the final assistant response, so their
    # common character prefix approximates the shared prompt.
    prefix_frac = np.array([
        len(os.path.commonprefix([c, r])) / max(len(c), 1)
        for c, r in zip(chosen_texts, rejected_texts)
    ])

    structure = Table(title="Conversation structure (per pair, from the 'chosen' side)")
    structure.add_column("statistic")
    structure.add_column("value", justify="right", style="bold")
    structure.add_row("median Human turns", f"{np.median(human_turns):.0f}")
    structure.add_row("median Assistant turns", f"{np.median(asst_turns):.0f}")
    structure.add_row("max Human turns", f"{human_turns.max()}")
    structure.add_row("single-turn pairs (1 Human turn)", f"{100 * np.mean(human_turns == 1):.1f}%")
    structure.add_row("multi-turn pairs (>1 Human turn)", f"{100 * np.mean(human_turns > 1):.1f}%")
    structure.add_row("median shared-prefix fraction", f"{100 * np.median(prefix_frac):.1f}%")
    console.print(structure)
    turn_hist = ", ".join(f"{k}:{v}" for k, v in sorted(Counter(human_turns.tolist()).items())[:8])
    console.print(f"Human-turn counts (turns:pairs): {turn_hist}\n")

    n_empty = sum(1 for c, r in zip(chosen_texts, rejected_texts) if not c.strip() or not r.strip())
    n_identical = sum(1 for c, r in zip(chosen_texts, rejected_texts) if c.strip() == r.strip())
    n_duplicate = n - len({hash((c, r)) for c, r in zip(chosen_texts, rejected_texts)})

    quality = Table(title="Data quality checks")
    quality.add_column("check")
    quality.add_column("count", justify="right")
    quality.add_column("% of sample", justify="right", style="bold")
    quality.add_row("empty / whitespace side", f"{n_empty:,}", f"{100 * n_empty / n:.2f}%")
    quality.add_row("chosen == rejected", f"{n_identical:,}", f"{100 * n_identical / n:.2f}%")
    quality.add_row("duplicate pairs", f"{n_duplicate:,}", f"{100 * n_duplicate / n:.2f}%")
    console.print(quality)
    if n_empty or n_identical:
        console.print(
            "[yellow]Note:[/yellow] empty or identical pairs carry no preference signal; "
            "RewardTrainer's loss on an identical pair is fixed at -log sigma(0) = 0.69 and "
            "contributes no gradient.\n"
        )
    else:
        console.print("No empty or degenerate (chosen == rejected) pairs found.\n")


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - main: reads as the EDA outline - schema and splits, a random sample preview,
#   per-split token lengths and the max_length trade-off, length bias, then
#   conversation structure and data quality - before dumping the whole report
#   to a text file.
# - Both splits, different decisions: the train split fixes the training
#   configuration (its cap column is what RewardTrainer's filter keeps), while
#   the test split is the acceptance-gate population (the gate filters it on
#   the same max_length), so the combined cap table shows both retentions side
#   by side. The deeper checks (bias, structure, quality) stay train-only:
#   they inform training choices, and duplicating them for the test split
#   would add numbers with no decision attached.
# - Shared rendering: schema/splits tables, the sample preview, the percentile
#   table, the cap trade-off, and the file dump come from eda_utils, so all three
#   stage EDAs (SFT, RM, PPO) print in one consistent format.
# - Faithful lengths: token_lengths reuses RMTrainingConfig and the training
#   tokeniser and appends EOS exactly as TRL does, so the lengths analysed are
#   the lengths RewardTrainer will filter on; the binding length per pair is the
#   longer of the two sides, since a pair survives only if both sides fit.
# - Length bias: the chosen-vs-rejected comparison is the highest-value check
#   here - a systematic length gap lets the RM score length instead of quality
#   and previews length-based reward hacking downstream.
# - Structure and quality: turns are counted from the dialogue markers, the
#   shared prompt prefix is the common character prefix of the two sides, and
#   identical/empty/duplicate pairs are flagged as zero-signal data.
# - Design choice: text-level statistics run in plain Python over the loaded
#   lists (cheap even on the full split) while only the tokenisation uses a
#   parallel map, keeping the slow path on the work that actually needs it.
# =============================================================================
