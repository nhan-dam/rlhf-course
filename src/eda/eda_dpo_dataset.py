"""
Exploratory Data Analysis - DPO Preference Dataset
==================================================
Inspect the preference pairs the DPO stage trains on and is judged on before
running it, so the configuration rests on the data. DPO consumes
Anthropic/hh-rlhf whole: the shared prompt (implicit in the two dialogues) and
both full sides, so THREE lengths matter, each with filter semantics -- the
prompt against max_prompt_tokens (PPO-cap parity) and each side with EOS
against max_pair_tokens (RM-cap parity).

Both splits are analysed on every run, as the RM stage's EDA does. The train
split drives the training configuration and gets the full treatment,
including a random preview. The test split is the evaluation population (the
in-training eval subsample, the post-training gate, and the PPO-vs-DPO
comparison prompts), so it gets every aggregate check -- lengths, retention,
length bias, quality -- but no preview, since reading evaluation examples is
what a pre-registered protocol should not do. A cross-split overlap check
closes the leakage question.

Inputs
------
Command-line flags (see parse_args): how many training pairs to sample,
candidate caps for both filters, preview size, and the output path. Defaults
are pulled from DPOTrainingConfig so tokenisation matches training.

Outputs
-------
A printed report (schema, splits, train preview, two pairs with a shared
response opening, per-split length distributions, per-cap and compound
retention for both splits, per-split length bias and data quality, and
train-test overlap), also written verbatim to
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
from rich.panel import Panel
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
    probe = load_dataset(config.dataset_name, split="train[:1]")[0]
    report_schema(console, config.dataset_name, probe)
    console.print(
        "DPO consumes both columns whole: the pipeline splits off the shared prompt "
        "at the final Assistant marker and DPOTrainer appends EOS to both sides, so "
        "the units analysed below are preference pairs, with three lengths each "
        "(prompt, chosen, rejected).\n"
    )
    report_splits(console, config.dataset_name, {
        "train": "DPO training pairs (after the two-cap filter)",
        "test":  f"in-training eval ({config.eval_examples:,}-pair subsample) + "
                 "post-training gate + PPO-vs-DPO comparison prompts",
    })

    # Load both splits. Train may be subsampled for speed; test is the
    # evaluation population and is always analysed in full.
    train_split = _load_split(config, "train", args.sample, console)
    test_split  = _load_split(config, "test", 0, console)

    # 3. Qualitative look at the pairs (train only: the test split is the
    #    evaluation population, and its examples are deliberately not read).
    preview_samples(console, train_split, ["chosen", "rejected"],
                    args.num_samples, args.sample_chars, config.seed)
    _report_shared_openings(console, train_split, config.seed, n=2)

    # 4. The three length distributions per split, tokenised as training will see them.
    lengths = {}
    shares = {}
    for name, split in (("train", train_split), ("test", test_split)):
        prompt_len, chosen_len, rejected_len = pair_token_lengths(split, tokenizer, args.num_proc)
        pair_len = np.maximum(chosen_len, rejected_len)   # the binding pair length
        lengths[name] = (prompt_len, chosen_len, rejected_len, pair_len)
        # Whether the rejected side begins with the chosen side's final-turn
        # prompt; the pipeline's split_prompt drops pairs where it does not.
        shares[name] = np.array([r.startswith(extract_prompt(c))
                                 for c, r in zip(split["chosen"], split["rejected"])])
        length_percentiles(console, f"Token-length distributions, {name} split", {
            "prompt": prompt_len,
            "chosen (+EOS)": chosen_len,
            "rejected (+EOS)": rejected_len,
            "pair max": pair_len,
        })

    # 5. Per-cap trade-offs on both splits, then the compound filter both caps
    #    apply together. The train column sets the caps; the test column sizes
    #    the gate population under the same caps.
    console.print("[bold]Prompt cap[/bold] (parity with the PPO prompt filter):")
    cap_tradeoff(console, {name: v[0] for name, v in lengths.items()},
                 prompt_candidates, config.max_prompt_tokens, semantics="filter")
    console.print("[bold]Pair cap[/bold] on max(chosen, rejected), both with EOS "
                  "(parity with the RM filter):")
    cap_tradeoff(console, {name: v[3] for name, v in lengths.items()},
                 pair_candidates, config.max_pair_tokens, semantics="filter")
    _report_compound_filter(console, {name: (v[0], v[3], shares[name]) for name, v in lengths.items()},
                            config.max_prompt_tokens, config.max_pair_tokens)

    # 6. DPO-specific bias check: chosen-vs-rejected length asymmetry, per split.
    _report_length_bias(console, {name: (v[1], v[2]) for name, v in lengths.items()})

    # 7. Data-quality checks, per split.
    _report_quality(console, {"train": train_split, "test": test_split})

    # 8. Leakage: exact overlap between the splits.
    _report_overlap(console, train_split, test_split)

    dump(console, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0,
                        help="Training pairs to sample; 0 = whole split (default: 0). "
                             "Pass e.g. --sample 20000 for a faster partial run. The test "
                             "split is always analysed in full.")
    parser.add_argument("--prompt-candidates", default="128,192,256,320,384",
                        help="Comma-separated max_prompt_tokens values to evaluate.")
    parser.add_argument("--pair-candidates", default="384,448,512,640,768",
                        help="Comma-separated max_pair_tokens values to evaluate.")
    parser.add_argument("--num-samples", type=int, default=6,
                        help="Random training pairs to print (default: 6).")
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
    the pipeline's split_prompt hands DPOTrainer), and EOS is appended to both
    sides before measuring, exactly as DPOTrainer's add_eos map and the
    training-time filter_pairs do -- so every length here matches the training
    decision within a token.
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


def _report_shared_openings(console, dataset, seed: int, n: int = 2, width: int = 120) -> None:
    """Show pairs whose two responses share leading characters past the marker.

    HH-RLHF responses are conversational openers, so the two sides of a pair
    often begin with the same characters ('Yes, ...' against 'Yeah, ...'). A
    prompt extracted as the longest common character prefix (TRL's default
    when no 'prompt' column is supplied) would then end inside a word, which is
    why the pipeline splits at the marker instead. Two real examples are
    printed so the report can illustrate the case; the count is not the point.
    """
    chosen = dataset["chosen"]
    rejected = dataset["rejected"]
    order = np.random.default_rng(seed).permutation(len(chosen))
    shown = 0
    console.print(
        f"[bold]Shared response openings[/bold] ({n} examples): pairs whose responses "
        "share leading characters, where a common-character-prefix prompt would end "
        "inside a word.\n"
    )
    for index in order:
        prompt = extract_prompt(chosen[int(index)])
        if not rejected[int(index)].startswith(prompt):
            continue
        c = chosen[int(index)][len(prompt):]
        r = rejected[int(index)][len(prompt):]
        k = 0
        while k < min(len(c), len(r)) and c[k] == r[k]:
            k += 1
        # Only the mid-word case matters: at least two shared characters, the
        # last shared one alphanumeric, and the first differing character
        # alphanumeric on at least one side. Divergence at punctuation or a
        # space is a pre-tokenisation boundary and tokenises consistently.
        inside_word = (k >= 2 and c[k - 1].isalnum()
                       and ((k < len(c) and c[k].isalnum()) or (k < len(r) and r[k].isalnum())))
        if not inside_word:
            continue
        body = (f"[bold]shared opening[/bold] {c[:k]!r}\n\n"
                f"[bold]chosen[/bold]   {c[:width].strip()!r}\n"
                f"[bold]rejected[/bold] {r[:width].strip()!r}")
        console.print(Panel(body, title=f"example #{int(index)}", border_style="dim"))
        shown += 1
        if shown == n:
            break
    if shown == 0:
        console.print("[green]None found[/green] in this split.")
    console.print()


def _report_compound_filter(
    console, splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], prompt_cap: int, pair_cap: int,
) -> None:
    """Decompose the joint retention of the two caps at their current values, per split.

    The two caps overlap (a long pair usually has a long prompt), so the
    compound retention is not the product of the marginals; this table shows
    what each cap uniquely costs, which is the number to weigh when
    considering moving either cap alone. A final row applies the pipeline's
    one non-length exclusion on top of the caps: pairs whose sides diverge
    before the final assistant turn (two conversations, not two responses to
    one prompt) are dropped by split_prompt, so that row is the training set
    in the train column and the gate population in the test column.
    """
    outcomes = {}
    for name, (prompt_len, pair_len, shares_prompt) in splits.items():
        prompt_ok = prompt_len <= prompt_cap
        pair_ok = pair_len <= pair_cap
        outcomes[name] = {
            "kept (both caps pass)":      prompt_ok & pair_ok,
            "dropped by prompt cap only": ~prompt_ok & pair_ok,
            "dropped by pair cap only":   prompt_ok & ~pair_ok,
            "dropped by both":            ~prompt_ok & ~pair_ok,
            "kept, sides share the final-turn prompt (trainer's set)":
                prompt_ok & pair_ok & shares_prompt,
        }

    table = Table(title=f"Compound filter at the current caps "
                        f"(prompt<={prompt_cap}, pair<={pair_cap})")
    table.add_column("outcome")
    for name in outcomes:
        table.add_column(f"{name} pairs", justify="right")
        table.add_column(f"{name} %", justify="right", style="bold")
    for outcome in next(iter(outcomes.values())):
        row = [outcome]
        for name in outcomes:
            mask = outcomes[name][outcome]
            row += [f"{int(mask.sum()):,}", f"{100 * mask.mean():.2f}%"]
        table.add_row(*row)
    console.print(table)

    train_prompt_len, train_pair_len, _ = splits["train"]
    pair_cap_only = 100 * outcomes["train"]["dropped by pair cap only"].mean()
    console.print(
        f"The train 'kept' row is what DPO trains on; PPO kept "
        f"{100 * np.mean(train_prompt_len <= prompt_cap):.2f}% under its prompt cap alone, "
        f"so the pair cap uniquely costs {pair_cap_only:.2f} percentage points of "
        f"{len(train_prompt_len):,} training pairs -- the price of RM parity. "
        f"The last row is what the trainer sees: its train count is the "
        f"training set and its test count is the post-training gate population.\n"
    )


def _report_length_bias(console, splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Report chosen-vs-rejected length asymmetry, per split.

    DPO's implicit reward is a sum of per-token log-probability ratios, so a
    systematic length gap between the two sides lets the policy buy margin
    with verbosity instead of quality -- the DPO analogue of the RM's
    length-bias trap. Train is the shortcut the policy could learn; test is
    the population the gate scores, so a bias there would let a length-driven
    implicit reward pass the gate for the wrong reason.
    """
    diffs = {name: chosen - rejected for name, (chosen, rejected) in splits.items()}
    table = Table(title="Chosen-vs-rejected length bias")
    table.add_column("statistic")
    for name in diffs:
        table.add_column(name, justify="right", style="bold")
    rows = [
        ("chosen longer",                    lambda d: f"{100 * np.mean(d > 0):.1f}%"),
        ("rejected longer",                  lambda d: f"{100 * np.mean(d < 0):.1f}%"),
        ("equal length",                     lambda d: f"{100 * np.mean(d == 0):.1f}%"),
        ("mean(chosen - rejected) tokens",   lambda d: f"{d.mean():+.1f}"),
        ("median(chosen - rejected) tokens", lambda d: f"{np.median(d):+.0f}"),
    ]
    for label, fn in rows:
        table.add_row(label, *[fn(d) for d in diffs.values()])
    console.print(table)
    for name, diff in diffs.items():
        if abs(np.mean(diff > 0) - np.mean(diff < 0)) < 0.10:
            console.print(f"[green]{name}: near-symmetric[/green], length is weakly "
                          "informative of preference.")
        else:
            longer = "chosen" if np.mean(diff > 0) > np.mean(diff < 0) else "rejected"
            console.print(f"[yellow]{name}: asymmetric[/yellow], '{longer}' is longer in a "
                          "clear majority of pairs -- watch generated response lengths.")
    console.print()


def _report_quality(console, splits: dict) -> None:
    """Report empty sides, duplicate pairs, and degenerate identical pairs, per split."""
    counts = {}
    for name, dataset in splits.items():
        chosen = dataset["chosen"]
        rejected = dataset["rejected"]
        n = len(chosen)
        counts[name] = (n, {
            "empty / whitespace side":      sum(1 for c, r in zip(chosen, rejected) if not c.strip() or not r.strip()),
            "identical chosen == rejected": sum(1 for c, r in zip(chosen, rejected) if c == r),
            "duplicate pairs":              n - len(set(zip(chosen, rejected))),
        })

    table = Table(title="Data quality checks")
    table.add_column("check")
    for name in counts:
        table.add_column(f"{name} count", justify="right")
        table.add_column(f"{name} %", justify="right", style="bold")
    for check in next(iter(counts.values()))[1]:
        row = [check]
        for name, (n, c) in counts.items():
            row += [f"{c[check]:,}", f"{100 * c[check] / n:.2f}%"]
        table.add_row(*row)
    console.print(table)
    console.print(
        "Identical pairs carry no preference signal (the DPO loss sees a zero "
        "margin by construction) and exact duplicates re-weight their pair; both "
        "are defects here, unlike the expected prompt reuse across different "
        "pairs, which is the dataset's design. In the test split an identical "
        "pair can only tie, and the gate counts a tie as a failure.\n"
    )


def _report_overlap(console, train_split, test_split) -> None:
    """Report exact train-test overlap: shared prompts, shared responses, shared pairs.

    The test split is the only population no stage of either pipeline trains
    on, and the comparison protocol rests on that. Exact matches are the
    leakage a reader will ask about first; near-duplicates are out of scope.
    """
    train_prompts = {extract_prompt(t) for t in train_split["chosen"]}
    train_texts = set(train_split["chosen"]) | set(train_split["rejected"])
    train_pairs = set(zip(train_split["chosen"], train_split["rejected"]))

    test_chosen = test_split["chosen"]
    test_rejected = test_split["rejected"]
    n = len(test_chosen)
    n_prompt = sum(1 for t in test_chosen if extract_prompt(t) in train_prompts)
    n_text = sum(1 for c, r in zip(test_chosen, test_rejected) if c in train_texts or r in train_texts)
    n_pair = sum(1 for p in zip(test_chosen, test_rejected) if p in train_pairs)

    table = Table(title="Train-test overlap (exact match, as a fraction of the test split)")
    table.add_column("overlap")
    table.add_column("test pairs", justify="right")
    table.add_column("% of test", justify="right", style="bold")
    table.add_row("prompt also in train", f"{n_prompt:,}", f"{100 * n_prompt / n:.2f}%")
    table.add_row("either full dialogue also in train", f"{n_text:,}", f"{100 * n_text / n:.2f}%")
    table.add_row("whole pair also in train", f"{n_pair:,}", f"{100 * n_pair / n:.2f}%")
    console.print(table)
    if n_pair == 0 and n_text == 0:
        console.print("[green]No exact leakage[/green]: no test dialogue or pair appears in train. "
                      "Shared prompts, if any, are the dataset's prompt reuse, not leakage of a "
                      "judged response.\n")
    else:
        console.print("[yellow]Exact overlap found[/yellow]: the gate and comparison populations "
                      "contain material the policy trained on. Exclude it before scoring.\n")


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - main: reads as the EDA outline - schema and splits, a train preview plus
#   two shared-opening examples, per-split length distributions, the two
#   per-cap trade-offs plus their compound decomposition (both splits), the
#   length-bias and quality checks (both splits), and train-test overlap -
#   then dumps the report to a file.
# - Two splits, asymmetric treatment: train gets the preview, test does not.
#   The test split is the evaluation population, so its examples are not
#   read, but every aggregate is reported because two of them correct
#   numbers the report cites: the compound 'kept' row is the gate population,
#   and identical test pairs can only tie, which the gate counts as failures.
# - Three binding lengths: DPO filters on the prompt (PPO-cap parity) and on
#   max(chosen, rejected) with EOS (RM-cap parity), so both are measured with
#   the exact tokenisation training applies (extract_prompt + EOS append,
#   mirroring filter_pairs, split_prompt and DPOTrainer's add_eos map).
# - Shared openings: the two sides of an HH-RLHF pair often begin with the
#   same characters, which is why the pipeline supplies a marker-split
#   'prompt' column rather than letting DPOTrainer take the common character
#   prefix. Two real examples are printed for the report; no count.
# - Compound decomposition: the caps overlap, so the joint retention is shown
#   split into 'prompt cap only', 'pair cap only', and 'both', making the
#   unique cost of each cap visible before either is moved.
# - Overlap: exact prompt, dialogue, and pair matches between the splits, the
#   leakage question the comparison protocol depends on. Prompt reuse across
#   splits is the dataset's design and is reported but not flagged.
# - Shared rendering: tables, preview, percentiles, cap trade-off, and the
#   file dump come from eda_utils, matching the other stages' EDA output.
# =============================================================================
