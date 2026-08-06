"""
Checkpoint Sweep for a Completed PPO Run
=====================================================================
Evaluates every saved checkpoint of a PPO run against the run's held-out
eval prompts and tabulates, per checkpoint, the reward-model score alongside
mechanical degeneracy statistics that the reward model cannot fake. The
purpose is monitoring, not selection: it answers 'did the run decline, and
when?' after the fact. Selecting a checkpoint purely by held-out RM score
would select for reward hacking, since the RM is the metric being optimised
(see reports/report_ppo_rlhf_loop.md, Section 6.2).

The mechanical statistics target the failure signatures documented for this
pipeline: repetition (degenerate loops), list-marker frequency (the RM's
format-gaming blind spot), response length drift, and empty completions.

Inputs
------
--label LABEL : the PPO run to sweep. Resolves to
    results/ppo_rlhf_loop/checkpoints_<label>/checkpoint-* plus the final
    adapter_<label>, all produced by ppo_rlhf_loop.py's train().
--num-prompts N : how many eval prompts per checkpoint (default 20).
--samples-per-prompt K : completions drawn per prompt per checkpoint
    (default 1). At temperature 0.7 a single draw is noisy, so per-prompt
    scores are means over K samples when K > 1, which tightens the
    checkpoint-to-checkpoint comparisons the prompt count alone cannot
    resolve (the held-out split caps the prompt count at eval_examples).

Outputs
-------
results/ppo_rlhf_loop/checkpoint_sweep_<label>_n<prompts>_k<samples>.md -- one
table row per checkpoint. To read the actual completions for a suspicious
checkpoint, run generate_ppo_completions.py with --adapter-path pointing at it.
results/ppo_rlhf_loop/checkpoint_sweep_<label>_n<prompts>_k<samples>.json --
the same sweep with every per-prompt and per-sample score, so a later reader
can compute confidence intervals rather than eyeball the table.

Two sweeps of one run at different prompt or sample counts are not comparable,
so the settings are in the filename as well as in both file headers, and
neither can overwrite the other.

Public API
----------
list_checkpoints(label)           -- ordered (step, path) pairs to sweep.
evaluate_checkpoint(...)          -- one checkpoint's row of statistics.
save_sweep(rows, label, config)   -- write the markdown table.
save_sweep_json(...)              -- write the per-prompt JSON record.
"""

# stdlib
import argparse
import gc
import json
import os
import re
import statistics
import sys

# third-party
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# local
from ..pipeline.ppo_rlhf_loop import RESULT_PATH, PPORunConfig
from .generate_ppo_completions import (
    _generate,
    _pick_device,
    _score,
    load_policy,
    load_reward_model,
    load_run_config,
    select_eval_prompts,
)

console = Console()

LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#{1,6}\s)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Sweep every checkpoint of a completed PPO run and tabulate the results."""
    args = parse_args()
    config = load_run_config(args.label)
    checkpoints = list_checkpoints(args.label)
    console.print(f"Sweeping {len(checkpoints)} checkpoints of run [bold]{args.label}[/bold]")

    device = _pick_device()

    # The reference column is constant across checkpoints, so it is generated
    # once from the final policy with its adapters disabled and reused as the
    # comparison baseline for every row.
    final_policy, tokenizer = load_policy(checkpoints[-1][1], device)
    reward_model = load_reward_model(config, tokenizer, device)
    prompts = select_eval_prompts(config, tokenizer, args.num_prompts)

    rows = []
    # One policy per bar step, the reference included, since it costs a full
    # generation pass like any checkpoint. Rows are printed through the
    # progress console so they scroll above the bar rather than corrupting it.
    with _progress() as progress:
        overall = progress.add_task(
            "Scoring the SFT reference", total=len(checkpoints) + 1
        )
        with final_policy.disable_adapter():
            ref_scores, _, ref_samples = _sample_mean_scores(
                final_policy, reward_model, tokenizer, prompts, config, device,
                args.samples_per_prompt, progress,
            )
        _free(final_policy)
        progress.advance(overall)

        for step, path in checkpoints:
            progress.update(overall, description=f"Checkpoint {step}")
            row = evaluate_checkpoint(
                step, path, reward_model, tokenizer, prompts, ref_scores, config, device,
                args.samples_per_prompt, progress,
            )
            rows.append(row)
            progress.advance(overall)
            progress.console.print(
                f"  step {row['step']:>5}: RM {row['mean_score']:+.3f} "
                f"(win {row['win_rate']:.0%}) rep {row['repetition']:.3f} "
                f"list {row['list_fraction']:.0%} len {row['mean_words']:.0f}w"
            )
        progress.update(overall, description="Sweep complete")

    output_path = save_sweep(
        rows, args.label, config, len(prompts), statistics.mean(ref_scores),
        args.samples_per_prompt,
    )
    json_path = save_sweep_json(
        rows, args.label, config, len(prompts), ref_scores, ref_samples,
        args.samples_per_prompt,
    )
    _print_table(rows)
    console.print(f"Wrote the sweep table to [bold]{output_path}[/bold]")
    console.print(f"Wrote the per-prompt record to [bold]{json_path}[/bold]")


# ---------------------------------------------------------------------------
# Argument parsing and checkpoint discovery
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (plain argparse; a one-off diagnostic, no config)."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", required=True,
        help="PPO run label; sweeps results/ppo_rlhf_loop/checkpoints_<label>/*.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=20,
        help="Number of eval prompts per checkpoint (default 20).",
    )
    parser.add_argument(
        "--samples-per-prompt", type=int, default=1,
        help="Completions drawn per prompt per checkpoint (default 1). "
             "Per-prompt scores are means over the samples.",
    )
    return parser.parse_args(argv)


def list_checkpoints(label: str) -> list[tuple[int, str]]:
    """Return (step, adapter_path) pairs to sweep, in step order.

    Includes every checkpoints_<label>/checkpoint-<step> directory that holds
    a LoRA adapter, plus the final adapter_<label> (ordered last, one step
    after the highest checkpoint, so the table reads chronologically even
    though it duplicates the final checkpoint's weights).

    Raises:
        FileNotFoundError: If the run has no checkpoints to sweep.
    """
    checkpoint_root = f"{RESULT_PATH}/checkpoints_{label}"
    pairs: list[tuple[int, str]] = []
    if os.path.isdir(checkpoint_root):
        for name in os.listdir(checkpoint_root):
            match = re.fullmatch(r"checkpoint-(\d+)", name)
            path = os.path.join(checkpoint_root, name)
            if match and os.path.isfile(os.path.join(path, "adapter_config.json")):
                pairs.append((int(match.group(1)), path))
    pairs.sort()

    final_path = f"{RESULT_PATH}/adapter_{label}"
    if os.path.isfile(os.path.join(final_path, "adapter_config.json")):
        final_step = (pairs[-1][0] + 1) if pairs else 0
        pairs.append((final_step, final_path))

    if not pairs:
        raise FileNotFoundError(
            f"No checkpoints or final adapter found for run {label} under "
            f"{RESULT_PATH}. Run the PPO stage first (`uv run rlhf-ppo`)."
        )
    return pairs


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def _sample_mean_scores(
    model:        torch.nn.Module,
    reward_model: torch.nn.Module,
    tokenizer,
    prompts:      list[str],
    config:       PPORunConfig,
    device:       str,
    num_samples:  int,
    progress:     "Progress | None" = None,
) -> tuple[list[float], list[str], list[list[float]]]:
    """Draw num_samples scored completions per prompt and average the scores.

    The generator is reseeded once per policy, so every checkpoint (and the
    reference) consumes the identical sampling stream, while successive
    passes within a policy remain independent draws.

    Returns:
        Per-prompt mean scores, every generated completion (num_samples
        passes concatenated) for the mechanical statistics, and the raw
        per-sample scores per prompt, which the JSON artefact records so a
        later reader can recompute spread rather than trust the mean.
    """
    torch.manual_seed(config.seed)
    per_sample: list[list[float]] = [[] for _ in prompts]
    all_completions: list[str] = []
    # One inner bar counting prompts generated across every pass, rather than
    # one tick per pass. Generation dominates the runtime, and at 100 prompts
    # a pass-level bar would sit still for minutes at a time.
    prompt_task = (
        progress.add_task("  prompts", total=num_samples * len(prompts))
        if progress is not None else None
    )
    on_batch = (
        (lambda size: progress.advance(prompt_task, size))
        if prompt_task is not None else None
    )
    for pass_index in range(num_samples):
        if prompt_task is not None:
            progress.update(
                prompt_task, description=f"  prompts (pass {pass_index + 1}/{num_samples})"
            )
        completions = _generate(model, tokenizer, prompts, config, device, on_batch)
        scores = _score(reward_model, tokenizer, prompts, completions, device)
        all_completions.extend(completions)
        for index, score in enumerate(scores):
            per_sample[index].append(score)
    if prompt_task is not None:
        progress.remove_task(prompt_task)
    means = [statistics.mean(samples) for samples in per_sample]
    return means, all_completions, per_sample


def evaluate_checkpoint(
    step:            int,
    adapter_path:    str,
    reward_model:    torch.nn.Module,
    tokenizer,
    prompts:         list[str],
    ref_scores:      list[float],
    config:          PPORunConfig,
    device:          str,
    num_samples:     int = 1,
    progress:        "Progress | None" = None,
) -> dict:
    """Generate, score, and summarise one checkpoint's held-out completions.

    Sampling is reseeded identically for every checkpoint, so differences
    between rows reflect the weights, not the draw. Per-prompt scores are
    means over num_samples draws, compared against the reference's
    per-prompt means. The policy is loaded and freed per call to keep one
    policy in memory at a time.

    Returns:
        A row dict: step, label, mean_score, win_rate, repetition,
        list_fraction, mean_words, empty_fraction, plus a per_prompt list
        carrying each prompt's mean score, its per-sample scores, the
        reference score it was compared against, and whether it counted as
        a win. The aggregates alone cannot support a confidence interval,
        which is what left an earlier single-sample sweep impossible to
        reconcile with its successor.
    """
    policy, _ = load_policy(adapter_path, device)
    scores, completions, per_sample = _sample_mean_scores(
        policy, reward_model, tokenizer, prompts, config, device, num_samples, progress
    )
    _free(policy)

    return {
        "step": step,
        "label": "final" if os.path.basename(adapter_path).startswith("adapter_") else str(step),
        "mean_score": statistics.mean(scores),
        "win_rate": sum(s > r for s, r in zip(scores, ref_scores)) / len(scores),
        "repetition": statistics.mean(_repetition_fraction(c) for c in completions),
        "list_fraction": sum(bool(LIST_MARKER_PATTERN.search(c)) for c in completions) / len(completions),
        "mean_words": statistics.mean(len(c.split()) for c in completions),
        "empty_fraction": sum(not c.strip() for c in completions) / len(completions),
        "per_prompt": [
            {
                "index":           index,
                "mean_score":      score,
                "sample_scores":   samples,
                "reference_score": reference,
                "win":             score > reference,
            }
            for index, (score, samples, reference)
            in enumerate(zip(scores, per_sample, ref_scores))
        ],
    }


def _repetition_fraction(text: str) -> float:
    """Fraction of duplicated word 4-grams, i.e. 0 for no repetition.

    A degenerate loop (the same phrase repeated to the token budget) drives
    this towards 1, while normal prose stays near 0. Texts shorter than four
    words score 0.
    """
    words = text.split()
    if len(words) < 4:
        return 0.0
    ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _free(model: torch.nn.Module) -> None:
    """Drop a model and release cached device memory before the next load."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _artefact_stem(label: str, num_prompts: int, num_samples: int) -> str:
    """Filename stem keyed by the run AND the evaluation settings.

    Two sweeps of one run at different prompt counts or sample counts are not
    comparable, so they must not share a path. Keying on the label alone once
    let a 4-sample sweep overwrite the single-sample sweep whose numbers were
    still being quoted in the report, leaving the discrepancy undiagnosable.
    """
    return f"checkpoint_sweep_{label}_n{num_prompts}_k{num_samples}"


def save_sweep(
    rows: list[dict], label: str, config: PPORunConfig, num_prompts: int,
    ref_mean: float, num_samples: int = 1,
) -> str:
    """Write the markdown sweep table. Returns the path."""
    # Always state the sample count, including when it is 1. Two sweeps of the
    # same run at different sample counts are not comparable, and an artefact
    # that omits the count cannot be told apart from one that used another.
    sampling = (
        f"{num_samples} sample per prompt, " if num_samples == 1 else
        f"{num_samples} samples per prompt, per-prompt scores are means over samples, "
    )
    lines = [
        f"# PPO checkpoint sweep: run `{label}`",
        "",
        f"Generated by `src/diagnostics/sweep_ppo_checkpoints.py` over "
        f"{num_prompts} held-out eval prompts per checkpoint ({sampling}seed "
        f"{config.seed}, temperature {config.temperature}, up to "
        f"{config.response_length} new tokens). The reference column baseline "
        f"is the SFT policy (adapters disabled), mean RM score "
        f"{ref_mean:.3f}. Win rate is the fraction of prompts where the "
        f"checkpoint outscores the reference.",
        "",
        "This table is for monitoring, not selection: choosing the "
        "checkpoint with the highest RM score would select for reward "
        "hacking, since the RM is the metric PPO optimises. The mechanical "
        "columns (repetition, list fraction, length, empty) catch the "
        "degeneracy signatures the RM cannot judge. To read a suspicious "
        "checkpoint's completions, run `generate_ppo_completions.py "
        "--adapter-path <checkpoint directory>`.",
        "",
        "| Step | Mean RM score | Win rate vs SFT | Repetition | List fraction | Mean words | Empty |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['mean_score']:+.3f} | {row['win_rate']:.0%} "
            f"| {row['repetition']:.3f} | {row['list_fraction']:.0%} "
            f"| {row['mean_words']:.0f} | {row['empty_fraction']:.0%} |"
        )
    lines.append("")

    output_path = f"{RESULT_PATH}/{_artefact_stem(label, num_prompts, num_samples)}.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def save_sweep_json(
    rows: list[dict], label: str, config: PPORunConfig, num_prompts: int,
    ref_scores: list[float], ref_samples: list[list[float]], num_samples: int = 1,
) -> str:
    """Write the per-prompt sweep record as JSON. Returns the path.

    The markdown table reports aggregates only, which is enough to read a
    trend but not enough to test one: a difference of two win rates cannot
    be given a confidence interval from the rates alone. This file records
    every per-prompt and per-sample score, plus the settings that make one
    sweep comparable with another.
    """
    payload = {
        "label": label,
        "generated_by": "src/diagnostics/sweep_ppo_checkpoints.py",
        "settings": {
            "num_prompts":        num_prompts,
            "samples_per_prompt": num_samples,
            "seed":               config.seed,
            "temperature":        config.temperature,
            "response_length":    config.response_length,
            "reference":          "SFT policy (adapters disabled)",
        },
        "reference": {
            "mean_score": statistics.mean(ref_scores),
            "per_prompt": [
                {"index": index, "mean_score": score, "sample_scores": samples}
                for index, (score, samples) in enumerate(zip(ref_scores, ref_samples))
            ],
        },
        "checkpoints": rows,
    }
    output_path = f"{RESULT_PATH}/{_artefact_stem(label, num_prompts, num_samples)}.json"
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return output_path


def _progress() -> Progress:
    """Two-level progress display for a sweep.

    The outer task counts policies scored (every checkpoint plus the shared
    SFT reference), the inner one the sampling passes within the policy in
    hand. A sweep is long enough, minutes per checkpoint at 100 prompts, that
    a bare spinner cannot answer 'how much is left'; the remaining-time
    estimate is the point of the display.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _print_table(rows: list[dict]) -> None:
    """Render the sweep as a rich table on the console."""
    table = Table(title="PPO checkpoint sweep (held-out prompts)")
    for column in ("Step", "Mean RM score", "Win rate", "Repetition", "List", "Words", "Empty"):
        table.add_column(column, justify="right")
    for row in rows:
        table.add_row(
            row["label"],
            f"{row['mean_score']:+.3f}",
            f"{row['win_rate']:.0%}",
            f"{row['repetition']:.3f}",
            f"{row['list_fraction']:.0%}",
            f"{row['mean_words']:.0f}",
            f"{row['empty_fraction']:.0%}",
        )
    console.print(table)


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - Scope: a one-off diagnostic utility, like its siblings in this package.
#   It reuses generate_ppo_completions.py's loading, prompt-selection,
#   generation, and scoring helpers (imported, not reimplemented), so the
#   sweep and the single-checkpoint probe can never drift apart.
# - Monitoring, not selection: the header of the output file restates why
#   the best row by RM score must not be promoted (the RM is the optimised,
#   gameable metric). The sweep exists to answer 'did the run decline, and
#   when?', with the mechanical columns as the RM-independent evidence.
# - One policy in memory at a time: checkpoints are loaded and freed
#   sequentially (the RM stays resident), keeping the footprint near the
#   single-probe case even for runs with many checkpoints.
# - Constant baseline: the reference completions are generated once from the
#   final policy with adapters disabled, which is the same SFT policy every
#   checkpoint trained against, so win rates are comparable across rows.
# =============================================================================
