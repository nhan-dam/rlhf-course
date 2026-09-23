"""
Generation-Health Probe for a Trained DPO Policy
=====================================================================
Generates completions from a trained DPO adapter and the reference (SFT)
policy on the same held-out prompts, and reports the diagnostics that say
whether likelihood displacement has damaged generation.

Why this is not compare_policies.py
-----------------------------------
compare_policies.py runs the pre-registered four-policy comparison (DPO
report, Section 6), whose protocol fixes the arm, the judges, and the
tie-breaker in advance. Two pieces of that protocol are still undecided, so
running it now would expose its aggregate statistics before the protocol
that reads them exists. This script therefore computes NO comparison
statistic: no reward-model score, no win rate, no cross-policy aggregate.
It answers one question about one policy, i.e. does it still generate well,
and it is deliberately incapable of answering any other.

The question it answers
-----------------------
The gate for run 75047d16 showed the implicit reward negative on BOTH sides,
i.e. the policy sat below the reference on chosen responses as well as
rejected ones, which is likelihood displacement. That is a statement about
log-probabilities on a fixed corpus, not about generated text, and the two
can diverge. Degenerate generation shows up as short or never-terminating
responses, collapsing diversity, or a policy that has moved its mass away
from anything the reference would produce. All three are measured here.

Inputs
------
--label LABEL : the DPO run; resolves to results/dpo_lora_hh/adapter_<label>
    and config_<label>.json. Beta, dataset, and prompt cap are read from the
    saved config, so the probe cannot drift from the run it probes.
--num-prompts N : distinct held-out prompts (default 20).
--samples-per-prompt K : draws per prompt per policy (default 1).
--temperature T, --response-length R, --seed S : the sampling regime
    (defaults 0.7 / 128 / the run's own seed). DPO does not generate at
    training time, so it has no regime of its own to inherit; these defaults
    match the PPO run's so the output reads alongside completions_<ppo>.md.
--skip-reverse : omit the log p_dpo pass over the reference's completions,
    which costs one extra model load.

Outputs
-------
results/dpo_lora_hh/generation_probe_<label>_n<N>_k<K>_t<T>_r<R>.md and .json
-- a per-prompt section with both policies' text, and a summary table of
mean response length, EOS rate, distinct-4 ratio, mean per-token
log-probability, and the implicit reward each policy's own text receives.

Public API
----------
GenerationSettings          -- the sampling regime passed to _generate.
select_probe_prompts(...)   -- held-out prompts, via compare_policies.
probe(...)                  -- generate and diagnose; returns records.
save_probe(records, ...)    -- write the markdown and JSON artefacts.
"""

# stdlib
import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

# third-party
import torch
from rich.console import Console
from transformers import PreTrainedTokenizer

# local
from ..common.model_utils import resolve_model_path
from ..pipeline.dpo_lora_hh import RESULT_PATH as DPO_RESULT_PATH
from ..pipeline.dpo_lora_hh import parse_config as parse_dpo_config
from ..pipeline.ppo_rlhf_loop import _load_tokenizer
from .compare_policies import (
    _free,
    _generate,
    _load_adapter,
    _load_causal,
    _load_run_config,
    _logprob_pass,
    _pick_device,
    select_test_prompts,
)

console = Console()

# Length of the n-gram used for the repetition statistic. Four is the usual
# choice for degeneracy checks: long enough that ordinary English repetition
# ('one of the') does not dominate, short enough to catch a looping phrase
# within a 128-token budget.
REPETITION_N = 4


# ---------------------------------------------------------------------------
# Sampling regime
# ---------------------------------------------------------------------------

@dataclass
class GenerationSettings:
    """The three fields _generate reads off a run config.

    DPOTrainingConfig has none of them, because DPO never samples during
    training, so they are supplied explicitly rather than inherited. Keeping
    them in a dataclass rather than passing loose arguments means _generate
    is reused unmodified.
    """
    seed:            int
    temperature:     float
    response_length: int


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Probe one DPO adapter's generation health against the SFT reference."""
    args = parse_args()
    dpo_config = _load_run_config(DPO_RESULT_PATH, args.label, parse_dpo_config)
    settings = GenerationSettings(
        seed=dpo_config.seed if args.seed is None else args.seed,
        temperature=args.temperature,
        response_length=args.response_length,
    )

    device = _pick_device()
    sft_path = resolve_model_path(dpo_config.sft_model_path, "causal-lm")
    tokenizer = _load_tokenizer(sft_path)
    prompts = select_probe_prompts(dpo_config, tokenizer, args.num_prompts)
    expanded = [prompt for prompt in prompts for _ in range(args.samples_per_prompt)]

    console.print(
        f"Probing [bold]{args.label}[/bold] against the SFT reference on "
        f"{len(prompts)} prompts x {args.samples_per_prompt} samples on "
        f"[bold]{device}[/bold] (temperature {settings.temperature}, up to "
        f"{settings.response_length} new tokens, seed {settings.seed})"
    )

    records = probe(args, dpo_config, settings, sft_path, tokenizer, expanded, device)
    json_path, md_path = save_probe(records, prompts, args, dpo_config, settings)
    _print_summary(records)
    console.print(f"Wrote [bold]{json_path}[/bold] and [bold]{md_path}[/bold]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Plain argparse, as in the other diagnostics: this is a one-off utility
    with no config label of its own. Everything that describes the RUN comes
    from the saved config; everything that describes the PROBE is a flag, and
    every flag that makes two probes incomparable appears in the filename.
    """
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description="Generation-health probe for a trained DPO policy adapter."
    )
    parser.add_argument("--label", required=True,
                        help="DPO run label; resolves to results/dpo_lora_hh/adapter_<label>.")
    parser.add_argument("--num-prompts", type=int, default=20,
                        help="Distinct held-out prompts to complete (default 20).")
    parser.add_argument("--samples-per-prompt", type=int, default=1,
                        help="Draws per prompt per policy (default 1).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default 0.7, matching the PPO run).")
    parser.add_argument("--response-length", type=int, default=128,
                        help="Maximum new tokens (default 128, matching the PPO run).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Sampling seed; defaults to the DPO run's own seed.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Generation and scoring batch size (default 4).")
    parser.add_argument("--skip-reverse", action="store_true",
                        help="Omit the log p_dpo pass over the reference's completions.")
    return parser.parse_args(argv)


def select_probe_prompts(
    dpo_config, tokenizer: PreTrainedTokenizer, num_prompts: int,
) -> list[str]:
    """Held-out prompts, under the DPO run's own dataset and prompt cap.

    Delegates to compare_policies.select_test_prompts rather than
    reimplementing it, so the probe inherits the same exclusions the
    comparison uses: deduplication, and the removal of test prompts HH-RLHF
    reuses from the training split. Sharing the selection does not share any
    statistic, since this script computes none.
    """
    return select_test_prompts(
        dpo_config.dataset_name, tokenizer, dpo_config.max_prompt_tokens,
        num_prompts, dpo_config.seed,
    )


# ---------------------------------------------------------------------------
# Generation and diagnosis (one model loaded at a time)
# ---------------------------------------------------------------------------

def probe(
    args, dpo_config, settings: GenerationSettings, sft_path: str,
    tokenizer: PreTrainedTokenizer, prompts: list[str], device: str,
) -> dict[str, list[dict]]:
    """Generate from both policies and attach the displacement diagnostics.

    Three model loads, never two at once, so peak memory stays at one 0.5B
    model. The order is chosen so that each model is loaded once for
    generation and once for scoring the other policy's text:

      1. DPO   -- generate its completions, and score them under pi_dpo.
      2. SFT   -- generate its completions, score them under pi_ref, and
                  score the DPO completions under pi_ref.
      3. DPO   -- score the SFT completions under pi_dpo (unless skipped).

    Step 3 is what makes the displacement reading two-sided. Its absence
    would leave the probe unable to distinguish a policy that has merely
    sharpened from one that has moved its mass off the reference's outputs
    entirely, which is the failure mode worth catching.
    """
    adapter_path = f"{DPO_RESULT_PATH}/adapter_{args.label}"

    dpo = _load_adapter(adapter_path, tokenizer, device)
    dpo_text = _generate(dpo, tokenizer, prompts, settings, device, args.batch_size)
    logp_dpo_on_dpo = _logprob_pass(dpo, tokenizer, prompts, dpo_text, device,
                                    args.batch_size, desc="log p_dpo on 'dpo'")
    _free(dpo, device)

    sft = _load_causal(sft_path, tokenizer, device)
    sft_text = _generate(sft, tokenizer, prompts, settings, device, args.batch_size)
    logp_ref_on_sft = _logprob_pass(sft, tokenizer, prompts, sft_text, device,
                                    args.batch_size, desc="log p_ref on 'sft'")
    logp_ref_on_dpo = _logprob_pass(sft, tokenizer, prompts, dpo_text, device,
                                    args.batch_size, desc="log p_ref on 'dpo'")
    _free(sft, device)

    if args.skip_reverse:
        logp_dpo_on_sft = [None] * len(prompts)
    else:
        dpo = _load_adapter(adapter_path, tokenizer, device)
        logp_dpo_on_sft = _logprob_pass(dpo, tokenizer, prompts, sft_text, device,
                                        args.batch_size, desc="log p_dpo on 'sft'")
        _free(dpo, device)

    beta = dpo_config.beta
    return {
        "dpo": _diagnose(dpo_text, logp_dpo_on_dpo, logp_dpo_on_dpo, logp_ref_on_dpo, beta),
        "sft": _diagnose(sft_text, logp_ref_on_sft, logp_dpo_on_sft, logp_ref_on_sft, beta),
    }


def _diagnose(
    completions: list[dict], own_logp: list[float],
    logp_dpo: list[float | None], logp_ref: list[float], beta: float,
) -> list[dict]:
    """Attach per-completion health and displacement figures.

    'own_logp' is the summed log-probability under the policy that produced
    the text, so its per-token mean is that policy's confidence in its own
    output; a collapse here accompanies degenerate sampling. The implicit
    reward is the DPO objective's own quantity, beta * (log pi_dpo - log
    pi_ref), evaluated on generated text rather than on corpus pairs, and is
    None where the reverse pass was skipped.
    """
    out: list[dict] = []
    for i, record in enumerate(completions):
        n_tokens = max(record["n_tokens"], 1)
        implicit = (None if logp_dpo[i] is None
                    else beta * (logp_dpo[i] - logp_ref[i]))
        out.append({
            **record,
            "logp_own":          own_logp[i],
            "logp_per_token":    own_logp[i] / n_tokens,
            "implicit_reward":   implicit,
            "distinct_n":        _distinct_n(record["text"]),
        })
    return out


def _distinct_n(text: str, n: int = REPETITION_N) -> float:
    """Fraction of n-grams in the text that are unique.

    One is a text with no repeated n-gram, and a value falling towards zero
    is a looping generation. Texts shorter than one full n-gram return 1.0,
    since there is nothing to repeat and scoring them as degenerate would
    conflate brevity with looping; response length is reported separately.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def _stem(args, settings: GenerationSettings) -> str:
    """Artefact stem carrying every setting that makes two probes incomparable.

    The run label alone is not enough: the same adapter probed at a different
    temperature, sample count, or response budget produces different text,
    and a stem keyed only on the label would let one probe overwrite another.
    """
    return (f"generation_probe_{args.label}_n{args.num_prompts}"
            f"_k{args.samples_per_prompt}_t{settings.temperature}"
            f"_r{settings.response_length}")


def save_probe(
    records: dict[str, list[dict]], prompts: list[str],
    args, dpo_config, settings: GenerationSettings,
) -> tuple[str, str]:
    """Write the JSON record and the markdown review file. Returns both paths."""
    os.makedirs(DPO_RESULT_PATH, exist_ok=True)
    stem = _stem(args, settings)
    json_path = f"{DPO_RESULT_PATH}/{stem}.json"
    md_path = f"{DPO_RESULT_PATH}/{stem}.md"

    payload = {
        "label":             args.label,
        "beta":              dpo_config.beta,
        "loss_type":         dpo_config.loss_type,
        "dataset_name":      dpo_config.dataset_name,
        "max_prompt_tokens": dpo_config.max_prompt_tokens,
        "num_prompts":       args.num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "temperature":       settings.temperature,
        "response_length":   settings.response_length,
        "seed":              settings.seed,
        "reverse_pass":      not args.skip_reverse,
        "timestamp_utc":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary":           {name: _summarise(rows) for name, rows in records.items()},
        "prompts":           prompts,
        "completions":       records,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(md_path, "w") as f:
        f.write("\n".join(_markdown(records, prompts, args, dpo_config, settings)))
    return json_path, md_path


def _summarise(rows: list[dict]) -> dict:
    """Mean health figures over one policy's completions."""
    implicit = [r["implicit_reward"] for r in rows if r["implicit_reward"] is not None]
    return {
        "n":                 len(rows),
        "mean_tokens":       statistics.fmean(r["n_tokens"] for r in rows),
        "eos_rate":          statistics.fmean(1.0 if r["emitted_eos"] else 0.0 for r in rows),
        "mean_distinct_n":   statistics.fmean(r["distinct_n"] for r in rows),
        "mean_logp_per_token": statistics.fmean(r["logp_per_token"] for r in rows),
        "mean_implicit_reward": statistics.fmean(implicit) if implicit else None,
    }


def _markdown(
    records: dict[str, list[dict]], prompts: list[str],
    args, dpo_config, settings: GenerationSettings,
) -> list[str]:
    """Build the review file: header, summary table, then one section per prompt."""
    lines = [
        f"# DPO generation probe: `{args.label}`",
        "",
        f"> Generated on: {datetime.now(timezone.utc).strftime('%d %B %Y')}",
        "",
        f"Beta {dpo_config.beta}, loss `{dpo_config.loss_type}`, "
        f"{args.num_prompts} held-out prompts x {args.samples_per_prompt} samples, "
        f"temperature {settings.temperature}, up to {settings.response_length} new "
        f"tokens, seed {settings.seed}.",
        "",
        "This probe reports generation health only. It carries no reward-model "
        "score, no win rate, and no cross-policy aggregate, so it cannot be read "
        "as a comparison of the two policies (see DPO report, Section 6).",
        "",
        "## Summary",
        "",
        "| policy | mean tokens | EOS rate | distinct-4 | log p / token | implicit reward |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("sft", "dpo"):
        s = _summarise(records[name])
        implicit = "n/a" if s["mean_implicit_reward"] is None else f"{s['mean_implicit_reward']:.3f}"
        lines.append(
            f"| {name} | {s['mean_tokens']:.1f} | {s['eos_rate']:.2f} | "
            f"{s['mean_distinct_n']:.3f} | {s['mean_logp_per_token']:.3f} | {implicit} |"
        )
    lines += [
        "",
        "What to read for: responses that never terminate (EOS rate falling "
        "against the reference), responses that collapse in length, a "
        "distinct-4 ratio falling towards zero, and text that has drifted into "
        "a single register regardless of the prompt.",
        "",
        "## Completions",
        "",
    ]
    for i, prompt in enumerate(prompts):
        lines += [f"### Prompt {i + 1}", "", "```", prompt.strip(), "```", ""]
        for j in range(args.samples_per_prompt):
            k = i * args.samples_per_prompt + j
            suffix = f" (sample {j + 1})" if args.samples_per_prompt > 1 else ""
            for name in ("sft", "dpo"):
                lines += _completion_section(name, records[name][k], suffix)
    return lines


def _completion_section(name: str, row: dict, suffix: str) -> list[str]:
    """One policy's completion for one prompt, with its per-completion figures."""
    implicit = ("" if row["implicit_reward"] is None
                else f", implicit reward {row['implicit_reward']:.3f}")
    return [
        f"**{name}**{suffix} -- {row['n_tokens']} tokens, "
        f"EOS {'yes' if row['emitted_eos'] else 'no'}, "
        f"distinct-4 {row['distinct_n']:.3f}, "
        f"log p/token {row['logp_per_token']:.3f}{implicit}",
        "",
        row["text"] if row["text"] else "_(empty)_",
        "",
    ]


def _print_summary(records: dict[str, list[dict]]) -> None:
    """Print the same summary table to the console."""
    for name in ("sft", "dpo"):
        s = _summarise(records[name])
        implicit = "n/a" if s["mean_implicit_reward"] is None else f"{s['mean_implicit_reward']:+.3f}"
        console.print(
            f"[bold]{name}[/bold]: {s['mean_tokens']:.1f} tokens, EOS "
            f"{s['eos_rate']:.0%}, distinct-4 {s['mean_distinct_n']:.3f}, "
            f"log p/token {s['mean_logp_per_token']:.3f}, implicit reward {implicit}"
        )


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - Scope: a one-off diagnostic, not a pipeline stage. It reads a completed
#   DPO run's adapter and saved config, and answers only whether that policy
#   still generates well. It deliberately computes no reward-model score and
#   no cross-policy statistic, so it cannot pre-empt the pre-registered
#   comparison in compare_policies.py whose protocol is not yet settled.
# - Reuse over reimplementation: prompt selection, model loading, sampling,
#   and the teacher-forcing log-probability pass are imported from
#   compare_policies.py. Sharing them means the probe and the eventual
#   comparison see the same prompt population and the same tokenisation
#   convention, so a reading taken here stays true there.
# - Why a GenerationSettings dataclass: _generate reads seed, temperature,
#   and response_length off a run config, and DPOTrainingConfig has none of
#   them, because DPO never samples during training. Supplying them in a
#   duck-typed dataclass reuses _generate unmodified and keeps the sampling
#   regime an explicit, recorded input rather than a hard-coded constant.
# - Three loads, never two at once: DPO generates and scores itself, SFT
#   generates and scores both its own and the DPO text, then DPO returns to
#   score the SFT text. That last pass is what makes the displacement reading
#   two-sided, distinguishing a policy that has sharpened from one that has
#   moved its mass off anything the reference would produce.
# - Why distinct-4 and not perplexity: a looping generation can hold a high
#   per-token log-probability precisely because it is repeating itself, so
#   confidence and degeneracy have to be read together rather than either
#   alone. Texts shorter than one n-gram score 1.0, and length is reported
#   separately, so brevity is never mistaken for looping.
# =============================================================================
