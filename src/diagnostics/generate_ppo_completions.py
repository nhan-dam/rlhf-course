"""
Standalone Qualitative Completion Probe for a Trained PPO Policy
=====================================================================
Generates reward-scored completions from an already-trained PPO policy
adapter, side by side with the reference (SFT) policy and, optionally, the
raw pre-SFT base model, and saves them to a markdown file for manual review.

The PPO stage's own post-training check (trainer.generate_completions() in
ppo_rlhf_loop.py) prints five samples to the console and, with
report_to="tensorboard", persists nothing. This script exists to backfill a
reviewable artefact for a completed run: the curriculum (Phase 2, Section
5.2) requires inspecting the actual generated text for reward hacking, and
the reward-model report (Section 6.3) lists the two specific weaknesses to
look for (near-chance discrimination on confidently-wrong content, and a
preference for bulleted or shouty-header formatting over substantive prose).

Inputs
------
--label LABEL or --adapter-path PATH : which PPO run to probe. --label
    resolves to results/ppo_rlhf_loop/adapter_<label> (the directory
    ppo_rlhf_loop.py's train() saves to); --adapter-path points at that
    directory (or any other PPO LoRA adapter directory) directly.
--num-prompts N : how many eval prompts to complete (default 20).
--samples-per-prompt K : completions drawn per prompt per policy (default
    1). At temperature 0.7 a single draw is noisy, so per-prompt score
    comparisons average over K samples when K > 1.
--skip-base : omit the pre-SFT base model column (faster; two columns).

Outputs
-------
results/ppo_rlhf_loop/completions_<label>.md -- one section per prompt with
the base (unless skipped), reference, and PPO completions and their
reward-model scores, preceded by a summary (mean scores, win rate) and the
review checklist. Base-model scores are reported for completeness but are
not calibrated: the RM was initialised from the SFT model and trained on
HH-RLHF dialogue, so raw base-model text is out of its training
distribution. Read the base column's text, not its numbers.

Public API
----------
resolve_adapter_path(args)        -- turn --label/--adapter-path into a directory.
load_policy(adapter_path)         -- load the PPO adapter for generation.
generate_probe_records(...)       -- generate and score paired completions.
save_completions(records, ...)    -- write the markdown review file.
"""

# stdlib
import argparse
import os
import statistics
import sys
from collections.abc import Callable

# third-party
import torch
from peft import AutoPeftModelForCausalLM
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
)

# local
from ..common.config import BASE_MODEL
from ..common.model_utils import resolve_model_path
from ..pipeline.ppo_rlhf_loop import (
    RESULT_PATH,
    PPORunConfig,
    _build_prompt_datasets,
    _load_tokenizer,
    parse_config,
)

console = Console()

GENERATION_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load a trained PPO adapter and generate scored completions for review."""
    args = parse_args()

    adapter_path = resolve_adapter_path(args)
    label = _label_from_adapter_path(adapter_path)
    config = load_run_config(label)
    console.print(f"Probing PPO adapter: [bold]{adapter_path}[/bold]")

    device = _pick_device()
    policy, tokenizer = load_policy(adapter_path, device)
    reward_model = load_reward_model(config, tokenizer, device)
    base_model = None if args.skip_base else load_base_model(tokenizer, device)
    prompts = select_eval_prompts(config, tokenizer, args.num_prompts)
    columns = "base, reference, and PPO" if base_model is not None else "reference and PPO"
    console.print(
        f"Generating {columns} completions for {len(prompts)} prompts "
        f"({args.samples_per_prompt} per prompt per policy) on [bold]{device}[/bold] "
        f"(temperature {config.temperature}, up to {config.response_length} new tokens)"
    )

    records = generate_probe_records(
        policy, reward_model, tokenizer, prompts, config, device,
        base_model, args.samples_per_prompt,
    )
    output_path = save_completions(records, label, config)
    _print_summary(records)
    console.print(f"Wrote {len(records)} completions to [bold]{output_path}[/bold]")


# ---------------------------------------------------------------------------
# Argument parsing and run resolution
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Plain argparse, not HfArgumentParser: this is a one-off diagnostic
    utility, not a config-driven training run, so it has no dataclass and no
    config label of its own. Generation settings (temperature, response
    length, eval split) are read from the probed run's own saved config so
    the completions reflect what that run actually optimised.
    """
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--label", help="PPO run label; resolves to results/ppo_rlhf_loop/adapter_<label>."
    )
    source.add_argument(
        "--adapter-path", help="Explicit path to a PPO LoRA adapter directory."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=20,
        help="Number of eval prompts to complete (default 20).",
    )
    parser.add_argument(
        "--samples-per-prompt", type=int, default=1,
        help="Completions drawn per prompt per policy (default 1). Scores "
             "are compared as per-prompt means over the samples.",
    )
    parser.add_argument(
        "--skip-base", action="store_true",
        help="Omit the pre-SFT base model column (faster; two columns).",
    )
    return parser.parse_args(argv)


def resolve_adapter_path(args: argparse.Namespace) -> str:
    """Turn --label/--adapter-path into a validated adapter directory.

    Args:
        args: Parsed CLI arguments (see parse_args).

    Returns:
        Absolute path to a directory containing adapter_config.json.

    Raises:
        FileNotFoundError: If no adapter exists there yet.
    """
    if args.adapter_path:
        adapter_path = os.path.abspath(args.adapter_path)
    else:
        adapter_path = f"{RESULT_PATH}/adapter_{args.label}"

    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(
            f"No LoRA adapter found at {adapter_path}. Run the PPO stage first "
            f"(`uv run rlhf-ppo`) so that results/ppo_rlhf_loop/adapter_<label>/ "
            f"exists, then pass --label <that label> or --adapter-path <that "
            f"directory>."
        )
    return adapter_path


def load_run_config(label: str) -> PPORunConfig:
    """Load the probed run's saved config so the probe mirrors its settings.

    The seed and eval_examples reproduce the exact eval split the run held
    out; temperature and response_length reproduce its generation regime.

    Raises:
        FileNotFoundError: If the run never saved a config (was never trained).
    """
    config_path = f"{RESULT_PATH}/config_{label}.json"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"No saved config at {config_path}; cannot reproduce the run's "
            f"eval split or generation settings."
        )
    return parse_config([config_path])


def _label_from_adapter_path(adapter_path: str) -> str:
    """Extract the run label from an 'adapter_<label>' directory name."""
    return os.path.basename(adapter_path.rstrip("/")).removeprefix("adapter_")


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_policy(adapter_path: str, device: str) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """Load the PPO LoRA adapter directly, without merging, for generation.

    Keeping the policy as a PeftModel is what makes the paired probe cheap:
    the reference policy is recovered exactly by disabling the adapters
    (the base weights are the SFT model), so one loaded model yields both
    sides of the comparison. Merging would also leave an unneeded '-merged'
    cache directory behind for a one-off diagnostic.
    """
    tokenizer = _load_tokenizer(adapter_path)
    policy = AutoPeftModelForCausalLM.from_pretrained(adapter_path, dtype=torch.bfloat16)
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.to(device)
    policy.eval()
    return policy, tokenizer


def load_base_model(tokenizer: PreTrainedTokenizer, device: str) -> torch.nn.Module:
    """Load the raw pre-SFT base model for the 'how far have we come' column.

    This cannot be recovered from the policy: disable_adapter() yields the
    merged SFT model (the policy's base weights), not BASE_MODEL. The base
    shares the SFT model's tokenizer family, so the probe's single tokenizer
    is reused for its generation and scoring.
    """
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.to(device)
    base_model.eval()
    return base_model


def load_reward_model(
    config: PPORunConfig, tokenizer: PreTrainedTokenizer, device: str
) -> torch.nn.Module:
    """Load the frozen reward model the run was trained against.

    resolve_model_path reuses the existing '<rm_model_path>-merged' cache
    created when the PPO run itself resolved the RM, so no new merge happens
    here. pad_token_id must be set for decoder-only sequence classification
    to pool the correct final token.
    """
    rm_path = resolve_model_path(config.rm_model_path, "seq-cls")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        rm_path, num_labels=1, dtype=torch.bfloat16
    )
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    reward_model.to(device)
    reward_model.eval()
    return reward_model


def select_eval_prompts(
    config: PPORunConfig, tokenizer: PreTrainedTokenizer, num_prompts: int
) -> list[str]:
    """Reproduce the run's held-out eval prompts and take the first N.

    _build_prompt_datasets splits with the run's own seed, so the eval side
    here is exactly the split the run held out: these prompts were never
    trained on, making them fair game for qualitative review.
    """
    _, eval_ds = _build_prompt_datasets(config, tokenizer)
    num_prompts = min(num_prompts, len(eval_ds))
    return [
        tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        for example in eval_ds.select(range(num_prompts))
    ]


# ---------------------------------------------------------------------------
# Generation and scoring
# ---------------------------------------------------------------------------

def generate_probe_records(
    policy:       torch.nn.Module,
    reward_model: torch.nn.Module,
    tokenizer:    PreTrainedTokenizer,
    prompts:      list[str],
    config:       PPORunConfig,
    device:       str,
    base_model:   torch.nn.Module | None = None,
    num_samples:  int = 1,
) -> list[dict]:
    """Generate base (optional), reference, and PPO completions and score all.

    Every policy draws num_samples completions per prompt under the same
    reseeded sampling settings, so any systematic difference between columns
    reflects training, not the sampling draw. Multiple samples tighten
    per-prompt score comparisons by averaging over the draw.

    Returns:
        One dict per prompt: 'prompt' plus, per policy ('ref', 'ppo', and
        'base' when base_model is given), a list of (completion, score)
        pairs of length num_samples.
    """
    ppo = _sample_policy(policy, reward_model, tokenizer, prompts, config, device, num_samples)
    with policy.disable_adapter():
        ref = _sample_policy(policy, reward_model, tokenizer, prompts, config, device, num_samples)

    records = [
        {"prompt": prompt, "ref": ref_samples, "ppo": ppo_samples}
        for prompt, ref_samples, ppo_samples in zip(prompts, ref, ppo)
    ]

    if base_model is not None:
        base = _sample_policy(
            base_model, reward_model, tokenizer, prompts, config, device, num_samples
        )
        for record, base_samples in zip(records, base):
            record["base"] = base_samples

    return records


def _sample_policy(
    model:        torch.nn.Module,
    reward_model: torch.nn.Module,
    tokenizer:    PreTrainedTokenizer,
    prompts:      list[str],
    config:       PPORunConfig,
    device:       str,
    num_samples:  int,
) -> list[list[tuple[str, float]]]:
    """Draw num_samples scored completions per prompt from one policy.

    The generator is reseeded once per policy, so every policy consumes the
    identical sampling stream while successive passes within a policy remain
    independent draws.

    Returns:
        One list of (completion, score) pairs per prompt.
    """
    torch.manual_seed(config.seed)
    passes = []
    for _ in range(num_samples):
        completions = _generate(model, tokenizer, prompts, config, device)
        scores = _score(reward_model, tokenizer, prompts, completions, device)
        passes.append(list(zip(completions, scores)))
    return [list(per_prompt) for per_prompt in zip(*passes)]


def _mean_score(samples: list[tuple[str, float]]) -> float:
    """Mean RM score over one prompt's samples from one policy."""
    return statistics.mean(score for _, score in samples)


def _generate(
    model:     torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts:   list[str],
    config:    PPORunConfig,
    device:    str,
    on_batch:  Callable[[int], None] | None = None,
) -> list[str]:
    """Sample one completion per prompt, batched, at the run's temperature.

    Prompts are left-padded (see _load_tokenizer) so generation continues
    from the true final prompt token. Responses are cut at the '\\n\\nHuman:'
    turn marker when the model runs on past its own turn, mirroring the stop
    behaviour used at export time (see ppo_to_ollama.py's Modelfile).

    on_batch, if given, is called with the batch size after each batch, so a
    caller can drive a prompt-level progress bar. It must not alter the
    batching, since the sampling stream is consumed per generate() call and
    reproducibility against an earlier sweep depends on that order.
    """
    completions: list[str] = []
    for start in range(0, len(prompts), GENERATION_BATCH_SIZE):
        batch = prompts[start : start + GENERATION_BATCH_SIZE]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=config.temperature,
                top_k=0,
                top_p=1.0,
                max_new_tokens=config.response_length,
                pad_token_id=tokenizer.pad_token_id,
            )
        responses = output[:, encoded["input_ids"].shape[1]:]
        for response in tokenizer.batch_decode(responses, skip_special_tokens=True):
            completions.append(response.split("\n\nHuman:")[0].strip())
        if on_batch is not None:
            on_batch(len(batch))
    return completions


def _score(
    reward_model: torch.nn.Module,
    tokenizer:    PreTrainedTokenizer,
    prompts:      list[str],
    completions:  list[str],
    device:       str,
) -> list[float]:
    """Score prompt+completion texts with the reward model, batched.

    Decoder-only sequence classification pools the rightmost non-pad token,
    which is position-correct under the tokenizer's left padding, so the
    generation tokenizer is reused as-is.
    """
    texts = [prompt + completion for prompt, completion in zip(prompts, completions)]
    scores: list[float] = []
    for start in range(0, len(texts), GENERATION_BATCH_SIZE):
        batch = texts[start : start + GENERATION_BATCH_SIZE]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            logits = reward_model(**encoded).logits
        scores.extend(logits[:, 0].float().cpu().tolist())
    return scores


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_completions(records: list[dict], label: str, config: PPORunConfig) -> str:
    """Write the markdown review file. Returns the path.

    The review checklist from the reward-model report, Section 6.3, is embedded at the top
    so the file is self-contained: it states what the reader should look for
    without needing the notes open alongside.
    """
    has_base = "base" in records[0]
    num_samples = len(records[0]["ppo"])
    ref_means = [_mean_score(record["ref"]) for record in records]
    ppo_means = [_mean_score(record["ppo"]) for record in records]
    wins = sum(p > r for p, r in zip(ppo_means, ref_means))
    mean_ref = statistics.mean(ref_means)
    mean_ppo = statistics.mean(ppo_means)

    lines = [
        f"# PPO completion review: run `{label}`",
        "",
        f"Generated by `src/diagnostics/generate_ppo_completions.py` from the "
        f"run's held-out eval prompts (seed {config.seed}, temperature "
        f"{config.temperature}, up to {config.response_length} new tokens). "
        f"Reference completions come from the same model with the LoRA "
        f"adapters disabled, i.e. the SFT policy."
        + (
            f" Each policy draws {num_samples} samples per prompt, and score "
            f"comparisons use per-prompt means over the samples."
            if num_samples > 1 else ""
        ),
        "",
        "## Summary",
        "",
        f"- Prompts: {len(records)}",
        f"- Mean reward-model score: reference {mean_ref:.3f}, PPO {mean_ppo:.3f}",
        f"- PPO scores higher on {wins}/{len(records)} prompts",
        "",
    ]
    if has_base:
        mean_base = statistics.mean(_mean_score(record["base"]) for record in records)
        lines += [
            f"- Mean reward-model score, pre-SFT base model (`{BASE_MODEL}`): "
            f"{mean_base:.3f}. Not calibrated: the RM was initialised from "
            f"the SFT model and trained on HH-RLHF dialogue, so raw "
            f"base-model text is out of its training distribution. Read the "
            f"base column's text, not its numbers.",
            "",
        ]
    lines += [
        "## What to look for (reward-model report, Section 6.3)",
        "",
        "- Format gaming: PPO completions trending towards bulleted lists and "
        "shouty headers regardless of prompt. The RM reliably over-scores "
        "these, so a formatting drift here is reward hacking, not quality.",
        "- Confidently wrong content: the RM is close to guessing on fluent "
        "but false text, so check factual claims rather than trusting the "
        "score.",
        "- The generic check: repetition, degenerate phrasing, or responses "
        "that stop matching the question.",
        "",
    ]
    for index, (record, ref_mean, ppo_mean) in enumerate(
        zip(records, ref_means, ppo_means), start=1
    ):
        delta = ppo_mean - ref_mean
        lines += [
            f"## Prompt {index} (PPO {ppo_mean:.3f} vs "
            f"reference {ref_mean:.3f}, delta {delta:+.3f})",
            "",
            "### Prompt",
            "",
            "```text",
            record["prompt"].strip(),
            "```",
            "",
        ]
        if has_base:
            lines += _policy_section("Pre-SFT base completion", record["base"], " (uncalibrated)")
        lines += _policy_section("Reference (SFT) completion", record["ref"])
        lines += _policy_section("PPO completion", record["ppo"])

    output_path = f"{RESULT_PATH}/completions_{label}.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def _policy_section(title: str, samples: list[tuple[str, float]], suffix: str = "") -> list[str]:
    """Markdown lines for one policy's samples on one prompt.

    A single sample keeps the original single-block layout, so default runs
    produce files shaped like earlier reviews. Multiple samples get a
    mean-score heading with one sub-block per sample.
    """
    if len(samples) == 1:
        completion, score = samples[0]
        return [
            f"### {title} — score {score:.3f}{suffix}",
            "",
            "```text",
            completion or "(empty completion)",
            "```",
            "",
        ]
    lines = [f"### {title} — mean score {_mean_score(samples):.3f}{suffix}", ""]
    for sample_index, (completion, score) in enumerate(samples, start=1):
        lines += [
            f"#### Sample {sample_index} — score {score:.3f}",
            "",
            "```text",
            completion or "(empty completion)",
            "```",
            "",
        ]
    return lines


def _print_summary(records: list[dict]) -> None:
    """Print the headline comparison so the console shows the verdict shape."""
    ref_means = [_mean_score(record["ref"]) for record in records]
    ppo_means = [_mean_score(record["ppo"]) for record in records]
    wins = sum(p > r for p, r in zip(ppo_means, ref_means))
    mean_ref = statistics.mean(ref_means)
    mean_ppo = statistics.mean(ppo_means)
    console.print(
        f"Mean RM score: reference [bold]{mean_ref:.3f}[/bold] vs "
        f"PPO [bold]{mean_ppo:.3f}[/bold]; PPO higher on "
        f"[bold]{wins}/{len(records)}[/bold] prompts"
    )
    if "base" in records[0]:
        mean_base = statistics.mean(_mean_score(record["base"]) for record in records)
        console.print(
            f"Pre-SFT base mean RM score: [bold]{mean_base:.3f}[/bold] "
            f"(uncalibrated; see the output file's summary note)"
        )


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - Scope: a one-off diagnostic utility, not a pipeline stage -- it reads an
#   already-trained PPO adapter plus that run's saved config, and writes the
#   reviewable completions file that generate_completions() (console print
#   only under report_to="tensorboard") never persisted.
# - One model, two policies: the policy is loaded as a PeftModel and the
#   reference side is produced under policy.disable_adapter(), the same
#   mechanism PPOTrainer used during training (ref_model=None), so the
#   comparison is against the exact reference the KL penalty anchored to.
# - Same data, same regime: the eval prompts are rebuilt with the run's own
#   seed and filters via _build_prompt_datasets (imported, not reimplemented),
#   so they are the run's true held-out split; sampling uses the run's own
#   temperature and response length, and both policies sample from the same
#   reseeded generator, so column differences reflect training, not the draw.
# - Scoring parity: completions are scored by the same frozen RM the run
#   optimised against (resolved through the existing '-merged' cache), with
#   the left-padded tokenizer, which decoder-only sequence classification
#   pooling handles correctly (rightmost non-pad token).
# - Base-model column (on by default, --skip-base to omit): the raw pre-SFT
#   BASE_MODEL is loaded separately, because disable_adapter() recovers the
#   merged SFT model, not the original base. Its RM scores are reported but
#   flagged uncalibrated: the RM was initialised from the SFT model and
#   trained on HH-RLHF dialogue, so raw base-model text is out of its
#   distribution, and the column's value is the visible before/after of the
#   whole SFT -> RM -> PPO pipeline, judged by reading, not by score.
# =============================================================================
