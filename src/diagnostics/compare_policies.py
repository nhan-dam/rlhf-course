"""
Four-Policy Comparison: Base vs SFT vs PPO vs DPO
==================================================
The standing evaluation for the PPO-vs-DPO comparison (see
reports/report_dpo_lora_hh.md, Section 6). Generates completions from four
policies on prompts drawn from the HH-RLHF test split, which no stage of
either pipeline has trained on, and evaluates them with BOTH available
judges, since each judge is biased towards the arm it belongs to:

- the frozen reward model (biased towards PPO, which optimised against it);
- the DPO implicit reward beta * (log pi_dpo - log pi_ref) (biased towards
  DPO, whose loss maximises exactly this margin).

The four policies are the two comparison arms plus two anchors: the SFT
model shows how much preference optimisation added on top of instruction
tuning, and the raw pre-SFT base model shows how much the entire pipeline
added on top of the pre-trained model. Each policy also gets a sampled KL
from pi_ref (the matched-KL axis of the comparison) and dependency-free
diversity statistics (mode collapse is a way to buy margin without quality).

Inputs
------
--ppo-label LABEL : PPO run; resolves to results/ppo_rlhf_loop/adapter_<label>.
--dpo-label LABEL : DPO run; resolves to results/dpo_lora_hh/adapter_<label>.
--num-prompts N   : distinct test-split prompts to evaluate (default 100).
--md-prompts N    : per-prompt sections written to the markdown file
    (default 20); the JSON always records every prompt.
--batch-size N    : generation/scoring batch size (default 4).

Outputs
-------
results/policy_comparison/comparison_ppo_<ppo_label>_dpo_<dpo_label>.json
    -- run settings, per-policy aggregates, and every per-prompt record.
results/policy_comparison/comparison_ppo_<ppo_label>_dpo_<dpo_label>.md
    -- summary table, head-to-head win rates, judge-bias notes, and the
    first --md-prompts prompts with all four completions for reading.

Public API
----------
main()                      -- run the full comparison and write both files.
select_test_prompts(...)    -- distinct, length-filtered test-split prompts.
sequence_logprob(...)       -- summed completion log-probability under a model.
"""

# stdlib
import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

# third-party
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
)

# local
from ..common.config import PROJECT_ROOT, BASE_MODEL
from ..common.model_utils import resolve_model_path
from ..pipeline.dpo_lora_hh import (
    RESULT_PATH as DPO_RESULT_PATH,
    DPOTrainingConfig,
)
from ..pipeline.dpo_lora_hh import parse_config as parse_dpo_config
from ..pipeline.ppo_rlhf_loop import (
    RESULT_PATH as PPO_RESULT_PATH,
    PPORunConfig,
    _load_tokenizer,
    extract_prompt,
)
from ..pipeline.ppo_rlhf_loop import parse_config as parse_ppo_config

OUTPUT_PATH = f"{PROJECT_ROOT}/results/policy_comparison"

POLICIES = ["base", "sft", "ppo", "dpo"]

console = Console()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate from all four policies, judge with both judges, write files."""
    args = parse_args()
    ppo_config = _load_run_config(PPO_RESULT_PATH, args.ppo_label, parse_ppo_config)
    dpo_config = _load_run_config(DPO_RESULT_PATH, args.dpo_label, parse_dpo_config)
    if ppo_config.sft_model_path != dpo_config.sft_model_path:
        console.print(
            "[yellow]Warning:[/yellow] the two runs record different SFT paths "
            f"({ppo_config.sft_model_path} vs {dpo_config.sft_model_path}); "
            "using the PPO run's as pi_ref. The comparison is only controlled "
            "if both runs share one SFT initialisation."
        )

    device = _pick_device()
    sft_path = resolve_model_path(ppo_config.sft_model_path, "causal-lm")
    tokenizer = _load_tokenizer(sft_path)
    prompts = select_test_prompts(
        ppo_config.dataset_name, tokenizer, ppo_config.max_prompt_tokens,
        args.num_prompts, ppo_config.seed,
    )
    console.print(
        f"Comparing {POLICIES} on {len(prompts)} distinct test-split prompts "
        f"(prompt cap {ppo_config.max_prompt_tokens}) on [bold]{device}[/bold] "
        f"(temperature {ppo_config.temperature}, up to "
        f"{ppo_config.response_length} new tokens)"
    )

    completions, own_logprobs = _generate_all(
        args, ppo_config, sft_path, tokenizer, prompts, device
    )
    judges = _judge_all(
        args, ppo_config, dpo_config, sft_path, tokenizer, prompts,
        completions, own_logprobs, device,
    )

    records, aggregates = _aggregate(prompts, completions, own_logprobs, judges, dpo_config)
    json_path, md_path = _save(records, aggregates, args, ppo_config, dpo_config)
    _print_summary(aggregates)
    console.print(f"Wrote [bold]{json_path}[/bold] and [bold]{md_path}[/bold]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Plain argparse, as in the other diagnostics: this is a one-off evaluation
    utility with no config label of its own. Generation settings (seed,
    temperature, response length, prompt cap) are read from the PPO run's
    saved config, since that is the regime PPO was optimised in; DPO does not
    generate at training time, so it has no competing regime of its own.
    """
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-label", required=True,
                        help="PPO run label; resolves to results/ppo_rlhf_loop/adapter_<label>.")
    parser.add_argument("--dpo-label", required=True,
                        help="DPO run label; resolves to results/dpo_lora_hh/adapter_<label>.")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Distinct test-split prompts to evaluate (default 100).")
    parser.add_argument("--md-prompts", type=int, default=20,
                        help="Per-prompt sections in the markdown file (default 20).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Generation/scoring batch size (default 4).")
    return parser.parse_args(argv)


def _load_run_config(result_path: str, label: str, parse_fn):
    """Load a run's saved config; the adapter must exist too."""
    config_path = f"{result_path}/config_{label}.json"
    adapter_path = f"{result_path}/adapter_{label}"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No saved config at {config_path}; was the run completed?")
    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(f"No adapter at {adapter_path}; was the run completed?")
    return parse_fn([config_path])


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _free(model, device: str) -> None:
    """Drop a model and release its cached device memory before the next load."""
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------

def select_test_prompts(
    dataset_name:      str,
    tokenizer:         PreTrainedTokenizer,
    max_prompt_tokens: int,
    num_prompts:       int,
    seed:              int,
) -> list[str]:
    """Distinct, length-filtered prompts from the dedicated test split.

    The test split is the only population no stage of either pipeline has
    trained on: PPO and DPO both trained on (different views of) the train
    split, and the RM's gate merely evaluated on the test split. Prompts are
    deduplicated first, because HH-RLHF pairs several responses to each
    prompt and duplicate prompts would silently weight the comparison, then
    filtered with PPO's own prompt cap so every policy generates under the
    length regime PPO was trained for, then sampled with a seeded shuffle.
    """
    dataset = load_dataset(dataset_name, split="test")
    distinct = sorted({extract_prompt(text) for text in dataset["chosen"]})
    lengths = tokenizer(distinct)["input_ids"]
    admissible = [p for p, ids in zip(distinct, lengths) if len(ids) <= max_prompt_tokens]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(admissible), generator=generator).tolist()
    selected = [admissible[i] for i in order[: min(num_prompts, len(admissible))]]
    console.print(
        f"Test split: {len(dataset):,} pairs -> {len(distinct):,} distinct prompts "
        f"-> {len(admissible):,} within the prompt cap -> {len(selected)} sampled"
    )
    return selected


# ---------------------------------------------------------------------------
# Generation sweep (one policy loaded at a time)
# ---------------------------------------------------------------------------

def _generate_all(
    args, ppo_config: PPORunConfig, sft_path: str,
    tokenizer: PreTrainedTokenizer, prompts: list[str], device: str,
) -> tuple[dict[str, list[dict]], dict[str, list[float]]]:
    """Generate one completion per prompt per policy; record own log-probs.

    Policies are loaded one at a time and freed before the next, so peak
    memory stays at one 0.5B model regardless of how many policies are
    compared. While each policy is loaded, the summed log-probability of its
    own sampled completion is computed (needed for its KL from pi_ref), so
    no policy has to be loaded twice.
    """
    loaders = {
        "base": lambda: _load_causal(BASE_MODEL, tokenizer, device),
        "sft":  lambda: _load_causal(sft_path, tokenizer, device),
        "ppo":  lambda: _load_adapter(f"{PPO_RESULT_PATH}/adapter_{args.ppo_label}", tokenizer, device),
        "dpo":  lambda: _load_adapter(f"{DPO_RESULT_PATH}/adapter_{args.dpo_label}", tokenizer, device),
    }
    completions: dict[str, list[dict]] = {}
    own_logprobs: dict[str, list[float]] = {}
    for name in POLICIES:
        console.print(f"[cyan]Generating[/cyan] from '{name}'")
        model = loaders[name]()
        completions[name] = _generate(model, tokenizer, prompts, ppo_config, device, args.batch_size)
        own_logprobs[name] = _logprob_pass(
            model, tokenizer, prompts, completions[name], device, args.batch_size,
            desc=f"log p_{name} on its own completions",
        )
        _free(model, device)
    return completions, own_logprobs


def _load_causal(path: str, tokenizer: PreTrainedTokenizer, device: str):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model


def _load_adapter(adapter_path: str, tokenizer: PreTrainedTokenizer, device: str):
    """Load a policy adapter without merging (no cache directories left behind)."""
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model


def _generate(
    model, tokenizer: PreTrainedTokenizer, prompts: list[str],
    config: PPORunConfig, device: str, batch_size: int,
) -> list[dict]:
    """Sample one completion per prompt; record text, token count, EOS emission.

    The generator is reseeded per policy so every policy consumes the same
    sampling stream, and the display text is cut at the '\\n\\nHuman:' marker
    when a policy runs past its turn, both as in
    generate_ppo_completions.py. Whether EOS was emitted within the budget is
    read off the raw output ids BEFORE the cut: a policy that never closes
    its turn is degenerate even when the cut hides it, and the PPO stage
    penalised exactly this at training time (missing_eos_penalty).
    """
    torch.manual_seed(config.seed)
    records: list[dict] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
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
        for ids in responses:
            id_list = ids.tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            records.append({
                "text": text.split("\n\nHuman:")[0].strip(),
                "n_tokens": sum(1 for i in id_list if i != tokenizer.pad_token_id),
                "emitted_eos": tokenizer.eos_token_id in id_list,
            })
    return records


# ---------------------------------------------------------------------------
# Judging sweep (one judge loaded at a time)
# ---------------------------------------------------------------------------

def _judge_all(
    args, ppo_config: PPORunConfig, dpo_config: DPOTrainingConfig,
    sft_path: str, tokenizer: PreTrainedTokenizer, prompts: list[str],
    completions: dict[str, list[dict]], own_logprobs: dict[str, list[float]],
    device: str,
) -> dict[str, dict[str, list[float]]]:
    """Score every policy's completions under pi_ref, pi_dpo, and the RM.

    Returns judges[name] with, per policy, aligned per-prompt lists:
    'ref_logprob' (for the KL), 'dpo_logprob' (for the implicit reward), and
    'rm_score'. The SFT policy's own log-probs already ARE pi_ref on its own
    completions, and likewise for the DPO policy, so those two passes are
    skipped rather than recomputed.
    """
    judges: dict[str, dict[str, list[float]]] = {name: {} for name in POLICIES}

    console.print("[cyan]Judging[/cyan] with pi_ref (SFT) log-probabilities")
    ref = _load_causal(sft_path, tokenizer, device)
    for name in POLICIES:
        judges[name]["ref_logprob"] = (
            own_logprobs["sft"] if name == "sft"
            else _logprob_pass(ref, tokenizer, prompts, completions[name], device,
                               args.batch_size, desc=f"log p_ref on '{name}'")
        )
    _free(ref, device)

    console.print("[cyan]Judging[/cyan] with the DPO implicit reward")
    dpo = _load_adapter(f"{DPO_RESULT_PATH}/adapter_{args.dpo_label}", tokenizer, device)
    for name in POLICIES:
        judges[name]["dpo_logprob"] = (
            own_logprobs["dpo"] if name == "dpo"
            else _logprob_pass(dpo, tokenizer, prompts, completions[name], device,
                               args.batch_size, desc=f"log p_dpo on '{name}'")
        )
    _free(dpo, device)

    console.print("[cyan]Judging[/cyan] with the frozen reward model")
    rm_path = resolve_model_path(ppo_config.rm_model_path, "seq-cls")
    rm = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1, dtype=torch.bfloat16)
    rm.config.pad_token_id = tokenizer.pad_token_id
    rm.to(device)
    rm.eval()
    for name in POLICIES:
        judges[name]["rm_score"] = _rm_score(
            rm, tokenizer, prompts, completions[name], device, args.batch_size,
        )
    _free(rm, device)

    return judges


def sequence_logprob(
    model, tokenizer: PreTrainedTokenizer, prompt: str, completion: str, device: str
) -> float:
    """Summed log-probability of the completion tokens given the prompt.

    Single-example convenience wrapper around _logprob_pass; see there for
    the tokenisation-boundary note.
    """
    return _logprob_pass(model, tokenizer, [prompt], [{"text": completion}], device, 1, desc=None)[0]


def _logprob_pass(
    model, tokenizer: PreTrainedTokenizer, prompts: list[str],
    completions: list[dict], device: str, batch_size: int, desc: str | None,
) -> list[float]:
    """Teacher-forcing pass: summed log p(completion | prompt) per example.

    prompt+completion is re-tokenised and the prompt length measured by
    tokenising the prompt alone, the same convention the RM scoring path has
    always used. The boundary token can differ from the generation-time
    tokenisation in rare cases, but every judge scores the SAME re-tokenised
    ids, so all comparisons remain internally consistent. Sums (not means)
    are returned: the KL penalty PPO optimised was a per-response total, and
    the DPO implicit reward is a sum of per-token log-ratios by definition.
    """
    if desc:
        console.print(f"  {desc}")
    results: list[float] = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        completion_batch = completions[start : start + batch_size]
        texts = [p + c["text"] for p, c in zip(prompt_batch, completion_batch)]
        prompt_lens = [len(ids) for ids in tokenizer(prompt_batch)["input_ids"]]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        # Left padding: content occupies the rightmost positions of each row.
        total_len = encoded["input_ids"].shape[1]
        content_lens = encoded["attention_mask"].sum(dim=1).tolist()
        for row, (p_len, c_len) in enumerate(zip(prompt_lens, content_lens)):
            # Positions of completion tokens within the padded row.
            first = total_len - c_len + p_len   # index of first completion token
            token_ids = encoded["input_ids"][row, first:total_len]
            # log p(token at t) is read from the logits at t-1.
            token_logprobs = logprobs[row, first - 1 : total_len - 1].gather(
                1, token_ids.unsqueeze(-1)
            ).squeeze(-1)
            results.append(float(token_logprobs.sum()))
    return results


def _rm_score(
    rm, tokenizer: PreTrainedTokenizer, prompts: list[str],
    completions: list[dict], device: str, batch_size: int,
) -> list[float]:
    """Score prompt+completion with the RM, truncated to its training cap."""
    texts = [p + c["text"] for p, c in zip(prompts, completions)]
    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(device)
        with torch.inference_mode():
            logits = rm(**encoded).logits
        scores.extend(logits[:, 0].float().cpu().tolist())
    return scores


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def _aggregate(
    prompts: list[str], completions: dict[str, list[dict]],
    own_logprobs: dict[str, list[float]], judges: dict[str, dict[str, list[float]]],
    dpo_config: DPOTrainingConfig,
) -> tuple[list[dict], dict]:
    """Fold the raw passes into per-prompt records and per-policy aggregates."""
    records = []
    for i, prompt in enumerate(prompts):
        record = {"prompt": prompt}
        for name in POLICIES:
            record[name] = {
                "completion":      completions[name][i]["text"],
                "n_tokens":        completions[name][i]["n_tokens"],
                "emitted_eos":     completions[name][i]["emitted_eos"],
                "rm_score":        judges[name]["rm_score"][i],
                "implicit_reward": dpo_config.beta
                                   * (judges[name]["dpo_logprob"][i] - judges[name]["ref_logprob"][i]),
                "kl_from_ref":     own_logprobs[name][i] - judges[name]["ref_logprob"][i],
            }
        records.append(record)

    aggregates = {}
    for name in POLICIES:
        rows = [record[name] for record in records]
        texts = [row["completion"] for row in rows]
        aggregates[name] = {
            "rm_score_mean":        statistics.mean(row["rm_score"] for row in rows),
            "rm_score_sd":          statistics.pstdev(row["rm_score"] for row in rows),
            "implicit_reward_mean": statistics.mean(row["implicit_reward"] for row in rows),
            "kl_from_ref_mean":     statistics.mean(row["kl_from_ref"] for row in rows),
            "response_tokens_mean": statistics.mean(row["n_tokens"] for row in rows),
            "missing_eos_rate":     statistics.mean(0.0 if row["emitted_eos"] else 1.0 for row in rows),
            "distinct_2":           _distinct_n(texts, 2),
        }
    # Head-to-head win rates on the two comparison arms, under both judges.
    aggregates["head_to_head"] = {
        "rm_judge_ppo_wins":  statistics.mean(
            1.0 if r["ppo"]["rm_score"] > r["dpo"]["rm_score"] else 0.0 for r in records
        ),
        "dpo_judge_ppo_wins": statistics.mean(
            1.0 if r["ppo"]["implicit_reward"] > r["dpo"]["implicit_reward"] else 0.0 for r in records
        ),
    }
    return records, aggregates


def _distinct_n(texts: list[str], n: int) -> float:
    """Distinct n-gram ratio over whitespace tokens, pooled across the corpus.

    A cheap, dependency-free mode-collapse signal: policies that keep emitting
    the same phrases score low regardless of per-response fluency.
    """
    ngrams = []
    for text in texts:
        words = text.split()
        ngrams.extend(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def _save(records, aggregates, args, ppo_config, dpo_config) -> tuple[str, str]:
    """Write the JSON record and the markdown summary. Returns both paths."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    stem = f"comparison_ppo_{args.ppo_label}_dpo_{args.dpo_label}"

    payload = {
        "ppo_label":       args.ppo_label,
        "dpo_label":       args.dpo_label,
        "base_model":      BASE_MODEL,
        "sft_model_path":  ppo_config.sft_model_path,
        "rm_model_path":   ppo_config.rm_model_path,
        "ppo_kl_coef":     ppo_config.kl_coef,
        "dpo_beta":        dpo_config.beta,
        "num_prompts":     len(records),
        "temperature":     ppo_config.temperature,
        "response_length": ppo_config.response_length,
        "seed":            ppo_config.seed,
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aggregates":      aggregates,
        "records":         records,
    }
    json_path = f"{OUTPUT_PATH}/{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md_path = f"{OUTPUT_PATH}/{stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_markdown_lines(records, aggregates, args, ppo_config, dpo_config)))
    return json_path, md_path


def _markdown_lines(records, aggregates, args, ppo_config, dpo_config) -> list[str]:
    """Render the summary table, head-to-head lines, and reading sections."""
    h2h = aggregates["head_to_head"]
    lines = [
        f"# Policy comparison: PPO `{args.ppo_label}` vs DPO `{args.dpo_label}`",
        "",
        f"Generated by `src/diagnostics/compare_policies.py` on {len(records)} "
        f"distinct HH-RLHF test-split prompts (unseen by every stage), "
        f"temperature {ppo_config.temperature}, up to {ppo_config.response_length} "
        f"new tokens, seed {ppo_config.seed}. PPO kl_coef "
        f"{ppo_config.kl_coef}; DPO beta {dpo_config.beta}.",
        "",
        "## Summary",
        "",
        "| policy | RM score (mean +/- sd) | DPO implicit reward | KL from pi_ref | "
        "resp. tokens | missing-EOS | distinct-2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in POLICIES:
        a = aggregates[name]
        lines.append(
            f"| {name} | {a['rm_score_mean']:.3f} +/- {a['rm_score_sd']:.3f} "
            f"| {a['implicit_reward_mean']:.3f} | {a['kl_from_ref_mean']:.1f} "
            f"| {a['response_tokens_mean']:.0f} | {100 * a['missing_eos_rate']:.0f}% "
            f"| {a['distinct_2']:.3f} |"
        )
    lines += [
        "",
        f"Head-to-head (PPO vs DPO): the RM judge prefers PPO on "
        f"{100 * h2h['rm_judge_ppo_wins']:.0f}% of prompts; the DPO implicit-reward "
        f"judge prefers PPO on {100 * h2h['dpo_judge_ppo_wins']:.0f}%.",
        "",
        "## How to read this",
        "",
        "- Each judge is biased towards its own arm: PPO optimised the RM's score "
        "directly, and DPO optimised its implicit-reward margin directly. Agreement "
        "between the judges is informative; disagreement means an external judge "
        "(a strong LLM or a human) should break the tie.",
        "- The SFT row anchors what preference optimisation added; the base row "
        "anchors what the whole pipeline added. Neither judge is calibrated on raw "
        "base-model text (the RM was initialised from the SFT model and trained on "
        "HH-RLHF dialogue), so read the base row's completions, not its numbers.",
        "- KL from pi_ref is a sampled per-response total under each policy's own "
        "completions; compare the PPO and DPO rows at similar KL (or sweep DPO's "
        "beta until they match) before comparing their rewards. The SFT row's KL "
        "is zero by construction; the base row's is not a drift measure (it "
        "drifted before pi_ref existed) and is reported only for completeness.",
        "- Low distinct-2 or a high missing-EOS rate flags margin bought with "
        "degeneracy rather than quality.",
        "",
    ]
    for index, record in enumerate(records[: args.md_prompts], start=1):
        lines += [
            f"## Prompt {index} (RM: ppo {record['ppo']['rm_score']:.3f} vs "
            f"dpo {record['dpo']['rm_score']:.3f}; implicit: ppo "
            f"{record['ppo']['implicit_reward']:.3f} vs dpo "
            f"{record['dpo']['implicit_reward']:.3f})",
            "",
            "### Prompt",
            "",
            "```text",
            record["prompt"].strip(),
            "```",
            "",
        ]
        for name in POLICIES:
            row = record[name]
            lines += [
                f"### {name} — RM {row['rm_score']:.3f}, implicit "
                f"{row['implicit_reward']:.3f}, KL {row['kl_from_ref']:.1f}",
                "",
                "```text",
                row["completion"] or "(empty completion)",
                "```",
                "",
            ]
    return lines


def _print_summary(aggregates: dict) -> None:
    """Print the headline numbers so the console shows the verdict shape."""
    for name in POLICIES:
        a = aggregates[name]
        console.print(
            f"{name:>4}: RM [bold]{a['rm_score_mean']:.3f}[/bold], "
            f"implicit [bold]{a['implicit_reward_mean']:.3f}[/bold], "
            f"KL {a['kl_from_ref_mean']:.1f}, "
            f"tokens {a['response_tokens_mean']:.0f}, "
            f"missing-EOS {100 * a['missing_eos_rate']:.0f}%, "
            f"distinct-2 {a['distinct_2']:.3f}"
        )
    h2h = aggregates["head_to_head"]
    console.print(
        f"Head-to-head PPO wins: RM judge [bold]{100 * h2h['rm_judge_ppo_wins']:.0f}%[/bold], "
        f"DPO judge [bold]{100 * h2h['dpo_judge_ppo_wins']:.0f}%[/bold]"
    )


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - Scope: the standing evaluation for the PPO-vs-DPO comparison (DPO report
#   Section 6), not a pipeline stage. It takes two completed run labels and
#   writes comparison_ppo_<ppo>_dpo_<dpo>.{json,md} under
#   results/policy_comparison/.
# - Prompts: distinct prompts from the dedicated test split, which no stage
#   trained on (PPO's old train-carved eval prompts sit inside DPO's training
#   set, so they cannot serve this comparison). Deduplicated (HH-RLHF repeats
#   prompts across pairs), filtered with PPO's own 256-token prompt cap, and
#   sampled with the PPO run's seed.
# - Four policies, two anchors: base and SFT are not comparison arms but
#   reference points -- SFT isolates what preference optimisation added, base
#   what the whole pipeline added. Policies are loaded one at a time
#   (adapters unmerged, via AutoPeftModelForCausalLM) and freed before the
#   next load, so peak memory is one 0.5B model plus activations.
# - Two judges, symmetric bias: the RM (PPO's training signal) and the DPO
#   implicit reward beta*(log pi_dpo - log pi_ref) (DPO's training signal)
#   both score all four policies; the markdown states the bias explicitly and
#   defers disagreements to an external judge.
# - KL from pi_ref: summed per-response log-ratio of each policy's own
#   completions under itself vs under the SFT reference -- the matched-KL
#   axis. Sums, not means, to match how both training objectives price drift.
# - Log-prob mechanics: teacher-forcing passes over re-tokenised
#   prompt+completion text (the repo's established scoring convention), with
#   completion positions located from the attention mask under left padding
#   and log p(token_t) read from logits at t-1. Every judge scores the same
#   re-tokenised ids, so cross-policy comparisons are internally consistent
#   even where re-tokenisation shifts a boundary token.
# - Generation parity: all policies sample under the PPO run's regime
#   (temperature, response length, reseeded per policy), the same convention
#   as generate_ppo_completions.py; EOS emission is read off the raw ids
#   before the display cut, since a never-closing policy is degenerate even
#   when the cut hides it.
# - Diversity: distinct-2 over pooled whitespace bigrams and the missing-EOS
#   rate are deliberately dependency-free stand-ins for self-BLEU; low
#   distinct-2 flags margin bought with mode collapse.
# =============================================================================
