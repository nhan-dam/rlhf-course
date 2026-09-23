"""
Direct Preference Optimisation on Anthropic HH-RLHF
====================================================
Implements the Phase 3 alternative to the RM + PPO stages (Rafailov et al.,
2023): fine-tune the SFT policy directly on pairwise preferences with the DPO
loss, -log sigmoid(beta * (implicit reward margin)), where the implicit reward
is beta * log(pi_theta / pi_ref). No reward model and no rollouts: one
supervised pass over the preference pairs replaces the whole PPO loop. The
policy is a LoRA PEFT model on the merged SFT backbone; with ref_model=None the
frozen reference pi_ref is recovered by disabling the adapters, exactly as in
the PPO stage.

The data view deliberately matches the earlier stages so the PPO-vs-DPO
comparison is fair: pairs are FILTERED (never truncated) to those whose prompt
fits the PPO prompt cap (256) and whose both sides, with EOS appended, fit the
RM cap (512). Trainable capacity matches the PPO policy adapter exactly
(rank 32, alpha 64, q_proj/v_proj).

Inputs
------
config : DPOTrainingConfig — SFT model path (adapter directories are merged
         automatically), LoRA settings, learning rate, beta, loss type,
         length caps, batch sizes, and logging cadence.

Outputs
-------
DPO policy LoRA adapter (and tokenizer) saved to
./results/dpo_lora_hh/adapter_<label>/; DPO loss, implicit-reward accuracy,
margins, and chosen/rejected log-probabilities logged to TensorBoard; a
post-training gate evaluation (implicit-reward pairwise accuracy over the
whole length-admissible test split) saved to metrics_<label>.json. If the
EXPORT_CANONICAL environment variable is set to '1', the adapter is also
exported to the pipeline-shared DPO_MODEL path (opt-in, as in every stage).

Dataset
-------
Anthropic/hh-rlhf — human preference pairs in implicit-prompt format: each
example holds 'chosen' and 'rejected' full-dialogue texts. The pipeline splits
off the shared prompt at the final '\\n\\nAssistant:' marker (split_prompt)
before handing the pairs to trl.DPOTrainer, which appends EOS itself.

Public API
----------
train(config)   — run DPO; return (trainer, gate_dataset, n_test).
filter_pairs(dataset, tokenizer, max_prompt_tokens, max_pair_tokens)
                — drop pairs violating the prompt or pair length caps.
exclude_train_pairs(gate_ds, train_raw)
                — drop gate pairs that appear verbatim in the training split.
split_prompt(dataset)
                — add the explicit 'prompt' column, cut at the final Assistant marker.
"""

# stdlib
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# Reduce CUDA allocator fragmentation on long, variable-length runs. Must be set
# before torch initialises the CUDA context; harmless on non-CUDA backends.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# third-party
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer

# local
from ..common.config import PROJECT_ROOT, SFT_ADAPTER, DPO_ADAPTER
from ..common.model_utils import resolve_model_path, CacheCleaner, export_canonical
from .ppo_rlhf_loop import extract_prompt

RESULT_PATH = f"{PROJECT_ROOT}/results/dpo_lora_hh"

console = Console()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPOTrainingConfig:
    # None → the adapter produced by sft_lora_dolly.py with default settings.
    sft_model_path: str | None = None
    dataset_name:   str = "Anthropic/hh-rlhf"
    seed:           int = 42

    # LoRA — identical to the PPO policy adapter (rank, alpha, dropout,
    # targets), so the PPO-vs-DPO comparison holds trainable capacity fixed.
    lora_r:       int   = 32
    lora_alpha:   int   = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # DPO objective. beta prices drift from pi_ref inside the implicit reward.
    # Lower beta raises held-out accuracy and chosen-side drift together, so
    # beta is chosen on the trade-off between them, not on accuracy alone.
    # loss_type='ipo' is the documented switch if likelihood displacement
    # (logps/chosen falling alongside logps/rejected) becomes large.
    beta:      float = 0.1
    loss_type: str   = "sigmoid"

    # Optimisation. The DPO paper's 5e-7 is a full-fine-tuning rate; a LoRA
    # adapter starts at zero and needs a larger step (see the report). The
    # schedule is the Trainer's linear one: ramp over warmup_ratio of the
    # steps, then decay to zero.
    learning_rate:               float = 1e-4
    warmup_ratio:                float = 0.03
    n_epochs:                    int   = 1
    per_device_train_batch_size: int   = 4
    per_device_eval_batch_size:  int   = 2   # small to cap the eval-time logit peak
    gradient_accumulation_steps: int   = 4   # effective batch size = 4 x 4 = 16
    # Off by default like the RM baseline: one 0.5B backbone with adapters and
    # <=512-token sequences is not memory-bound; enable for capacity sweeps.
    gradient_checkpointing:      bool  = False
    # Score pi_ref once before training instead of once per step. Same numbers
    # (the base is frozen and adapter-disabled either way); removes the
    # reference forward's transient logits from every step.
    precompute_ref_log_probs:    bool  = False

    # Length caps, both with FILTER semantics (a pair is dropped, never
    # truncated). max_prompt_tokens mirrors the PPO prompt cap so DPO trains
    # on the same prompt distribution PPO optimised on; max_pair_tokens
    # mirrors RewardConfig.max_length so DPO sees the pairs the RM saw.
    max_prompt_tokens: int = 256
    max_pair_tokens:   int = 512

    # Evaluation and logging. eval_examples pairs are subsampled (seeded) from
    # the length-filtered test split for the periodic in-training evaluation;
    # the post-training gate scores that whole filtered split, mirroring the
    # RM stage's two-role split of the same population.
    eval_examples: int = 1_000
    logging_steps: int = 50
    eval_steps:    int = 500
    save_steps:    int = 500
    report_to:     str = "tensorboard"

    def __post_init__(self) -> None:
        if self.sft_model_path is None:
            self.sft_model_path = SFT_ADAPTER
        if self.eval_examples <= 0:
            raise ValueError(f"eval_examples ({self.eval_examples}) must be positive.")
        if self.max_prompt_tokens >= self.max_pair_tokens:
            raise ValueError(
                f"max_prompt_tokens ({self.max_prompt_tokens}) must be below "
                f"max_pair_tokens ({self.max_pair_tokens}); the response needs room."
            )
        if self.save_steps % self.eval_steps != 0:
            raise ValueError(
                f"save_steps ({self.save_steps}) must be a multiple of eval_steps "
                f"({self.eval_steps}); best-checkpoint selection needs an evaluation "
                f"at every save."
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio ({self.warmup_ratio}) must be in [0, 1).")

    @property
    def label(self) -> str:
        # Hash the FULL config so any hyperparameter change (e.g. a beta sweep
        # step) yields a distinct label and its own results directory,
        # preventing different experiments from overwriting each other.
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: DPOTrainingConfig) -> tuple[DPOTrainer, Dataset, int]:
    """Run DPO fine-tuning and save the policy adapter.

    Reads as the algorithm outline: resolve the SFT backbone (merging the SFT
    LoRA adapter if needed), load and length-filter the preference pairs,
    assemble the trainer (reference recovered by adapter disabling), train,
    and persist.

    Args:
        config: Hyperparameter configuration.

    Returns:
        trainer: The fitted DPOTrainer.
        gate_ds: The gate population: the filtered test split minus the
                 in-training evaluation subsample, scored once after training.
        n_test:  The unfiltered test-split size, for the retention record.
    """
    sft_path  = resolve_model_path(config.sft_model_path, "causal-lm")
    tokenizer = _load_tokenizer(sft_path)
    policy    = AutoModelForCausalLM.from_pretrained(sft_path, dtype=torch.bfloat16)
    policy.config.pad_token_id = tokenizer.pad_token_id
    train_ds, eval_ds, gate_ds, n_test = _load_preference_datasets(config, tokenizer)
    trainer   = _build_trainer(policy, tokenizer, train_ds, eval_ds, config)

    # Resume from the latest checkpoint if one exists, else start fresh. Unlike
    # the experimental PPOTrainer, DPOTrainer is a standard Trainer subclass,
    # so interrupted runs resume and load_best_model_at_end applies.
    output_dir = trainer.args.output_dir
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint:
        console.print(f"[yellow]Resuming from checkpoint[/yellow] {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    adapter_path = f"{RESULT_PATH}/adapter_{config.label}"
    trainer.save_model(adapter_path)
    console.print(f"[green]DPO policy adapter saved to[/green] {adapter_path}")

    return trainer, gate_ds, n_test


def _load_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """Load the tokenizer, reusing EOS as the pad token if none is set."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_preference_datasets(
    config: DPOTrainingConfig, tokenizer: PreTrainedTokenizer
) -> tuple[Dataset, Dataset, Dataset, int]:
    """Load HH-RLHF splits and filter BOTH to the same length caps.

    The train split is filtered too, not just the evaluation side: matching
    the RM's data view (pairs within 512) and PPO's prompt view (prompts
    within 256) is the point of the caps, and DPOConfig.max_length would
    otherwise TRUNCATE over-long pairs, which corrupts the preference signal
    (the two sides of an HH-RLHF pair differ mainly in the final assistant
    turn; clipping tends to leave two near-identical prefixes).

    Two further exclusions follow the caps: gate pairs that appear verbatim
    in the training split (exclude_train_pairs), and pairs whose sides
    diverge before the final assistant turn (split_prompt). The filtered
    test split is then divided into two DISJOINT roles:

    - eval_ds: a seeded eval_examples-pair subsample, evaluated every
      eval_steps during training and used to select the best checkpoint.
    - gate_ds: every other filtered test pair, scored once post-training.
      Keeping it disjoint from eval_ds means the checkpoint is never
      selected on pairs the gate then scores.

    Returns (train_ds, eval_ds, gate_ds, n_test_total), the last being the
    unfiltered test-split size for the retention record.
    """
    train_raw = load_dataset(config.dataset_name, split="train")
    test_raw  = load_dataset(config.dataset_name, split="test")
    train_ds  = split_prompt(filter_pairs(train_raw, tokenizer, config.max_prompt_tokens, config.max_pair_tokens))
    gate_ds   = filter_pairs(test_raw, tokenizer, config.max_prompt_tokens, config.max_pair_tokens)
    test_ds   = split_prompt(exclude_train_pairs(gate_ds, train_raw))
    n_eval    = min(config.eval_examples, len(test_ds))
    shuffled  = test_ds.shuffle(seed=config.seed)
    eval_ds   = shuffled.select(range(n_eval))
    gate_ds   = shuffled.select(range(n_eval, len(shuffled)))
    console.print(
        f"Data view (caps prompt<={config.max_prompt_tokens}, pair<={config.max_pair_tokens}, "
        f"prompt split, overlap): train {len(train_ds):,}/{len(train_raw):,} "
        f"({100 * len(train_ds) / len(train_raw):.1f}% kept); "
        f"test {len(test_ds):,}/{len(test_raw):,} "
        f"({100 * len(test_ds) / len(test_raw):.1f}% kept) = "
        f"{len(eval_ds):,} eval + {len(gate_ds):,} gate"
    )
    return train_ds, eval_ds, gate_ds, len(test_raw)


def filter_pairs(
    dataset:           Dataset,
    tokenizer:         PreTrainedTokenizer,
    max_prompt_tokens: int,
    max_pair_tokens:   int,
) -> Dataset:
    """Drop pairs whose prompt or either side exceeds the caps.

    Two caps, both with filter semantics, reproducing the earlier stages'
    data view for a fair PPO-vs-DPO comparison:

    - prompt cap: the shared prompt (everything up to and including the final
      '\\n\\nAssistant:' marker, via the PPO stage's extract_prompt) must fit
      max_prompt_tokens, mirroring the PPO prompt filter.
    - pair cap: each full dialogue with EOS appended must fit
      max_pair_tokens, mirroring the RM stage's _filter_gate_pairs. EOS is
      appended before measuring exactly as DPOTrainer does internally, so the
      length decision matches training tokenisation.
    """
    eos = tokenizer.eos_token

    def _fits(batch: dict) -> list[bool]:
        prompts  = [extract_prompt(text) for text in batch["chosen"]]
        chosen   = [t if t.endswith(eos) else t + eos for t in batch["chosen"]]
        rejected = [t if t.endswith(eos) else t + eos for t in batch["rejected"]]
        prompt_ids   = tokenizer(prompts)["input_ids"]
        chosen_ids   = tokenizer(chosen)["input_ids"]
        rejected_ids = tokenizer(rejected)["input_ids"]
        return [
            len(p) <= max_prompt_tokens
            and len(c) <= max_pair_tokens
            and len(r) <= max_pair_tokens
            for p, c, r in zip(prompt_ids, chosen_ids, rejected_ids)
        ]

    return dataset.filter(
        _fits, batched=True,
        desc=f"Filtering pairs to prompt<={max_prompt_tokens}, pair<={max_pair_tokens}",
    )


def exclude_train_pairs(gate_ds: Dataset, train_raw: Dataset) -> Dataset:
    """Drop any gate pair that also appears, verbatim, in the training split.

    The gate's claim is that no scored pair was trained on. The EDA observes
    zero overlap in HH-RLHF; this turns the observation into a guarantee, on
    raw texts before the prompt split, against the whole training split
    rather than its filtered view. A non-zero count is printed so leakage
    would be visible in the log rather than silently scored.
    """
    train_pairs = set(zip(train_raw["chosen"], train_raw["rejected"]))

    def _unseen(batch: dict) -> list[bool]:
        return [(c, r) not in train_pairs for c, r in zip(batch["chosen"], batch["rejected"])]

    n_before = len(gate_ds)
    gate_ds = gate_ds.filter(_unseen, batched=True,
                             desc="Excluding gate pairs present in the training split")
    n_dropped = n_before - len(gate_ds)
    colour = "yellow" if n_dropped else "green"
    console.print(f"[{colour}]{n_dropped} gate pair(s) also present in the training split, excluded[/{colour}]")
    return gate_ds


def split_prompt(dataset: Dataset) -> Dataset:
    """Add an explicit 'prompt' column and strip it from both sides.

    Given no 'prompt' column, DPOTrainer extracts one as the longest common
    CHARACTER prefix of chosen and rejected. In HH-RLHF that prefix routinely
    runs past the '\\n\\nAssistant:' marker into the shared opening words
    of the two responses, and can end mid-word. The trainer then tokenises
    prompt and prompt+completion separately and slices the completion off
    by prompt length, so a mid-word cut changes the BPE merges at the
    boundary and the sliced completion loses or corrupts its first tokens
    (TRL warns 'Mismatch between tokenized prompt and the start of tokenized
    prompt+chosen').

    Splitting at the marker instead ends the prompt on ':' with the response
    keeping its leading space, which is a pre-tokenisation boundary, so the
    two tokenisations agree. It is also the prompt filter_pairs measured, so
    the prompt cap applies to the object DPOTrainer actually sees. Pairs
    whose rejected side does not share the prompt are transcripts that
    diverge at an earlier assistant turn, i.e. two conversations rather than
    two responses to one prompt; they are dropped rather than mis-split, and
    the count is printed.
    """
    def _shares_prompt(example: dict) -> bool:
        return example["rejected"].startswith(extract_prompt(example["chosen"]))

    def _split(batch: dict) -> dict:
        prompts = [extract_prompt(t) for t in batch["chosen"]]
        return {
            "prompt":   prompts,
            "chosen":   [t[len(p):] for t, p in zip(batch["chosen"], prompts)],
            "rejected": [t[len(p):] for t, p in zip(batch["rejected"], prompts)],
        }

    n_before = len(dataset)
    dataset = dataset.filter(_shares_prompt, desc="Checking both sides share the prompt")
    if len(dataset) != n_before:
        console.print(f"[yellow]{n_before - len(dataset)} pair(s) dropped: rejected side does not share the prompt[/yellow]")
    return dataset.map(_split, batched=True, desc="Splitting prompt from responses")


def _build_trainer(
    policy:    torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    train_ds:  Dataset,
    eval_ds:   Dataset,
    config:    DPOTrainingConfig,
) -> DPOTrainer:
    """Assemble the DPOTrainer with a LoRA policy and no explicit reference."""
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        task_type="CAUSAL_LM",
    )
    # Derive absolute warmup steps from the ratio (warmup_ratio is deprecated in
    # transformers >=5.2). Single-device (MPS), so effective batch = batch x accum.
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    warmup_steps    = round(config.warmup_ratio * steps_per_epoch * config.n_epochs)

    # logging_dir is deprecated; the TensorBoard integration now reads this env var.
    os.environ["TENSORBOARD_LOGGING_DIR"] = f"{RESULT_PATH}/tb/{config.label}"

    dpo_config = DPOConfig(
        output_dir=f"{RESULT_PATH}/checkpoints_{config.label}",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        beta=config.beta,
        loss_type=config.loss_type,
        precompute_ref_log_probs=config.precompute_ref_log_probs,
        # Truncation backstop only: filter_pairs guarantees every surviving
        # sequence fits, so this never binds. (TRL v1 has no filtering cap and
        # no max_prompt_length; truncation is its only length mechanism.)
        max_length=config.max_pair_tokens,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        # Parallelise tokenisation/collation so the GPU is not data-starved.
        dataloader_num_workers=4,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        # Keep the best checkpoint rather than the final one (save_steps must
        # be a multiple of eval_steps). The per-pair loss is monotone in that
        # pair's margin, but mean loss is not monotone in accuracy, so the two
        # can pick different checkpoints; the post-training gate records both.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.report_to,
        seed=config.seed,
    )
    return DPOTrainer(
        model=policy,
        ref_model=None,             # pi_ref = policy with adapters disabled
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[CacheCleaner()],
    )


# ---------------------------------------------------------------------------
# Post-training gate evaluation
# ---------------------------------------------------------------------------

def save_metrics(
    trainer: DPOTrainer,
    gate:    dict,
    n_gate:  int,
    n_test:  int,
    config:  DPOTrainingConfig,
) -> str:
    """Persist the gate evaluation to JSON, keyed by run label.

    The implicit-reward pairwise accuracy over the whole filtered test split
    is DPO's analogue of the RM stage's accuracy gate: the fraction of
    held-out pairs where beta * log(pi_theta / pi_ref) ranks chosen above
    rejected. The chosen and rejected log-probabilities are recorded because
    their joint drift is the likelihood-displacement signal: both falling
    together means probability mass is leaving the preference pair entirely,
    the documented trigger for switching loss_type to 'ipo' or raising beta.
    """
    metrics = {
        "label":                 config.label,
        "gate_accuracy":         gate["eval_rewards/accuracies"],
        "gate_margin":           gate["eval_rewards/margins"],
        "gate_loss":             gate["eval_loss"],
        "gate_logps_chosen":     gate["eval_logps/chosen"],
        "gate_logps_rejected":   gate["eval_logps/rejected"],
        # Implicit rewards on each side; divided by beta, the drift from pi_ref
        # that configurations are compared on.
        "gate_rewards_chosen":   gate["eval_rewards/chosen"],
        "gate_rewards_rejected": gate["eval_rewards/rejected"],
        "beta":                  config.beta,
        "loss_type":             config.loss_type,
        # The gate is the filtered test split minus the in-training evaluation
        # subsample; the counts make the evaluation population auditable.
        "n_eval_pairs":          n_gate,
        "n_eval_subsample_pairs": len(trainer.eval_dataset),
        "n_test_split_pairs":    n_test,
        "length_retention":      (n_gate + len(trainer.eval_dataset)) / n_test,
        # The training set after both caps and the prompt split, i.e. what
        # one epoch iterates over; cross-check against the EDA's kept count.
        "n_train_pairs":         len(trainer.train_dataset),
        "max_prompt_tokens":     config.max_prompt_tokens,
        "max_pair_tokens":       config.max_pair_tokens,
        "sft_model_path":        config.sft_model_path,
        "dataset_name":          config.dataset_name,
        "global_step":           trainer.state.global_step,
        # The weights the gate scored: load_best_model_at_end restores this
        # checkpoint, which need not be the final step.
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "timestamp_utc":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = f"{RESULT_PATH}/metrics_{config.label}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def parse_config(argv: list[str] | None = None) -> DPOTrainingConfig:
    """Build a DPOTrainingConfig from the command line or a JSON config file.

    Two invocation styles are supported, so experiments (e.g. the beta
    sensitivity sweep) are driven by config rather than by editing the
    source:

        python -m src.pipeline.dpo_lora_hh --beta 0.05
        python -m src.pipeline.dpo_lora_hh configs/dpo_default.json

    A single argument ending in '.json' is read as a config file; otherwise the
    arguments are parsed as CLI overrides. With no arguments, the dataclass
    defaults are used.
    """
    argv = sys.argv[1:] if argv is None else argv
    parser = HfArgumentParser(DPOTrainingConfig)
    if len(argv) == 1 and argv[0].endswith(".json"):
        (config,) = parser.parse_json_file(os.path.abspath(argv[0]))
    else:
        (config,) = parser.parse_args_into_dataclasses(argv)
    return config


def save_config(config: DPOTrainingConfig) -> str:
    """Persist the resolved config to JSON for reproducibility. Returns the path."""
    path = f"{RESULT_PATH}/config_{config.label}.json"
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULT_PATH, exist_ok=True)

    config = parse_config()
    config_path = save_config(config)
    console.print(
        f"DPO run [bold]{config.label}[/bold]: policy from {config.sft_model_path}, "
        f"beta={config.beta}, loss={config.loss_type}"
    )
    console.print(f"[green]Resolved config saved to[/green] {config_path}")

    trainer, gate_ds, n_test = train(config)

    # Post-training gate: implicit-reward pairwise accuracy over the WHOLE
    # filtered test split (the periodic in-training evaluation used only a
    # subsample, whose precision suffices for checkpoint selection but not for
    # the run's record). Re-running an already-completed config recomputes
    # exactly this block on the best checkpoint without retraining, via the
    # same resume mechanism as the RM stage.
    console.print(
        f"Gate set: {len(gate_ds):,} of {n_test:,} test pairs (within caps, prompt-split, "
        f"unseen in training, and outside the in-training evaluation subsample)"
    )
    # DPOTrainer tokenises its datasets in __init__ only, so a dataset handed
    # to evaluate() later must be put through the same preparation explicitly
    # (prompt extraction, EOS append, tokenisation). _prepare_dataset is
    # TRL-private, but the alternative -- registering the full gate split as a
    # second init-time eval_dataset -- would re-score all of it every
    # eval_steps; verified against the pinned TRL v1 source.
    gate_prepared = trainer._prepare_dataset(
        gate_ds, trainer.processing_class, trainer.args, "gate"
    )
    gate = trainer.evaluate(eval_dataset=gate_prepared)
    # TRL's DPOTrainer.log rebinds its `logs` argument rather than mutating it
    # (dpo_trainer.py line 1482 in the pinned version), so the DPO metrics reach
    # the callbacks and TensorBoard but never the dict evaluate() returns. The
    # callbacks did receive them, so merge them back from the last state entry.
    # Subscript rather than .get() throughout: a missing key must fail here,
    # not travel on as None into the printed line and the metrics file.
    gate = {**gate, **trainer.state.log_history[-1]}
    accuracy = gate["eval_rewards/accuracies"]
    margin   = gate["eval_rewards/margins"]
    console.print(
        f"Held-out implicit-reward accuracy over {len(gate_ds):,} pairs: "
        f"[bold]{accuracy:.3f}[/bold] (mean margin {margin:.3f}) -- "
        f"the curriculum's expectation band is 0.6-0.7"
    )

    metrics_path = save_metrics(trainer, gate, len(gate_ds), n_test, config)
    console.print(f"[green]Metrics saved to[/green] {metrics_path}")

    # Opt-in canonical export, as in every stage: sweeping beta must not
    # silently overwrite dpo-model with whichever run happened last.
    if os.environ.get("EXPORT_CANONICAL") == "1":
        export_canonical(trainer, DPO_ADAPTER)
        console.print(f"[green]DPO policy adapter exported to canonical path[/green] {DPO_ADAPTER}")
    else:
        console.print(
            f"[dim]EXPORT_CANONICAL not set[/dim] -- canonical path {DPO_ADAPTER} left "
            f"untouched (promote later with: uv run rlhf-promote dpo {config.label})"
        )


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - train: resolve the SFT backbone (merging its LoRA adapter via
#   model_utils.resolve_model_path), load and cap-filter the HH-RLHF pairs,
#   fit with DPOTrainer, save the adapter.
# - One backbone, two policies: the policy is a LoRA PEFT model and
#   ref_model=None, so DPOTrainer computes pi_ref by disabling the adapters —
#   the same memory trick as the PPO stage, and exact for the same reason
#   (a fresh adapter contributes Delta W = BA = 0, so the adapter-disabled
#   policy IS the merged SFT model).
# - Filtering, not truncating: TRL v1's DPOConfig.max_length TRUNCATES
#   (keep_start), which would corrupt pairs whose sides differ mainly in the
#   final assistant turn. filter_pairs therefore drops, before the trainer
#   sees them, any pair whose prompt exceeds 256 tokens (PPO's cap) or whose
#   either side with EOS exceeds 512 (the RM's cap); max_length is kept only
#   as a backstop that never binds.
# - Comparability: the caps and the LoRA setup (32/64/q,v) exist so the
#   PPO-vs-DPO comparison is not confounded by data view or capacity. The
#   default beta is the arm compared against PPO, fixed in advance; other
#   beta values are a sensitivity check, never a pool to select the arm from.
# - Gate: after training, trainer.evaluate over the whole filtered test split
#   records the implicit-reward pairwise accuracy (DPO's analogue of the RM
#   accuracy gate; expectation band 0.6-0.7), plus the chosen/rejected
#   log-probabilities whose joint fall is the likelihood-displacement signal
#   (switch loss_type to 'ipo' or raise beta if it fires).
# - Resume + best checkpoint: DPOTrainer is a standard Trainer subclass, so
#   unlike the experimental PPOTrainer it resumes from interruption and keeps
#   the best (lowest eval-loss) checkpoint; eval loss is monotone in the
#   implicit margin, so it is a sound selection metric.
# - Canonical export: opt-in via EXPORT_CANONICAL=1 or post-hoc via
#   `rlhf-promote dpo <label>`, as in every other stage.
# =============================================================================
