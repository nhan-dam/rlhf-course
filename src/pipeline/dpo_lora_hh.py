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
example holds 'chosen' and 'rejected' full-dialogue texts, consumed directly
by trl.DPOTrainer, which extracts the shared prompt and appends EOS itself.

Public API
----------
train(config)   — run DPO; return (trainer, gate_dataset, n_test).
filter_pairs(dataset, tokenizer, max_prompt_tokens, max_pair_tokens)
                — drop pairs violating the prompt or pair length caps.
"""

# stdlib
import hashlib
import json
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

    # DPO objective. beta prices drift from pi_ref inside the implicit reward;
    # it is the sweep axis for the matched-KL comparison against PPO (the two
    # coefficients are not numerically comparable, so runs are compared on the
    # reward-vs-KL plane instead). loss_type='ipo' is the documented switch if
    # likelihood displacement appears (logps/chosen falling alongside
    # logps/rejected).
    beta:      float = 0.1
    loss_type: str   = "sigmoid"

    # Optimisation — 5e-7, two to three orders below SFT values: the implicit
    # rewards are extremely sensitive to log-probability changes, and larger
    # steps reliably destabilise DPO.
    learning_rate:               float = 5e-7
    n_epochs:                    int   = 1
    per_device_train_batch_size: int   = 4
    per_device_eval_batch_size:  int   = 2   # small to cap the eval-time logit peak
    gradient_accumulation_steps: int   = 4   # effective batch size = 4 x 4 = 16
    # Off by default like the RM baseline: one 0.5B backbone with adapters and
    # <=512-token sequences is not memory-bound; enable for capacity sweeps.
    gradient_checkpointing:      bool  = False

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
        gate_ds: The held-out test split filtered to cap-admissible pairs,
                 for the post-training gate evaluation (training-time
                 evaluation uses a seeded subsample of it).
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

    The filtered test split then serves two roles, as in the RM stage:

    - eval_ds: a seeded eval_examples-pair subsample, evaluated every
      eval_steps during training (cheap, enough precision for checkpoint
      selection).
    - gate_ds: the whole filtered split, scored once post-training where
      precision matters.

    Returns (train_ds, eval_ds, gate_ds, n_test_total), the last being the
    unfiltered test-split size for the retention record.
    """
    train_raw = load_dataset(config.dataset_name, split="train")
    test_raw  = load_dataset(config.dataset_name, split="test")
    train_ds  = filter_pairs(train_raw, tokenizer, config.max_prompt_tokens, config.max_pair_tokens)
    gate_ds   = filter_pairs(test_raw,  tokenizer, config.max_prompt_tokens, config.max_pair_tokens)
    eval_ds   = gate_ds.shuffle(seed=config.seed).select(
        range(min(config.eval_examples, len(gate_ds)))
    )
    console.print(
        f"Length filter (prompt<={config.max_prompt_tokens}, pair<={config.max_pair_tokens}): "
        f"train {len(train_ds):,}/{len(train_raw):,} "
        f"({100 * len(train_ds) / len(train_raw):.1f}% kept), "
        f"test {len(gate_ds):,}/{len(test_raw):,} "
        f"({100 * len(gate_ds) / len(test_raw):.1f}% kept)"
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
    # logging_dir is deprecated; the TensorBoard integration now reads this env var.
    os.environ["TENSORBOARD_LOGGING_DIR"] = f"{RESULT_PATH}/tb/{config.label}"

    dpo_config = DPOConfig(
        output_dir=f"{RESULT_PATH}/checkpoints_{config.label}",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        loss_type=config.loss_type,
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
        # be a multiple of eval_steps). The DPO eval loss is monotone in the
        # implicit reward margin, so lower loss tracks higher pairwise
        # accuracy and is a sound selection metric; the post-training gate
        # evaluation remains the real acceptance record.
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
        "gate_accuracy":         gate.get("eval_rewards/accuracies"),
        "gate_margin":           gate.get("eval_rewards/margins"),
        "gate_loss":             gate.get("eval_loss"),
        "gate_logps_chosen":     gate.get("eval_logps/chosen"),
        "gate_logps_rejected":   gate.get("eval_logps/rejected"),
        "beta":                  config.beta,
        "loss_type":             config.loss_type,
        # The gate filters the test split to cap-admissible pairs (parity with
        # training); the counts make the evaluation population auditable.
        "n_eval_pairs":          n_gate,
        "n_test_split_pairs":    n_test,
        "length_retention":      n_gate / n_test,
        "max_prompt_tokens":     config.max_prompt_tokens,
        "max_pair_tokens":       config.max_pair_tokens,
        "sft_model_path":        config.sft_model_path,
        "dataset_name":          config.dataset_name,
        "global_step":           trainer.state.global_step,
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

    Two invocation styles are supported, so experiments (e.g. the beta sweep
    for the matched-KL comparison against PPO) are driven by config rather
    than by editing the source:

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
        f"Gate set: {len(gate_ds):,} of {n_test:,} test pairs within caps "
        f"(prompt<={config.max_prompt_tokens}, pair<={config.max_pair_tokens})"
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
    accuracy = gate.get("eval_rewards/accuracies")
    margin   = gate.get("eval_rewards/margins")
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
#   PPO-vs-DPO comparison is not confounded by data view or capacity; beta is
#   the sweep axis, and runs are compared on the reward-vs-KL plane rather
#   than at equal coefficients.
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
