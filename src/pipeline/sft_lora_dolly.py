"""
Supervised Fine-Tuning with LoRA on Dolly-15k
==============================================
Implements the SFT stage of the classical RLHF pipeline (Stiennon et al., 2020;
Ouyang et al., 2022): fine-tune a pre-trained causal language model on an
instruction dataset to produce the reference policy pi_ref used by the
subsequent reward-modelling and PPO stages. Low-Rank Adaptation (LoRA, Hu et
al., 2021) keeps the base weights frozen and trains only low-rank adapter
matrices injected into the attention projections.

Inputs
------
config : SFTTrainingConfig — base model name, LoRA rank/alpha/dropout/targets,
         learning rate, epochs, batch size, gradient accumulation, sequence
         length, validation fraction, and logging cadence.

Outputs
-------
LoRA adapter weights (and tokenizer) saved to ./results/sft_lora_dolly/
adapter_<label>/; training and validation loss logged to TensorBoard;
qualitative sample generations printed for held-out prompts. If the
EXPORT_CANONICAL environment variable is set to '1', the adapter is also
exported to the pipeline-shared SFT_ADAPTER path that the reward-modelling
stage reads by default (opt-in, since every run would otherwise overwrite it
regardless of whether that run was the one actually chosen).

Dataset
-------
databricks/databricks-dolly-15k — 15,011 human-written (instruction, context,
response) triples, permissively licensed. Mapped to prompt/completion pairs so
the loss is computed on response tokens only.

Public API
----------
train(config)                  — run SFT; return (trainer, tokenizer, val_dataset).
sample_responses(model, tokenizer, prompts, max_new_tokens)
                               — generate responses for qualitative inspection.
format_prompt(instruction, context) — render one example into the prompt template.
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
from peft import LoraConfig, get_peft_model
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

# local
from ..common.config import BASE_MODEL, PROJECT_ROOT, SFT_ADAPTER
from ..common.model_utils import CacheCleaner, export_canonical

RESULT_PATH = f"{PROJECT_ROOT}/results/sft_lora_dolly"

console = Console()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SFTTrainingConfig:
    model_name:   str = BASE_MODEL  # shared across the pipeline; see config.py
    dataset_name: str = "databricks/databricks-dolly-15k"
    seed:         int = 42

    # LoRA — applied to the attention query/value projections (Hu et al., 2021)
    lora_r:              int   = 32
    lora_alpha:          int   = 64       # alpha = 2r is the conventional scaling
    lora_dropout:        float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Optimisation
    learning_rate:               float = 2e-4
    n_epochs:                    int   = 3
    per_device_train_batch_size: int   = 8
    per_device_eval_batch_size:  int   = 2
    gradient_accumulation_steps: int   = 2    # effective batch size = 8 x 2 = 16
    gradient_checkpointing:      bool  = True  # trades recompute for activation memory
    warmup_ratio:                float = 0.03
    # Cap on the (prompt + completion) sequence. Longer sequences are truncated
    # (keep_start); shorter ones are not padded up to this. Batches are padded
    # dynamically to the longest member, not to max_length.
    max_length:                  int   = 512

    # Evaluation and logging
    val_fraction:  float = 0.05
    logging_steps: int   = 10
    eval_steps:    int   = 200
    save_steps:    int   = 200
    report_to:     str   = "tensorboard"

    def __post_init__(self) -> None:
        # Fail fast at construction rather than mid-download or mid-training.
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError(
                f"val_fraction ({self.val_fraction}) must be in (0, 0.5): the "
                f"validation split exists to detect overfitting, not to starve training."
            )
        if self.lora_r <= 0:
            raise ValueError(f"lora_r ({self.lora_r}) must be positive.")

    @property
    def label(self) -> str:
        # Hash the FULL config so any hyperparameter change yields a distinct
        # label and its own results directory, preventing different experiments
        # from silently overwriting each other's artefacts.
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: SFTTrainingConfig) -> tuple[SFTTrainer, PreTrainedTokenizer, Dataset]:
    """Run the full SFT-with-LoRA pipeline and save the adapter.

    The function reads as the algorithm outline: load the frozen base model,
    inject trainable LoRA adapters, prepare prompt/completion data, train with
    completion-only loss, and persist the adapter weights.

    Args:
        config: Hyperparameter configuration.

    Returns:
        trainer:   The fitted SFTTrainer (gives access to log history and model).
        tokenizer: Tokenizer with padding configured.
        val_ds:    Held-out split, for qualitative probing after training.
    """
    tokenizer        = _load_tokenizer(config.model_name)
    model            = _load_base_model(config.model_name)
    model            = _attach_lora(model, config)
    train_ds, val_ds = _build_datasets(config)
    trainer          = _build_trainer(model, tokenizer, train_ds, val_ds, config)

    # Resume from the latest checkpoint if one exists, else start fresh.
    output_dir = trainer.args.output_dir
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint:
        console.print(f"[yellow]Resuming from checkpoint[/yellow] {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # trainer.save_model() writes three things to the target directory: the LoRA
    # adapter (adapter_model.safetensors + adapter_config.json), the tokenizer
    # files (the trainer's processing_class), and training_args.bin. So no
    # separate tokenizer save is needed.
    adapter_path = f"{RESULT_PATH}/adapter_{config.label}"
    trainer.save_model(adapter_path)
    console.print(f"[green]LoRA adapter saved to[/green] {adapter_path}")

    return trainer, tokenizer, val_ds


def _load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load the tokenizer and configure padding.

    Some tokenizers (e.g. Llama) ship without a pad token; the fallback reuses EOS
    in that case. Qwen2.5 provides a pad token, so the fallback does not fire here.
    Reusing EOS would be safe regardless, because the completion-only loss masks
    padded positions out of the objective anyway.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(model_name: str) -> PreTrainedModel:
    """Load the frozen base model in bfloat16."""
    return AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)


def _attach_lora(model: PreTrainedModel, config: SFTTrainingConfig) -> PreTrainedModel:
    """Freeze base weights and inject trainable low-rank adapters.

    Targeting q_proj and v_proj follows Hu et al. (2021), who found adapting
    the query/value projections gives the best quality per trainable parameter.
    """
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def format_prompt(instruction: str, context: str) -> str:
    """Render one Dolly example into the instruction-following prompt template.

    The 'Context' block is omitted entirely when empty (roughly 60% of Dolly)
    rather than left as an empty header, so the model never learns to expect
    a vacuous section.
    """
    context_block = f"### Context:\n{context}\n\n" if context else ""
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"{context_block}"
        "### Response:\n"
    )


def _build_datasets(config: SFTTrainingConfig) -> tuple[Dataset, Dataset]:
    """Load Dolly-15k, map to prompt/completion pairs, and split off validation.

    The prompt/completion column format is what tells SFTTrainer where the
    prompt ends, enabling completion-only loss. Dolly ships a single 'train'
    split, so a held-out fraction is carved out here for overfitting detection.
    """
    dataset = load_dataset(config.dataset_name, split="train")
    dataset = dataset.map(
        lambda example: {
            "prompt":     format_prompt(example["instruction"], example["context"]),
            "completion": example["response"],
        },
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=config.val_fraction, seed=config.seed)
    return split["train"], split["test"]


def _build_trainer(
    model:     PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_ds:  Dataset,
    val_ds:    Dataset,
    config:    SFTTrainingConfig,
) -> SFTTrainer:
    """Assemble the SFTTrainer with completion-only loss and periodic evaluation."""
    # Derive absolute warmup steps from the ratio (warmup_ratio is deprecated in
    # transformers >=5.2). Single-device (MPS), so effective batch = batch x accum.
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    total_steps     = steps_per_epoch * config.n_epochs
    warmup_steps    = round(config.warmup_ratio * total_steps)

    # logging_dir is deprecated; the TensorBoard integration now reads this env var.
    os.environ["TENSORBOARD_LOGGING_DIR"] = f"{RESULT_PATH}/tb/{config.label}"

    sft_config = SFTConfig(
        output_dir=f"{RESULT_PATH}/checkpoints_{config.label}",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        max_length=config.max_length,
        completion_only_loss=True,   # gradient signal on response tokens only
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        # Keep the lowest eval-loss checkpoint rather than the final one, so the
        # exported adapter is the best on the held-out split (save_steps must be a
        # multiple of eval_steps for this to work).
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.report_to,
        seed=config.seed,
    )
    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[CacheCleaner()],
    )


# ---------------------------------------------------------------------------
# Qualitative evaluation
# ---------------------------------------------------------------------------

def sample_responses(
    model:          PreTrainedModel,
    tokenizer:      PreTrainedTokenizer,
    prompts:        list[str],
    max_new_tokens: int = 256,
    do_sample:      bool = True,
) -> list[str]:
    """Generate one response per prompt for qualitative inspection.

    Sampling (the default) matches how the SFT model will be used as the policy
    in the PPO stage, so the inspection previews the actual generation
    distribution. Greedy decoding (do_sample=False) is deterministic and is used
    for the base-vs-SFT comparison, where randomness would obscure the effect of
    fine-tuning.

    Args:
        model:          The fine-tuned (or base, for comparison) model.
        tokenizer:      Tokenizer with padding configured.
        prompts:        Already-formatted prompt strings (see format_prompt).
        max_new_tokens: Generation budget per response.
        do_sample:      Sample (top-p/temperature) if True, else greedy decoding.

    Returns:
        Decoded responses, excluding the prompt tokens.
    """
    model.eval()
    gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
    if do_sample:
        gen_kwargs.update(do_sample=True, top_p=0.9, temperature=0.7)
    else:
        gen_kwargs["do_sample"] = False
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def probe_sft_responses(model, tokenizer, examples) -> list[dict]:
    """Sample one response per prompt from the fine-tuned model and pair it with the
    human reference. Sampling previews the generation distribution used in the PPO
    stage, so this is the practical-use view of the model.
    """
    prompts = [ex["prompt"] for ex in examples]
    responses = sample_responses(model, tokenizer, prompts, do_sample=True)
    return [
        {"prompt": p, "model_response": r, "reference_response": ex["completion"]}
        for p, r, ex in zip(prompts, responses, examples)
    ]


def compare_base_and_sft(model, tokenizer, examples) -> list[dict]:
    """Greedily decode each prompt from the base model (LoRA adapter disabled) and
    the fine-tuned model (adapter enabled), for a like-for-like before/after view.

    Uses one model in memory via the PEFT adapter toggle, so no second base-model
    copy is loaded.
    """
    prompts = [ex["prompt"] for ex in examples]
    with model.disable_adapter():
        base_responses = sample_responses(model, tokenizer, prompts, do_sample=False)
    sft_responses = sample_responses(model, tokenizer, prompts, do_sample=False)
    return [
        {
            "prompt": p,
            "base_response": b,
            "sft_response": s,
            "reference_response": ex["completion"],
        }
        for p, b, s, ex in zip(prompts, base_responses, sft_responses, examples)
    ]


# ---------------------------------------------------------------------------
# Configuration loading and run records
# ---------------------------------------------------------------------------

def parse_config(argv: list[str] | None = None) -> SFTTrainingConfig:
    """Build an SFTTrainingConfig from the command line or a JSON config file.

    A single argument ending in '.json' is read as a config file; otherwise the
    arguments are parsed as CLI overrides. With no arguments, the dataclass
    defaults are used. This keeps experiments config-driven rather than requiring
    source edits.
    """
    argv = sys.argv[1:] if argv is None else argv
    parser = HfArgumentParser(SFTTrainingConfig)
    if len(argv) == 1 and argv[0].endswith(".json"):
        (config,) = parser.parse_json_file(os.path.abspath(argv[0]))
    else:
        (config,) = parser.parse_args_into_dataclasses(argv)
    return config


def save_config(config: SFTTrainingConfig) -> str:
    """Persist the resolved config to JSON for reproducibility. Returns the path."""
    path = f"{RESULT_PATH}/config_{config.label}.json"
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    return path


def save_metrics(trainer: SFTTrainer, config: SFTTrainingConfig) -> str:
    """Persist the held-out training metrics to JSON, keyed by run label.

    The best (lowest) validation loss is the model-selection criterion, so it is
    recorded alongside its perplexity and the final logged losses, mirroring the
    per-run metrics written by the other pipeline stages.
    """
    state = trainer.state
    evals = [e for e in state.log_history if "eval_loss" in e]
    trains = [e for e in state.log_history if "loss" in e and "eval_loss" not in e]
    best_eval_loss = state.best_metric
    metrics = {
        "label":               config.label,
        "best_eval_loss":      best_eval_loss,
        "best_eval_perplexity": math.exp(best_eval_loss) if best_eval_loss is not None else None,
        "final_eval_loss":     evals[-1]["eval_loss"] if evals else None,
        "final_train_loss":    trains[-1]["loss"] if trains else None,
        "best_checkpoint":     state.best_model_checkpoint,
        "global_step":         state.global_step,
        "model_name":          config.model_name,
        "dataset_name":        config.dataset_name,
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = f"{RESULT_PATH}/metrics_{config.label}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULT_PATH, exist_ok=True)

    config = parse_config()
    config_path = save_config(config)
    console.print(f"SFT run [bold]{config.label}[/bold]: {config.model_name} on {config.dataset_name}")
    console.print(f"[green]Resolved config saved to[/green] {config_path}")

    trainer, tokenizer, validation_set = train(config)

    metrics_path = save_metrics(trainer, config)
    console.print(f"[green]Metrics saved to[/green] {metrics_path}")

    # Exporting to the canonical path is opt-in (see model_utils.export_canonical):
    # every run would otherwise overwrite sft-model regardless of whether it was
    # the config actually chosen after comparing metrics across runs. Re-running
    # an already-completed config with EXPORT_CANONICAL=1 resumes instantly
    # (resume_from_checkpoint finds nothing left to train), loads the best
    # checkpoint via load_best_model_at_end, and exports that.
    if os.environ.get("EXPORT_CANONICAL") == "1":
        export_canonical(trainer, SFT_ADAPTER)
        console.print(f"[green]LoRA adapter exported to canonical path[/green] {SFT_ADAPTER}")
    else:
        console.print(
            f"[dim]EXPORT_CANONICAL not set[/dim] -- canonical path {SFT_ADAPTER} left "
            f"untouched (set EXPORT_CANONICAL=1 to promote this run)"
        )

    # Qualitative probe on held-out prompts: a well-tuned SFT model should
    # produce coherent, on-format responses before reward modelling begins.
    n_probe        = 10
    probe_examples = validation_set.select(range(n_probe))
    
    probe_records  = probe_sft_responses(trainer.model, tokenizer, probe_examples)
    probe_path = f"{RESULT_PATH}/inference_sft_ref_{config.label}.json"
    with open(probe_path, "w", encoding="utf-8") as fout:
        json.dump(probe_records, fout, indent=2, ensure_ascii=False)
    console.print(f"Wrote {len(probe_records)} inference samples to [bold]{probe_path}[/bold]")

    # Show the first few on screen.
    for record in probe_records[:3]:
        console.rule()
        console.print(f"[bold cyan]Prompt[/bold cyan]\n{record['prompt']}")
        console.print(f"[bold green]Model response[/bold green]\n{record['model_response']}")
        console.print(f"[bold yellow]Reference response[/bold yellow]\n{record['reference_response']}")

    # Pre-/post-SFT comparison: greedy decoding on the same held-out prompts, with
    # the LoRA adapter disabled (base) then enabled (fine-tuned), so the contrast
    # isolates the effect of fine-tuning rather than sampling noise.
    comparison_records = compare_base_and_sft(trainer.model, tokenizer, probe_examples)
    comparison_path = f"{RESULT_PATH}/comparison_base_sft_{config.label}.json"
    with open(comparison_path, "w", encoding="utf-8") as fout:
        json.dump(comparison_records, fout, indent=2, ensure_ascii=False)
    console.print(f"Wrote {len(comparison_records)} base-vs-SFT comparisons to [bold]{comparison_path}[/bold]")

    for record in comparison_records[:3]:
        console.rule()
        console.print(f"[bold cyan]Prompt[/bold cyan]\n{record['prompt']}")
        console.print(f"[bold red]Base (pre-SFT)[/bold red]\n{record['base_response']}")
        console.print(f"[bold green]SFT (post-SFT)[/bold green]\n{record['sft_response']}")
        console.print(f"[bold yellow]Reference[/bold yellow]\n{record['reference_response']}")


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - train: reads as the pipeline outline — load tokenizer and frozen bf16 base
#   model, inject LoRA adapters, build prompt/completion datasets, train with
#   SFTTrainer, save the adapter (a few hundred MB rather than the full model).
# - LoRA: only low-rank matrices B·A injected into q_proj/v_proj are trainable
#   (r=32, alpha=64); base weights stay frozen, cutting trainable parameters
#   to well under 1% of the base model's parameters and fitting comfortably in
#   unified memory.
# - Completion-only loss: the dataset is mapped to explicit prompt/completion
#   columns and SFTConfig(completion_only_loss=True) masks prompt tokens out
#   of the cross-entropy, focusing the gradient on generation behaviour.
# - Prompt template: Dolly's optional context field is dropped (not left as an
#   empty header) when absent, so the model never conditions on vacuous text.
# - Padding: if the tokenizer has no pad token, EOS is reused, which is harmless
#   because padded positions are excluded from the loss by the completion-only mask.
# - Validation: Dolly has no eval split, so 5% is held out (seeded) for
#   periodic eval loss — flat validation loss with falling training loss is
#   the overfitting signal that says to reduce epochs or raise dropout.
# - Qualitative probe: after training, sampled generations on held-out prompts
#   are printed beside the human references, previewing the policy that the
#   PPO stage will start from.
# - Canonical export: opt-in via EXPORT_CANONICAL=1 (main(), not train()), so
#   sweeping configs does not silently overwrite sft-model with whichever run
#   happened last. Promoting a completed run is just re-running its exact
#   config with the flag set: same label, same checkpoints_<label>, so
#   trainer.train(resume_from_checkpoint=...) finds nothing left to train and
#   load_best_model_at_end loads the best checkpoint before export.
# =============================================================================
