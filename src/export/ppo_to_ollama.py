"""
Export a PPO Policy Adapter for Local Testing in Ollama
=========================================================
Converts a PPO-stage LoRA adapter (see src/pipeline/ppo_rlhf_loop.py) into a
GGUF model plus an Ollama Modelfile, so the aligned policy can be tried
interactively with `ollama run` rather than only through the training
pipeline.

Ollama can import safetensors directly with `FROM <directory>`, but only for
a handful of architectures (Llama, Mistral, Gemma, Phi3); Qwen2 -- this
project's base model -- is not among them. So this script takes the
llama.cpp route that Ollama's own documentation points to for everything
else: merge the LoRA adapter into the base model, convert the merged model
to GGUF with llama.cpp's convert_hf_to_gguf.py, optionally quantize it, and
write a Modelfile that points Ollama at the result.

This is a one-off packaging step, not a pipeline stage: it has no config
label of its own and reads an already-trained PPO adapter rather than
training anything.

Inputs
------
--label LABEL or --adapter-path PATH : which PPO run to export. Both are
    optional: with neither, the promoted policy at ppo-model (populated by
    `rlhf-promote ppo <label>`, see README Section 7) is exported, and the
    run label is recovered from its promoted_from.json provenance record.
    --label resolves to results/ppo_rlhf_loop/adapter_<label> (the directory
    ppo_rlhf_loop.py's train() saves to); --adapter-path points at that
    directory (or any other LoRA adapter directory) directly.
--llama-cpp-dir PATH  : local llama.cpp checkout, for convert_hf_to_gguf.py
    and (optionally) the llama-quantize binary. Falls back to the
    LLAMA_CPP_DIR environment variable, then '<repo>/../llama.cpp'.
--quantize TYPE        : optional GGUF quantization (e.g. q4_K_M). Skipped
    (fp16 GGUF kept) if omitted, or if llama-quantize is not found -- for a
    0.5B model fp16 is already under 1 GB, so quantization is a convenience,
    not a requirement.
--ollama-name NAME     : model name to register with Ollama. Defaults to
    '<base-model>-<dataset>-ppo-<label>' (see default_ollama_name), so a
    generic name does not collide once this project trains PPO on a
    different base model or dataset.
--no-create            : write the GGUF and Modelfile but skip invoking
    `ollama create` (e.g. if Ollama is not installed on this machine yet).

Outputs
-------
results/ppo_rlhf_loop/ollama_<label>/model-f16.gguf (or model-<quantize>.gguf)
results/ppo_rlhf_loop/ollama_<label>/Modelfile
An Ollama model registered under the resolved name, unless --no-create is
passed.

Public API
----------
resolve_adapter_path(args)  -- turn --label/--adapter-path into a directory,
                                erroring clearly if no PPO adapter exists yet.
merge_adapter(adapter_path) -- merge the LoRA adapter into the base model.
convert_to_gguf(...)        -- run llama.cpp's HF -> GGUF converter.
quantize_gguf(...)          -- optionally quantize with llama-quantize.
write_modelfile(...)        -- render the Ollama Modelfile.
default_ollama_name(...)    -- build a collision-resistant default model name.
create_ollama_model(...)    -- run `ollama create`, if the CLI is present.
"""

# stdlib
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

# third-party
from rich.console import Console

# local
from ..common.config import BASE_MODEL, PROJECT_ROOT, PPO_ADAPTER
from ..common.model_utils import resolve_model_path

RESULT_PATH = f"{PROJECT_ROOT}/results/ppo_rlhf_loop"

# Mirrors PPORunConfig's generation defaults (src/pipeline/ppo_rlhf_loop.py),
# so a quick Ollama test approximates the sampling behaviour seen during
# training rather than Ollama's own defaults.
DEFAULT_TEMPERATURE = 0.7

# The HH-RLHF turn marker (see extract_prompt() in ppo_rlhf_loop.py). Stopping
# here keeps the model from inventing a new "Human:" turn and answering it
# itself, which raw completion models tend to do once given a chat-style REPL.
STOP_SEQUENCE = "\n\nHuman:"

console = Console()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Merge, convert, package, and register a completed PPO run in Ollama."""
    args = parse_args()

    adapter_path = resolve_adapter_path(args)
    console.print(f"Exporting PPO adapter: [bold]{adapter_path}[/bold]")

    merged_path = merge_adapter(adapter_path)
    console.print(f"[green]Merged model at[/green] {merged_path}")

    export_dir = f"{RESULT_PATH}/ollama_{_label_from_adapter_path(adapter_path)}"
    os.makedirs(export_dir, exist_ok=True)

    gguf_path = convert_to_gguf(merged_path, export_dir, args.llama_cpp_dir)
    console.print(f"[green]Converted to GGUF[/green] {gguf_path}")

    if args.quantize:
        gguf_path = quantize_gguf(gguf_path, export_dir, args.quantize, args.llama_cpp_dir)
        console.print(f"[green]Quantized ({args.quantize}) GGUF[/green] {gguf_path}")

    modelfile_path = write_modelfile(gguf_path, export_dir)
    console.print(f"[green]Modelfile written to[/green] {modelfile_path}")

    ollama_name = args.ollama_name or default_ollama_name(adapter_path)
    if args.no_create:
        print_manual_instructions(ollama_name, modelfile_path)
    else:
        create_ollama_model(ollama_name, modelfile_path)


# ---------------------------------------------------------------------------
# Argument parsing and adapter resolution
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Plain argparse, not HfArgumentParser: this is a one-off packaging utility,
    not a config-driven training run, so it has no dataclass and no config
    label of its own.
    """
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--label",
        help="PPO run label; resolves to results/ppo_rlhf_loop/adapter_<label>. "
        "Omit both source flags to export the promoted policy at ppo-model.",
    )
    source.add_argument(
        "--adapter-path", help="Explicit path to a LoRA adapter directory."
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Local llama.cpp checkout. Falls back to LLAMA_CPP_DIR, then '<repo>/../llama.cpp'.",
    )
    parser.add_argument(
        "--quantize",
        default=None,
        help="GGUF quantization type for llama-quantize (e.g. q4_K_M). Omit to keep fp16.",
    )
    parser.add_argument(
        "--ollama-name",
        default=None,
        help="Model name to register with Ollama. Defaults to '<base-model>-<dataset>-ppo-<label>'.",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Write the GGUF and Modelfile but skip invoking `ollama create`.",
    )
    return parser.parse_args(argv)


def resolve_adapter_path(args: argparse.Namespace) -> str:
    """Turn --label/--adapter-path into a validated adapter directory.

    With neither flag, the promoted policy at ppo-model is used, mirroring
    how the pipeline's other consumers read the canonical paths by default.

    Args:
        args: Parsed CLI arguments (see parse_args).

    Returns:
        Absolute path to a directory containing adapter_config.json.

    Raises:
        FileNotFoundError: If no adapter exists there yet, with a message
            pointing at how to train and promote first -- this script is
            meant to be created and committed before a PPO run has ever
            completed.
    """
    if args.adapter_path:
        adapter_path = os.path.abspath(args.adapter_path)
    elif args.label:
        adapter_path = f"{RESULT_PATH}/adapter_{args.label}"
    else:
        adapter_path = PPO_ADAPTER

    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(
            f"No LoRA adapter found at {adapter_path}. Train PPO first "
            f"(`uv run rlhf-ppo` or a configs/ppo_*.json variant), then either "
            f"promote the chosen run (`uv run rlhf-promote ppo <label>`) so that "
            f"{PPO_ADAPTER} exists, or pass --label <label> / --adapter-path "
            f"<directory> to export an unpromoted run directly."
        )
    return adapter_path


# ---------------------------------------------------------------------------
# Merge and conversion
# ---------------------------------------------------------------------------

def merge_adapter(adapter_path: str) -> str:
    """Merge the LoRA adapter into the base model.

    Reuses resolve_model_path from model_utils.py, the same helper the RM and
    PPO stages use internally, so the merged model and its caching (a
    '<adapter_path>-merged' sibling, reused if present) match the rest of the
    pipeline.
    """
    return resolve_model_path(adapter_path, "causal-lm")


def convert_to_gguf(merged_path: str, export_dir: str, llama_cpp_dir_arg: str | None) -> str:
    """Run llama.cpp's HF -> GGUF converter on the merged model.

    Args:
        merged_path:       Plain (non-PEFT) HF model directory from merge_adapter.
        export_dir:        Directory to write the GGUF file into.
        llama_cpp_dir_arg: --llama-cpp-dir value, or None to use the
            fallbacks in _locate_llama_cpp.

    Returns:
        Path to the written fp16 GGUF file.
    """
    llama_cpp_dir = _locate_llama_cpp(llama_cpp_dir_arg)
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    out_path = f"{export_dir}/model-f16.gguf"
    subprocess.run(
        [sys.executable, convert_script, merged_path, "--outfile", out_path, "--outtype", "f16"],
        check=True,
    )
    return out_path


def quantize_gguf(
    gguf_path: str, export_dir: str, quant_type: str, llama_cpp_dir_arg: str | None
) -> str:
    """Quantize the fp16 GGUF with llama.cpp's llama-quantize binary, if found.

    Args:
        gguf_path:          fp16 GGUF from convert_to_gguf.
        export_dir:         Directory to write the quantized file into.
        quant_type:         llama-quantize type, e.g. 'q4_K_M'.
        llama_cpp_dir_arg: --llama-cpp-dir value, or None to use the
            fallbacks in _locate_llama_cpp.

    Returns:
        Path to the quantized GGUF, or the original fp16 path if
        llama-quantize is not found (quantization is a size/speed
        convenience for this 0.5B model, not a requirement, so a missing
        binary degrades gracefully rather than failing the run).
    """
    llama_cpp_dir = _locate_llama_cpp(llama_cpp_dir_arg)
    quantize_bin = shutil.which("llama-quantize") or os.path.join(
        llama_cpp_dir, "build", "bin", "llama-quantize"
    )
    if not (os.path.isfile(quantize_bin) or shutil.which(quantize_bin)):
        console.print(
            f"[yellow]llama-quantize not found (looked at {quantize_bin} and PATH); "
            f"skipping quantization, keeping the fp16 GGUF.[/yellow]"
        )
        return gguf_path

    out_path = f"{export_dir}/model-{quant_type}.gguf"
    subprocess.run([quantize_bin, gguf_path, out_path, quant_type], check=True)
    return out_path


def _locate_llama_cpp(cli_arg: str | None) -> str:
    """Resolve the llama.cpp checkout directory.

    Checks the CLI flag, then the LLAMA_CPP_DIR environment variable, then a
    sibling directory next to this repo. llama.cpp is a separate C++ project
    and is not vendored into or installed alongside this repo, so a missing
    checkout raises a clear, actionable error rather than a bare
    FileNotFoundError from subprocess.
    """
    candidate = cli_arg or os.environ.get("LLAMA_CPP_DIR") or f"{PROJECT_ROOT}/../llama.cpp"
    candidate = os.path.abspath(candidate)
    convert_script = os.path.join(candidate, "convert_hf_to_gguf.py")
    if not os.path.isfile(convert_script):
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found under {candidate}. Clone llama.cpp "
            f"(git clone https://github.com/ggml-org/llama.cpp) and either place it "
            f"at {PROJECT_ROOT}/../llama.cpp, set LLAMA_CPP_DIR, or pass "
            f"--llama-cpp-dir. See OLLAMA.md for the full setup."
        )
    return candidate


# ---------------------------------------------------------------------------
# Ollama packaging
# ---------------------------------------------------------------------------

def write_modelfile(gguf_path: str, export_dir: str) -> str:
    """Render the Ollama Modelfile for the exported GGUF.

    No TEMPLATE is set: this policy was fine-tuned on raw prompt/completion
    text (Dolly's instruction format in SFT, then HH-RLHF-style dialogue in
    RM/PPO), never on Qwen2.5's chat-template tokens, so Ollama's default chat
    templating would feed it a format it has never seen. A stop sequence and
    the training-time temperature are set instead, so `ollama run`
    approximates how the policy was actually sampled.
    """
    modelfile_path = f"{export_dir}/Modelfile"
    contents = (
        f"FROM {os.path.abspath(gguf_path)}\n"
        "\n"
        "# This is a completion model (SFT -> RM -> PPO on Anthropic/hh-rlhf), not a\n"
        "# chat/instruction model: it was never trained on Qwen2.5's chat-template\n"
        "# tokens, so no TEMPLATE is set here. Prompt it directly in the format used\n"
        "# during training, ending in the literal turn marker, e.g.:\n"
        "#\n"
        "#   \\n\\nHuman: What's a good icebreaker for a team meeting?\\n\\nAssistant:\n"
        "#\n"
        "# (see extract_prompt() in src/pipeline/ppo_rlhf_loop.py). Prompts that don't\n"
        "# end in that marker will get less coherent completions.\n"
        "\n"
        f'PARAMETER stop "{STOP_SEQUENCE}"\n'
        f"PARAMETER temperature {DEFAULT_TEMPERATURE}\n"
    )
    with open(modelfile_path, "w") as f:
        f.write(contents)
    return modelfile_path


def default_ollama_name(adapter_path: str) -> str:
    """Build a collision-resistant default Ollama model name.

    Args:
        adapter_path: The PPO LoRA adapter directory being exported.

    Returns:
        A slug of the form '<base-model>-<dataset>-ppo-<label>', e.g.
        'qwen2.5-0.5b-hh-rlhf-ppo-3f9a2b7c'. A bare name like 'rlhf-ppo' would
        collide the moment this project trains PPO on a different base model
        (BASE_MODEL in common/config.py) or dataset
        (PPORunConfig.dataset_name in ppo_rlhf_loop.py), so both are encoded
        alongside the run label.
    """
    label = _label_from_adapter_path(adapter_path)
    dataset_name = _read_dataset_name(adapter_path, label)
    return f"{_slugify(BASE_MODEL)}-{_slugify(dataset_name)}-ppo-{label}"


def _label_from_adapter_path(adapter_path: str) -> str:
    """Recover the run label for an adapter directory.

    For a results directory the label is in the name ('adapter_<label>').
    For the promoted canonical directory (ppo-model) the name carries no
    label, so it is read from the promoted_from.json provenance record that
    rlhf-promote writes. Failing both, the directory name itself is used, so
    exports of arbitrary --adapter-path directories still get stable output
    names rather than an error.
    """
    base = os.path.basename(adapter_path.rstrip("/"))
    if base.startswith("adapter_"):
        return base.removeprefix("adapter_")
    provenance_path = os.path.join(adapter_path, "promoted_from.json")
    if os.path.isfile(provenance_path):
        with open(provenance_path) as f:
            label = json.load(f).get("label")
        if label:
            return label
    return _slugify(base)


def _read_dataset_name(adapter_path: str, label: str) -> str:
    """Read dataset_name from this run's config_<label>.json, if present.

    config_<label>.json is written as a sibling of adapter_<label>/ (both
    directly under results/ppo_rlhf_loop/), not inside the adapter directory
    itself. For the promoted ppo-model directory (or any other location) the
    sibling lookup fails, so the run's results directory is tried next; if
    both miss, a placeholder is used rather than failing.
    """
    candidates = [
        os.path.join(os.path.dirname(adapter_path.rstrip("/")), f"config_{label}.json"),
        f"{RESULT_PATH}/config_{label}.json",
    ]
    for config_path in candidates:
        if os.path.isfile(config_path):
            with open(config_path) as f:
                return json.load(f).get("dataset_name", "unknown-dataset")
    return "unknown-dataset"


def _slugify(text: str) -> str:
    """Lowercase text and replace characters outside [a-z0-9.-] with '-'.

    Used to turn Hub identifiers such as 'Qwen/Qwen2.5-0.5B' or
    'Anthropic/hh-rlhf' into Ollama-safe model-name components.
    """
    text = text.rsplit("/", maxsplit=1)[-1].lower()
    return re.sub(r"[^a-z0-9.-]+", "-", text).strip("-")


def create_ollama_model(name: str, modelfile_path: str) -> None:
    """Register the model with a running Ollama install via `ollama create`.

    Falls back to printing the manual command rather than raising if the
    `ollama` CLI is not on PATH, since Ollama is an external application this
    script cannot install.
    """
    if shutil.which("ollama") is None:
        console.print("[yellow]`ollama` CLI not found on PATH.[/yellow]")
        print_manual_instructions(name, modelfile_path)
        return
    subprocess.run(["ollama", "create", name, "-f", modelfile_path], check=True)
    console.print(f"[green]Registered with Ollama as[/green] '{name}'. Try: ollama run {name}")


def print_manual_instructions(name: str, modelfile_path: str) -> None:
    """Print the manual `ollama create`/`ollama run` commands."""
    console.print("Once Ollama is installed and on PATH, run:")
    console.print(f"  ollama create {name} -f {modelfile_path}")
    console.print(f"  ollama run {name}")


if __name__ == "__main__":
    main()


# =============================================================================
# How it works
# =============================================================================
# - Scope: this is a packaging utility, not a pipeline stage -- it reads an
#   already-trained PPO adapter and produces artefacts for an external tool
#   (Ollama); it has no config dataclass, label, or checkpoints of its own.
# - Merge: merge_adapter reuses resolve_model_path from model_utils.py, the
#   same helper the RM and PPO stages use internally, so the merged model and
#   its caching (a '<adapter_path>-merged' sibling, reused if present) are
#   consistent with the rest of the pipeline.
# - Why llama.cpp: Ollama's direct safetensors import (`FROM <directory>`)
#   only covers Llama/Mistral/Gemma/Phi3; Qwen2 needs converting to GGUF via
#   llama.cpp's convert_hf_to_gguf.py first. llama.cpp itself is not vendored
#   here (a separate, independently built C++ project); _locate_llama_cpp
#   looks for a checkout via --llama-cpp-dir, LLAMA_CPP_DIR, or a sibling
#   directory.
# - Quantization is optional and best-effort: a 0.5B model is under 1 GB even
#   at fp16, and llama-quantize is a compiled binary that may not be built,
#   so a missing binary degrades to "keep fp16" rather than failing the run.
# - No chat template: the model was fine-tuned on raw prompt/completion text,
#   never Qwen2.5's chat-template tokens, so the Modelfile omits TEMPLATE and
#   documents the exact "\n\nHuman: ...\n\nAssistant:" format it expects,
#   plus a matching stop sequence and the PPO temperature.
# - Default source: with no --label/--adapter-path the promoted policy at
#   ppo-model is exported; its run label is recovered from the
#   promoted_from.json record that rlhf-promote writes, so output paths and
#   model names stay keyed by the underlying run.
# - Default naming: default_ollama_name encodes BASE_MODEL and the run's
#   dataset_name (read back from config_<label>.json, falling back to the
#   results directory for canonical exports) plus the run label, so a
#   generic name cannot collide once this project trains PPO on a
#   different base model, dataset, or config.
# =============================================================================
