# Testing the PPO Policy in Ollama

This is a walkthrough for `src/export/ppo_to_ollama.py`, which packages a completed PPO run so it can be tried interactively with `ollama run` instead of only through the training pipeline. It assumes you have already run the PPO stage (`uv run rlhf-ppo` or a `configs/ppo_*.json` variant), and ideally promoted the chosen run to `ppo-model` with `uv run rlhf-promote ppo <label>` (README, Section 7), which is what the exporter reads by default.

## 1. Why a conversion step is needed at all

PPO (like SFT and RM) only ever saves a LoRA adapter directory, not a model Ollama can load directly. Ollama can import safetensors directly with `FROM <directory>` in a Modelfile, but only for a handful of architectures: Llama, Mistral, Gemma, and Phi3. Qwen2, this project's base model (Qwen2.5-0.5B), is not among them.

The route that works for Qwen2 is the one Ollama's own documentation points to for everything else: merge the LoRA adapter into the base model, convert the merged model to GGUF with llama.cpp's `convert_hf_to_gguf.py`, and point a Modelfile at the resulting `.gguf` file. `ppo_to_ollama.py` automates all of it bar the one-time setup below.

## 2. One-time setup

You need three things beyond the project's own `uv sync`, none of which are part of this repository:

1. **The converter's Python dependencies** (the `gguf` container-format writer and `sentencepiece`, both imported by llama.cpp's converter):

   ```bash
   uv sync --extra ollama
   ```

2. **A local llama.cpp checkout**, for `convert_hf_to_gguf.py`:

   ```bash
   git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
   ```

   Placing it at `../llama.cpp` (a sibling of this repository) means `ppo_to_ollama.py` finds it with no extra flags. If you keep it elsewhere, either set `LLAMA_CPP_DIR` or pass `--llama-cpp-dir` each time (see Section 4).

   Do **not** install llama.cpp's own `requirements-convert_hf_to_gguf.txt` into this project's environment: the export script runs the converter with the project venv's Python, and that file pins exact torch/transformers versions that can downgrade the pipeline's. The venv already provides most of what the converter needs (torch, transformers, and `gguf` from step 1). If the converter fails with a `ModuleNotFoundError` for something the project does not ship (e.g. `sentencepiece` or `mistral-common`), add just that package with `uv pip install <name>` and rerun.

   Building llama.cpp (`cmake -B build && cmake --build build`) is only needed if you want the `llama-quantize` binary for the optional quantisation step in Section 4. The base model is 0.5B parameters, so the unquantised fp16 GGUF is well under 1 GB and is fine to run as-is. Quantisation is a convenience, not a requirement.

3. **Ollama itself.** Install it from [ollama.com](https://ollama.com) and confirm `ollama` is on your `PATH` (`ollama --version`).

## 3. Running the export

From the project root, with the same `uv` environment used for training:

```bash
uv run rlhf-ppo-ollama
```

With no flags, this exports the promoted policy at `ppo-model`, recovering the run label from the `promoted_from.json` record that `rlhf-promote` writes there, so output paths and the registered model name are still keyed by the underlying run. To export a specific run that has not been promoted, pass its label instead:

```bash
uv run rlhf-ppo-ollama --label <ppo_run_label>
```

`<ppo_run_label>` is the hash printed by the PPO run and used in `results/ppo_rlhf_loop/adapter_<label>/` and `config_<label>.json` (the same labelling scheme used by SFT and RM). If you would rather point at a specific directory directly (a renamed export, say), use `--adapter-path` instead:

```bash
uv run rlhf-ppo-ollama --adapter-path /path/to/some/adapter/dir
```

This does four things in order:

1. Merges the LoRA adapter into the base model (reusing the same `<adapter>-merged` caching that the RM and PPO stages already use internally).
2. Converts the merged model to a fp16 GGUF file.
3. Writes an Ollama Modelfile next to it.
4. Runs `ollama create`, registering the model, unless `ollama` isn't on `PATH` or `--no-create` was passed, in which case it prints the two commands to run manually.

Everything lands in `results/ppo_rlhf_loop/ollama_<label>/`: `model-f16.gguf` (or `model-<quantize>.gguf`, see below) and `Modelfile`.

Useful flags:

| Flag | Purpose |
|---|---|
| `--llama-cpp-dir PATH` | Point at a llama.cpp checkout that isn't at `../llama.cpp`. Also settable via the `LLAMA_CPP_DIR` environment variable. |
| `--quantize q4_K_M` | Quantise the GGUF with `llama-quantize` (build it first, see Section 2). Skipped automatically, with a warning, if the binary isn't found. |
| `--ollama-name NAME` | Name to register with Ollama. Defaults to `<base-model>-<dataset>-ppo-<label>` (e.g. `qwen2.5-0.5b-hh-rlhf-ppo-3f9a2b7c`), built by `default_ollama_name` from `BASE_MODEL`, the run's `dataset_name`, and its label, so a generic name can't collide once this project trains PPO on a different base model or dataset. |
| `--no-create` | Write the GGUF and Modelfile but don't call `ollama create` (e.g. to inspect them first, or if Ollama isn't installed on this machine). |

## 4. Talking to the model

Once registered:

```bash
ollama run qwen2.5-0.5b-hh-rlhf-ppo-<label>
```

The Modelfile deliberately sets no chat `TEMPLATE`. The policy was fine-tuned on raw prompt/completion text throughout the pipeline (Dolly's instruction format for SFT, then HH-RLHF-style dialogue for RM and PPO), never on Qwen2.5's chat-template tokens, so applying Ollama's default chat formatting would feed it a format it has never seen and produce worse completions than the model is actually capable of.

Prompt it directly in the format PPO trained against, i.e. text ending in the literal HH-RLHF turn marker (see `extract_prompt()` in `src/pipeline/ppo_rlhf_loop.py`):

```
$ ollama run qwen2.5-0.5b-hh-rlhf-ppo-<label>
>>> Human: What's a good icebreaker for a team meeting?

Assistant:
```

The Modelfile sets `PARAMETER stop "\n\nHuman:"` so the model stops once it would otherwise start inventing the next human turn, and `PARAMETER temperature 0.7` to match the sampling temperature used during PPO training (see `PPORunConfig.temperature` in `ppo_rlhf_loop.py`), rather than Ollama's own default. Prompts that don't end in the `Assistant:` marker will tend to produce less coherent completions, since that is not the distribution the model was trained on.

## 5. Troubleshooting

- **`FileNotFoundError: No LoRA adapter found at ...`**: with no flags, this means no run has been promoted to `ppo-model` yet (run `uv run rlhf-promote ppo <label>` first). With `--label`/`--adapter-path`, PPO hasn't produced that run, or the label/path is wrong. Check `results/ppo_rlhf_loop/` for the actual `adapter_<label>` directories.
- **`convert_hf_to_gguf.py not found under ...`**: the llama.cpp checkout isn't where the script is looking. Pass `--llama-cpp-dir`, set `LLAMA_CPP_DIR`, or clone it to `../llama.cpp`.
- **`ModuleNotFoundError: gguf`** (raised inside `convert_hf_to_gguf.py`, not this script): run `uv sync --extra ollama`. For any other `ModuleNotFoundError` from the converter, add just the missing package with `uv pip install <name>` rather than installing llama.cpp's pinned requirements file (see Section 2).
- **`llama-quantize not found ... skipping quantization`**: expected if you only cloned llama.cpp without building it. Not an error: the fp16 GGUF is used instead. Build llama.cpp with `cmake` if you want a smaller quantised file.
- **`ollama` CLI not found on PATH**: the script still writes the GGUF and Modelfile. Install Ollama, then run the two printed commands manually.
- **Responses look like generic chat-assistant boilerplate, not what PPO actually learned**: double-check the prompt ends in `\n\nAssistant:` with no other formatting added. Ollama's interactive prompt does not add this for you.
