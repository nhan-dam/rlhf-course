"""
Model Utilities for the RLHF Pipeline
=====================================
Shared helpers for the SFT / reward-model / PPO scripts in this repository.
Currently provides adapter resolution: stages of the pipeline hand each other
LoRA adapter directories, but downstream consumers (sequence-classification
heads, PPO's frozen reference) need a plain merged model directory.

Inputs
------
model_path : str — a Hub model id, a full model directory, or a LoRA adapter
             directory (identified by the presence of adapter_config.json).
model_kind : 'causal-lm' or 'seq-cls' — how to load the base when merging.

Outputs
-------
A path loadable with AutoModelFor*.from_pretrained: the input path unchanged,
or a sibling '<path>-merged' directory created on first call and reused after.

Public API
----------
resolve_model_path(model_path, model_kind) — merge a LoRA adapter into its
    base model if needed; return a directly loadable model path.
export_canonical(trainer, canonical_path) — overwrite a pipeline-shared
    canonical path with a run's model, clearing any stale merge cache.
"""

# stdlib
import os
import shutil
import threading
import time
from typing import Literal

# third-party
import torch
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer, TrainerCallback


class CacheCleaner(TrainerCallback):
    """Release cached-but-unused GPU memory once reserved usage crosses a threshold.

    PyTorch's caching allocator (on both CUDA and Apple Silicon MPS) holds freed
    blocks rather than returning them, so the reserved high-water mark creeps up
    over a long run. Rather than clearing on a fixed cadence, we poll the (cheap)
    reserved-memory counter each step and empty the cache when it exceeds
    THRESHOLD_RATIO of the device capacity, plus once after each eval. On CPU the
    callback is inert. Shared by every training stage in the pipeline.

    A one-line memory report is printed at the trainer's ``logging_steps``
    cadence and whenever the cache is cleared, so it sits beside the loss lines.
    Each figure is the maximum observed since the previous line, i.e. live (the
    footprint at the sampled points) and reserved (the allocator pool), and the
    windows reset after every print. It is a plain print, so it stays out of the
    metric and TensorBoard stream.

    Setting the environment variable ``TRACK_PEAK_MEM=1`` adds the in-step peak
    live memory, which the step-end ``live`` sample misses. The peak is measured
    within the step: on CUDA via the built-in high-water counter, on MPS via a
    background polling thread. It is off by default, as the thread adds overhead.
    """

    THRESHOLD_RATIO = 0.8
    SAMPLE_INTERVAL_S = 0.005   # MPS peak-sampler polling period

    def __init__(self, track_peak: bool | None = None) -> None:
        self.track_peak = (
            track_peak if track_peak is not None
            else os.environ.get("TRACK_PEAK_MEM", "0") == "1"
        )
        # Running maxima since the last printed line; reset after each print.
        self._live_max = 0
        self._reserved_max = 0
        self._peak_max = 0                   # MPS in-step peak (CUDA uses the built-in counter)
        self._host_max = 0                   # whole-process resident memory (host side)
        self._sampler: threading.Thread | None = None
        self._stop = threading.Event()

    @staticmethod
    def _backend() -> str | None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return None

    @staticmethod
    def _reserved_and_capacity(backend: str) -> tuple[int, int]:
        if backend == "cuda":
            return torch.cuda.memory_reserved(), torch.cuda.get_device_properties(0).total_memory
        return torch.mps.driver_allocated_memory(), torch.mps.recommended_max_memory()

    @staticmethod
    def _live(backend: str) -> int:
        if backend == "cuda":
            return torch.cuda.memory_allocated()
        return torch.mps.current_allocated_memory()

    @staticmethod
    def _host_rss() -> int:
        """CPU-side resident memory in bytes (psutil RSS).

        This is the host-side footprint: interpreter, libraries, dataset, and
        dataloader buffers. On macOS it EXCLUDES the Metal/GPU allocator pool, so
        it is much smaller than Activity Monitor, whose figure (phys_footprint)
        is approximately this plus `reserved`. It isolates host-side growth that
        the device counters miss, e.g. a Python leak or shader-cache growth.
        Uses psutil if available (current RSS), else the stdlib peak RSS.
        """
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except Exception:
            import resource
            import sys
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak if sys.platform == "darwin" else peak * 1024  # macOS bytes, Linux KiB

    @staticmethod
    def _empty(backend: str) -> None:
        if backend == "cuda":
            torch.cuda.empty_cache()
        elif backend == "mps":
            torch.mps.empty_cache()

    # --- running maxima (reset after each printed line) --------------------

    def _update_maxima(self, backend: str) -> None:
        """Fold the current live and reserved readings into the running maxima."""
        live = self._live(backend)
        reserved, _ = self._reserved_and_capacity(backend)
        self._live_max = max(self._live_max, live)
        self._reserved_max = max(self._reserved_max, reserved)
        self._host_max = max(self._host_max, self._host_rss())
        if backend == "mps":
            self._peak_max = max(self._peak_max, live)

    def _peak_value(self, backend: str) -> int:
        return torch.cuda.max_memory_allocated() if backend == "cuda" else self._peak_max

    def _reset_maxima(self, backend: str) -> None:
        """Restart the windows at the current readings after a line is printed."""
        self._live_max = self._live(backend)
        self._reserved_max = self._reserved_and_capacity(backend)[0]
        self._host_max = self._host_rss()
        if backend == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            self._peak_max = self._live(backend)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._peak_max = max(self._peak_max, torch.mps.current_allocated_memory())
            time.sleep(self.SAMPLE_INTERVAL_S)

    def on_train_begin(self, args, state, control, **kwargs):
        backend = self._backend()
        if not self.track_peak or backend is None:
            return
        if backend == "cuda":
            # The CUDA peak is the allocator's high-water mark since the last
            # reset. Reset here so the first window excludes model-loading and
            # warm-up allocations; _reset_maxima resets it after every print.
            torch.cuda.reset_peak_memory_stats()
        else:  # mps has no peak counter, so sample current usage in a thread
            self._stop.clear()
            self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
            self._sampler.start()

    def on_train_end(self, args, state, control, **kwargs):
        self._stop.set()

    # --- reporting and clearing -------------------------------------------

    def _log(self, backend: str, state, when: str, cleared: bool = False) -> None:
        """Print one line of running-max memory, main process only.

        Each figure is the maximum since the previous line: live (footprint at
        the sampled points), peak (the true in-step high, when TRACK_PEAK_MEM is
        set), reserved (the device allocator pool, including free cache), and
        host (CPU-side resident memory; on macOS Activity Monitor is roughly
        host + reserved). A steadily rising host while reserved stays flat points
        to a host-side leak. A plain print, so it stays out of the metric and
        TensorBoard stream.
        """
        if not getattr(state, "is_world_process_zero", True):
            return
        _, capacity = self._reserved_and_capacity(backend)
        peak = f"gpu_peak={self._peak_value(backend) / 1e9:.1f}GB " if self.track_peak else ""
        note = " (cleared)" if cleared else ""
        print(
            f"[mem] {when}: gpu_live={self._live_max / 1e9:.1f}GB {peak}"
            f"gpu_reserved={self._reserved_max / 1e9:.1f}GB "
            f"({100 * self._reserved_max / capacity:.0f}% of GPU cap) "
            f"cpu_host={self._host_max / 1e9:.1f}GB{note}"
        )

    def _report(self, backend: str, state, when: str, cleared: bool) -> None:
        """Print the current window and start a fresh one."""
        self._log(backend, state, when, cleared)
        self._reset_maxima(backend)

    def on_step_end(self, args, state, control, **kwargs):
        backend = self._backend()
        if backend is None:
            return
        self._update_maxima(backend)
        reserved, capacity = self._reserved_and_capacity(backend)
        cleared = reserved > self.THRESHOLD_RATIO * capacity
        if cleared:
            self._empty(backend)
        # Report on a clear, or at the trainer's own loss-logging cadence.
        cadence = bool(args.logging_steps) and state.global_step % args.logging_steps == 0
        if cleared or cadence:
            self._report(backend, state, f"step {state.global_step}", cleared)

    def on_evaluate(self, args, state, control, **kwargs):
        backend = self._backend()
        if backend is None:
            return
        self._update_maxima(backend)
        self._empty(backend)   # a clear always triggers a report (rule 1)
        self._report(backend, state, f"post-eval (step {state.global_step})", cleared=True)


def resolve_model_path(
    model_path: str,
    model_kind: Literal["causal-lm", "seq-cls"] = "causal-lm",
) -> str:
    """Return a model path that loads as a plain (non-PEFT) model.

    If model_path is a LoRA adapter directory, the adapter is merged into its
    base model and saved to '<model_path>-merged' (tokenizer included). The
    merge runs once; later calls reuse the cached directory. Hub ids and full
    model directories pass through unchanged.

    Args:
        model_path: Hub id, full model directory, or LoRA adapter directory.
        model_kind: Architecture to merge into. 'seq-cls' loads a single-label
                    sequence-classification base (for reward/value models).

    Returns:
        A path accepted by AutoModelFor*.from_pretrained.
    """
    if not _is_adapter_dir(model_path):
        return model_path

    merged_path = model_path.rstrip("/") + "-merged"
    if os.path.isdir(merged_path):
        return merged_path

    if model_kind == "causal-lm":
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        )
    else:
        peft_model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1, dtype=torch.bfloat16
        )

    peft_model.merge_and_unload().save_pretrained(merged_path)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(merged_path)
    return merged_path


def _is_adapter_dir(model_path: str) -> bool:
    """A local directory containing adapter_config.json is a LoRA adapter."""
    return os.path.isdir(model_path) and os.path.isfile(
        os.path.join(model_path, "adapter_config.json")
    )


def export_canonical(trainer, canonical_path: str) -> None:
    """Overwrite a pipeline-shared canonical path with this run's model.

    The canonical paths (SFT_ADAPTER, RM_MODEL in config.py) are what the next
    stage reads by default, so exporting to them is how a run's artefact
    becomes "the" input for downstream stages. Callers are expected to gate
    this behind an explicit, opt-in choice (e.g. an EXPORT_CANONICAL=1
    environment variable) rather than calling it unconditionally after every
    run: unconditional export means the canonical path always reflects
    whichever config was run most recently, not whichever config was chosen
    after comparing metrics.

    This also deletes any stale '<canonical_path>-merged' cache, if present.
    resolve_model_path() treats the mere existence of that sibling directory
    as "already merged, reuse it," with no check that it was derived from the
    adapter currently at canonical_path. Left in place, it would make
    downstream stages silently keep loading the merge of whatever used to be
    exported here instead of the model this call just wrote.

    Args:
        trainer:        The fitted trainer (SFTTrainer, RewardTrainer, ...).
        canonical_path: The pipeline-shared path to overwrite, e.g. SFT_ADAPTER.
    """
    merged_path = canonical_path.rstrip("/") + "-merged"
    if os.path.isdir(merged_path):
        shutil.rmtree(merged_path)
    trainer.save_model(canonical_path)


# =============================================================================
# How it works
# =============================================================================
# - resolve_model_path: the pipeline's stages save LoRA adapters (small, fast
#   to checkpoint), but AutoModelForSequenceClassification and PPO's frozen
#   reference need a self-contained model; this helper bridges the two by
#   merging W + BA into plain weights exactly once and caching the result.
# - Merge is idempotent: the '-merged' sibling directory acts as the cache
#   key, so repeated pipeline runs skip the (slow) merge-and-save step.
# - model_kind exists because the same adapter directory may be merged into
#   different heads: 'causal-lm' for the PPO policy/reference, 'seq-cls' for
#   reward and value models (scalar head, randomly initialised on first load).
# - export_canonical: pairs with resolve_model_path's caching assumption —
#   whenever a canonical adapter is replaced, its '-merged' cache must be
#   invalidated too, or downstream stages keep silently reading the old merge.
#   Kept opt-in (callers check an env var, not a config field) so promoting a
#   run never changes that run's config hash/label, which would point
#   get_last_checkpoint() at an empty directory instead of the completed run.
# =============================================================================
