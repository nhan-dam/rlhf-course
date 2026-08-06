"""
Promote a Completed Run to its Canonical Artefact Path
======================================================
Copies a completed run's saved adapter (results/<stage>/adapter_<label>/)
over the stage's canonical path (sft-model, rm-model, or ppo-model), which is
where downstream consumers look by default. This is the file-based
counterpart of the EXPORT_CANONICAL=1 mechanism in the training scripts:

- For SFT and RM it is an equivalent, faster alternative to the re-run
  route: the saved adapter is already the best held-out checkpoint (the
  trainers save after load_best_model_at_end), so promotion is a pure copy.
- For PPO it is the only route: the experimental PPOTrainer cannot resume
  (train() takes no arguments) and has no best-checkpoint selection (PPO
  deliberately ships the final adapter; selecting by reward-model score
  would select for reward hacking), so a re-run would repeat the whole run.

Like model_utils.export_canonical, promotion deletes any stale
'<canonical>-merged' cache so downstream stages regenerate the merge from
the newly promoted adapter instead of silently reusing the old one. A
promoted_from.json provenance record (stage, label, source path, timestamp)
is written into the canonical directory; extra files are ignored by
from_pretrained, and the file answers "which run is this?" later.

Usage
-----
uv run rlhf-promote ppo e71b6d13
uv run rlhf-promote sft <label>      # post-hoc alternative to EXPORT_CANONICAL=1

Public API
----------
promote_adapter(adapter_path, canonical_path, provenance)
    — copy an adapter directory over a canonical path, invalidating caches.
"""

# stdlib
import argparse
import json
import os
import shutil
from datetime import datetime, timezone

# third-party
from rich.console import Console

# local
from ..common.config import PROJECT_ROOT, SFT_ADAPTER, RM_MODEL, PPO_ADAPTER, DPO_ADAPTER

# stage -> (results directory, canonical path). The results directory names
# mirror each stage script's RESULT_PATH; they are repeated here so this
# tool stays importable without pulling in the stages' torch-heavy modules.
STAGES = {
    "sft": (f"{PROJECT_ROOT}/results/sft_lora_dolly", SFT_ADAPTER),
    "rm":  (f"{PROJECT_ROOT}/results/reward_model_hh", RM_MODEL),
    "ppo": (f"{PROJECT_ROOT}/results/ppo_rlhf_loop", PPO_ADAPTER),
    "dpo": (f"{PROJECT_ROOT}/results/dpo_lora_hh", DPO_ADAPTER),
}

console = Console()


def promote_adapter(
    adapter_path: str, canonical_path: str, provenance: dict | None = None
) -> None:
    """Copy a run's saved adapter directory over the canonical path.

    Deletes the stale '<canonical_path>-merged' cache first (see
    export_canonical in model_utils.py for why: resolve_model_path treats
    that directory's mere existence as "already merged, reuse it"), then
    replaces the canonical directory with a copy of the adapter, then writes
    the provenance record, if given, into the copy.

    Args:
        adapter_path:   Source adapter directory (must exist).
        canonical_path: Canonical destination; replaced if present.
        provenance:     Optional record written as promoted_from.json.
    """
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"No adapter directory at {adapter_path} - is the label right, "
            f"and did the run complete?"
        )
    merged_path = canonical_path.rstrip("/") + "-merged"
    if os.path.isdir(merged_path):
        shutil.rmtree(merged_path)
    if os.path.isdir(canonical_path):
        shutil.rmtree(canonical_path)
    shutil.copytree(adapter_path, canonical_path)
    if provenance is not None:
        with open(os.path.join(canonical_path, "promoted_from.json"), "w") as f:
            json.dump(provenance, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a completed run's adapter to its stage's canonical path."
    )
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("label", help="run label, e.g. e71b6d13")
    args = parser.parse_args()

    results_dir, canonical_path = STAGES[args.stage]
    adapter_path = f"{results_dir}/adapter_{args.label}"
    provenance = {
        "stage":         args.stage,
        "label":         args.label,
        "promoted_from": adapter_path,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    promote_adapter(adapter_path, canonical_path, provenance)
    console.print(
        f"[green]Promoted[/green] {adapter_path} [green]->[/green] {canonical_path} "
        f"(stale merge cache cleared, provenance recorded)"
    )


if __name__ == "__main__":
    main()
