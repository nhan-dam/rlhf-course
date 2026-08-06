"""Shared configuration for the RLHF course pipeline (SFT -> RM -> PPO).

Single source of truth for values that cross project boundaries: the base model
and the on-disk artefacts exchanged between the supervised fine-tuning (SFT),
reward-modelling (RM), and proximal policy optimisation (PPO) stages. Importing
these here keeps the three scripts from drifting out of sync.
"""

# Base model shared by every stage. SFT trains a LoRA adapter on it; the reward
# model and the PPO policy/reference all initialise from it via the merged SFT
# checkpoint. A small model is used so the four-model PPO stage fits on a single
# workstation.
BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# Root directory for all on-disk artefacts.
PROJECT_ROOT = "/Volumes/ML_Workspace/projects/rlhf-course"

# Cross-stage artefacts (produced by one stage, consumed by the next).
SFT_ADAPTER      = f"{PROJECT_ROOT}/sft-model"          # LoRA adapter from SFT; merged by the RM stage
SFT_MERGED_MODEL = f"{PROJECT_ROOT}/sft-model-merged"   # base + merged adapter; policy/reference init for PPO
RM_MODEL         = f"{PROJECT_ROOT}/rm-model"           # trained reward model; reward + critic init for PPO
# End of the pipeline: nothing downstream trains on these, but the export and
# review tooling (and the human) need one well-known place to find "the"
# policy. Populated by `rlhf-promote <stage> <label>` or EXPORT_CANONICAL=1.
PPO_ADAPTER      = f"{PROJECT_ROOT}/ppo-model"          # promoted PPO policy adapter
DPO_ADAPTER      = f"{PROJECT_ROOT}/dpo-model"          # promoted DPO policy adapter (Phase 3)
