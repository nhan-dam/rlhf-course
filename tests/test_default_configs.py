"""
Default-Config Drift Tests
==========================
The files configs/<stage>_default.json list every configurable hyperparameter
of a pipeline stage at its default value, doubling as runnable configs and as
reference documentation for custom runs (see README, 'configs/'). The defaults
themselves live in each stage's config dataclass, so the JSON files are
snapshots that can silently drift when a dataclass field is added, removed,
renamed, or its default changed. These tests pin the invariant: running a
stage with its default file must be identical to running it with no config
file at all.

Run with: uv run pytest
"""

# stdlib
import json
from dataclasses import asdict, fields
from pathlib import Path

# third-party
import pytest

# local
from src.pipeline.dpo_lora_hh import DPOTrainingConfig
from src.pipeline.ppo_rlhf_loop import PPORunConfig
from src.pipeline.reward_model_hh import RMTrainingConfig
from src.pipeline.sft_lora_dolly import SFTTrainingConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

STAGES = [
    ("sft_default.json", SFTTrainingConfig),
    ("rm_default.json", RMTrainingConfig),
    ("ppo_default.json", PPORunConfig),
    ("dpo_default.json", DPOTrainingConfig),
]


@pytest.mark.parametrize(("filename", "config_cls"), STAGES, ids=[s[0] for s in STAGES])
def test_default_config_reproduces_no_arg_run(filename, config_cls):
    """The default file must construct a config identical to Config().

    Three layers, from most to least diagnostic on failure:
    1. Key sets match exactly. This catches a dataclass field missing from the
       file, which dict equality alone would miss (a missing key falls back to
       the dataclass default and compares equal).
    2. Full dict equality, for a readable per-field diff when a value drifts.
    3. Label equality, the semantic guarantee: the label hashes the full
       resolved config, so equal labels mean the file maps to the same run
       directory and artefacts as a no-arg run.
    """
    with open(CONFIGS_DIR / filename) as f:
        data = json.load(f)

    assert set(data) == {f.name for f in fields(config_cls)}

    from_file = config_cls(**data)   # __post_init__ resolves null path fields
    default = config_cls()
    assert asdict(from_file) == asdict(default)
    assert from_file.label == default.label
