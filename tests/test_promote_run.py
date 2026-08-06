"""
Promotion Tests
===============
promote_adapter (src/pipeline/promote_run.py) replaces a stage's canonical
artefact with a completed run's adapter. The invariants pinned here: the
canonical directory becomes an exact copy of the source adapter, any stale
'<canonical>-merged' cache is deleted (resolve_model_path treats its mere
existence as "already merged, reuse it"), provenance is recorded, and a
missing source fails loudly rather than half-promoting.

Run with: uv run pytest
"""

# stdlib
import json

# third-party
import pytest

# local
from src.pipeline.promote_run import promote_adapter


@pytest.fixture
def adapter(tmp_path):
    """A fake completed-run adapter directory with a nested file."""
    source = tmp_path / "adapter_abc12345"
    (source / "sub").mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"r": 32}')
    (source / "sub" / "weights.bin").write_text("new-weights")
    return source


def test_promote_copies_replaces_and_invalidates(adapter, tmp_path):
    canonical = tmp_path / "ppo-model"
    merged = tmp_path / "ppo-model-merged"

    # Pre-existing canonical content and a stale merge cache, both of which
    # must be gone after promotion.
    canonical.mkdir()
    (canonical / "old.bin").write_text("old-weights")
    merged.mkdir()
    (merged / "stale.bin").write_text("stale-merge")

    provenance = {"stage": "ppo", "label": "abc12345"}
    promote_adapter(str(adapter), str(canonical), provenance)

    assert (canonical / "adapter_config.json").read_text() == '{"r": 32}'
    assert (canonical / "sub" / "weights.bin").read_text() == "new-weights"
    assert not (canonical / "old.bin").exists()
    assert not merged.exists()
    assert json.loads((canonical / "promoted_from.json").read_text()) == provenance
    # The source adapter is copied, not moved.
    assert (adapter / "adapter_config.json").exists()


def test_promote_missing_source_fails_loudly(adapter, tmp_path):
    canonical = tmp_path / "ppo-model"
    canonical.mkdir()
    (canonical / "old.bin").write_text("old-weights")

    with pytest.raises(FileNotFoundError):
        promote_adapter(str(tmp_path / "adapter_nonexistent"), str(canonical))

    # A failed promotion must leave the existing canonical artefact intact.
    assert (canonical / "old.bin").read_text() == "old-weights"
