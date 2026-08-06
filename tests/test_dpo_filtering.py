"""
DPO Length-Filter Semantics Tests
=================================
The fairness of the PPO-vs-DPO comparison rests on filter_pairs reproducing
the earlier stages' data view exactly: prompt cap with PPO's semantics, pair
cap with the RM's both-sides-fit semantics, EOS counted as DPOTrainer counts
it, and filtering (never truncation) throughout. These tests pin that
behaviour with a stub tokenizer (one token per whitespace-separated word, EOS
tokenised separately), so the cap arithmetic is exact and no model download
is needed.

Run with: uv run pytest
"""

# third-party
import pytest
from datasets import Dataset

# local
from src.pipeline.dpo_lora_hh import DPOTrainingConfig, filter_pairs

EOS = "<eos>"


class StubTokenizer:
    """Whitespace tokenizer counting EOS as its own token, like a real one."""

    eos_token = EOS

    def __call__(self, texts: list[str]) -> dict:
        return {"input_ids": [t.replace(EOS, f" {EOS} ").split() for t in texts]}


def pair(prompt_words: int, chosen_words: int, rejected_words: int) -> dict:
    """Build an HH-RLHF-style pair with exact token counts.

    The prompt is `prompt_words` words ending in the '\\n\\nAssistant:' marker
    (which the stub counts as one word); each side appends its response words.
    """
    prompt = " ".join(["w"] * (prompt_words - 1)) + "\n\nAssistant:"
    return {
        "chosen":   prompt + " " + " ".join(["c"] * chosen_words),
        "rejected": prompt + " " + " ".join(["r"] * rejected_words),
    }


def surviving(pairs: list[dict], max_prompt: int, max_pair: int) -> int:
    dataset = Dataset.from_dict({
        "chosen":   [p["chosen"] for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    })
    return len(filter_pairs(dataset, StubTokenizer(), max_prompt, max_pair))


def test_pair_within_both_caps_is_kept():
    # prompt 4; sides 4+3+1(EOS)=8 and 4+2+1=7, caps 10/20 -> kept.
    assert surviving([pair(4, 3, 2)], max_prompt=10, max_pair=20) == 1


def test_overlong_prompt_is_dropped_regardless_of_sides():
    # prompt 11 > 10 even though both sides fit the pair cap comfortably.
    assert surviving([pair(11, 1, 1)], max_prompt=10, max_pair=50) == 0


def test_one_overlong_side_drops_the_whole_pair():
    # chosen 4+20+1=25 > 20; rejected fits. Both-sides-fit means the pair goes.
    assert surviving([pair(4, 20, 1)], max_prompt=10, max_pair=20) == 0


def test_eos_is_counted_at_the_boundary():
    # chosen without EOS is exactly the cap (4+16=20); with EOS it is 21.
    # A filter that forgot the EOS append would keep this pair.
    assert surviving([pair(4, 16, 1)], max_prompt=10, max_pair=20) == 0
    # One word shorter fits: 4+15+1(EOS)=20 <= 20.
    assert surviving([pair(4, 15, 1)], max_prompt=10, max_pair=20) == 1


def test_filtering_never_truncates():
    # The surviving dataset carries the original texts untouched.
    p = pair(4, 3, 2)
    dataset = Dataset.from_dict({"chosen": [p["chosen"]], "rejected": [p["rejected"]]})
    kept = filter_pairs(dataset, StubTokenizer(), 10, 20)
    assert kept[0]["chosen"] == p["chosen"]
    assert kept[0]["rejected"] == p["rejected"]


def test_config_rejects_prompt_cap_at_or_above_pair_cap():
    with pytest.raises(ValueError):
        DPOTrainingConfig(max_prompt_tokens=512, max_pair_tokens=512)


def test_beta_sweep_yields_distinct_labels():
    # The matched-KL comparison sweeps beta; each step must get its own
    # results directory, so the full-config hash must move with beta.
    assert DPOTrainingConfig(beta=0.05).label != DPOTrainingConfig(beta=0.1).label
