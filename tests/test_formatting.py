"""Tokenization / chat formatting tests with the real Qwen2.5 tokenizer (needs Hub access)."""
import pytest

from conftest import make_pair
from saferl import data
from saferl.formatting import (FORBIDDEN_IN_PROMPT, assert_no_objective_leakage, collate_dpo, encode_dpo_row,
                               encode_example, full_text, prompt_text, tokenize_completion, tokenize_prompt)

pytestmark = pytest.mark.hf


def test_prompt_is_context_and_masked(tokenizer):
    enc = encode_example(tokenizer, "How do I boil an egg?", "Put it in boiling water for 7 minutes.")
    p = enc["prompt_len"]
    assert p > 0 and enc["completion_len"] > 0
    assert enc["loss_mask"][:p] == [0] * p
    assert enc["loss_mask"][p:] == [1] * enc["completion_len"]
    assert enc["input_ids"][:p] == tokenize_prompt(tokenizer, "How do I boil an egg?")
    # user prompt text is actually present in the serialized context
    assert "How do I boil an egg?" in tokenizer.decode(enc["input_ids"][:p])
    assert enc["attention_mask"] == [1] * len(enc["input_ids"])


def test_eos_is_last_token_and_counted_in_loss(tokenizer):
    enc = encode_example(tokenizer, "hi", "hello")
    assert enc["input_ids"][-1] == tokenizer.eos_token_id
    assert enc["loss_mask"][-1] == 1
    assert tokenizer.eos_token == "<|im_end|>"
    assert tokenize_completion(tokenizer, "hello")[-1] == tokenizer.eos_token_id
    assert tokenize_completion(tokenizer, "hello").count(tokenizer.eos_token_id) == 1


def test_split_tokenization_matches_joint_serialization(tokenizer):
    prompt, resp = "Explain photosynthesis briefly.", "Plants convert light into chemical energy."
    joint = tokenizer(full_text(tokenizer, prompt, resp), add_special_tokens=False)["input_ids"]
    split = tokenize_prompt(tokenizer, prompt) + tokenize_completion(tokenizer, resp)
    assert joint == split


def test_chosen_rejected_share_identical_prompt_serialization(tokenizer):
    p = make_pair(0, better=0, safer=1)
    for row in data.pair_to_dpo_rows(p):
        enc = encode_dpo_row(tokenizer, row)
        pc, pr = enc["chosen"]["prompt_len"], enc["rejected"]["prompt_len"]
        assert pc == pr
        assert enc["chosen"]["input_ids"][:pc] == enc["rejected"]["input_ids"][:pr]


def test_no_custom_system_prompt_only_official_template(tokenizer):
    text = prompt_text(tokenizer, "Hello there")
    official = tokenizer.apply_chat_template([{"role": "user", "content": "Hello there"}], tokenize=False, add_generation_prompt=True)
    assert text == official
    assert text.endswith("<|im_start|>assistant\n")


def test_no_objective_leakage_in_prompt(tokenizer):
    p = make_pair(0, better=0, safer=1)
    rows = data.pair_to_dpo_rows(p)
    h, s = (prompt_text(tokenizer, r["prompt"]) for r in rows)
    assert h == s  # both objectives see byte-identical prompts
    for r in rows:
        assert_no_objective_leakage(tokenizer, r)
    scaffold = h.replace(p["prompt"], "").lower()
    for tok in FORBIDDEN_IN_PROMPT:
        assert tok not in scaffold
    # a prompt that itself contains the word is fine (user content), but the objective name must not be added
    row = dict(rows[0]); row["prompt"] = "Is safety important?"
    assert_no_objective_leakage(tokenizer, row)


def test_overflow_errors_by_default_and_truncates_on_request(tokenizer):
    with pytest.raises(ValueError):
        encode_example(tokenizer, "word " * 300, "x", max_prompt_length=64, max_length=128)
    enc = encode_example(tokenizer, "word " * 300, "x", max_prompt_length=64, max_length=128, on_overflow="truncate")
    assert enc["truncated"] and enc["prompt_len"] == 64


def test_collate_pads_and_keeps_alignment(tokenizer):
    rows = data.pairs_to_dpo_rows([make_pair(0, r0="short", r1="a much longer response " * 5), make_pair(1)])
    enc = [encode_dpo_row(tokenizer, r) for r in rows]
    b = collate_dpo(enc, tokenizer.pad_token_id)
    B = b["batch_size"]
    assert b["input_ids"].shape[0] == 2 * B
    for i in range(B):
        L = len(enc[i]["chosen"]["input_ids"])
        assert b["attention_mask"][i, :L].sum() == L and b["attention_mask"][i, L:].sum() == 0
        assert b["loss_mask"][i, L:].sum() == 0
        assert b["loss_mask"][i].sum() == enc[i]["chosen"]["completion_len"]
        assert b["loss_mask"][B + i].sum() == enc[i]["rejected"]["completion_len"]
