"""DPO loss math (hand-computed), monotonicity, beta usage, masking, and reference-no-grad."""
import math

import pytest
import torch

from saferl.dpo_loss import dpo_loss, sequence_logprobs


def _t(x):
    return torch.tensor([x], dtype=torch.float32)


def test_hand_computed_values():
    beta = 0.1
    # logits = beta * ((pc - rc) - (pr - rr)) = 0.1 * ((-10 - -12) - (-20 - -15)) = 0.1 * (2 - (-5)) = 0.7
    out = dpo_loss(_t(-10.0), _t(-20.0), _t(-12.0), _t(-15.0), beta)
    assert math.isclose(float(out["margin"]), 0.7, rel_tol=1e-6)
    assert math.isclose(float(out["loss"]), -math.log(1 / (1 + math.exp(-0.7))), rel_tol=1e-6)
    assert math.isclose(float(out["chosen_reward"]), 0.2, rel_tol=1e-6)
    assert math.isclose(float(out["rejected_reward"]), -0.5, rel_tol=1e-6)
    # policy == reference -> margin 0, loss log 2
    out0 = dpo_loss(_t(-3.0), _t(-4.0), _t(-3.0), _t(-4.0), beta)
    assert math.isclose(float(out0["loss"]), math.log(2), rel_tol=1e-6)
    assert float(out0["margin"]) == 0.0


def test_loss_decreases_when_policy_prefers_chosen_and_increases_otherwise():
    beta = 0.1
    base = float(dpo_loss(_t(-10.0), _t(-10.0), _t(-10.0), _t(-10.0), beta)["loss"])
    prefers_chosen = float(dpo_loss(_t(-5.0), _t(-45.0), _t(-10.0), _t(-10.0), beta)["loss"])  # margin +4
    prefers_rejected = float(dpo_loss(_t(-45.0), _t(-5.0), _t(-10.0), _t(-10.0), beta)["loss"])  # margin -4
    assert prefers_chosen < base < prefers_rejected
    assert math.isclose(base, math.log(2), rel_tol=1e-6)
    assert prefers_chosen < 0.02 and prefers_rejected > 4.0


def test_beta_is_used():
    o1 = dpo_loss(_t(-5.0), _t(-15.0), _t(-10.0), _t(-10.0), beta=0.1)
    o2 = dpo_loss(_t(-5.0), _t(-15.0), _t(-10.0), _t(-10.0), beta=0.2)
    assert math.isclose(float(o1["margin"]), 1.0, rel_tol=1e-6)
    assert math.isclose(float(o2["margin"]), 2.0, rel_tol=1e-6)
    assert float(o2["loss"]) < float(o1["loss"])


def test_label_smoothing_zero_is_plain_sigmoid():
    a = dpo_loss(_t(-5.0), _t(-15.0), _t(-10.0), _t(-10.0), 0.1, label_smoothing=0.0)["loss"]
    b = -torch.nn.functional.logsigmoid(_t(1.0))
    assert torch.allclose(a, b)


def test_sequence_logprobs_masks_prompt_positions():
    torch.manual_seed(0)
    B, T, V = 2, 6, 11
    logits = torch.randn(B, T, V)
    ids = torch.randint(0, V, (B, T))
    mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0]])
    out = sequence_logprobs(logits, ids, mask)
    logp = torch.log_softmax(logits, dim=-1)
    for b in range(B):
        manual = sum(float(logp[b, t - 1, ids[b, t]]) for t in range(1, T) if mask[b, t] == 1)
        assert math.isclose(float(out["sum_logprob"][b]), manual, rel_tol=1e-5)
        assert int(out["num_tokens"][b]) == int(mask[b, 1:].sum())
    # changing prompt-position logits (positions whose next token is a prompt token) does not change the sum
    logits2 = logits.clone()
    logits2[:, 0, :] += 5.0  # predicts position 1 (masked in both rows)
    assert torch.allclose(sequence_logprobs(logits2, ids, mask)["sum_logprob"], out["sum_logprob"])
    # changing a response-position logit does change it
    logits3 = logits.clone()
    logits3[0, 3, :] += torch.randn(V)
    assert not torch.allclose(sequence_logprobs(logits3, ids, mask)["sum_logprob"][0], out["sum_logprob"][0])


@pytest.mark.hf
def test_prompt_conditions_response_logprob_with_model(tokenizer, tiny_model):
    """Prompt tokens must influence the response log-prob (context) but contribute no terms."""
    from saferl.formatting import collate_dpo, encode_dpo_row
    from saferl.modeling import forward_logprobs

    row_a = {"row_id": "a", "prompt": "Tell me about cats.", "chosen": "Cats are small mammals.", "rejected": "No."}
    row_b = {"row_id": "b", "prompt": "Tell me about the moon.", "chosen": "Cats are small mammals.", "rejected": "No."}
    enc = [encode_dpo_row(tokenizer, r) for r in (row_a, row_b)]
    batch = collate_dpo(enc, tokenizer.pad_token_id)
    with torch.no_grad():
        out = forward_logprobs(tiny_model, batch, "cpu", "fp32")
    # same response, different prompt -> different conditional log-prob (prompt is context)
    assert not math.isclose(float(out["chosen_logps"][0]), float(out["chosen_logps"][1]), rel_tol=1e-4)
    # number of scored tokens equals completion length (prompt contributes zero terms)
    assert int(out["chosen_tokens"][0]) == enc[0]["chosen"]["completion_len"]
    # manual recomputation: sum over completion positions only
    ids = batch["input_ids"][0:1]
    with torch.no_grad():
        logits = tiny_model(input_ids=ids, attention_mask=batch["attention_mask"][0:1]).logits
    logp = torch.log_softmax(logits[0].float(), -1)
    p = enc[0]["chosen"]["prompt_len"]
    L = len(enc[0]["chosen"]["input_ids"])
    manual = sum(float(logp[t - 1, ids[0, t]]) for t in range(p, L))
    assert math.isclose(float(out["chosen_logps"][0]), manual, rel_tol=1e-4, abs_tol=1e-4)


@pytest.mark.hf
def test_reference_receives_no_gradient_and_policy_does(tokenizer, tiny_model):
    import copy

    from saferl.formatting import collate_dpo, encode_dpo_row
    from saferl.modeling import forward_logprobs

    policy = copy.deepcopy(tiny_model).train()
    reference = copy.deepcopy(tiny_model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    row = {"row_id": "a", "prompt": "Q?", "chosen": "Yes indeed.", "rejected": "Nope."}
    batch = collate_dpo([encode_dpo_row(tokenizer, row)], tokenizer.pad_token_id)
    with torch.no_grad():
        ref = forward_logprobs(reference, batch, "cpu", "fp32")
    pol = forward_logprobs(policy, batch, "cpu", "fp32")
    out = dpo_loss(pol["chosen_logps"], pol["rejected_logps"], ref["chosen_logps"], ref["rejected_logps"], beta=0.1)
    out["loss"].sum().backward()
    assert all(p.grad is None for p in reference.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    # at init policy == reference -> loss is exactly log 2
    assert math.isclose(float(out["loss"]), math.log(2), rel_tol=1e-5)
