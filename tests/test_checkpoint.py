"""Checkpoint save/load must not alter the live model (regression for an MPS in-place corruption
seen with a fused ``.to("cpu", dtype)`` on parameters) and must round-trip the weights."""
import pytest
import torch

from saferl.modeling import load_model, save_checkpoint
from saferl.utils import pick_device

pytestmark = pytest.mark.hf


def _param_fingerprint(model):
    return [float(p.detach().float().abs().sum()) for p in model.parameters()]


def test_save_checkpoint_leaves_parameters_intact_and_round_trips(tokenizer, tiny_model, tmp_path):
    import copy

    device = pick_device("auto")
    model = copy.deepcopy(tiny_model).to(device).train()
    before = _param_fingerprint(model)
    save_checkpoint(model, tokenizer, tmp_path / "ckpt", "bfloat16", {"step": 1})
    after = _param_fingerprint(model)
    assert before == after, "save_checkpoint modified live parameters"
    assert (tmp_path / "ckpt" / "model.safetensors").exists()
    assert (tmp_path / "ckpt" / "trainer_state.json").exists()

    reloaded = load_model(str(tmp_path / "ckpt"), device, torch.float32)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), reloaded.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.detach().float().cpu(), p2.detach().float().cpu(), atol=2e-2, rtol=2e-2), n1  # bf16 round-trip
