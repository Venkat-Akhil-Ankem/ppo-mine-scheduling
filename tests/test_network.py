"""
test_network.py — Unit tests for ActorCriticNetwork.
Run:  pytest tests/ -v
"""

import sys
import pytest
import tempfile
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from network import ActorCriticNetwork, count_parameters


N_BLOCKS  = 12
STATE_DIM = 5 * N_BLOCKS + 2


@pytest.fixture
def net():
    return ActorCriticNetwork(state_dim=STATE_DIM, n_blocks=N_BLOCKS, hidden_dim=64)


class TestForward:

    def test_output_shapes(self, net):
        state = torch.randn(4, STATE_DIM)
        mask  = torch.ones(4, N_BLOCKS, dtype=torch.bool)
        dist, value = net(state, mask)
        assert value.shape == (4,)
        assert dist.probs.shape == (4, N_BLOCKS)

    def test_probs_sum_to_one(self, net):
        state = torch.randn(8, STATE_DIM)
        mask  = torch.ones(8, N_BLOCKS, dtype=torch.bool)
        dist, _ = net(state, mask)
        sums = dist.probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_masked_actions_have_zero_prob(self, net):
        state = torch.randn(1, STATE_DIM)
        mask  = torch.zeros(1, N_BLOCKS, dtype=torch.bool)
        mask[:, :3] = True   # only first 3 available
        dist, _ = net(state, mask)
        assert dist.probs[0, 3:].sum().item() < 1e-5

    def test_no_nan_in_output(self, net):
        state = torch.randn(16, STATE_DIM)
        mask  = torch.ones(16, N_BLOCKS, dtype=torch.bool)
        dist, value = net(state, mask)
        assert not torch.isnan(dist.probs).any()
        assert not torch.isnan(value).any()


class TestAct:

    def test_act_returns_valid_action(self, net):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        mask  = np.ones(N_BLOCKS, dtype=bool)
        mask[5:] = False
        action, lp, val = net.act(state, mask, torch.device("cpu"))
        assert action in range(5), f"Action {action} outside available set"
        assert isinstance(lp, float)
        assert isinstance(val, float)

    def test_greedy_act_deterministic(self, net):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        mask  = np.ones(N_BLOCKS, dtype=bool)
        a1, _, _ = net.act(state, mask, torch.device("cpu"), greedy=True)
        a2, _, _ = net.act(state, mask, torch.device("cpu"), greedy=True)
        assert a1 == a2


class TestEvaluateActions:

    def test_log_probs_shape(self, net):
        states  = torch.randn(8, STATE_DIM)
        masks   = torch.ones(8, N_BLOCKS, dtype=torch.bool)
        actions = torch.randint(0, N_BLOCKS, (8,))
        lp, val, ent = net.evaluate_actions(states, masks, actions)
        assert lp.shape  == (8,)
        assert val.shape == (8,)
        assert ent.shape == (8,)

    def test_log_probs_non_positive(self, net):
        states  = torch.randn(4, STATE_DIM)
        masks   = torch.ones(4, N_BLOCKS, dtype=torch.bool)
        actions = torch.zeros(4, dtype=torch.long)
        lp, _, _ = net.evaluate_actions(states, masks, actions)
        assert (lp <= 0).all(), "Log probabilities must be non-positive"

    def test_entropy_non_negative(self, net):
        states  = torch.randn(4, STATE_DIM)
        masks   = torch.ones(4, N_BLOCKS, dtype=torch.bool)
        actions = torch.zeros(4, dtype=torch.long)
        _, _, ent = net.evaluate_actions(states, masks, actions)
        assert (ent >= 0).all()


class TestSaveLoad:

    def test_save_and_load_weights_match(self, net):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.pth")
            net.save(path)
            loaded = ActorCriticNetwork.load(path, torch.device("cpu"))
            s = torch.randn(2, STATE_DIM)
            m = torch.ones(2, N_BLOCKS, dtype=torch.bool)
            with torch.no_grad():
                _, v1 = net(s, m)
                _, v2 = loaded(s, m)
            assert torch.allclose(v1, v2)

    def test_parameters_nonzero_after_init(self, net):
        total = sum(p.abs().sum().item() for p in net.parameters())
        assert total > 0

    def test_parameter_count(self, net):
        n = count_parameters(net)
        assert n > 1000
