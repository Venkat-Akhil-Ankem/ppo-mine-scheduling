"""
test_ppo.py — Unit tests for PPO agent (GAE, update, save/load).
Run:  pytest tests/ -v
"""

import sys
import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ppo_agent import PPOAgent, Trajectory


N_BLOCKS  = 10
STATE_DIM = 5 * N_BLOCKS + 2


def make_agent(**kwargs) -> PPOAgent:
    defaults = dict(
        state_dim=STATE_DIM, n_blocks=N_BLOCKS,
        hidden_dim=32, lr=1e-3, n_epochs=2, batch_size=8,
    )
    defaults.update(kwargs)
    return PPOAgent(**defaults)


def make_trajectory(n: int = 20) -> Trajectory:
    traj = Trajectory()
    for _ in range(n):
        traj.states.append(np.random.randn(STATE_DIM).astype(np.float32))
        mask = np.zeros(N_BLOCKS, dtype=bool)
        mask[:5] = True
        traj.available_masks.append(mask)
        traj.actions.append(np.random.randint(0, 5))
        traj.rewards.append(float(np.random.randn()))
        traj.values.append(float(np.random.randn()))
        traj.log_probs.append(float(np.random.randn()))
        traj.dones.append(False)
    traj.dones[-1] = True
    return traj


class TestTrajectory:

    def test_len(self):
        traj = make_trajectory(15)
        assert len(traj) == 15

    def test_clear(self):
        traj = make_trajectory(10)
        traj.clear()
        assert len(traj) == 0


class TestGAE:

    def test_gae_output_shape(self):
        agent = make_agent()
        traj  = make_trajectory(20)
        adv, ret = agent.compute_gae(traj)
        assert adv.shape == (20,)
        assert ret.shape == (20,)

    def test_gae_returns_are_finite(self):
        agent = make_agent()
        traj  = make_trajectory(20)
        adv, ret = agent.compute_gae(traj)
        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

    def test_terminal_no_future_value(self):
        """With done=True at last step, last future value should be 0."""
        agent = make_agent(gamma=0.99, gae_lambda=0.95)
        traj  = make_trajectory(5)
        traj.dones[-1] = True
        traj.rewards   = [0.0, 0.0, 0.0, 0.0, 10.0]
        traj.values    = [0.0, 0.0, 0.0, 0.0, 0.0]
        adv, ret = agent.compute_gae(traj, last_value=0.0)
        # Final return should equal final reward (no bootstrap)
        assert abs(ret[-1] - 10.0) < 1e-4


class TestUpdate:

    def test_update_returns_metrics_dict(self):
        agent   = make_agent()
        traj    = make_trajectory(20)
        metrics = agent.update(traj)
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_update_increments_counter(self):
        agent = make_agent()
        assert agent.update_count == 0
        agent.update(make_trajectory(16))
        assert agent.update_count == 1

    def test_update_changes_weights(self):
        agent  = make_agent()
        before = [p.clone() for p in agent.network.parameters()]
        agent.update(make_trajectory(20))
        after  = list(agent.network.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed, "Update should change network weights"

    def test_loss_history_grows(self):
        agent = make_agent()
        for _ in range(3):
            agent.update(make_trajectory(16))
        assert len(agent.loss_history) == 3

    def test_metrics_are_finite(self):
        agent   = make_agent()
        metrics = agent.update(make_trajectory(20))
        for k, v in metrics.items():
            assert np.isfinite(v), f"Metric {k} = {v} is not finite"


class TestSelectAction:

    def test_action_in_available_set(self):
        agent = make_agent()
        state = np.random.randn(STATE_DIM).astype(np.float32)
        mask  = np.zeros(N_BLOCKS, dtype=bool)
        mask[:4] = True
        action, lp, val = agent.select_action(state, mask)
        assert action < 4
        assert isinstance(lp,  float)
        assert isinstance(val, float)

    def test_greedy_deterministic(self):
        agent = make_agent()
        state = np.random.randn(STATE_DIM).astype(np.float32)
        mask  = np.ones(N_BLOCKS, dtype=bool)
        a1, _, _ = agent.select_action(state, mask, greedy=True)
        a2, _, _ = agent.select_action(state, mask, greedy=True)
        assert a1 == a2


class TestSaveLoad:

    def test_save_and_load(self):
        agent = make_agent()
        agent.update(make_trajectory(16))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ppo.pth")
            agent.save(path)
            loaded = PPOAgent.load(path)
            assert loaded.update_count == agent.update_count
