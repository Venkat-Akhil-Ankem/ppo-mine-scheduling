"""
test_environment.py — Unit tests for MineSchedulingEnv.
Run:  pytest tests/ -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from instance_loader import generate_instance
from environment import MineSchedulingEnv


@pytest.fixture
def tiny_env():
    """2-bench × 2-row × 2-col = 8 blocks, 2 periods."""
    inst = generate_instance(n_benches=2, n_rows=2, n_cols=2,
                             n_periods=2, seed=0)
    return MineSchedulingEnv(inst)


@pytest.fixture
def small_env():
    inst = generate_instance(n_benches=3, n_rows=4, n_cols=4,
                             n_periods=3, seed=42)
    return MineSchedulingEnv(inst)


class TestReset:

    def test_returns_state_and_mask(self, tiny_env):
        state, avail = tiny_env.reset()
        assert isinstance(state, np.ndarray)
        assert isinstance(avail, np.ndarray)

    def test_state_dimension(self, tiny_env):
        state, _ = tiny_env.reset()
        assert len(state) == tiny_env.state_dim

    def test_only_surface_blocks_available_at_start(self, tiny_env):
        """Bench-0 blocks have no predecessors; they should be available."""
        _, avail = tiny_env.reset()
        for b in tiny_env.instance.blocks:
            if b["bench"] == 0:
                assert avail[b["id"]], f"Surface block {b['id']} should be available"

    def test_deep_blocks_not_available_at_start(self, tiny_env):
        _, avail = tiny_env.reset()
        for b in tiny_env.instance.blocks:
            if b["bench"] > 0 and b["predecessors"]:
                assert not avail[b["id"]], \
                    f"Block {b['id']} at bench {b['bench']} should not be available yet"

    def test_no_blocks_extracted_after_reset(self, tiny_env):
        tiny_env.reset()
        assert not tiny_env._extracted.any()

    def test_period_starts_at_one(self, tiny_env):
        tiny_env.reset()
        assert tiny_env._period == 1

    def test_capacity_used_zero_after_reset(self, tiny_env):
        tiny_env.reset()
        assert tiny_env._cap_used == 0.0


class TestStep:

    def test_valid_action_succeeds(self, tiny_env):
        state, avail = tiny_env.reset()
        action = int(np.where(avail)[0][0])
        result = tiny_env.step(action)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

    def test_invalid_action_raises(self, tiny_env):
        _, avail = tiny_env.reset()
        unavail = int(np.where(~avail)[0][0])
        with pytest.raises(ValueError):
            tiny_env.step(unavail)

    def test_extracted_block_not_in_available(self, tiny_env):
        _, avail = tiny_env.reset()
        action = int(np.where(avail)[0][0])
        result = tiny_env.step(action)
        assert not result.available_mask[action], \
            "Extracted block should not be in available set"

    def test_reward_positive_for_ore_block(self, tiny_env):
        _, avail = tiny_env.reset()
        # Find an ore block that is available
        ore_avail = [b["id"] for b in tiny_env.instance.blocks
                     if b["is_ore"] and avail[b["id"]]]
        if ore_avail:
            result = tiny_env.step(ore_avail[0])
            assert result.reward > 0

    def test_npv_accumulates(self, tiny_env):
        _, avail = tiny_env.reset()
        action = int(np.where(avail)[0][0])
        tiny_env.step(action)
        assert tiny_env._total_npv != 0.0

    def test_state_vector_length_consistent(self, tiny_env):
        state, avail = tiny_env.reset()
        action = int(np.where(avail)[0][0])
        result = tiny_env.step(action)
        assert len(result.next_state) == tiny_env.state_dim

    def test_episode_ends_when_no_blocks_left(self, tiny_env):
        """Run until done; all blocks should be extracted."""
        state, avail = tiny_env.reset()
        tiny_env.capacity = 1e9  # remove capacity constraint
        done = False
        steps = 0
        while not done and steps < 1000:
            if not avail.any():
                break
            action = int(np.where(avail)[0][0])
            result = tiny_env.step(action)
            avail  = result.available_mask
            done   = result.done
            steps += 1
        # Either all extracted or periods exhausted
        assert done


class TestGreedyRollout:

    def test_greedy_rollout_returns_float(self, small_env):
        npv = small_env.greedy_rollout()
        assert isinstance(npv, float)

    def test_greedy_rollout_positive_npv(self, small_env):
        npv = small_env.greedy_rollout()
        assert npv > 0, "Greedy should extract mostly ore blocks → positive NPV"

    def test_greedy_rollout_not_added_to_history(self, small_env):
        small_env.greedy_rollout()
        assert len(small_env.episode_npvs) == 0


class TestStateDimension:

    def test_state_dim_formula(self, tiny_env):
        assert tiny_env.state_dim == 5 * tiny_env.n_blocks + 2

    def test_state_in_reasonable_range(self, tiny_env):
        state, _ = tiny_env.reset()
        # Normalised features should mostly be in [0,1]
        binary_part = state[:2 * tiny_env.n_blocks]
        assert binary_part.min() >= 0.0
        assert binary_part.max() <= 1.0
