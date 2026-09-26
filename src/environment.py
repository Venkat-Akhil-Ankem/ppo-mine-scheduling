"""
environment.py
--------------
Custom RL environment for open-pit mine block sequencing.

This implements the mine scheduling problem as a Markov Decision Process
(MDP) suitable for policy-gradient algorithms like PPO.

STATE VECTOR (normalised, fixed-size)
--------------------------------------
The state fed to the neural network is a concatenation of:

  [1]  extracted_mask        : float[n_blocks]  — 1.0 if block already mined
  [2]  available_mask        : float[n_blocks]  — 1.0 if block is exposed
                               (all predecessors extracted)
  [3]  block_npvs_norm       : float[n_blocks]  — normalised NPV per block
  [4]  block_tonnage_norm    : float[n_blocks]  — normalised tonnage
  [5]  block_grade_norm      : float[n_blocks]  — normalised ore grade
  [6]  capacity_remaining    : float[1]          — fraction of period capacity left
  [7]  period_fraction       : float[1]          — current period / total periods

Total state dim = 5 * n_blocks + 2

ACTION
------
Select one block index i ∈ {0, ..., n_blocks-1}.
The environment enforces that only available blocks can be chosen.
Invalid actions raise an exception (used in tests); in training, the
PPO agent uses a masked softmax to only sample from valid actions.

REWARD
------
  r = discounted_npv(block) = npv[i] / (1 + discount_rate) ^ (period - 1)
  + capacity_utilisation_bonus at period end

Waste blocks (npv < 0) give negative reward, incentivising the agent to
avoid unnecessary waste extraction.

PERIOD TRANSITIONS
------------------
After the capacity for the current period is used up (total tonnage extracted
≥ capacity_per_period), the period advances by 1 and capacity resets.
The episode ends when all periods are exhausted or no blocks remain.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from instance_loader import MineInstance, build_predecessor_map, compute_npv_bounds


@dataclass
class StepResult:
    next_state:   np.ndarray
    reward:       float
    done:         bool
    available_mask: np.ndarray   # boolean mask of valid next actions
    info:         dict


class MineSchedulingEnv:
    """
    Open-pit mine block sequencing environment for PPO.

    Parameters
    ----------
    instance        : MineInstance (loaded from JSON or generated)
    discount_rate   : annual NPV discount rate (overrides instance value if set)
    capacity_bonus  : reward bonus per period for good capacity utilisation
    """

    def __init__(
        self,
        instance:       MineInstance,
        discount_rate:  Optional[float] = None,
        capacity_bonus: float = 2.0,
    ):
        self.instance       = instance
        self.n_blocks       = instance.n_blocks
        self.n_periods      = instance.n_periods
        self.capacity       = instance.capacity_tonnes
        self.discount_rate  = discount_rate or instance.discount_rate
        self.capacity_bonus = capacity_bonus

        # Block data arrays (indexed by block id)
        self.npvs      = np.array([b["npv"]     for b in instance.blocks], dtype=np.float32)
        self.tonnages  = np.array([b["tonnage"] for b in instance.blocks], dtype=np.float32)
        self.grades    = np.array([b["grade"]   for b in instance.blocks], dtype=np.float32)

        # Normalisation constants
        self._npv_min, self._npv_max = compute_npv_bounds(instance.blocks)
        self._ton_max  = float(self.tonnages.max())
        self._grad_max = float(self.grades.max()) if self.grades.max() > 0 else 1.0

        # Predecessor lookup {block_id: set of predecessor ids}
        self._preds = build_predecessor_map(instance.blocks)

        # State dimension
        self.state_dim = 5 * self.n_blocks + 2

        # Episode state (initialised in reset())
        self._extracted:    np.ndarray = np.zeros(self.n_blocks, dtype=bool)
        self._period:       int        = 1
        self._cap_used:     float      = 0.0
        self._total_npv:    float      = 0.0
        self._steps:        int        = 0

        # History
        self.episode_npvs:  List[float] = []
        self.episode_steps: List[int]   = []

    # ── Core MDP interface ─────────────────────────────────────────────────────

    def reset(self) -> tuple:
        """Reset to the start of a new episode."""
        self._extracted = np.zeros(self.n_blocks, dtype=bool)
        self._period    = 1
        self._cap_used  = 0.0
        self._total_npv = 0.0
        self._steps     = 0

        available = self._compute_available()
        state     = self._build_state(available)
        return state, available

    def step(self, action: int) -> StepResult:
        """
        Extract block `action`.

        Returns StepResult with next_state, reward, done, available_mask, info.
        """
        available = self._compute_available()

        if not available[action]:
            raise ValueError(
                f"Block {action} is not available. "
                f"Predecessors not yet extracted: "
                f"{self._preds[action] - set(np.where(self._extracted)[0])}"
            )

        # Extract block
        self._extracted[action] = True
        self._cap_used         += self.tonnages[action]
        self._steps            += 1

        # Discounted NPV reward
        discount = 1.0 / (1.0 + self.discount_rate) ** (self._period - 1)
        reward   = float(self.npvs[action]) * discount
        self._total_npv += reward

        info = {
            "block_id":    action,
            "block_npv":   self.npvs[action],
            "period":      self._period,
            "cap_used":    self._cap_used,
            "total_npv":   self._total_npv,
        }

        # Period transition: advance when capacity is used up
        period_end_bonus = 0.0
        if self._cap_used >= self.capacity:
            utilisation = min(1.0, self._cap_used / self.capacity)
            period_end_bonus = self.capacity_bonus * utilisation
            reward += period_end_bonus
            self._period   += 1
            self._cap_used  = 0.0
            info["period_ended"]  = True
            info["utilisation"]   = utilisation

        # Check terminal condition
        all_extracted = bool(self._extracted.all())
        periods_done  = self._period > self.n_periods
        done          = all_extracted or periods_done

        if done:
            self.episode_npvs.append(self._total_npv)
            self.episode_steps.append(self._steps)

        next_available = self._compute_available() if not done else np.zeros(self.n_blocks, dtype=bool)
        next_state     = self._build_state(next_available)

        return StepResult(
            next_state=next_state,
            reward=reward,
            done=done,
            available_mask=next_available,
            info=info,
        )

    # ── State construction ─────────────────────────────────────────────────────

    def _compute_available(self) -> np.ndarray:
        """
        Compute which blocks are currently available to mine.
        A block is available iff:
          1. It has not yet been extracted.
          2. All its predecessors have been extracted.
          3. The current period has remaining capacity for its tonnage.
        """
        available = np.zeros(self.n_blocks, dtype=bool)
        extracted_set = set(np.where(self._extracted)[0])

        for bid in range(self.n_blocks):
            if self._extracted[bid]:
                continue
            if not self._preds[bid].issubset(extracted_set):
                continue
            if self._cap_used + self.tonnages[bid] > self.capacity * 1.05:
                # Allow slight over-run to avoid deadlock at period end
                continue
            available[bid] = True

        return available

    def _build_state(self, available: np.ndarray) -> np.ndarray:
        """Construct the fixed-size normalised state vector."""
        npv_range = max(self._npv_max - self._npv_min, 1e-6)

        state = np.concatenate([
            self._extracted.astype(np.float32),                          # extracted mask
            available.astype(np.float32),                                # available mask
            ((self.npvs - self._npv_min) / npv_range).astype(np.float32),  # NPV normalised
            (self.tonnages / self._ton_max).astype(np.float32),          # tonnage norm
            (self.grades   / self._grad_max).astype(np.float32),         # grade norm
            np.array([
                max(0.0, 1.0 - self._cap_used / self.capacity),          # capacity fraction
                (self._period - 1) / max(1, self.n_periods - 1),         # period fraction
            ], dtype=np.float32),
        ])
        return state

    # ── Utility ────────────────────────────────────────────────────────────────

    @property
    def best_episode_npv(self) -> float:
        return max(self.episode_npvs) if self.episode_npvs else 0.0

    @property
    def avg_episode_npv(self) -> float:
        return float(np.mean(self.episode_npvs[-50:])) if self.episode_npvs else 0.0

    def greedy_rollout(self) -> float:
        """
        Run a greedy-NPV episode (highest available NPV first).
        Returns total discounted NPV. Does NOT modify episode history.
        """
        state, available = self.reset()
        total = 0.0
        done  = False
        while not done:
            if not available.any():
                break
            # Greedy: pick available block with highest NPV
            masked_npv = np.where(available, self.npvs, -np.inf)
            action = int(np.argmax(masked_npv))
            result = self.step(action)
            total  = result.info["total_npv"]
            done   = result.done
            available = result.available_mask
        # Remove from history (this is just a baseline measurement)
        if self.episode_npvs:
            self.episode_npvs.pop()
            self.episode_steps.pop()
        return total

    def __repr__(self) -> str:
        return (f"MineSchedulingEnv(blocks={self.n_blocks}, "
                f"periods={self.n_periods}, "
                f"capacity={self.capacity:.0f}t)")


if __name__ == "__main__":
    from instance_loader import generate_instance
    inst = generate_instance(n_benches=2, n_rows=3, n_cols=3, n_periods=2, seed=0)
    env  = MineSchedulingEnv(inst)
    state, avail = env.reset()
    print(f"Env: {env}")
    print(f"State dim: {len(state)}")
    print(f"Available blocks at start: {np.where(avail)[0].tolist()}")

    # Take one step
    action = int(np.where(avail)[0][0])
    result = env.step(action)
    print(f"Extracted block {action} | reward={result.reward:.2f} | done={result.done}")
    print("✅ Environment check passed.")
