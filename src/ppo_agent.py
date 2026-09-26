"""
ppo_agent.py
------------
Proximal Policy Optimisation (PPO) agent.
Schulman et al., 2017 — https://arxiv.org/abs/1707.06347

PPO is a policy-gradient algorithm that:
  1. Collects a trajectory (sequence of states, actions, rewards)
  2. Computes advantages using Generalised Advantage Estimation (GAE)
  3. Updates the policy using a CLIPPED surrogate objective
     (this is the key PPO innovation — prevents large policy changes)
  4. Repeats steps 1–3 until convergence

WHY PPO OVER OTHER ALGORITHMS?
-------------------------------
  - REINFORCE (vanilla PG): high variance, no value function baseline
  - A2C: better variance reduction, but no constraint on update size
  - TRPO: constrained optimisation, complex and slow
  - PPO: simple clipping constraint, stable, parallelisable → industry standard

THE CLIPPED SURROGATE OBJECTIVE
---------------------------------
Let:
  ρ_t = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)   (importance sampling ratio)
  Â_t = advantage estimate at step t

Unclipped objective (like REINFORCE):
  L = E[ ρ_t · Â_t ]
  Problem: if ρ_t is large (policy changed a lot), this can cause
           catastrophic updates.

PPO clips the ratio:
  L_CLIP = E[ min( ρ_t · Â_t,  clip(ρ_t, 1-ε, 1+ε) · Â_t ) ]

  The clip(1-ε, 1+ε) prevents ρ_t from being too large or too small.
  The min() ensures we never benefit from going OUTSIDE the clip range.

GENERALISED ADVANTAGE ESTIMATION (GAE)
---------------------------------------
Balances bias vs variance in advantage estimation:

  δ_t  = r_t + γ V(s_{t+1}) - V(s_t)        ← TD error
  Â_t  = Σ_{k=0}^{T-t} (γλ)^k δ_{t+k}      ← GAE

  λ=0: pure TD (low variance, high bias)
  λ=1: pure MC (high variance, low bias)
  λ=0.95: good empirical balance (standard)

FULL LOSS FUNCTION
-------------------
  L = -L_CLIP  +  c₁ · L_VALUE  -  c₂ · L_ENTROPY

  L_VALUE   = MSE(V(s_t), R_t)   ← fit critic to returns
  L_ENTROPY = H(π(·|s_t))        ← encourage exploration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List

from network import ActorCriticNetwork


@dataclass
class Trajectory:
    """Stores one episode's worth of experience for PPO update."""
    states:          List[np.ndarray] = field(default_factory=list)
    available_masks: List[np.ndarray] = field(default_factory=list)
    actions:         List[int]        = field(default_factory=list)
    rewards:         List[float]      = field(default_factory=list)
    values:          List[float]      = field(default_factory=list)
    log_probs:       List[float]      = field(default_factory=list)
    dones:           List[bool]       = field(default_factory=list)

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.states.clear()
        self.available_masks.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


class PPOAgent:
    """
    PPO agent for the mine scheduling environment.

    Parameters
    ----------
    state_dim          : observation vector dimension
    n_blocks           : number of mine blocks (= action space size)
    hidden_dim         : actor-critic hidden layer width
    lr                 : learning rate for Adam
    gamma              : discount factor γ
    gae_lambda         : GAE smoothing parameter λ
    clip_epsilon       : PPO clip range ε
    value_coef         : weight c₁ for value loss
    entropy_coef       : weight c₂ for entropy bonus
    n_epochs           : number of update epochs per trajectory
    batch_size         : mini-batch size for PPO updates
    max_grad_norm      : gradient clipping norm
    device             : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        state_dim:    int,
        n_blocks:     int,
        hidden_dim:   int   = 128,
        lr:           float = 3e-4,
        gamma:        float = 0.99,
        gae_lambda:   float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef:   float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs:     int   = 4,
        batch_size:   int   = 32,
        max_grad_norm:float = 0.5,
        device:       str   = "cpu",
    ):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.max_grad_norm = max_grad_norm
        self.device       = torch.device(device)

        # Network
        self.network = ActorCriticNetwork(
            state_dim=state_dim, n_blocks=n_blocks, hidden_dim=hidden_dim
        ).to(self.device)

        # Optimiser
        self.optimiser = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Training history
        self.update_count = 0
        self.loss_history:         List[float] = []
        self.policy_loss_history:  List[float] = []
        self.value_loss_history:   List[float] = []
        self.entropy_history:      List[float] = []

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(
        self,
        state:          np.ndarray,
        available_mask: np.ndarray,
        greedy:         bool = False,
    ) -> tuple:
        """
        Select an action and return (action, log_prob, value).
        Used during trajectory collection.
        """
        return self.network.act(state, available_mask, self.device, greedy=greedy)

    # ── GAE computation ────────────────────────────────────────────────────────

    def compute_gae(
        self,
        trajectory: Trajectory,
        last_value: float = 0.0,
    ) -> tuple:
        """
        Compute Generalised Advantage Estimates (GAE) and returns.

        GAE:  Â_t = Σ_{k≥0} (γλ)^k · δ_{t+k}
              where δ_t = r_t + γ V(s_{t+1}) - V(s_t)

        Returns (advantages, returns) as numpy arrays.
        """
        T         = len(trajectory)
        rewards   = trajectory.rewards
        values    = trajectory.values + [last_value]
        dones     = trajectory.dones

        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(T)):
            next_val   = values[t + 1] * (1.0 - float(dones[t]))
            delta      = rewards[t] + self.gamma * next_val - values[t]
            gae        = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages[t] = gae

        returns = advantages + np.array(values[:-1], dtype=np.float32)
        return advantages, returns

    # ── PPO update ─────────────────────────────────────────────────────────────

    def update(self, trajectory: Trajectory) -> dict:
        """
        Run PPO update over the collected trajectory.

        Steps:
          1. Compute GAE advantages and returns
          2. Convert trajectory to tensors
          3. For n_epochs: sample mini-batches and apply clipped update

        Returns dict of mean losses for logging.
        """
        # GAE
        last_val = 0.0   # terminal state has no future value
        advantages, returns = self.compute_gae(trajectory, last_val)

        # Normalise advantages (standard practice — reduces variance)
        adv_std  = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        T = len(trajectory)

        # Convert to tensors
        states_t   = torch.tensor(np.array(trajectory.states),
                                  dtype=torch.float32, device=self.device)
        masks_t    = torch.tensor(np.array(trajectory.available_masks),
                                  dtype=torch.bool,    device=self.device)
        actions_t  = torch.tensor(trajectory.actions,
                                  dtype=torch.long,    device=self.device)
        old_lps_t  = torch.tensor(trajectory.log_probs,
                                  dtype=torch.float32, device=self.device)
        advs_t     = torch.tensor(advantages,
                                  dtype=torch.float32, device=self.device)
        returns_t  = torch.tensor(returns,
                                  dtype=torch.float32, device=self.device)

        # K epochs of mini-batch updates
        total_loss_sum  = 0.0
        policy_loss_sum = 0.0
        value_loss_sum  = 0.0
        entropy_sum     = 0.0
        n_updates       = 0

        indices = np.arange(T)
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                if len(batch_idx) == 0:
                    continue

                b_states   = states_t[batch_idx]
                b_masks    = masks_t[batch_idx]
                b_actions  = actions_t[batch_idx]
                b_old_lps  = old_lps_t[batch_idx]
                b_advs     = advs_t[batch_idx]
                b_returns  = returns_t[batch_idx]

                # Evaluate actions under current policy
                log_probs, values, entropy = self.network.evaluate_actions(
                    b_states, b_masks, b_actions
                )

                # ── PPO clipped policy loss ────────────────────────────────────
                ratio       = torch.exp(log_probs - b_old_lps)
                surr1       = ratio * b_advs
                surr2       = torch.clamp(ratio, 1 - self.clip_epsilon,
                                                  1 + self.clip_epsilon) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value function loss (clipped) ──────────────────────────────
                value_loss  = nn.functional.mse_loss(values, b_returns)

                # ── Entropy bonus ──────────────────────────────────────────────
                entropy_loss = -entropy.mean()

                # ── Total loss ─────────────────────────────────────────────────
                loss = (policy_loss
                        + self.value_coef   * value_loss
                        + self.entropy_coef * entropy_loss)

                self.optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimiser.step()

                total_loss_sum  += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum  += value_loss.item()
                entropy_sum     += entropy.mean().item()
                n_updates       += 1

        self.update_count += 1
        n = max(1, n_updates)
        metrics = {
            "loss":        total_loss_sum  / n,
            "policy_loss": policy_loss_sum / n,
            "value_loss":  value_loss_sum  / n,
            "entropy":     entropy_sum     / n,
        }
        self.loss_history.append(metrics["loss"])
        self.policy_loss_history.append(metrics["policy_loss"])
        self.value_loss_history.append(metrics["value_loss"])
        self.entropy_history.append(metrics["entropy"])
        return metrics

    # ── Save / load ────────────────────────────────────────────────────────────

    def save(self, path: str):
        self.network.save(path, extra={"update_count": self.update_count})

    @classmethod
    def load(cls, path: str, **kwargs) -> "PPOAgent":
        device = kwargs.get("device", "cpu")
        import torch
        ckpt   = torch.load(path, map_location=device, weights_only=True)
        agent  = cls(
            state_dim  = ckpt["state_dim"],
            n_blocks   = ckpt["n_blocks"],
            hidden_dim = ckpt["hidden_dim"],
            device     = device,
        )
        agent.network.load_state_dict(ckpt["state_dict"])
        agent.update_count = ckpt.get("update_count", 0)
        print(f"✅ PPO agent loaded ({agent.update_count} updates)")
        return agent
