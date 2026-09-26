"""
network.py
----------
Actor-Critic neural network for PPO mine scheduling.

ACTOR-CRITIC ARCHITECTURE
--------------------------
PPO uses an Actor-Critic design with a SHARED backbone:

  State s
    │
    ▼
  ┌─────────────────────────────────┐
  │  Shared Backbone (MLP)          │  Extracts features common to
  │  Linear → LayerNorm → ReLU      │  both policy and value estimation
  │  Linear → LayerNorm → ReLU      │
  └──────────┬──────────────────────┘
             │  shared features
      ┌──────┴──────────┐
      │                 │
      ▼                 ▼
  Actor Head        Critic Head
  Linear(n_actions) Linear(1)
  + masked softmax  (no activation)
      │                 │
      ▼                 ▼
  π(a|s)            V(s)
  (policy)       (state value)

WHY SHARED BACKBONE?
  - Shared layers learn representations useful for BOTH tasks
  - More parameter-efficient than two separate networks
  - Standard practice in PPO implementations (e.g. OpenAI baselines)

WHY LAYER NORM (not batch norm)?
  - Layer norm operates per-sample, not per-batch
  - More stable when batch sizes are small or vary
  - Does not require running statistics (simpler)

ACTION MASKING
--------------
The mine environment has variable action sets — only blocks satisfying
precedence constraints and capacity limits are valid at each step.

We implement this as a MASKED SOFTMAX:
  1. Compute raw logits for all n_blocks actions
  2. Set logits of INVALID actions to -1e9 (effectively -infinity)
  3. Apply softmax → invalid actions get probability ≈ 0
  4. Sample from the resulting distribution (or take argmax for greedy)

This is the standard approach for combinatorial RL with varying action sets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class ActorCriticNetwork(nn.Module):
    """
    Shared-backbone Actor-Critic for PPO mine scheduling.

    Parameters
    ----------
    state_dim   : dimension of the state observation vector
    n_blocks    : number of blocks = number of possible actions
    hidden_dim  : width of each hidden layer
    n_layers    : number of shared backbone layers
    """

    def __init__(
        self,
        state_dim:  int,
        n_blocks:   int,
        hidden_dim: int = 128,
        n_layers:   int = 2,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.n_blocks   = n_blocks
        self.hidden_dim = hidden_dim

        # ── Shared backbone ────────────────────────────────────────────────────
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # ── Actor head: outputs logits for each block ──────────────────────────
        self.actor_head  = nn.Linear(hidden_dim, n_blocks)

        # ── Critic head: outputs scalar state value ────────────────────────────
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialisation — recommended for PPO by OpenAI."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Smaller gain for output heads
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(
        self,
        state:          torch.Tensor,
        available_mask: torch.Tensor,
    ) -> tuple:
        """
        Forward pass.

        Parameters
        ----------
        state          : (batch, state_dim) float tensor
        available_mask : (batch, n_blocks) bool tensor —
                         True = block is available to mine

        Returns
        -------
        dist  : masked categorical distribution over blocks
        value : (batch,) state value estimates
        """
        features = self.backbone(state)
        logits   = self.actor_head(features)    # (batch, n_blocks)
        value    = self.critic_head(features).squeeze(-1)  # (batch,)

        # Mask invalid actions: set their logits to very negative number
        MASK_VALUE = -1e9
        masked_logits = logits.masked_fill(~available_mask, MASK_VALUE)

        dist = torch.distributions.Categorical(logits=masked_logits)
        return dist, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return critic value only (used during GAE computation)."""
        features = self.backbone(state)
        return self.critic_head(features).squeeze(-1)

    def act(
        self,
        state:          np.ndarray,
        available_mask: np.ndarray,
        device:         torch.device,
        greedy:         bool = False,
    ) -> tuple:
        """
        Select one action for a single state (no gradient).

        Parameters
        ----------
        state          : 1-D numpy array
        available_mask : 1-D boolean numpy array
        device         : torch device
        greedy         : if True, argmax instead of sampling

        Returns
        -------
        action      : int
        log_prob    : float (log probability of chosen action)
        value       : float (critic estimate)
        """
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.tensor(available_mask, dtype=torch.bool,  device=device).unsqueeze(0)
            dist, val = self.forward(s, m)
            action    = dist.probs.argmax(dim=-1) if greedy else dist.sample()
            log_prob  = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(val.item())

    def evaluate_actions(
        self,
        states:          torch.Tensor,
        available_masks: torch.Tensor,
        actions:         torch.Tensor,
    ) -> tuple:
        """
        Evaluate a batch of (state, action) pairs for PPO update.

        Returns
        -------
        log_probs : (batch,) log probabilities of actions under current policy
        values    : (batch,) state value estimates
        entropy   : (batch,) policy entropy (for entropy bonus)
        """
        dist, values = self.forward(states, available_masks)
        log_probs    = dist.log_prob(actions)
        entropy      = dist.entropy()
        return log_probs, values, entropy

    def save(self, path: str, extra: dict = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": self.state_dict(),
            "state_dim":  self.state_dim,
            "n_blocks":   self.n_blocks,
            "hidden_dim": self.hidden_dim,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        print(f"💾 Model saved → {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> "ActorCriticNetwork":
        ckpt  = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            state_dim  = ckpt["state_dim"],
            n_blocks   = ckpt["n_blocks"],
            hidden_dim = ckpt["hidden_dim"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"✅ Model loaded from {path}")
        return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    n_blocks  = 48
    state_dim = 5 * n_blocks + 2
    net = ActorCriticNetwork(state_dim=state_dim, n_blocks=n_blocks)
    print(f"Network: state_dim={state_dim}, n_blocks={n_blocks}")
    print(f"Parameters: {count_parameters(net):,}")

    state   = torch.randn(4, state_dim)
    avail   = torch.ones(4, n_blocks, dtype=torch.bool)
    avail[:, 10:] = False   # only first 10 blocks available

    dist, value = net(state, avail)
    action  = dist.sample()
    lp      = dist.log_prob(action)
    entropy = dist.entropy()

    print(f"Action shape:  {action.shape}")
    print(f"Value shape:   {value.shape}")
    print(f"Entropy mean:  {entropy.mean().item():.3f}")

    # Verify masked actions have prob ≈ 0
    probs = dist.probs[0]
    assert probs[10:].sum().item() < 1e-5, "Masked actions should have ~0 probability"
    print("✅ Action masking verified.")
    print("✅ Network check passed.")
