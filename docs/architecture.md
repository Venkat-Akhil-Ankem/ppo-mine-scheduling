# Architecture & Design Notes

## Why PPO for Mine Scheduling?

Mine block sequencing is a **combinatorial sequential decision problem**:
at each step, choose one of the currently-available blocks to extract.
The action space changes every step (availability depends on what has been
extracted so far), and the reward is the discounted NPV of the chosen block.

PPO is well-suited because:
1. **Variable action sets** — handled via action masking (masked softmax)
2. **Long episodes** — GAE provides variance-reduced advantage estimates
3. **Stable training** — clipped surrogate prevents large policy jumps
4. **No model of the environment needed** — pure model-free RL

## MDP Formulation

```
State s_t:
  extracted_mask    [n_blocks]   — which blocks already mined
  available_mask    [n_blocks]   — which blocks legally mineable now
  npv_norm          [n_blocks]   — normalised block NPV values
  tonnage_norm      [n_blocks]   — normalised tonnages
  grade_norm        [n_blocks]   — normalised ore grades
  cap_remaining     [1]          — fraction of period capacity left
  period_fraction   [1]          — normalised current period

  Total: 5*n_blocks + 2 dimensions

Action a_t:
  Block index i ∈ available_blocks  (masked softmax enforces validity)

Reward r_t:
  npv[i] / (1 + discount_rate)^(period-1)   +   capacity_utilisation_bonus
```

## PPO Algorithm

```
Collect T-step trajectory τ = {(s_t, a_t, r_t, V_t, log π(a_t|s_t))}

Compute GAE advantages:
  δ_t  = r_t + γ V(s_{t+1}) - V(s_t)
  Â_t  = Σ_{k≥0} (γλ)^k δ_{t+k}
  Normalise Â (subtract mean, divide by std)

For K epochs:
  Sample mini-batch B ⊂ τ
  Compute:
    ρ_t = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))
    L_CLIP = E[ min(ρ_t Â_t, clip(ρ_t, 1-ε, 1+ε) Â_t) ]
    L_VF   = E[ (V_φ(s_t) - R_t)² ]
    L_ENT  = E[ H(π_θ(·|s_t)) ]
    Loss   = -L_CLIP + c₁ L_VF - c₂ L_ENT
  Gradient step with gradient clipping
```

## Action Masking

```python
# Logits for all n_blocks actions
logits = actor_head(features)           # (batch, n_blocks)

# Set unavailable blocks to -infinity
MASK = -1e9
masked = logits.masked_fill(~available, MASK)

# Softmax ignores -inf entries → prob ≈ 0 for masked blocks
dist = Categorical(logits=masked)
```

This is critical for correctness: the agent must NEVER select a block
that violates precedence constraints, or the extracted sequence would
violate pit slope stability.

## Hyperparameter Choices

| Parameter | Value | Rationale |
|---|---|---|
| γ (discount) | 0.99 | Long episode; future blocks matter |
| λ (GAE) | 0.95 | Standard balance of bias/variance |
| ε (clip) | 0.2 | Standard PPO clip range |
| K (epochs) | 4 | Re-use each trajectory 4 times |
| c₁ (value) | 0.5 | Standard value loss weight |
| c₂ (entropy) | 0.02 | Light exploration bonus |
| Backbone | 2-layer MLP, 128 hidden, LayerNorm | Stable for tabular RL |
| Init | Orthogonal | Recommended for PPO (OpenAI) |

## File Map

| File | Role |
|---|---|
| `instance_loader.py` | MineLib-format data; generation + I/O |
| `environment.py` | MDP: state, action, reward, period transitions |
| `network.py` | Actor-Critic MLP with masked softmax |
| `ppo_agent.py` | PPO: GAE, clipped surrogate, K-epoch update |
| `train.py` | Episode loop, logging, model saving, plots |
| `evaluate.py` | PPO vs Random, Greedy-NPV, Greedy-Grade |
