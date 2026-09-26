# ⛏️ PPO Agent for Open-Pit Mine Block Sequencing

> A **Proximal Policy Optimization (PPO)** deep reinforcement learning agent
> that learns to sequence the extraction of ore blocks in an open-pit mine
> to **maximise Net Present Value (NPV)**, subject to geomechanical
> precedence constraints and per-period extraction capacity limits.
>
> Inspired by the **MineLib** open-pit mining benchmark library
> (Espinoza et al., 2013 — https://mansci-web.uai.cl/minelib/).
> Instance data follows the MineLib format: blocks with (x, y, z) coordinates,
> economic value, tonnage, and slope-based precedence constraints.

---

## 🌟 What It Does

The agent observes the current state of the pit (which blocks have been
extracted, which are exposed and available, remaining capacity) and decides
which block to extract next. Over thousands of training episodes it learns
a **policy** — a probability distribution over available actions — that
maximises the total discounted NPV of the extraction sequence.

This is a direct application of RL to the **Open-Pit Mine Production
Scheduling Problem (OPMPSP)**, one of the most studied NP-hard problems
in mining engineering.

---

## 🎯 Why PPO for Mine Scheduling?

Classical solvers (CPLEX, Gurobi) struggle with large OPMPSP instances
because the problem is NP-hard and constraint counts grow super-linearly.
Recent research (e.g., Zhang et al. 2020; Schulman et al. 2017) shows
that RL agents can learn high-quality heuristics that:

- Scale to large instances without re-solving from scratch
- Generalise across problem instances with similar structure
- Provide solutions in milliseconds at inference time

PPO specifically is chosen because:
- It is **stable and sample-efficient** (proximal clipping prevents large policy jumps)
- It handles **variable action spaces** (available blocks change each step)
- It is the **industry standard** for continuous and combinatorial control tasks

---

## 🏗️ Project Structure

```
ppo-mine-scheduling/
│
├── src/
│   ├── instance_loader.py  # Loads/generates MineLib-format mine instances
│   ├── environment.py      # Custom Gym-style mine scheduling MDP
│   ├── network.py          # Actor-Critic neural network (shared backbone)
│   ├── ppo_agent.py        # PPO algorithm: clipped surrogate objective
│   ├── train.py            # Training loop + curriculum + plots
│   └── evaluate.py         # Evaluate PPO vs greedy baselines
│
├── data/
│   ├── small_instance.json # 30-block, 3-period MineLib-format instance
│   └── medium_instance.json # 60-block, 5-period instance
│
├── outputs/                # Saved models + plots (auto-generated)
│
├── tests/
│   ├── test_environment.py # Unit tests for the mine MDP
│   ├── test_network.py     # Unit tests for Actor-Critic network
│   └── test_ppo.py         # Unit tests for PPO update logic
│
├── notebooks/
│   └── walkthrough.ipynb   # End-to-end Jupyter walkthrough
│
├── docs/
│   └── architecture.md     # PPO algorithm + mine scheduling formulation
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/ppo-mine-scheduling.git
cd ppo-mine-scheduling

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate MineLib-format instances
python src/instance_loader.py

# Train PPO agent
python src/train.py

# Evaluate vs baselines
python src/evaluate.py
```

---

## 💡 The Mine Scheduling MDP

```
┌────────────────────────────────────────────────────────────┐
│  OPEN-PIT MINE SCHEDULING as a Markov Decision Process     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  STATE  s_t:                                               │
│    • extracted[i]     — binary vector (which blocks done)  │
│    • available[i]     — binary vector (pit-slope exposed)  │
│    • period_remaining — capacity left this period          │
│    • period_idx       — current time period (normalised)   │
│    • block_features   — NPV, grade, tonnage per block      │
│                                                            │
│  ACTION  a_t:                                              │
│    • Select one block i from the available set             │
│    • Constraint: i must be geomechanically exposed         │
│      (all blocks directly above i already extracted)       │
│                                                            │
│  REWARD  r_t:                                              │
│    • Discounted NPV of extracted block:                    │
│      r_t = npv[i] / (1 + discount_rate)^period            │
│    • Penalty for extracting waste blocks                   │
│    • Period-end bonus for good capacity utilisation        │
│                                                            │
│  DONE:                                                     │
│    • All periods exhausted  OR  no blocks remain           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🧠 PPO Algorithm Flow

```
Collect trajectory (T steps):
  For t = 1..T:
    state  s_t     ← env.observe()
    action a_t     ~ π_θ(· | s_t)        [Actor samples from policy]
    value  V_t     ← V_φ(s_t)            [Critic estimates state value]
    r_t, s_{t+1}  ← env.step(a_t)

Compute advantages (GAE):
  δ_t = r_t + γ V(s_{t+1}) - V(s_t)    [TD error]
  Â_t = Σ_{k≥0} (γλ)^k δ_{t+k}        [GAE: exponential advantage smoothing]

PPO Update (K epochs over the trajectory):
  ratio   ρ_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   [importance ratio]

  L_CLIP = E[ min( ρ_t Â_t,  clip(ρ_t, 1-ε, 1+ε) Â_t ) ]  [clipped surrogate]
  L_VF   = E[ (V_φ(s_t) - R_t)² ]                            [value loss]
  L_ENT  = E[ H(π_θ(·|s_t)) ]                                [entropy bonus]

  Loss = -L_CLIP + c₁ L_VF - c₂ L_ENT

  θ, φ ← θ, φ - α ∇ Loss
```

The **clip(ρ, 1-ε, 1+ε)** is the key PPO innovation — it prevents the
policy from changing too drastically in one update, making training stable
without needing a KL-divergence constraint (unlike TRPO).

---

## 📊 Expected Results (small instance: 30 blocks, 3 periods)

| Policy | Total NPV | Gap to optimum |
|---|---|---|
| Random | ~35–45 | ~40–55% |
| Greedy (highest NPV first) | ~55–65 | ~15–25% |
| **PPO Agent** | **~70–80** | **~5–15%** |
| LP Relaxation (upper bound) | ~85 | 0% |

---

## 🤖 RL & Deep Learning Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **Policy gradient** | REINFORCE foundation of PPO |
| **Actor-Critic architecture** | Shared backbone, separate policy & value heads |
| **PPO clipped surrogate** | Core algorithm: prevents large policy updates |
| **Generalised Advantage Estimation (GAE)** | Variance-reduced advantage estimation |
| **Masked softmax** | Only available (precedence-valid) blocks selectable |
| **Action masking** | Critical for combinatorial RL environments |
| **Entropy regularisation** | Encourages exploration during training |
| **Value function baseline** | Reduces variance in policy gradient |
| **Mini-batch PPO updates** | K epochs per trajectory collection |
| **Curriculum learning** | Start small, grow instance size during training |

---

## 📦 Tech Stack

- **[PyTorch](https://pytorch.org/)** — Neural networks and autograd
- **[NumPy](https://numpy.org/)** — Numerical operations
- **[Matplotlib](https://matplotlib.org/)** — Training visualisation
- **[pytest](https://pytest.org/)** — Unit testing

No Gymnasium / Stable-Baselines dependency — environment and PPO are
implemented from scratch for full transparency.

---

## 🔗 Key References

> Espinoza, D. et al. (2013). **MineLib: a library of open pit mining problems.**
> *Annals of Operations Research*, 206, 93–114.

> Schulman, J. et al. (2017). **Proximal Policy Optimization Algorithms.**
> *arXiv:1707.06347.*

> Zhang, C. et al. (2020). **Learning to Dispatch for Job Shop Scheduling
> via Deep Reinforcement Learning.** *NeurIPS 2020.*

---

## 🔗 Related Projects

| Project | RL Method | Problem |
|---|---|---|
| [rl-job-scheduler](../rl-job-scheduler) | Tabular Q-learning | Parallel machine scheduling |
| [dqn-cartpole](../dqn-cartpole) | Deep Q-Network (DQN) | CartPole control |
| **ppo-mine-scheduling** (this) | PPO Actor-Critic | Open-pit mine sequencing |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 👤 Author

Built as a portfolio project at the intersection of **deep reinforcement
learning** and **open-pit mine production scheduling** — combining PhD
research expertise in large-scale MILP mine scheduling with modern
policy-gradient RL methods.
