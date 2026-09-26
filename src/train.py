"""
train.py
--------
Training loop for PPO mine scheduling agent.

Procedure:
  For each episode:
    1. Reset environment
    2. Collect full trajectory (until episode ends)
       - Agent selects actions using current policy (stochastic)
       - Store (state, mask, action, reward, value, log_prob, done)
    3. Run PPO update on collected trajectory
    4. Log NPV, loss, entropy
    5. Save best model

Run:
    python src/train.py
"""

import sys
import time
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from instance_loader import load_instance, generate_instance, save_instance
from environment import MineSchedulingEnv
from ppo_agent import PPOAgent, Trajectory


# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    # Instance
    "instance_path": "data/small_instance.json",

    # Network
    "hidden_dim":    128,

    # PPO hyperparameters
    "lr":            3e-4,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_epsilon":  0.2,
    "value_coef":    0.5,
    "entropy_coef":  0.02,
    "n_epochs":      4,
    "batch_size":    32,
    "max_grad_norm": 0.5,

    # Training
    "n_episodes":    3000,
    "print_every":   200,

    # Output
    "output_dir":    "outputs",
    "model_path":    "outputs/ppo_mine.pth",
    "plot_path":     "outputs/training_curve.png",
}


# ── Episode rollout ────────────────────────────────────────────────────────────

def collect_episode(env: MineSchedulingEnv, agent: PPOAgent) -> tuple:
    """
    Run one full episode and collect the trajectory.

    Returns (trajectory, total_npv)
    """
    traj = Trajectory()
    state, available = env.reset()
    done = False

    while not done:
        if not available.any():
            break

        action, log_prob, value = agent.select_action(state, available)

        result = env.step(action)

        traj.states.append(state.copy())
        traj.available_masks.append(available.copy())
        traj.actions.append(action)
        traj.rewards.append(result.reward)
        traj.values.append(value)
        traj.log_probs.append(log_prob)
        traj.dones.append(result.done)

        state     = result.next_state
        available = result.available_mask
        done      = result.done

    return traj, env._total_npv


# ── Training ───────────────────────────────────────────────────────────────────

def train(config: dict = None) -> PPOAgent:
    if config is None:
        config = CONFIG

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Load or generate instance
    inst_path = Path(config["instance_path"])
    if not inst_path.exists():
        print("Instance not found — generating...")
        inst = generate_instance(name="minelib_small",
                                 n_benches=3, n_rows=4, n_cols=4,
                                 n_periods=3, seed=42)
        save_instance(inst, str(inst_path))
    else:
        inst = load_instance(str(inst_path))

    env = MineSchedulingEnv(inst)

    # Greedy baseline (fixed reference)
    greedy_npv = env.greedy_rollout()
    print(f"\nGreedy NPV baseline: {greedy_npv:.2f}")

    agent = PPOAgent(
        state_dim    = env.state_dim,
        n_blocks     = env.n_blocks,
        hidden_dim   = config["hidden_dim"],
        lr           = config["lr"],
        gamma        = config["gamma"],
        gae_lambda   = config["gae_lambda"],
        clip_epsilon = config["clip_epsilon"],
        value_coef   = config["value_coef"],
        entropy_coef = config["entropy_coef"],
        n_epochs     = config["n_epochs"],
        batch_size   = config["batch_size"],
        max_grad_norm= config["max_grad_norm"],
    )

    print("=" * 68)
    print("  PPO Mine Scheduling — Training")
    print("=" * 68)
    print(f"  Instance: {inst.name}  |  {inst.n_blocks} blocks  |  "
          f"{inst.n_periods} periods")
    print(f"  State dim: {env.state_dim}  |  "
          f"lr={config['lr']}  ε={config['clip_epsilon']}  "
          f"λ={config['gae_lambda']}")
    print()

    npv_window  = deque(maxlen=100)
    all_npvs:   list = []
    avg_npvs:   list = []
    all_losses: list = []
    all_entropy:list = []
    best_npv    = -np.inf

    print(f"{'Ep':>6}  {'NPV':>9}  {'Avg100':>9}  {'vs Greedy':>10}  "
          f"{'Loss':>9}  {'Entropy':>8}")
    print("─" * 68)

    t0 = time.time()

    for ep in range(1, config["n_episodes"] + 1):
        # Collect trajectory
        traj, ep_npv = collect_episode(env, agent)

        # PPO update
        metrics = agent.update(traj)

        npv_window.append(ep_npv)
        avg_npv = float(np.mean(npv_window))
        all_npvs.append(ep_npv)
        avg_npvs.append(avg_npv)
        all_losses.append(metrics["loss"])
        all_entropy.append(metrics["entropy"])

        # Save best
        if avg_npv > best_npv and ep >= 50:
            best_npv = avg_npv
            agent.save(config["model_path"])

        if ep % config["print_every"] == 0 or ep == 1:
            gap = (avg_npv - greedy_npv) / abs(greedy_npv) * 100 if greedy_npv != 0 else 0
            print(f"{ep:>6}  {ep_npv:>9.2f}  {avg_npv:>9.2f}  "
                  f"{gap:>+9.1f}%  {metrics['loss']:>9.5f}  "
                  f"{metrics['entropy']:>8.4f}")

    elapsed = time.time() - t0
    print(f"\n⏱  Training time: {elapsed:.1f}s")
    print(f"🏆 Best avg NPV: {best_npv:.2f}  (greedy: {greedy_npv:.2f})")

    _plot(all_npvs, avg_npvs, all_losses, all_entropy,
          greedy_npv, config["plot_path"])

    return agent


# ── Plotting ───────────────────────────────────────────────────────────────────

def _smooth(v: list, w: int = 30) -> list:
    return [float(np.mean(v[max(0, i-w):i+1])) for i in range(len(v))]


def _plot(npvs, avg_npvs, losses, entropy, greedy_npv, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    eps = list(range(1, len(npvs) + 1))

    # NPV
    ax = axes[0]
    ax.plot(eps, npvs,     color="#b0c4de", alpha=0.3, linewidth=0.7, label="Episode NPV")
    ax.plot(eps, avg_npvs, color="#2b5be0", linewidth=2,              label="100-ep avg")
    ax.axhline(greedy_npv, color="orange", linestyle="--", linewidth=1.5,
               label=f"Greedy ({greedy_npv:.1f})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Discounted NPV")
    ax.set_title("NPV During Training")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(eps, _smooth(losses, 30), color="#c0392b", linewidth=2)
    ax.set_xlabel("Episode"); ax.set_ylabel("PPO Loss")
    ax.set_title("PPO Loss (smoothed)"); ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[2]
    ax.plot(eps, _smooth(entropy, 30), color="#f0a500", linewidth=2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Policy Entropy")
    ax.set_title("Entropy (exploration)"); ax.grid(True, alpha=0.3)

    plt.suptitle("PPO Training — Open-Pit Mine Scheduling", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training curve saved → {save_path}")


if __name__ == "__main__":
    train()
