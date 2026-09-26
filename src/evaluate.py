"""
evaluate.py
-----------
Evaluates the trained PPO agent and compares it against baselines.

Baselines:
  1. Random policy    — uniformly random available block
  2. Greedy NPV       — always mine highest-NPV available block
  3. Greedy Grade     — always mine highest-grade available block
  4. PPO Agent        — learned policy (greedy/deterministic at eval)

Run:
    python src/evaluate.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from instance_loader import load_instance, generate_instance
from environment import MineSchedulingEnv
from ppo_agent import PPOAgent
from train import collect_episode, CONFIG


def run_random(env: MineSchedulingEnv, rng: np.random.Generator) -> float:
    """Random policy: choose uniformly from available blocks."""
    _, available = env.reset()
    done = False
    while not done:
        if not available.any():
            break
        candidates = np.where(available)[0]
        action     = int(rng.choice(candidates))
        result     = env.step(action)
        available  = result.available_mask
        done       = result.done
    npv = env._total_npv
    if env.episode_npvs:
        env.episode_npvs.pop()
        env.episode_steps.pop()
    return npv


def run_greedy_grade(env: MineSchedulingEnv) -> float:
    """Greedy-grade policy: always mine highest-grade available block."""
    _, available = env.reset()
    done = False
    while not done:
        if not available.any():
            break
        masked = np.where(available, env.grades, -np.inf)
        action = int(np.argmax(masked))
        result = env.step(action)
        available = result.available_mask
        done = result.done
    npv = env._total_npv
    if env.episode_npvs:
        env.episode_npvs.pop()
        env.episode_steps.pop()
    return npv


def run_ppo(env: MineSchedulingEnv, agent: PPOAgent) -> float:
    """PPO agent (greedy/deterministic at evaluation)."""
    state, available = env.reset()
    done = False
    while not done:
        if not available.any():
            break
        action, _, _ = agent.select_action(state, available, greedy=True)
        result = env.step(action)
        state = result.next_state
        available = result.available_mask
        done = result.done
    return env._total_npv


def main(n_trials: int = 50, seed: int = 999):
    inst_path  = Path(CONFIG["instance_path"])
    model_path = Path(CONFIG["model_path"])

    if not inst_path.exists():
        print(f"⚠️  Instance not found at {inst_path}. Generating...")
        from instance_loader import save_instance
        inst = generate_instance(name="minelib_small",
                                 n_benches=3, n_rows=4, n_cols=4,
                                 n_periods=3, seed=42)
        save_instance(inst, str(inst_path))

    if not model_path.exists():
        print(f"❌ No model at {model_path}. Run train.py first.")
        return

    inst = load_instance(str(inst_path))
    env  = MineSchedulingEnv(inst)

    agent = PPOAgent.load(str(model_path))

    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"  PPO Evaluation — {n_trials} trials  |  {inst.name}")
    print(f"{'='*60}")

    results = {}

    print("  Running Random policy...")
    results["Random"]       = [run_random(env, rng) for _ in range(n_trials)]

    print("  Running Greedy-NPV policy...")
    results["Greedy-NPV"]   = [env.greedy_rollout() for _ in range(n_trials)]

    print("  Running Greedy-Grade policy...")
    results["Greedy-Grade"] = [run_greedy_grade(env) for _ in range(n_trials)]

    print("  Running PPO agent...")
    results["PPO Agent"]    = [run_ppo(env, agent) for _ in range(n_trials)]

    best_greedy = max(
        np.mean(results["Greedy-NPV"]),
        np.mean(results["Greedy-Grade"]),
    )

    print(f"\n  {'Policy':22s}  {'Mean NPV':>9}  {'Std':>7}  {'Best':>8}  {'vs Best Greedy':>15}")
    print(f"  {'─'*70}")
    for name, vals in results.items():
        mean = np.mean(vals)
        std  = np.std(vals)
        best = np.max(vals)
        gap  = (mean - best_greedy) / abs(best_greedy) * 100 if best_greedy != 0 else 0
        tag  = " ← PPO" if "PPO" in name else ""
        print(f"  {name:22s}  {mean:>9.2f}  {std:>7.2f}  {best:>8.2f}  {gap:>+13.1f}%{tag}")

    ppo_vals  = np.array(results["PPO Agent"])
    rand_vals = np.array(results["Random"])
    gnpv_vals = np.array(results["Greedy-NPV"])

    print(f"\n  PPO beats Random on    {(ppo_vals > rand_vals).mean()*100:.1f}% of trials")
    print(f"  PPO beats Greedy-NPV on {(ppo_vals > gnpv_vals).mean()*100:.1f}% of trials")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
