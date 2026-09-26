"""
instance_loader.py
------------------
Loads and generates open-pit mine instances in MineLib format.

MineLib Format Reference
-------------------------
Espinoza, D. et al. (2013). MineLib: a library of open pit mining problems.
Annals of Operations Research, 206, 93–114.
https://mansci-web.uai.cl/minelib/

A MineLib instance consists of:
  - Blocks arranged on a 3-D grid (bench × row × column)
  - Each block has: economic value, tonnage, ore grade, rock type
  - Slope constraints: block (i) cannot be mined before its
    "predecessor" blocks directly above it in the cone of influence
    (the geomechanical stability cone determines which blocks above
     must be removed first — called slope or precedence constraints)
  - Time periods with capacity (tonnage) limits per period

This module generates synthetic instances following the MineLib structure
and saves them as JSON for use by the RL environment.

BLOCK INDEXING
--------------
Blocks are indexed on a 2-D grid (row, col) across multiple benches.
Bench 0 = surface (topmost); higher bench index = deeper in the pit.

PRECEDENCE RULE (simplified 1:2 cone)
--------------------------------------
Block at (bench b, row r, col c) can only be mined after ALL of:
  (bench b-1, row r,   col c)
  (bench b-1, row r-1, col c)
  (bench b-1, row r+1, col c)
  (bench b-1, row r,   col c-1)
  (bench b-1, row r,   col c+1)
  ... (only those within grid bounds)

This ensures pit wall stability — you must mine the surface before
digging deeper, following the slope angle.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Tuple


@dataclass
class MineBlock:
    """One block in a mine instance."""
    id:         int
    bench:      int        # 0 = surface, increasing = deeper
    row:        int
    col:        int
    npv:        float      # economic value (may be negative = waste)
    tonnage:    float      # tonnes of material
    grade:      float      # ore grade (g/t)
    is_ore:     bool       # True if ore block (npv > 0)
    predecessors: List[int] = field(default_factory=list)
                            # block ids that must be mined first


@dataclass
class MineInstance:
    """A complete mine scheduling instance."""
    name:            str
    n_blocks:        int
    n_benches:       int
    n_rows:          int
    n_cols:          int
    n_periods:       int
    capacity_tonnes: float      # max tonnage extractable per period
    discount_rate:   float      # annual discount rate
    blocks:          List[dict] = field(default_factory=list)


# ── Instance generation ────────────────────────────────────────────────────────

def generate_instance(
    name:            str   = "synthetic_small",
    n_benches:       int   = 3,
    n_rows:          int   = 4,
    n_cols:          int   = 4,
    n_periods:       int   = 3,
    capacity_frac:   float = 0.40,   # fraction of total tonnage extractable/period
    discount_rate:   float = 0.10,
    ore_probability: float = 0.65,   # fraction of ore blocks
    seed:            int   = 42,
) -> MineInstance:
    """
    Generate a synthetic MineLib-style open-pit instance.

    The pit is a (n_benches × n_rows × n_cols) grid of blocks.
    Surface blocks (bench 0) have no predecessors — they are always available.
    Deeper blocks require surface blocks above them to be mined first.

    Parameters
    ----------
    name            : instance name
    n_benches       : number of vertical benches (depth levels)
    n_rows, n_cols  : horizontal grid dimensions
    n_periods       : number of planning periods
    capacity_frac   : max fraction of total tonnage mined per period
    discount_rate   : NPV discount rate per period
    ore_probability : probability a block has positive economic value
    seed            : random seed

    Returns
    -------
    MineInstance dataclass
    """
    rng = np.random.default_rng(seed)
    n_blocks = n_benches * n_rows * n_cols

    # Map (bench, row, col) → block id
    def block_id(b, r, c):
        return b * n_rows * n_cols + r * n_cols + c

    blocks = []
    total_tonnage = 0.0

    for b in range(n_benches):
        for r in range(n_rows):
            for c in range(n_cols):
                bid = block_id(b, r, c)

                # Economic value: ore blocks have positive NPV, waste negative
                is_ore = rng.random() < ore_probability
                if is_ore:
                    grade   = float(rng.uniform(0.8, 4.0))
                    tonnage = float(rng.uniform(800, 2000))
                    npv     = float(round(grade * tonnage * 0.04 - tonnage * 0.01, 2))
                    npv     = max(1.0, npv)   # ensure positive
                else:
                    grade   = float(rng.uniform(0.0, 0.5))
                    tonnage = float(rng.uniform(500, 1500))
                    npv     = float(round(-tonnage * 0.008, 2))  # mining cost only

                total_tonnage += tonnage

                # Precedence: find blocks at bench b-1 directly above
                predecessors = []
                if b > 0:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            pr, pc = r + dr, c + dc
                            if 0 <= pr < n_rows and 0 <= pc < n_cols:
                                predecessors.append(block_id(b - 1, pr, pc))

                blocks.append(MineBlock(
                    id=bid, bench=b, row=r, col=c,
                    npv=npv, tonnage=tonnage, grade=grade,
                    is_ore=is_ore, predecessors=predecessors,
                ))

    capacity = total_tonnage / n_blocks * n_rows * n_cols * capacity_frac

    instance = MineInstance(
        name=name,
        n_blocks=n_blocks,
        n_benches=n_benches,
        n_rows=n_rows,
        n_cols=n_cols,
        n_periods=n_periods,
        capacity_tonnes=round(capacity, 1),
        discount_rate=discount_rate,
        blocks=[asdict(b) for b in blocks],
    )
    return instance


def save_instance(instance: MineInstance, path: str):
    """Save a MineInstance to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(instance), f, indent=2)
    ore_count = sum(1 for b in instance.blocks if b["is_ore"])
    print(f"✅ Saved '{instance.name}' → {path}")
    print(f"   {instance.n_blocks} blocks "
          f"({ore_count} ore, {instance.n_blocks - ore_count} waste)  |  "
          f"{instance.n_periods} periods  |  "
          f"capacity {instance.capacity_tonnes:.0f} t/period")


def load_instance(path: str) -> MineInstance:
    """Load a MineInstance from JSON."""
    with open(path) as f:
        data = json.load(f)
    inst = MineInstance(**{k: v for k, v in data.items() if k != "blocks"})
    inst.blocks = data["blocks"]
    return inst


# ── Derived helpers used by the environment ────────────────────────────────────

def build_predecessor_map(blocks: list) -> dict:
    """Return {block_id: set_of_predecessor_ids}."""
    return {b["id"]: set(b["predecessors"]) for b in blocks}


def compute_npv_bounds(blocks: list) -> Tuple[float, float]:
    """Return (min_npv, max_npv) for normalisation."""
    npvs = [b["npv"] for b in blocks]
    return min(npvs), max(npvs)


def greedy_npv_sequence(instance: MineInstance) -> list:
    """
    Greedy baseline: always mine the available block with highest NPV.
    Returns list of block ids in extraction order (ignoring capacity).
    """
    extracted  = set()
    preds      = build_predecessor_map(instance.blocks)
    block_npvs = {b["id"]: b["npv"] for b in instance.blocks}
    sequence   = []

    while len(extracted) < instance.n_blocks:
        available = [
            bid for bid in block_npvs
            if bid not in extracted and preds[bid].issubset(extracted)
        ]
        if not available:
            break
        best = max(available, key=lambda bid: block_npvs[bid])
        sequence.append(best)
        extracted.add(best)

    return sequence


if __name__ == "__main__":
    data_dir = Path("data")

    # Small: 3×4×4 = 48 blocks, 3 periods
    small = generate_instance(
        name="minelib_small", n_benches=3, n_rows=4, n_cols=4,
        n_periods=3, seed=42,
    )
    save_instance(small, str(data_dir / "small_instance.json"))

    # Medium: 4×5×5 = 100 blocks, 5 periods
    medium = generate_instance(
        name="minelib_medium", n_benches=4, n_rows=5, n_cols=5,
        n_periods=5, seed=7,
    )
    save_instance(medium, str(data_dir / "medium_instance.json"))

    print("\nGreedy sequence (first 10 blocks of small instance):")
    seq = greedy_npv_sequence(small)
    print([seq[i] for i in range(min(10, len(seq)))])
