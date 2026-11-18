"""Task-clustered sensitivity analysis and a pass^k independence check.

  .venv/bin/python scripts/clustered_stats.py

Episode-level tests treat the 150 episodes of a cell as independent, but they
share 30 tasks (5 trials each). Here the paired difference is recomputed with
a cluster bootstrap: tasks are resampled with replacement, each keeping all
its trials, so the interval reflects task-to-task variation.

pass^k: if every attempt succeeded independently with the overall success
rate p, pass^k would be p^k. Observed pass^k above p^k means failures
concentrate on particular tasks (they repeat), not that consistency decays
faster than success.
"""

import csv
import os
import random
import sys
from collections import defaultdict
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab.stats import paired_diff_ci   # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
B = 10_000
COMPARISONS = [("baseline", "retry"), ("retry", "safe_retry"), ("baseline", "safe_retry")]


def load(model):
    rows = [r for r in csv.DictReader(open(os.path.join(RES, f"main-realistic-{model}", "episodes.csv")))
            if not r["stop"].startswith("error")]
    by = {}
    for r in rows:
        by[(r["condition"], r["fault_rate"], r["task"], r["trial"])] = int(r["passed"])
    return by


def clustered(by, c1, c2, rate, rng):
    tasks = sorted({k[2] for k in by if k[1] == rate})
    pairs = defaultdict(list)
    for t in tasks:
        for trial in range(1, 6):
            a, b = by.get((c1, rate, t, str(trial))), by.get((c2, rate, t, str(trial)))
            if a is not None and b is not None:
                pairs[t].append((a, b))
    flat = [p for t in tasks for p in pairs[t]]
    n = len(flat)
    both = sum(1 for a, b in flat if a and b)
    first = sum(1 for a, b in flat if a and not b)
    second = sum(1 for a, b in flat if b and not a)
    d, lo, hi = paired_diff_ci(both, first, second, n - both - first - second)
    boots = []
    for _ in range(B):
        sample = [p for t in rng.choices(tasks, k=len(tasks)) for p in pairs[t]]
        boots.append(sum(b - a for a, b in sample) / len(sample))
    boots.sort()
    return d, lo, hi, boots[int(0.025 * B)], boots[int(0.975 * B) - 1], n


def passk_check(by, cond, rate, k=5):
    per_task = defaultdict(list)
    for (c, x, t, _), v in by.items():
        if c == cond and x == rate:
            per_task[t].append(v)
    p = sum(sum(v) for v in per_task.values()) / sum(len(v) for v in per_task.values())
    observed = sum(comb(sum(v), k) / comb(len(v), k) for v in per_task.values() if len(v) >= k) / len(per_task)
    return p, observed, p ** k


def main():
    rng = random.Random(2026)
    for model in ("sonnet", "haiku", "gemma4-26b", "qwen3-coder-next-latest"):
        path = os.path.join(RES, f"main-realistic-{model}", "episodes.csv")
        if not os.path.exists(path):
            continue
        by = load(model)
        print(f"\n## {model}\n")
        print("| Comparison (20% faults) | Episode-level diff [95% CI] | Task-clustered bootstrap 95% CI |")
        print("|---|---|---|")
        for c1, c2 in COMPARISONS:
            d, lo, hi, blo, bhi, n = clustered(by, c1, c2, "0.2", rng)
            print(f"| {c1} → {c2} (n={n}) | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
                  f"[{100 * blo:+.1f}, {100 * bhi:+.1f}] |")
        print("\n| baseline | pass^1 | observed pass^5 | p^5 if attempts were independent |")
        print("|---|---|---|---|")
        for rate in ("0.1", "0.2"):
            p, obs, ind = passk_check(by, "baseline", rate)
            print(f"| {100 * float(rate):.0f}% faults | {100 * p:.0f}% | {100 * obs:.0f}% | {100 * ind:.0f}% |")


if __name__ == "__main__":
    main()
