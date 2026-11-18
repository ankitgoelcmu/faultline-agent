"""Exploratory paired comparisons against the main grid's baseline (§VI).

  .venv/bin/python scripts/exploratory_paired.py

Each exploratory run reuses the main grid's seeds (seed from task, fault
rate and trial), so an episode pairs with the main-grid baseline episode of
the same task and trial at 20% faults. Reports the paired difference in
task success with a Newcombe 95% CI and an exact McNemar p (unadjusted;
exploratory). Covers: exactly-once instruction (instruct-*), read-back
tools (obs-*), and injection through error messages (errinj-*), reported per
defense: each defense's attacked episodes (both goals) are paired with the
clean-error baseline episode of the same task and trial. Pooling defenses
would reuse each baseline episode several times, so pairs would not be
independent. Within one defense each clean episode still serves both
goals, so the paper's primary estimate averages the goals per task and
trial (60 units) with a task-clustered bootstrap, plus each goal alone.
"""

import collections
import csv
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lab.stats import mcnemar_exact, paired_diff_ci   # noqa: E402

RES = os.path.join(ROOT, "results")


def rows(path):
    return [r for r in csv.DictReader(open(path)) if not r["stop"].startswith("error")]


def baseline(model):
    return {(r["task"], r["trial"]): int(r["passed"])
            for r in rows(os.path.join(RES, f"main-realistic-{model}", "episodes.csv"))
            if r["condition"] == "baseline" and float(r["fault_rate"]) == 0.2}


def report(label, pairs):
    both = sum(1 for a, b in pairs if a and b)
    first = sum(1 for a, b in pairs if a and not b)
    second = sum(1 for a, b in pairs if b and not a)
    d, lo, hi = paired_diff_ci(both, first, second, len(pairs) - both - first - second)
    rate = 100 * sum(b for _, b in pairs) / len(pairs)
    print(f"| {label} | {len(pairs)} | {rate:.0f}% | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
          f"{first}/{second} | {mcnemar_exact(first, second):.2g} |")


KEY_RUNS = [("sonnet", "gen-sonnet55-keys", "main-realistic-sonnet"),
            ("sonnet45", "gen-sonnet45", "gen-sonnet45"),
            ("gemma4-26b", "main-realistic-gemma4-26b", "main-realistic-gemma4-26b")]


def by_cond(folder, cond):
    path = os.path.join(RES, folder, "episodes.csv")
    return {(r["task"], r["trial"]): int(r["passed"]) for r in rows(path)
            if r["condition"] == cond and float(r["fault_rate"]) == 0.2}


def key_comparisons():
    """Agent-supplied keys (key_available) vs. no key (baseline) and vs.
    harness keys (safe_retry), paired by task and trial at 20% faults."""
    for label, kdir, bdir in KEY_RUNS:
        if not os.path.exists(os.path.join(RES, kdir, "episodes.csv")):
            continue
        agent = by_cond(kdir, "key_available")
        for cond in ("baseline", "safe_retry"):
            other = by_cond(bdir, cond)
            pairs = [(other[k], agent[k]) for k in agent if k in other]
            report(f"{label} agent keys vs. {cond}", pairs)


def main():
    print("| Run (vs. main-grid baseline, 20% faults) | Pairs | Success | Difference [95% CI] | "
          "discordant (base+/run+) | McNemar p |")
    print("|---|---|---|---|---|---|")
    for model in ("sonnet", "gemma4-26b", "qwen3-coder-next-latest"):
        main_path = os.path.join(RES, f"main-realistic-{model}", "episodes.csv")
        if not os.path.exists(main_path):
            continue
        base = baseline(model)
        for kind in ("instruct", "obs"):
            path = os.path.join(RES, f"{kind}-{model}", "episodes.csv")
            if not os.path.exists(path):
                continue
            run = [r for r in rows(path) if r["condition"] == "baseline" and float(r["fault_rate"]) == 0.2]
            pairs = [(base[(r["task"], r["trial"])], int(r["passed"])) for r in run
                     if (r["task"], r["trial"]) in base]
            report(f"{model} {kind}", pairs)
        path = os.path.join(RES, f"errinj-{model}", "errinj.csv")
        if os.path.exists(path):
            run = rows(path)
            for d in sorted({r["defense"] for r in run}):
                pairs = [(base[(r["task"], r["trial"])], int(r["task_passed"])) for r in run
                         if r["defense"] == d and (r["task"], r["trial"]) in base]
                report(f"{model} error-channel injection, defense={d}", pairs)
                for g in sorted({r["goal"] for r in run}):
                    report(f"{model} error-channel, defense={d}, goal={g}",
                           [(base[(r["task"], r["trial"])], int(r["task_passed"])) for r in run
                            if r["defense"] == d and r["goal"] == g and (r["task"], r["trial"]) in base])
                by = collections.defaultdict(list)
                for r in run:
                    if r["defense"] == d and (r["task"], r["trial"]) in base:
                        by[(r["task"], r["trial"])].append(int(r["task_passed"]))
                diffs = {k: statistics.mean(v) - base[k] for k, v in by.items()}
                tasks = sorted({k[0] for k in diffs})
                rng = random.Random(2026)
                boots = sorted(statistics.mean([diffs[k] for t in rng.choices(tasks, k=len(tasks))
                                                for k in diffs if k[0] == t]) for _ in range(10_000))
                print(f"| {model} error-channel, defense={d}, goals averaged per task and trial ({len(diffs)} units), task-clustered bootstrap | "
                      f"| | {100 * statistics.mean(diffs.values()):+.1f} "
                      f"[{100 * boots[250]:+.1f}, {100 * boots[9749]:+.1f}] | | |")


if __name__ == "__main__":
    main()
    key_comparisons()
