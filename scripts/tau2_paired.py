"""Paired τ²-bench comparisons: idempotent vs. baseline, per model and domain.

  .venv/bin/python scripts/tau2_paired.py

A matched pair is the same (task_id, trial) under both conditions; both
conditions use the same seed. Each task has 2 trials. Reports the paired
difference with a Newcombe interval and exact McNemar p (these treat the 2
trials of a task as independent), a task-clustered bootstrap interval
(tasks resampled with both their trials; 10,000 resamples), and the H12
interaction (airline gain minus retail gain), bootstrapped the same way with
each domain resampled separately. For H12 it also gives a normal
approximation combining the two Newcombe intervals (SE = half-width / 1.96,
domains independent), the wider and primary interval. The same comparison at 0% faults is the
noise floor: there the two conditions are behaviourally identical.
"""

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lab.stats import mcnemar_exact, paired_diff_ci   # noqa: E402

RES = os.path.join(ROOT, "results", "tau2")
B = 10_000


def load(name):
    path = os.path.join(RES, name, "results.json")
    if not os.path.exists(path):
        return None
    sims = json.load(open(path))["simulations"]
    return {(s["task_id"], s.get("trial", 0)): int(((s.get("reward_info") or {}).get("reward") or 0) >= 1)
            for s in sims if s.get("reward_info")}


def pairs(model, domain, rate):
    a, b = load(f"{model}-{domain}-baseline-p{rate}"), load(f"{model}-{domain}-idempotent-p{rate}")
    if a is None or b is None:
        return None
    by_task = {}
    for k in sorted(set(a) & set(b)):
        by_task.setdefault(k[0], []).append((a[k], b[k]))
    return by_task


def summary(ps):
    both = sum(1 for x, y in ps if x and y)
    only_a = sum(1 for x, y in ps if x and not y)
    only_b = sum(1 for x, y in ps if y and not x)
    d, lo, hi = paired_diff_ci(both, only_a, only_b, len(ps) - both - only_a - only_b)
    return d, lo, hi, mcnemar_exact(only_a, only_b), only_a, only_b


def boot_gain(by_task, rng):
    tasks = list(by_task)
    s = [p for t in rng.choices(tasks, k=len(tasks)) for p in by_task[t]]
    return sum(y - x for x, y in s) / len(s)


def flat(by_task):
    return [p for ps in by_task.values() for p in ps]


def ci(boots):
    boots = sorted(boots)
    return boots[int(0.025 * B)], boots[int(0.975 * B) - 1]


def main():
    rng = random.Random(2026)
    print("| Model | Domain | Faults | tasks × trials | idempotent − baseline [Newcombe 95% CI] | "
          "task-clustered 95% CI | discordant (b+/i+) | McNemar p |")
    print("|---|---|---|---|---|---|---|---|")
    for model in ("sonnet", "haiku", "gemma", "qwen3coder"):   # folder prefixes used by scripts/run_tau2.sh
        got = {}
        for domain in ("airline", "retail"):
            for rate in ("0.2", "0"):
                bt = pairs(model, domain, rate)
                if not bt:
                    continue
                got[(domain, rate)] = bt
                d, lo, hi, p, oa, ob = summary(flat(bt))
                clo, chi = ci(boot_gain(bt, rng) for _ in range(B))
                print(f"| {model} | {domain} | {float(rate) * 100:.0f}% | {len(bt)} × {len(flat(bt)) // len(bt)} | "
                      f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | [{100 * clo:+.1f}, {100 * chi:+.1f}] | "
                      f"{oa}/{ob} | {p:.2f} |")
        if ("airline", "0.2") in got and ("retail", "0.2") in got:
            air, ret = got[("airline", "0.2")], got[("retail", "0.2")]
            fa, fr = flat(air), flat(ret)
            point = (sum(y - x for x, y in fa) / len(fa)) - (sum(y - x for x, y in fr) / len(fr))
            lo, hi = ci(boot_gain(air, rng) - boot_gain(ret, rng) for _ in range(B))
            se = sum((((h - l) / 2) / 1.96) ** 2 for _, l, h, *_ in (summary(fa), summary(fr))) ** 0.5
            print(f"| {model} | H12: airline gain − retail gain | 20% | | {100 * point:+.1f} "
                  f"[{100 * (point - 1.96 * se):+.1f}, {100 * (point + 1.96 * se):+.1f}] (Newcombe-combined) | "
                  f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | | |")


if __name__ == "__main__":
    main()
