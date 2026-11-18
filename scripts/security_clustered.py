"""Post-hoc: task-clustered bootstrap intervals for the security comparisons
(each defense against no defense, per model and goal family). Resamples the 14
tasks with all their levels, goals and trials; 10,000 resamples. Free.

  .venv/bin/python scripts/security_clustered.py
"""
import csv, os, random, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 10_000
MODELS = [("Sonnet 5.5", "sonnet"), ("Haiku 5.5", "haiku"), ("Gemma 4", "gemma4-26b"),
          ("qwen3-coder-next", "qwen3-coder-next-latest")]
DEFS = ["spotlight", "classifier", "llm_judge", "nemo", "least_privilege"]

def load(folder):
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", folder, "security.csv")))
            if not r["stop"].startswith("error") and int(r["level"]) > 0]
    return {(r["defense"], r["task"], r["level"], r["goal"], r["trial"]): r for r in rows}

def main():
    rng = random.Random(2026)
    print("| Model | Family | Defense | Metric | Diff. (points) | Task-clustered 95% CI |")
    print("|---|---|---|---|---|---|")
    for name, suf in MODELS:
        for fam, pre in (("cross-account", "full"), ("within-permission", "within")):
            path = os.path.join(ROOT, "results", f"sec-{pre}-{suf}", "security.csv")
            if not os.path.exists(path):
                continue
            D = load(f"sec-{pre}-{suf}")
            for d in DEFS:
                for metric, col in (("attack success", "attack_success"), ("task completion", "task_passed")):
                    by = defaultdict(list)
                    for (dd, t, lv, g, tr), r in D.items():
                        if dd != d: continue
                        base = D.get(("none", t, lv, g, tr))
                        if base: by[t].append(int(r[col]) - int(base[col]))
                    tasks = sorted(by)
                    if not tasks: continue
                    point = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values())
                    boots = []
                    for _ in range(B):
                        s = [rng.choice(tasks) for _ in tasks]
                        boots.append(sum(sum(by[t]) for t in s) / sum(len(by[t]) for t in s))
                    boots.sort()
                    print(f"| {name} | {fam} | {d} | {metric} | {100*point:+.1f} | "
                          f"[{100*boots[int(.025*B)]:+.1f}, {100*boots[int(.975*B)-1]:+.1f}] |")

if __name__ == "__main__":
    main()
