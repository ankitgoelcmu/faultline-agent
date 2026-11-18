"""Tables and charts for the prompt-injection experiment.

  python -m lab.security_analyze results/<security-run-folder>
"""

import argparse
import csv
import os
from collections import defaultdict

from .stats import holm, mcnemar_exact, paired_diff_ci, rate_ci

ORDER = ["none", "spotlight", "classifier", "llm_judge", "nemo", "least_privilege"]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    a = p.parse_args()
    with open(os.path.join(a.folder, "security.csv")) as f:
        rows = [r for r in csv.DictReader(f) if not str(r["stop"]).startswith("error")]
    if not rows:
        raise SystemExit("no finished episodes")
    for r in rows:
        for k in ("level", "attack_success", "attack_partial", "task_passed",
                  "outputs_flagged", "calls_blocked", "tool_calls"):
            r[k] = int(r[k])
        r["defense_ms"] = float(r["defense_ms"])
    defs = sorted({r["defense"] for r in rows},
                  key=lambda d: (ORDER.index(d) if d in ORDER else 99, d))
    levels = sorted({r["level"] for r in rows if r["level"] > 0})
    goals = sorted({r["goal"] for r in rows if r["level"] > 0})
    by = defaultdict(list)
    for r in rows:
        by[(r["defense"], r["level"])].append(r)

    L = [f"# Prompt-injection results: {os.path.basename(os.path.normpath(a.folder))}", "",
         f"{len(rows)} episodes.", ""]

    L += ["## Attack success rate (lower is better)", "",
          "Share of attacked episodes where the attacker got what they wanted. "
          "Brackets are 95% confidence intervals (Wilson).", "",
          "| Defense | " + " | ".join(f"L{lv}" for lv in levels) + " | All levels |",
          "|---|" + "---|" * (len(levels) + 1)]
    for d in defs:
        cells = [pct(mean(r["attack_success"] for r in by[(d, lv)])) for lv in levels]
        allr = [r for r in rows if r["defense"] == d and r["level"] > 0]
        L.append(f"| {d} | " + " | ".join(cells) + f" | {rate_ci(r['attack_success'] for r in allr)} |")
    L.append("")

    if len(goals) > 1:
        L += ["## Attack success by goal", "",
              "| Defense | " + " | ".join(goals) + " |", "|---|" + "---|" * len(goals)]
        for d in defs:
            L.append(f"| {d} | " + " | ".join(
                pct(mean(r["attack_success"] for r in rows
                         if r["defense"] == d and r["goal"] == g and r["level"] > 0)) for g in goals) + " |")
        L.append("")

    L += ["## Utility: does the real work still get done? (higher is better)", "",
          "No attack = normal tickets. A drop here means the defense gets in the way (false positives).", "",
          "| Defense | Task success, no attack | Task success under attack | Benign runs with a false flag |",
          "|---|---|---|---|"]
    for d in defs:
        benign = by[(d, 0)]
        attacked = [r for r in rows if r["defense"] == d and r["level"] > 0]
        fp = mean(1 if (r["outputs_flagged"] or r["calls_blocked"]) else 0 for r in benign)
        L.append(f"| {d} | {rate_ci(r['task_passed'] for r in benign)} | "
                 f"{rate_ci(r['task_passed'] for r in attacked)} | {pct(fp)} |")
    L.append("")

    # Paired tests against no defense: same task, level, goal and trial.
    tests = []
    for metric, label in (("attack_success", "attack success"), ("task_passed", "utility under attack")):
        for d in defs:
            if d == "none":
                continue
            key = lambda r: (r["task"], r["level"], r["goal"], r["trial"])
            base = {key(r): r[metric] for r in rows if r["defense"] == "none" and r["level"] > 0}
            other = {key(r): r[metric] for r in rows if r["defense"] == d and r["level"] > 0}
            ks = base.keys() & other.keys()
            if not ks:
                continue
            b_only = sum(1 for k in ks if base[k] and not other[k])
            d_only = sum(1 for k in ks if other[k] and not base[k])
            both = sum(1 for k in ks if base[k] and other[k])
            diff = paired_diff_ci(both, b_only, d_only, len(ks) - both - b_only - d_only)
            tests.append((label, d, len(ks), b_only, d_only, diff, mcnemar_exact(b_only, d_only)))
    if tests:
        # Holm families: the pre-registered defenses only; exploratory variants
        # (*_redact, *_parwithhold) are adjusted on their own, so adding them
        # can never change a pre-registered p-value.
        prereg = [i for i, t in enumerate(tests) if t[1] in ORDER]
        explo = [i for i, t in enumerate(tests) if t[1] not in ORDER]
        adj = [0.0] * len(tests)
        for fam in (prereg, explo):
            if fam:
                for i, v in zip(fam, holm([tests[i][-1] for i in fam])):
                    adj[i] = v
        L += ["## Paired comparisons against no defense (McNemar exact test)", "",
              "Same task, level, goal and trial with and without the defense. "
              "p-values are Holm-adjusted within two families: the pre-registered defenses, "
              "and separately the exploratory variants (*_redact, *_parwithhold).", "",
              "| Metric | Defense | Pairs | Only without defense | Only with defense | Difference [95% CI] | Holm p |",
              "|---|---|---|---|---|---|---|"]
        for (label, d, n, b_only, d_only, (df, lo, hi), _), pa in zip(tests, adj):
            L.append(f"| {label} | {d} | {n} | {b_only} | {d_only} | "
                     f"{100 * df:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {pa:.3g} |")
        L.append("")

    # H6 as registered: no defense, pooled L4-L5 vs. pooled L1-L2.
    none = [r for r in rows if r["defense"] == "none" and r["level"] > 0]
    crude = [r["attack_success"] for r in none if r["level"] in (1, 2)]
    subtle = [r["attack_success"] for r in none if r["level"] in (4, 5)]
    if crude and subtle:
        L += ["## H6: subtle vs. crude attacks, no defense (as registered)", "",
              "| Levels | Attack success |", "|---|---|",
              f"| L1-L2 (crude) | {rate_ci(crude)} |", f"| L4-L5 (subtle) | {rate_ci(subtle)} |", ""]

    L += ["## How each defense acted on attacks", "",
          "| Defense | Attacked runs where it flagged output | ... where it blocked a call | Mean added ms per run |",
          "|---|---|---|---|"]
    for d in defs:
        attacked = [r for r in rows if r["defense"] == d and r["level"] > 0]
        L.append(f"| {d} | {pct(mean(1 if r['outputs_flagged'] else 0 for r in attacked))} | "
                 f"{pct(mean(1 if r['calls_blocked'] else 0 for r in attacked))} | "
                 f"{mean(r['defense_ms'] for r in rows if r['defense'] == d) or 0:.0f} |")
    L.append("")

    with open(os.path.join(a.folder, "summary.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for d in defs:
        ys = [mean(r["attack_success"] for r in by[(d, lv)]) for lv in levels]
        ax.plot(levels, [100 * (y or 0) for y in ys], marker="o", lw=2, label=d)
    ax.set_xticks(levels, [f"L{lv}" for lv in levels])
    ax.set_xlabel("Attack level (L1 plain → L5 blended)")
    ax.set_ylabel("Attack success (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("Prompt injection success by defense and attack level")
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.folder, "chart_attack_success.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = range(len(defs))
    ben = [100 * (mean(r["task_passed"] for r in by[(d, 0)]) or 0) for d in defs]
    att = [100 * (mean(r["task_passed"] for r in rows if r["defense"] == d and r["level"] > 0) or 0)
           for d in defs]
    ax.bar([i - .2 for i in xs], ben, .4, label="no attack", color="#00798c")
    ax.bar([i + .2 for i in xs], att, .4, label="under attack", color="#d1495b")
    ax.set_xticks(list(xs), defs, rotation=20)
    ax.set_ylabel("Tasks fully correct (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Utility: real work done with each defense")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.folder, "chart_utility.png"), dpi=160)
    plt.close(fig)
    print(f"\nCharts written to {a.folder}")


if __name__ == "__main__":
    main()
