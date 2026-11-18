"""Export the paper's main results as machine-readable CSV files.

  .venv/bin/python scripts/export_tables.py            # writes results/summary/*.csv
  .venv/bin/python scripts/export_tables.py --compare main-realistic-<your-model>

Files (one row per cell; rates and CIs in percent, Wilson 95%):
  reliability.csv  model, condition, fault_rate, n, passed, success, ci_low, ci_high
  paired.csv       model, comparison, fault_rate, n_pairs, diff, ci_low, ci_high, mcnemar_p
  security.csv     model, goal_family, defense, attacked, attack_success, utility_under_attack, benign_utility
  tau2.csv         model, domain, condition, fault_rate, n, solved, success
  keys.csv         model, writes, keyed, lost, retry_same_key, retry_new_key, retry_no_key

--compare runs your own results folder (same layout as results/main-realistic-*,
e.g. produced by `python -m lab.run --preset realistic`) and prints its success
per condition at 20% faults next to the published models, with the same
intervals. Reads saved results only; no API calls.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lab.analyze import key_adoption                          # noqa: E402
from lab.stats import mcnemar_exact, paired_diff_ci, wilson   # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "summary")
MAIN = ("sonnet", "haiku", "gemma4-26b", "qwen3-coder-next-latest")
CONDS = ("baseline", "retry", "safe_retry", "guarded")
PAIRS = (("baseline", "retry"), ("retry", "safe_retry"), ("baseline", "safe_retry"), ("safe_retry", "guarded"))


def rows(path):
    if not os.path.exists(path):
        return []
    return [r for r in csv.DictReader(open(path)) if not str(r.get("stop", "")).startswith("error")]


def pct(x):
    return round(100 * x, 1)


def reliability(model, folder):
    out, by = [], defaultdict(dict)
    for r in rows(os.path.join(RES, folder, "episodes.csv")):
        by[(r["condition"], float(r["fault_rate"]))][(r["task"], r["trial"])] = int(r["passed"])
    for (c, x), d in sorted(by.items()):
        if c not in CONDS:
            continue
        k, n = sum(d.values()), len(d)
        lo, hi = wilson(k, n)
        out.append(dict(model=model, condition=c, fault_rate=pct(x), n=n, passed=k,
                        success=pct(k / n), ci_low=pct(lo), ci_high=pct(hi)))
    return out, by


def paired(model, by):
    out = []
    for c1, c2 in PAIRS:
        for x in sorted({x for _, x in by if x > 0}):
            a, b = by.get((c1, x), {}), by.get((c2, x), {})
            ks = a.keys() & b.keys()
            if not ks:
                continue
            both = sum(1 for k in ks if a[k] and b[k])
            first = sum(1 for k in ks if a[k] and not b[k])
            second = sum(1 for k in ks if b[k] and not a[k])
            d, lo, hi = paired_diff_ci(both, first, second, len(ks) - both - first - second)
            out.append(dict(model=model, comparison=f"{c1}->{c2}", fault_rate=pct(x), n_pairs=len(ks),
                            diff=pct(d), ci_low=pct(lo), ci_high=pct(hi),
                            mcnemar_p=f"{mcnemar_exact(first, second):.3g}"))
    return out


def security(model):
    out = []
    for fam, kind in (("cross-account", "full"), ("within-permission", "within")):
        rs = rows(os.path.join(RES, f"sec-{kind}-{model}", "security.csv"))
        for d in sorted({r["defense"] for r in rs}):
            att = [r for r in rs if r["defense"] == d and int(r["level"]) > 0]
            ben = [r for r in rs if r["defense"] == d and int(r["level"]) == 0]
            if att:
                out.append(dict(model=model, goal_family=fam, defense=d, attacked=len(att),
                                attack_success=pct(sum(int(r["attack_success"]) for r in att) / len(att)),
                                utility_under_attack=pct(sum(int(r["task_passed"]) for r in att) / len(att)),
                                benign_utility=pct(sum(int(r["task_passed"]) for r in ben) / len(ben)) if ben else ""))
    return out


def tau2():
    out, base = [], os.path.join(RES, "tau2")
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        parts = name.split("-")
        if len(parts) < 4 or parts[0] in ("pilot", "smoke"):
            continue
        path = os.path.join(base, name, "results.json")
        if not os.path.exists(path):
            continue
        sims = [s for s in json.load(open(path))["simulations"] if s.get("reward_info")]
        solved = sum(1 for s in sims if (s["reward_info"].get("reward") or 0) >= 1)
        model, domain, cond, rate = "-".join(parts[:-3]), parts[-3], parts[-2], parts[-1][1:]
        out.append(dict(model=model, domain=domain, condition=cond, fault_rate=pct(float(rate)),
                        n=len(sims), solved=solved, success=pct(solved / max(len(sims), 1))))
    return out


def keys():
    out = []
    for model, folder in (("sonnet", "gen-sonnet55-keys"), ("sonnet45", "gen-sonnet45"),
                          ("gemma4-26b", "main-realistic-gemma4-26b")):
        path = os.path.join(RES, folder, "traces.jsonl")
        if os.path.exists(path):
            c = key_adoption(path)
            out.append(dict(model=model, writes=c["writes"], keyed=c["keyed"], lost=c["lost"],
                            retry_same_key=c["reused"], retry_new_key=c["new_key"], retry_no_key=c["no_key"]))
    return out


def write(name, recs):
    if not recs:
        return
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"wrote results/summary/{name} ({len(recs)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", help="your results folder under results/ (main-grid layout)")
    a = p.parse_args()
    if a.compare:
        print("| Model | " + " | ".join(CONDS) + " |  (success at 20% faults, 95% Wilson CI)")
        print("|---|" + "---|" * len(CONDS))
        for model, folder in [(m, f"main-realistic-{m}") for m in MAIN] + [(a.compare, a.compare)]:
            recs, _ = reliability(model, folder)
            cell = {r["condition"]: f"{r['success']:.0f} [{r['ci_low']:.0f}–{r['ci_high']:.0f}]"
                    for r in recs if r["fault_rate"] == 20.0}
            if cell:
                print(f"| {model} | " + " | ".join(cell.get(c, "–") for c in CONDS) + " |")
        return
    rel, pr = [], []
    for m in MAIN:
        recs, by = reliability(m, f"main-realistic-{m}")
        rel += recs
        pr += paired(m, by)
    write("reliability.csv", rel)
    write("paired.csv", pr)
    write("security.csv", [r for m in ("sonnet", "haiku", "sonnet45", "redact-sonnet", "gemma4-26b",
                                       "qwen3-coder-next-latest") for r in security(m)])
    write("tau2.csv", tau2())
    write("keys.csv", keys())


if __name__ == "__main__":
    main()
