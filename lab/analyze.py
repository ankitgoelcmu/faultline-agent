"""Turn a results folder into tables and charts.

  python -m lab.analyze results/<run-folder> [--price-in 2 --price-out 10]

Writes summary.md and three PNG charts into the same folder.
Prices are dollars per million tokens. Check current prices before quoting cost.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

from .stats import holm, mcnemar_exact, paired_diff_ci, pass_hat_k, rate_ci

# Condition pairs compared episode by episode. Each adds one fix to the last.
COMPARISONS = [("baseline", "retry"), ("retry", "safe_retry"), ("baseline", "safe_retry"),
               ("safe_retry", "guarded"), ("baseline", "key_available")]

BUCKETS = [("short (≤7 calls)", 0, 7), ("medium (8–14)", 8, 14), ("long (15+)", 15, 10**6)]
ORDER = ["baseline", "retry", "safe_retry", "guarded", "key_available"]


def load(folder):
    with open(os.path.join(folder, "episodes.csv")) as f:
        rows = list(csv.DictReader(f))
    ok, errs = [], 0
    for r in rows:
        if str(r["stop"]).startswith("error"):
            errs += 1
            continue
        for k in ("passed", "planned_calls", "logical_calls", "attempts", "faults_injected",
                  "lost_responses", "input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens"):
            r[k] = int(float(r[k] or 0))
        r["stale_reads"] = int(float(r.get("stale_reads") or 0))   # absent in older runs
        r["fault_rate"] = float(r["fault_rate"])
        ok.append(r)
    return ok, errs


def bucket(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name


def pct(x):
    return f"{100 * x:.0f}%"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0



def key_adoption(traces_path, condition="key_available"):
    """Agent-supplied idempotency keys: writes sent with a key, and, after a
    lost response, what the agent did on its retry of the *same operation*
    (the next write with the same intent fields): reused the key, minted a
    new one, or sent none. Lost responses with no later retry of the same
    operation are counted in no_retry. Shared by analyze and the key figure.

    (Corrected 2026-11-07: an earlier version took the next write to the same
    tool, which can be a different operation; see DEVIATIONS.md.)"""
    from .harness import intent_key
    from .world import WRITE_TOOLS
    c = dict(writes=0, keyed=0, lost=0, reused=0, new_key=0, no_key=0, no_retry=0)
    with open(traces_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["row"]["condition"] != condition or str(rec["row"].get("stop", "")).startswith("error"):
                continue
            first = [t for t in rec["trace"] if t["attempt"] == 1 and t["tool"] in WRITE_TOOLS]
            c["writes"] += len(first)
            c["keyed"] += sum(1 for t in first if t["args"].get("idempotency_key"))
            for i, t in enumerate(first):
                if t.get("fault") != "lost_response":
                    continue
                c["lost"] += 1
                op = intent_key(t["tool"], t["args"])
                nxt = next((u for u in first[i + 1:]
                            if u["tool"] == t["tool"] and intent_key(u["tool"], u["args"]) == op), None)
                if nxt is None:
                    c["no_retry"] += 1
                    continue
                k0, k1 = t["args"].get("idempotency_key"), nxt["args"].get("idempotency_key")
                c["reused" if (k0 and k1 == k0) else "new_key" if k1 else "no_key"] += 1
    return c

def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--price-in", type=float, default=None, help="$ per M input tokens")
    p.add_argument("--price-out", type=float, default=None, help="$ per M output tokens")
    a = p.parse_args()
    rows, errs = load(a.folder)
    if not rows:
        raise SystemExit("no finished episodes found")

    conds = [c for c in ORDER if any(r["condition"] == c for r in rows)]
    rates = sorted({r["fault_rate"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["fault_rate"])].append(r)

    L = [f"# Results: {os.path.basename(os.path.normpath(a.folder))}", ""]
    L.append(f"{len(rows)} episodes" + (f" ({errs} API errors excluded)" if errs else "") + ".")
    meta_path = os.path.join(a.folder, "run.json")
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        L.append(f"Model: {m.get('model') or m.get('agent')}. Faults: {m.get('fault_model', 'uniform')}"
                 + (f" (mean outage {m.get('burst_len')} attempts)" if m.get("fault_model") == "bursty" else "")
                 + f", lost-response share {m.get('lost_share', 0.5)}, stale reads {m.get('stale_rate', 0.0)}."
                 + f" Grader v{m.get('grader', 1)}.")
    L.append("")

    # 0. What each cell actually faced. A nominal fault rate can mean very few
    # faults on short runs, so report the real counts next to the results.
    L += ["## Faults actually injected (mean per episode)", "",
          "| Condition | Fault rate | Faults | Lost responses | Stale reads | Episodes with ≥1 lost response |",
          "|---|---|---|---|---|---|"]
    for c in conds:
        for x in rates:
            g = by[(c, x)]
            if g and x > 0:
                L.append(f"| {c} | {pct(x)} | {mean(r['faults_injected'] for r in g):.2f} | "
                         f"{mean(r['lost_responses'] for r in g):.2f} | {mean(r['stale_reads'] for r in g):.2f} | "
                         f"{sum(1 for r in g if r['lost_responses'])}/{len(g)} |")
    L.append("")

    # 1. Task success by condition and fault rate
    L += ["## Task success rate", "",
          "Brackets are 95% confidence intervals (Wilson).", "",
          "| Condition | " + " | ".join(f"fault {pct(x)}" for x in rates) + " |",
          "|---|" + "---|" * len(rates)]
    for c in conds:
        cells = [rate_ci(r["passed"] for r in by[(c, x)]) for x in rates]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")

    # 1b. Paired tests. Every condition faces the same task and fault seed, so
    # compare outcomes episode by episode (McNemar), not as separate pools.
    pairs_out = []
    for c1, c2 in COMPARISONS:
        if c1 not in conds or c2 not in conds:
            continue
        for x in rates:
            if x == 0:
                continue
            o1 = {(r["task"], r["trial"]): r["passed"] for r in by[(c1, x)]}
            o2 = {(r["task"], r["trial"]): r["passed"] for r in by[(c2, x)]}
            keys = o1.keys() & o2.keys()
            b = sum(1 for k in keys if o1[k] and not o2[k])
            c = sum(1 for k in keys if o2[k] and not o1[k])
            both = sum(1 for k in keys if o1[k] and o2[k])
            diff = paired_diff_ci(both, b, c, len(keys) - both - b - c)
            pairs_out.append((c1, c2, x, len(keys), b, c, diff, mcnemar_exact(b, c)))
    if pairs_out:
        # Holm family = the pre-registered comparisons only; an exploratory
        # comparison (key_available) is adjusted on its own, so adding it
        # can never change a pre-registered p-value.
        prereg = [i for i, p in enumerate(pairs_out) if p[1] != "key_available"]
        explo = [i for i, p in enumerate(pairs_out) if p[1] == "key_available"]
        adj = [0.0] * len(pairs_out)
        for fam in (prereg, explo):
            for i, v in zip(fam, holm([pairs_out[i][-1] for i in fam])):
                adj[i] = v
        L += ["## Paired comparisons (McNemar exact test)", "",
              "Same task, trial and fault seed under both conditions. Only episodes where the two "
              "disagree count. p-values are Holm-adjusted across every row of this table.", "",
              "| Comparison | Fault rate | Pairs | Only first passed | Only second passed | Difference [95% CI] | Holm p |",
              "|---|---|---|---|---|---|---|"]
        for (c1, c2, x, n, b, c, (d, lo, hi), _), pa in zip(pairs_out, adj):
            L.append(f"| {c1} → {c2} | {pct(x)} | {n} | {b} | {c} | "
                     f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {pa:.3g} |")
        L.append("")

    # 1c. pass^k: does the agent get the same task right every time?
    trials = sorted({int(r["trial"]) for r in rows})
    if len(trials) > 1:
        ks = list(range(1, len(trials) + 1))
        L += ["## Consistency: pass^k (tau-bench)", "",
              "Chance that k attempts at the same task all succeed, averaged over tasks. "
              "pass^1 is the plain success rate; a steep drop means unreliable, not just imperfect.", "",
              "| Condition | Fault rate | " + " | ".join(f"pass^{k}" for k in ks) + " |",
              "|---|---|" + "---|" * len(ks)]
        for c in conds:
            for x in rates:
                per_task = defaultdict(list)
                for r in by[(c, x)]:
                    per_task[r["task"]].append(r["passed"])
                cells = []
                for k in ks:
                    vals = [pass_hat_k(sum(v), len(v), k) for v in per_task.values()]
                    vals = [v for v in vals if v is not None]
                    cells.append(pct(mean(vals)) if vals else "–")
                L.append(f"| {c} | {pct(x)} | " + " | ".join(cells) + " |")
        L.append("")

    # 1d. Key adoption (key_available, exploratory): does the agent use the
    # optional idempotency_key by itself, and reuse it after a lost response?
    if "key_available" in conds:
        ka = key_adoption(os.path.join(a.folder, "traces.jsonl"))
        writes, keyed, lost = ka["writes"], ka["keyed"], ka["lost"]
        reused, new_key, no_key = ka["reused"], ka["new_key"], ka["no_key"]
        L += ["## Key adoption (key_available, exploratory)", "",
              "Write tools accept an optional idempotency_key; the harness adds none.", "",
              "Retry = the agent's next write of the same operation (same intent fields) after a lost response.", "",
              "| Writes sent with a key | Lost responses | Retry reused the same key | Retry with a new key | Retry with no key | No retry of that operation |",
              "|---|---|---|---|---|---|",
              f"| {keyed}/{writes} ({100 * keyed / max(writes, 1):.0f}%) | {lost} | {reused} | {new_key} | {no_key} | {ka['no_retry']} |", ""]

    # 2. Per-call vs per-task, and the compounding prediction (baseline)
    L += ["## Per call vs per task (baseline)", "",
          "Naive prediction = (per-call success) ^ (tool calls per task): what you'd get if no error were ever recovered.", "",
          "| Fault rate | Per-call success | Planned calls/task | Naive prediction | Observed task success |",
          "|---|---|---|---|---|"]
    for x in rates:
        g = by[("baseline", x)]
        if not g:
            continue
        att = sum(r["attempts"] for r in g)
        per_call = 1 - sum(r["faults_injected"] for r in g) / att if att else 1.0
        n = mean(r["planned_calls"] for r in g)
        naive = mean(per_call ** r["planned_calls"] for r in g)
        L.append(f"| {pct(x)} | {100 * per_call:.1f}% | {n:.1f} | {pct(naive)} | "
                 f"{pct(mean(r['passed'] for r in g))} |")
    L.append("")

    # 3. Success by task length at the highest fault rate
    top = max(rates)
    L += [f"## Task success by length (fault {pct(top)})", "",
          "| Condition | " + " | ".join(b[0] for b in BUCKETS) + " |",
          "|---|" + "---|" * len(BUCKETS)]
    for c in conds:
        cells = [rate_ci(r["passed"] for r in by[(c, top)] if bucket(r["planned_calls"]) == name)
                 for name, _, _ in BUCKETS]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")

    # 4. Why tasks failed
    L += ["## Why tasks failed (all fault rates)", "",
          "| Condition | Top failure reasons |", "|---|---|"]
    fail_counts = {}
    for c in conds:
        cnt = Counter()
        for r in rows:
            if r["condition"] == c and r["fails"]:
                cnt.update(r["fails"].split(";"))
        fail_counts[c] = cnt
        L.append(f"| {c} | " + (", ".join(f"{k} ×{v}" for k, v in cnt.most_common(5)) or "none") + " |")
    L.append("")

    # 5. Cost and effort
    L += ["## Effort per task", "",
          "| Condition | Mean tool calls | Mean attempts | Mean input tokens | Mean output tokens |",
          "|---|---|---|---|---|"]
    for c in conds:
        g = [r for r in rows if r["condition"] == c]
        L.append(f"| {c} | {mean(r['logical_calls'] for r in g):.1f} | "
                 f"{mean(r['attempts'] for r in g):.1f} | "
                 f"{mean(r['input_tokens'] + r['cache_read_input_tokens'] + r['cache_creation_input_tokens'] for r in g):,.0f} | "
                 f"{mean(r['output_tokens'] for r in g):,.0f} |")
    tin = sum(r["input_tokens"] + r["cache_creation_input_tokens"] for r in rows)
    tcache = sum(r["cache_read_input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    L += ["", f"Total tokens: {tin:,} input, {tcache:,} cached input, {tout:,} output."]
    if a.price_in is not None and a.price_out is not None:
        cost = (tin * a.price_in + tcache * a.price_in * 0.1 + tout * a.price_out) / 1e6
        L.append(f"Estimated cost: ${cost:,.2f} (cached input priced at 10% of input; check current pricing).")
    L.append("")

    with open(os.path.join(a.folder, "summary.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    charts(a.folder, rows, by, conds, rates, fail_counts)


def charts(folder, rows, by, conds, rates, fail_counts):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts")
        return
    colors = {"baseline": "#d1495b", "retry": "#edae49", "safe_retry": "#00798c", "guarded": "#30638e",
              "key_available": "#8e5ea2"}

    # Chart 1: task success vs fault rate
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c in conds:
        ys = [100 * mean(r["passed"] for r in by[(c, x)]) for x in rates]
        ax.plot([100 * x for x in rates], ys, marker="o", label=c, color=colors.get(c), lw=2)
    ax.set_xlabel("Per-call fault rate (%)")
    ax.set_ylabel("Tasks fully correct (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Task success as tool calls get less reliable")
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "chart_success_vs_fault_rate.png"), dpi=160)
    plt.close(fig)

    # Chart 2: baseline success by task length, observed vs predicted
    top = max(rates)
    g = by[("baseline", top)]
    if g:
        att = sum(r["attempts"] for r in g)
        pc = 1 - sum(r["faults_injected"] for r in g) / att if att else 1.0
        names, obs, pred = [], [], []
        for name, _, _ in BUCKETS:
            gg = [r for r in g if bucket(r["planned_calls"]) == name]
            if not gg:
                continue
            names.append(name)
            obs.append(100 * mean(r["passed"] for r in gg))
            pred.append(100 * mean(pc ** r["planned_calls"] for r in gg))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        xs = range(len(names))
        ax.bar([i - .2 for i in xs], pred, .4, label="naive prediction (per-call ^ calls)", color="#cccccc")
        ax.bar([i + .2 for i in xs], obs, .4, label="observed", color=colors["baseline"])
        ax.set_xticks(list(xs), names)
        ax.set_ylabel("Tasks fully correct (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Baseline at {100 * top:.0f}% fault rate: longer tasks fail more")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "chart_compounding.png"), dpi=160)
        plt.close(fig)

    # Chart 3: failure reasons per condition
    reasons = sorted({k for c in fail_counts.values() for k in c})
    if reasons:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bottom = [0] * len(conds)
        cmap = plt.get_cmap("tab20")
        for i, reason in enumerate(reasons):
            vals = [fail_counts[c].get(reason, 0) for c in conds]
            ax.bar(conds, vals, bottom=bottom, label=reason, color=cmap(i))
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("Failure count (all fault rates)")
        ax.set_title("Why tasks failed")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, "chart_failures.png"), dpi=160)
        plt.close(fig)
    print(f"\nCharts written to {folder}")


if __name__ == "__main__":
    main()
