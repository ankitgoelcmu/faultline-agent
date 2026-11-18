"""Audit what the idempotency key deduplicated under safe_retry.

  .venv/bin/python scripts/dedup_audit.py

A key merges a repeated write with an earlier one. That is intended when the
earlier write's response was lost (the agent could not know it succeeded). It
would hide an agent error when the earlier write visibly succeeded and the
agent repeated it anyway. This counts both, per model and fault rate.
Exploratory, post hoc (review round 2); see DEVIATIONS.md.
"""

import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lab.harness import intent_key   # noqa: E402
from lab.world import WRITE_TOOLS    # noqa: E402

MODELS = ("sonnet", "haiku", "gemma4-26b", "qwen3-coder-next-latest")


def audit(path):
    """Per fault rate: deduplicated writes by what the first write with that key
    returned ("lost" or "visible"), how many had full arguments identical to
    the first write, and how many distinct episodes were affected."""
    n = collections.Counter()
    same = collections.Counter()
    eps = collections.defaultdict(set)
    for line in open(path):
        rec = json.loads(line)
        row = rec["row"]
        if row["condition"] != "safe_retry" or str(row.get("stop", "")).startswith("error"):
            continue
        calls = collections.defaultdict(list)
        for t in rec["trace"]:
            if t["tool"] in WRITE_TOOLS:
                calls[t["call"]].append(t)
        first = {}
        for call_no in sorted(calls):
            ats = calls[call_no]
            k = intent_key(ats[0]["tool"], ats[0]["args"])
            lost = any(a["fault"] == "lost_response" for a in ats)
            if not (any(a["ok"] for a in ats) or lost):
                continue          # never applied: nothing to deduplicate against
            if k in first:
                kind, args0 = first[k]
                key = (row["fault_rate"], kind)
                n[key] += 1
                same[key] += ats[0]["args"] == args0
                eps[key].add((row["task"], row["trial"]))
            else:
                first[k] = ("visible" if ats[-1]["ok"] else "lost", ats[0]["args"])
    return n, same, eps


def main():
    print("| Model | Fault rate | Deduplicated writes | First result lost | First result visible | "
          "Full arguments identical | Affected episodes (lost / visible) |")
    print("|---|---|---|---|---|---|---|")
    for m in MODELS:
        path = os.path.join(ROOT, "results", f"main-realistic-{m}", "traces.jsonl")
        if not os.path.exists(path):
            continue
        n, same, eps = audit(path)
        rates = sorted({r for r, _ in n}, key=float) + ["all"]
        for rate in rates:
            pick = (lambda r: True) if rate == "all" else (lambda r, x=rate: r == x)
            tot = lambda d, kind: sum(v if isinstance(v, int) else len(v)
                                      for (r, k), v in d.items() if pick(r) and k == kind)
            lo, vi = tot(n, "lost"), tot(n, "visible")
            label = "all" if rate == "all" else f"{100 * float(rate):.0f}%"
            print(f"| {m} | {label} | {lo + vi} | {lo} | {vi} | {tot(same, 'lost') + tot(same, 'visible')} | "
                  f"{tot(eps, 'lost')} / {tot(eps, 'visible')} |")


if __name__ == "__main__":
    main()
