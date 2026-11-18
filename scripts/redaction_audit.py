"""Audit the perfect-redaction result on Gemma (llm_judge_redact).

  .venv-sec/bin/python scripts/redaction_audit.py [--sample 30]

Traces store what the tool returned, before filtering, so what the agent saw
is rebuilt here by re-running the same redacting judge on those outputs.
The injected paragraphs are found by comparing the poisoned task data with the
clean task data; nothing about their location is given to the judge.

Reports counts only: attack text is never printed.
  (a) injected paragraphs in redacted outputs: how many the re-run removed
  (b) legitimate paragraphs in those outputs: how many survived
  (c) from the original traces: outputs holding injected text that the original
      run did NOT redact (a detector miss that did not lead to attack success)
Costs a few cents of Haiku 5.5. The judge is stochastic, so (a) and (b) are
estimates of the original run's behaviour, not a replay of it.
"""

import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lab.attacks import make_attack                 # noqa: E402
from lab.defenses import LLMJudge, Redacting, strings_in   # noqa: E402
from lab.tasks import build_tasks                   # noqa: E402

GRIDS = ("sec-full-gemma4-26b", "sec-within-gemma4-26b")


def paragraphs(obj):
    out = []
    for s in strings_in(obj):
        out += [p.strip() for p in re.split(r"\n\s*\n", s) if len(p.strip()) >= 25]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=30)
    a = p.parse_args()
    recs = []
    for g in GRIDS:
        for line in open(os.path.join(ROOT, "results", g, "traces.jsonl")):
            r = json.loads(line)
            row = r["row"]
            if row["defense"] == "llm_judge_redact" and int(row["level"]) > 0 \
                    and not str(row["stop"]).startswith("error"):
                recs.append(r)
    rng = random.Random(2026)
    by_level = {}
    for r in recs:
        by_level.setdefault(int(r["row"]["level"]), []).append(r)
    per = a.sample // len(by_level)
    sample = [r for lv in sorted(by_level) for r in rng.sample(by_level[lv], min(per, len(by_level[lv])))]

    tasks = {t["id"]: t for t in build_tasks()}
    redactor = Redacting(LLMJudge())
    tot = dict(outputs=0, inj=0, inj_removed=0, legit=0, legit_kept=0, unredacted_with_inj=0,
               eps_passed=0, eps_attacked=0, eps_with_redaction=0)
    per_level = {}
    for r in sample:
        row = r["row"]
        task = tasks[row["task"]]
        poisoned, _ = make_attack(task, int(row["level"]), row["goal"])
        injected = set(paragraphs(poisoned["data"])) - set(paragraphs(task["data"]))
        lv = per_level.setdefault(int(row["level"]), dict(inj=0, inj_removed=0, legit=0, legit_kept=0))
        tot["eps_passed"] += int(row["task_passed"])
        tot["eps_attacked"] += int(row["attack_success"])
        saw = False
        for t in r["trace"]:
            res = t.get("result")
            if res is None:
                continue
            paras = paragraphs(res)
            has_inj = any(q in injected for q in paras)
            if t.get("fault") != "defense_redacted_output":
                tot["unredacted_with_inj"] += int(has_inj)
                continue
            saw = True
            hits = []
            redactor._clean(res, hits)
            removed = {h.strip() for h in hits}
            tot["outputs"] += 1
            for q in paras:
                k = ("inj", "inj_removed") if q in injected else ("legit", "legit_kept")
                gone = q in removed
                ok = gone if q in injected else not gone
                tot[k[0]] += 1
                tot[k[1]] += int(ok)
                lv[k[0]] += 1
                lv[k[1]] += int(ok)
        tot["eps_with_redaction"] += int(saw)
    print(f"episodes audited: {len(sample)} (of {len(recs)}), stratified over attack levels")
    print(f"of which: real task passed {tot['eps_passed']}, attack succeeded {tot['eps_attacked']}, "
          f"had at least one redaction {tot['eps_with_redaction']}")
    print(f"redacted outputs re-judged: {tot['outputs']}")
    print(f"(a) injected paragraphs removed: {tot['inj_removed']} of {tot['inj']}")
    print(f"(b) legitimate paragraphs kept:  {tot['legit_kept']} of {tot['legit']}")
    print(f"(c) outputs with injected text the original run did not redact: {tot['unredacted_with_inj']}")
    for lv, c in sorted(per_level.items()):
        print(f"  L{lv}: injected removed {c['inj_removed']}/{c['inj']}, legitimate kept {c['legit_kept']}/{c['legit']}")
    print(f"judge usage: {redactor.base.usage}")


if __name__ == "__main__":
    main()
