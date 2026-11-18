"""Injection through the error channel (exploratory; DEVIATIONS.md, prediction P5).

Fault benchmarks inject faults and injection benchmarks inject hostile data,
but here the two meet: when a tool call faults, the error message the agent
gets carries attacker text phrased as recovery guidance. The agent is most
likely to take instructions while recovering from an error, and output
defenses (spotlighting, detectors) only inspect *successful* tool output.

Episodes use the main grid's `realistic` fault settings and seeds at 20%
faults, so the k-th call of an episode faults exactly where it did in the
main grid's baseline run; the only difference is the text of the error.

  python -m lab.error_injection --agent claude --model claude-sonnet-5-5 --out results/errinj-sonnet
"""

import argparse
import csv
import json
import os

from .agents import make_agent
from .attacks import _goal, attack_applies, attack_succeeded
from .defenses import make_defense
from .harness import Harness
from .run import PRESETS, parse_tasks, seed_for
from .stats import rate_ci
from .tasks import build_tasks, grade
from .world import World

FAULT_RATE = 0.20
SETTINGS = {k: PRESETS["realistic"][k] for k in ("fault_model", "burst_len", "lost_share", "stale_rate")}
FIELDS = ["task", "goal", "defense", "trial", "attack_success", "task_passed", "fails",
          "faults_injected", "stop", "input_tokens", "output_tokens",
          "cache_read_input_tokens", "cache_creation_input_tokens"]


def payload(g):
    return "[attack text withheld from the public release; see the paper, Section III-E] (error channel)"

def episode(agent, task, goal, defense_name, trial):
    g = _goal(task, goal)
    defense = make_defense(defense_name)
    defense.setup(task)
    world = World(task["data"])
    trace = []
    h = Harness(world, "baseline", FAULT_RATE, seed_for(task["id"], FAULT_RATE, trial), trace,
                defense=defense, error_payload=payload(g), **SETTINGS)
    try:
        info = agent.run(task, h)
    except Exception as e:
        info = {"stop": f"error: {type(e).__name__}: {e}"[:300]}
    passed, fails = grade(task, world)
    success, _ = attack_succeeded(g, world)
    row = {"task": task["id"], "goal": goal, "defense": defense_name, "trial": trial,
           "attack_success": int(success), "task_passed": int(passed), "fails": ";".join(fails),
           "faults_injected": sum(1 for x in trace if x["fault"] in ("transient", "lost_response")),
           "stop": info.get("stop"),
           **{k: info.get(k, 0) for k in ("input_tokens", "output_tokens",
                                          "cache_read_input_tokens", "cache_creation_input_tokens")}}
    return row, {"row": row, "final_text": info.get("final_text", ""), "trace": trace}


def summarize(folder):
    rows = [r for r in csv.DictReader(open(os.path.join(folder, "errinj.csv")))
            if not r["stop"].startswith("error")]
    L = [f"# Injection through error messages: {os.path.basename(os.path.normpath(folder))}", "",
         "Attack success over all episodes, and over episodes where at least one fault "
         "(so at least one poisoned error message) occurred. 95% Wilson intervals.", "",
         "| Defense | Goal | All episodes | Episodes with ≥1 fault | Real task done |", "|---|---|---|---|---|"]
    for d in sorted({r["defense"] for r in rows}):
        for g in sorted({r["goal"] for r in rows}):
            s = [r for r in rows if r["defense"] == d and r["goal"] == g]
            if not s:
                continue
            hit = [r for r in s if int(r["faults_injected"]) > 0]
            L.append(f"| {d} | {g} | {rate_ci(int(r['attack_success']) for r in s)} | "
                     f"{rate_ci(int(r['attack_success']) for r in hit)} | "
                     f"{rate_ci(int(r['task_passed']) for r in s)} |")
    L.append("")
    open(os.path.join(folder, "summary.md"), "w").write("\n".join(L))
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description="Injection through the error channel (exploratory)")
    p.add_argument("--agent", choices=["claude", "ollama"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="all")
    p.add_argument("--goals", default="refund,inflate")
    p.add_argument("--defenses", default="none,spotlight,least_privilege")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    agent = make_agent(a.agent, a.model)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "errinj.csv")
    done = set()
    if os.path.exists(path):
        done = {(r["task"], r["goal"], r["defense"], r["trial"]) for r in csv.DictReader(open(path))
                if not r["stop"].startswith("error")}
    json.dump({**vars(a), "fault_rate": FAULT_RATE, **SETTINGS}, open(os.path.join(a.out, "run.json"), "w"),
              indent=2)
    jobs = [(t, g, d, k) for t in parse_tasks(a.tasks, build_tasks()) for g in a.goals.split(",")
            for d in a.defenses.split(",") for k in range(1, a.trials + 1)
            if attack_applies(t, g) and (t["id"], g, d, str(k)) not in done]
    print(f"{len(jobs)} episodes to run -> {a.out}")
    new = not os.path.exists(path)
    from concurrent.futures import ThreadPoolExecutor
    with open(path, "a", newline="") as fc, open(os.path.join(a.out, "traces.jsonl"), "a") as ft:
        w = csv.DictWriter(fc, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ThreadPoolExecutor(a.workers) as ex:
            for i, (row, rec) in enumerate(ex.map(lambda j: episode(agent, *j), jobs), 1):
                w.writerow(row)
                ft.write(json.dumps(rec, default=str) + "\n")
                fc.flush(); ft.flush()
                print(f"[{i}/{len(jobs)}] {row['task']} {row['goal']:<8} {row['defense']:<16} "
                      f"{'ATTACK OK' if row['attack_success'] else 'blocked':<10} task={row['task_passed']}",
                      flush=True)
    summarize(a.out)


if __name__ == "__main__":
    main()
