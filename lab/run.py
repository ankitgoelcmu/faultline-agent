"""Run the experiment.

Examples:
  python -m lab.run --agent scripted                      # free, about a second
  python -m lab.run --agent claude --preset quick         # small paid run
  python -m lab.run --agent claude --preset full          # the main experiment
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import make_agent
from .harness import CONDITIONS, FAULT_MODELS, Harness

# The pre-registered conditions. Exploratory ones (key_available) are added
# to a run only by naming them in --conditions, never through a preset.
PREREGISTERED_CONDITIONS = ["baseline", "retry", "safe_retry", "guarded"]
from .tasks import GRADER_VERSION, build_tasks, grade
from .world import World

PRESETS = {
    # tasks, conditions, fault rates, trials
    "quick": dict(tasks="1,6,14,18,26", conditions="baseline,safe_retry",
                  fault_rates="0,0.05", trials=1),
    "full": dict(tasks="all", conditions=",".join(PREREGISTERED_CONDITIONS),
                 fault_rates="0,0.02,0.05,0.10", trials=2),
    # Outages that last several calls, more lost responses, and replicas that
    # lag one write behind. Closer to production than independent faults.
    "realistic": dict(tasks="all", conditions=",".join(PREREGISTERED_CONDITIONS),
                      fault_rates="0,0.05,0.10,0.20", trials=5, fault_model="bursty",
                      burst_len=4.0, lost_share=0.5, stale_rate=0.1),
}

# Settings that change what an episode faces. A results folder can only be
# resumed with the same values, or its episodes would mix two experiments.
FAULT_SETTINGS = ("fault_model", "burst_len", "lost_share", "stale_rate", "observable", "system_note")

# Exploratory (DEVIATIONS.md): an instruction that asks for exactly-once writes.
EXACTLY_ONCE_NOTE = ("\n- Tool calls can time out after the change was already applied. Never repeat a "
                     "write unless you can confirm the first attempt did not take effect.")

CSV_FIELDS = ["task", "kind", "condition", "fault_rate", "trial", "agent", "passed", "fails",
              "planned_calls", "logical_calls", "attempts", "faults_injected", "lost_responses",
              "calls_failed_final", "stop", "turns", "input_tokens", "output_tokens",
              "cache_read_input_tokens", "cache_creation_input_tokens", "seconds", "stale_reads"]


def parse_tasks(spec, tasks):
    if spec == "all":
        return tasks
    keep = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            keep |= set(range(int(a), int(b) + 1))
        else:
            keep.add(int(part))
    return [t for i, t in enumerate(tasks, 1) if i in keep]


def seed_for(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def run_episode(agent, task, condition, rate, trial, faults=None):
    world = World(task["data"])
    trace = []
    # Same seed for every condition, so they face the same fault dice.
    h = Harness(world, condition, rate, seed_for(task["id"], rate, trial), trace, **(faults or {}))
    t0 = time.time()
    try:
        info = agent.run(task, h)
    except Exception as e:  # API outage etc. Recorded, then excluded in analysis.
        info = {"stop": f"error: {type(e).__name__}: {e}"[:300]}
    passed, fails = grade(task, world)
    row = {
        "task": task["id"], "kind": task["kind"], "condition": condition, "fault_rate": rate,
        "trial": trial, "agent": agent.name, "passed": int(passed), "fails": ";".join(fails),
        "planned_calls": task["planned_calls"], "logical_calls": h.calls,
        "attempts": len(trace),
        "faults_injected": sum(1 for x in trace if x["fault"] in ("transient", "lost_response")),
        "lost_responses": sum(1 for x in trace if x["fault"] == "lost_response"),
        "calls_failed_final": len({x["call"] for x in trace} -
                                  {x["call"] for x in trace if x["ok"]}),
        "stop": info.get("stop"), "turns": info.get("turns", 0),
        "input_tokens": info.get("input_tokens", 0), "output_tokens": info.get("output_tokens", 0),
        "cache_read_input_tokens": info.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": info.get("cache_creation_input_tokens", 0),
        "seconds": round(time.time() - t0, 2),
        "stale_reads": sum(1 for x in trace if x["fault"] == "stale_read"),
    }
    trace_rec = {"key": episode_key(row), "row": row, "final_text": info.get("final_text", ""),
                 "trace": trace}
    return row, trace_rec


def episode_key(r):
    return f"{r['task']}|{r['condition']}|{r['fault_rate']}|{r['trial']}"


def main():
    p = argparse.ArgumentParser(description="Agent reliability lab")
    p.add_argument("--agent", choices=["scripted", "scripted-strict", "claude", "ollama"], default="scripted")
    p.add_argument("--model", default=os.environ.get("LAB_MODEL", "claude-haiku-5-5"))
    p.add_argument("--preset", choices=list(PRESETS))
    p.add_argument("--tasks", default="all", help='"all", "1-10" or "1,5,9"')
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--fault-rates", default="0,0.02,0.05,0.10")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--fault-model", choices=FAULT_MODELS, default="uniform")
    p.add_argument("--burst-len", type=float, default=4.0, help="mean outage length in attempts (bursty)")
    p.add_argument("--lost-share", type=float, default=0.5, help="share of write faults that are lost responses")
    p.add_argument("--stale-rate", type=float, default=0.0, help="chance a read is one write behind")
    p.add_argument("--instruct-exactly-once", dest="system_note", action="store_const",
                   const=EXACTLY_ONCE_NOTE, default="",
                   help="add an exactly-once instruction to the system prompt (exploratory)")
    p.add_argument("--observable", action="store_true",
                   help="also give the agent list_refunds and list_sent_emails (exploratory)")
    p.add_argument("--workers", type=int, default=4, help="parallel episodes (claude/ollama only)")
    p.add_argument("--out", default=None, help="results folder (reuse it to resume a run)")
    # A preset sets defaults; flags given explicitly still win.
    preset = p.parse_known_args()[0].preset
    if preset:
        p.set_defaults(**PRESETS[preset])
    a = p.parse_args()

    tasks = parse_tasks(a.tasks, build_tasks())
    conditions = [c.strip() for c in a.conditions.split(",")]
    rates = [float(x) for x in a.fault_rates.split(",")]
    agent = make_agent(a.agent, a.model)

    out = a.out or os.path.join(
        "results", f"{dt.datetime.now():%Y%m%d-%H%M%S}-{a.agent}")
    os.makedirs(out, exist_ok=True)
    csv_path, trace_path = os.path.join(out, "episodes.csv"), os.path.join(out, "traces.jsonl")

    faults = {k: getattr(a, k) for k in FAULT_SETTINGS}
    meta_path = os.path.join(out, "run.json")
    if os.path.exists(meta_path):
        prev = json.load(open(meta_path))
        prev_faults = {k: prev.get(k, d) for k, d in zip(FAULT_SETTINGS, ("uniform", 4.0, 0.5, 0.0, False, ""))}
        if prev_faults != faults:
            raise SystemExit(f"{out} was run with {prev_faults}, not {faults}. Use a new --out folder.")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            if next(csv.reader(f), None) != CSV_FIELDS:
                raise SystemExit(f"{csv_path} has an older column layout. Use a new --out folder.")

    done = set()
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            # Episodes that ended in an API error are retried, not kept as done.
            done = {r["key"] for r in map(json.loads, filter(str.strip, f))
                    if not str(r["row"]["stop"]).startswith("error")}

    jobs = [(t, c, r, k) for t in tasks for c in conditions for r in rates
            for k in range(1, a.trials + 1)]
    jobs = [j for j in jobs if f"{j[0]['id']}|{j[1]}|{j[2]}|{j[3]}" not in done]
    meta = {"agent": a.agent, "model": a.model if a.agent in ("claude", "ollama") else None,
            "tasks": [t["id"] for t in tasks], "conditions": conditions, "fault_rates": rates,
            "trials": a.trials, "grader": GRADER_VERSION, **faults,
            "started": dt.datetime.now().isoformat()}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{len(jobs)} episodes to run ({len(done)} already done) -> {out}")

    lock = threading.Lock()
    new_csv = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fc, open(trace_path, "a") as ft:
        w = csv.DictWriter(fc, fieldnames=CSV_FIELDS)
        if new_csv:
            w.writeheader()

        def save(res, i):
            row, rec = res
            with lock:
                w.writerow(row)
                ft.write(json.dumps(rec, default=str) + "\n")
                fc.flush(); ft.flush()
                mark = "PASS" if row["passed"] else "fail"
                print(f"[{i}/{len(jobs)}] {row['task']} {row['condition']:<10} "
                      f"p={row['fault_rate']:<4} {mark} {row['fails']}")

        workers = a.workers if a.agent in ("claude", "ollama") else 1
        if workers <= 1:
            for i, j in enumerate(jobs, 1):
                save(run_episode(agent, *j, faults), i)
        else:
            with ThreadPoolExecutor(workers) as ex:
                futs = [ex.submit(run_episode, agent, *j, faults) for j in jobs]
                for i, fut in enumerate(as_completed(futs), 1):
                    save(fut.result(), i)
    print(f"\nDone. Next: python -m lab.analyze {out}")


if __name__ == "__main__":
    main()
