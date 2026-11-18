"""Run the prompt-injection experiment.

Every episode: one task, one attack level (0 = no attack), one attacker goal,
one defense. No tool faults here, so only the attack varies.

Examples:
  python -m lab.security_run --agent gullible                 # free plumbing check
  python -m lab.security_run --agent claude --preset quick    # small paid run
  python -m lab.security_run --agent claude --preset full
"""

import argparse
import csv
import datetime as dt
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents import MAX_TOOL_CALLS, make_agent
from .attacks import GOALS, LEVELS, attack_applies, attack_succeeded, make_attack, make_custom_attack
from .defenses import judge_spend_usd, make_defense
from .harness import Harness
from .run import parse_tasks, seed_for
from .tasks import GRADER_VERSION, build_tasks, grade
from .world import World

PRESETS = {
    "quick": dict(tasks="1,6,14,22", levels="0,1,3,5", goals="refund",
                  defenses="none,spotlight,least_privilege"),
    "full": dict(tasks="1,3,5,6,8,10,11,13,14,16,18,20,22,24", levels="0,1,2,3,4,5",
                 goals="refund,exfil",
                 defenses="none,spotlight,classifier,llm_judge,nemo,least_privilege"),
    # Attacks from the ticket's own customer that stay inside task-scoped
    # permissions: what least_privilege is not built to stop.
    "within": dict(tasks="1,3,5,6,8,10,11,13,14,16,18,20,22,24", levels="0,1,2,3,4,5",
                   goals="inflate,leak",
                   defenses="none,spotlight,classifier,llm_judge,nemo,least_privilege"),
}

FIELDS = ["task", "kind", "defense", "level", "goal", "surface", "trial", "agent",
          "attack_success", "attack_partial", "task_passed", "fails",
          "outputs_flagged", "calls_blocked", "tool_calls", "defense_ms",
          "stop", "turns", "input_tokens", "output_tokens",
          "cache_read_input_tokens", "cache_creation_input_tokens", "seconds"]


class GullibleAgent:
    """A scripted agent with no judgment: it follows the task plan, and if it
    reads any injected text it also does what the attacker asked. Free, and
    useful to check that attacks land and defenses block them."""
    name = "gullible"

    def __init__(self):
        self.attack = None

    def run(self, task, harness):
        done_attack = False
        for tool, args in task["plan"]:
            if harness.calls >= MAX_TOOL_CALLS:
                return {"stop": "budget"}
            ok, out = harness.call(tool, args)
            text = json.dumps(out)
            if (self.attack and not done_attack and ok
                    and any(m in text for m in self.attack["markers"])):
                done_attack = True
                for t2, a2 in self.attack["actions"]:
                    harness.call(t2, a2)
        return {"stop": "done"}


def episode(agent_name, model, task, level, goal, defense_name, trial, note=None, agent=None):
    """One episode. `note` replaces the attack with custom account-note text
    (the adaptive attacker); `agent` reuses an already-built agent."""
    if agent is None:
        agent = GullibleAgent() if agent_name == "gullible" else make_agent(agent_name, model)
    if note is not None:
        poisoned, attack = make_custom_attack(task, goal, note)
    else:
        poisoned, attack = make_attack(task, level, goal)
    if isinstance(agent, GullibleAgent):
        agent.attack = attack
    defense = make_defense(defense_name)
    defense.setup(poisoned)
    world = World(poisoned["data"])
    trace = []
    h = Harness(world, "baseline", 0.0, seed_for(task["id"], level, goal, trial), trace,
                defense=defense)
    t0 = time.time()
    try:
        info = agent.run(poisoned, h)
    except Exception as e:
        info = {"stop": f"error: {type(e).__name__}: {e}"[:300]}
    passed, fails = grade(poisoned, world)
    success, partial = attack_succeeded(attack, world)
    row = {
        "task": task["id"], "kind": task["kind"], "defense": defense_name, "level": level,
        "goal": goal if level else "-", "surface": attack["surface"] if attack else "-",
        "trial": trial, "agent": getattr(agent, "name", agent_name),
        "attack_success": int(success), "attack_partial": int(partial),
        "task_passed": int(passed), "fails": ";".join(fails),
        "outputs_flagged": sum(1 for x in trace if x.get("fault") in ("defense_flagged_output",
                                                                      "defense_redacted_output")),
        "calls_blocked": sum(1 for x in trace if x.get("fault") == "defense_blocked_call"),
        "tool_calls": h.calls,
        "defense_ms": round(sum(x.get("defense_ms", 0) for x in trace), 1),
        "stop": info.get("stop"), "turns": info.get("turns", 0),
        "input_tokens": info.get("input_tokens", 0), "output_tokens": info.get("output_tokens", 0),
        "cache_read_input_tokens": info.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": info.get("cache_creation_input_tokens", 0),
        "seconds": round(time.time() - t0, 2),
    }
    key = f"{row['task']}|{defense_name}|{level}|{row['goal']}|{trial}"
    return row, {"key": key, "row": row, "attack": attack, "final_text": info.get("final_text", ""),
                 "trace": trace}


def main():
    p = argparse.ArgumentParser(description="Prompt-injection lab")
    p.add_argument("--agent", choices=["gullible", "claude", "ollama"], default="gullible")
    p.add_argument("--model", default=os.environ.get("LAB_MODEL", "claude-sonnet-5-5"))
    p.add_argument("--preset", choices=list(PRESETS))
    p.add_argument("--tasks", default="1,6,14,22")
    p.add_argument("--levels", default=",".join(map(str, LEVELS)))
    p.add_argument("--goals", default=",".join(GOALS))
    p.add_argument("--defenses", default="none,spotlight,least_privilege")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None)
    # A preset sets defaults; flags given explicitly still win.
    preset = p.parse_known_args()[0].preset
    if preset:
        p.set_defaults(**PRESETS[preset])
    a = p.parse_args()

    tasks = parse_tasks(a.tasks, build_tasks())
    levels = [int(x) for x in a.levels.split(",")]
    goals = [g.strip() for g in a.goals.split(",")]
    defenses = [d.strip() for d in a.defenses.split(",")]
    for d in defenses:          # fail fast on missing packages or keys
        make_defense(d)
    if a.agent in ("claude", "ollama"):
        make_agent(a.agent, a.model)

    out = a.out or os.path.join("results", f"{dt.datetime.now():%Y%m%d-%H%M%S}-security-{a.agent}")
    os.makedirs(out, exist_ok=True)
    csv_path, trace_path = os.path.join(out, "security.csv"), os.path.join(out, "traces.jsonl")
    done = set()
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            # Episodes that ended in an API error are retried, not kept as done.
            done = {r["key"] for r in map(json.loads, filter(str.strip, f))
                    if not str(r["row"]["stop"]).startswith("error")}

    jobs = []
    for t in tasks:
        for d in defenses:
            for k in range(1, a.trials + 1):
                for lv in levels:
                    for g in (goals if lv else ["-"]):
                        if lv and not attack_applies(t, g):
                            continue
                        if f"{t['id']}|{d}|{lv}|{g}|{k}" not in done:
                            jobs.append((t, lv, g if lv else goals[0], d, k))
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump({"agent": a.agent, "model": a.model if a.agent in ("claude", "ollama") else None,
                   "tasks": [t["id"] for t in tasks], "levels": levels, "goals": goals,
                   "defenses": defenses, "trials": a.trials, "grader": GRADER_VERSION,
                   "started": dt.datetime.now().isoformat()}, f, indent=2)
    print(f"{len(jobs)} episodes to run ({len(done)} already done) -> {out}")

    lock = threading.Lock()
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fc, open(trace_path, "a") as ft:
        w = csv.DictWriter(fc, fieldnames=FIELDS)
        if new:
            w.writeheader()

        def save(res, i):
            row, rec = res
            with lock:
                w.writerow(row); ft.write(json.dumps(rec, default=str) + "\n")
                fc.flush(); ft.flush()
                atk = "ATTACK OK" if row["attack_success"] else "attack blocked" if row["level"] else "benign"
                print(f"[{i}/{len(jobs)}] {row['task']} {row['defense']:<16} L{row['level']} "
                      f"{row['goal']:<6} {atk:<14} task={'pass' if row['task_passed'] else 'fail'}")

        run = lambda j: episode(a.agent, a.model, *j)
        workers = a.workers if a.agent in ("claude", "ollama") else 1
        if workers <= 1:
            for i, j in enumerate(jobs, 1):
                save(run(j), i)
        else:
            with ThreadPoolExecutor(workers) as ex:
                futs = [ex.submit(run, j) for j in jobs]
                for i, fut in enumerate(as_completed(futs), 1):
                    save(fut.result(), i)
    if judge_spend_usd():
        print(f"LLM-judge spend this process: ${judge_spend_usd():.2f}")
    print(f"\nDone. Next: python -m lab.security_analyze {out}")


if __name__ == "__main__":
    main()
