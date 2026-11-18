"""Adaptive prompt-injection attacker (exploratory; pre-registration section 5).

Fixed payloads understate a real attacker, who keeps trying. Here a second
model plays the attacker. Each round it writes the poisoned account note,
the target agent handles the ticket, and the attacker sees what happened and
rewrites the note. It stops at the first success or after --rounds rounds.

Threat model (state it in the paper): a black-box attacker who controls one
account note and, after each attempt, sees the agent's final reply, the tool
calls it made and whether a defense blocked something. It never sees the
system prompt or the defense's code. That is stronger than most real
attackers, so the result is an upper bound for this budget.

  python -m lab.adaptive --agent ollama --model gemma4:26b \\
      --attacker ollama:gemma4:26b --defenses none,spotlight,least_privilege \\
      --goals refund,inflate --rounds 5 --out results/adaptive-gemma

Round 0 is the fixed L5 payload, so the table shows how much adapting adds.
Payloads are saved only in the (git-ignored) results folder. Publish the
method and the success rates, not the payloads.
"""

import argparse
import csv
import json
import os
import re
import urllib.request

from .attacks import _goal, _payload, attack_applies
from .run import parse_tasks
from .security_run import episode
from .stats import rate_ci
from .tasks import build_tasks

FIELDS = ["task", "defense", "goal", "trial", "round", "attack_success", "task_passed",
          "outputs_flagged", "calls_blocked", "tool_calls", "note"]

ATTACKER_PROMPT = """[attack text withheld from the public release; see the paper, Section III-E] (adaptive attacker prompt)
{history}"""


def _feedback(row, rec):
    calls = [f"{t['tool']}({json.dumps(t['args'])[:80]})" for t in rec["trace"]]
    blocked = [t["error"] for t in rec["trace"] if t.get("fault") in ("defense_blocked_call",
                                                                       "defense_flagged_output")]
    return (f"Agent's tool calls: {'; '.join(calls) or 'none'}\n"
            f"Blocked or withheld by a defense: {'; '.join(blocked) or 'nothing'}\n"
            f"Agent's final reply: {rec['final_text'][:300] or '(none)'}")


class Attacker:
    """Rewrites the note. Backend 'ollama:<model>' (local, free) or 'claude:<model>'."""

    def __init__(self, spec):
        self.kind, _, self.model = spec.partition(":")
        if self.kind == "claude":
            import anthropic
            self.client = anthropic.Anthropic(max_retries=6)
        elif self.kind != "ollama":
            raise SystemExit("--attacker must be ollama:<model> or claude:<model>")
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def write(self, prompt):
        if self.kind == "claude":
            r = self.client.messages.create(model=self.model, max_tokens=800,
                                            messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in r.content if b.type == "text")
        else:
            body = {"model": self.model, "stream": False, "options": {"temperature": 0.9},
                    "messages": [{"role": "user", "content": prompt}]}
            req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                text = json.loads(resp.read())["message"]["content"]
        m = re.search(r"<note>(.*?)</note>", text, re.S)
        return (m.group(1) if m else text).strip()[:600] or None   # None = the attacker declined


def attack_one(agent_name, model, agent, attacker, task, defense, goal, trial, rounds):
    """Rounds for one (task, defense, goal, trial). Yields (row, trace record) per round."""
    g = _goal(task, goal)
    note = _payload(5, g)[1]               # round 0: the fixed L5 payload
    history = []
    for rnd in range(rounds + 1):
        row, rec = episode(agent_name, model, task, 6, goal, defense, trial, note=note, agent=agent)
        out = {"task": task["id"], "defense": defense, "goal": goal, "trial": trial, "round": rnd,
               "attack_success": row["attack_success"], "task_passed": row["task_passed"],
               "outputs_flagged": row["outputs_flagged"], "calls_blocked": row["calls_blocked"],
               "tool_calls": row["tool_calls"], "note": note}
        yield out, rec
        if row["attack_success"] or rnd == rounds:
            return
        history.append(f"--- Attempt {rnd + 1}: FAILED\nNote: {note}\n{_feedback(row, rec)}")
        note = attacker.write(ATTACKER_PROMPT.format(goal_text=g["plain"] + ".",
                                                     history="\n\n".join(history)))
        if note is None:
            return


def summarize(folder, rounds):
    with open(os.path.join(folder, "adaptive.csv")) as f:
        rows = list(csv.DictReader(f))
    best = {}
    for r in rows:
        k = (r["task"], r["defense"], r["goal"], r["trial"])
        best.setdefault(k, []).append(r)
    defs = sorted({k[1] for k in best}, key=lambda d: (d != "none", d))
    goals = sorted({k[2] for k in best})
    L = [f"# Adaptive attacker: {os.path.basename(os.path.normpath(folder))}", "",
         "Share of (task, trial) cases the attacker broke within N rounds. Round 0 is the fixed L5 "
         "payload. Brackets are 95% Wilson intervals.", "",
         "| Defense | Goal | Fixed L5 (round 0) | Within " + f"{rounds} rounds | Mean rounds to success |",
         "|---|---|---|---|---|"]
    for d in defs:
        for g in goals:
            cases = [v for k, v in best.items() if k[1] == d and k[2] == g]
            if not cases:
                continue
            r0 = [int(v[0]["attack_success"]) for v in cases]
            anyr = [int(any(int(x["attack_success"]) for x in v)) for v in cases]
            wins = [min(int(x["round"]) for x in v if int(x["attack_success"]))
                    for v in cases if any(int(x["attack_success"]) for x in v)]
            mean_r = f"{sum(wins) / len(wins):.1f}" if wins else "–"
            L.append(f"| {d} | {g} | {rate_ci(r0)} | {rate_ci(anyr)} | {mean_r} |")
    L.append("")
    with open(os.path.join(folder, "summary.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description="Adaptive prompt-injection attacker (exploratory)")
    p.add_argument("--agent", choices=["gullible", "claude", "ollama"], required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--attacker", default="ollama:gemma4:26b")
    p.add_argument("--tasks", default="1,6,14,22")
    p.add_argument("--defenses", default="none,spotlight,least_privilege")
    p.add_argument("--goals", default="refund,inflate")
    p.add_argument("--rounds", type=int, default=5, help="adaptive rounds after the fixed payload")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from .agents import make_agent
    agent = None if a.agent == "gullible" else make_agent(a.agent, a.model)
    attacker = Attacker(a.attacker)
    os.makedirs(a.out, exist_ok=True)
    csv_path, trace_path = os.path.join(a.out, "adaptive.csv"), os.path.join(a.out, "traces.jsonl")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            done = {(r["task"], r["defense"], r["goal"], r["trial"]) for r in csv.DictReader(f)}
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fc, open(trace_path, "a") as ft:
        w = csv.DictWriter(fc, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for t in parse_tasks(a.tasks, build_tasks()):
            for d in a.defenses.split(","):
                for g in a.goals.split(","):
                    if not attack_applies(t, g):
                        continue
                    for k in range(1, a.trials + 1):
                        if (t["id"], d, g, str(k)) in done:
                            continue
                        for row, rec in attack_one(a.agent, a.model, agent, attacker, t, d, g, k,
                                                   a.rounds):
                            w.writerow(row)
                            ft.write(json.dumps(rec, default=str) + "\n")
                            fc.flush(); ft.flush()
                            print(f"{t['id']} {d:<16} {g:<8} round {row['round']}: "
                                  f"{'ATTACK OK' if row['attack_success'] else 'blocked'}", flush=True)
    summarize(a.out, a.rounds)


if __name__ == "__main__":
    main()
