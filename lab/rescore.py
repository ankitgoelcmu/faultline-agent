"""Re-grade a finished results folder with the current grader. No model calls.

  python -m lab.rescore results/<folder> [--dry-run]

Every trace records each tool attempt and whether its write took effect, so
the final world state can be rebuilt exactly: replay the applied writes into
a fresh world (with the same idempotency keys the harness used), then grade.
The original files are kept as *.v1 the first time a folder is rescored.

Works for both reliability folders (episodes.csv) and security folders
(security.csv). Attack success is recomputed too.
"""

import argparse
import csv
import json
import os
import shutil

from .attacks import attack_succeeded, make_attack
from .harness import intent_key
from .run import CSV_FIELDS
from .security_run import FIELDS as SEC_FIELDS
from .tasks import GRADER_VERSION, build_tasks, grade
from .world import WRITE_TOOLS, BusinessError, World

def replay(task, trace, keyed):
    world = World(task["data"])
    for t in trace:
        # A write took effect if it succeeded (including one whose output a
        # defense then withheld) or if only its response was lost. Faults,
        # business errors and blocked calls never changed anything.
        if t["tool"] not in WRITE_TOOLS or not (t["ok"] or t["fault"] == "lost_response"):
            continue
        args = t["args"]
        key = intent_key(t["tool"], args) if keyed else None
        if "idempotency_key" in args:   # key_available: the agent's own key
            ck = args["idempotency_key"]
            args = {k: v for k, v in args.items() if k != "idempotency_key"}
            key = f"client:{ck}" if ck not in (None, "") else None
        try:
            world.call(t["tool"], args, idempotency_key=key)
        except BusinessError:
            pass
    return world


def rescore(folder, dry_run=False):
    security = os.path.exists(os.path.join(folder, "security.csv"))
    csv_name = "security.csv" if security else "episodes.csv"
    fields = SEC_FIELDS if security else CSV_FIELDS
    tasks = {t["id"]: t for t in build_tasks()}
    trace_path = os.path.join(folder, "traces.jsonl")
    with open(trace_path) as f:
        recs = [json.loads(line) for line in f if line.strip()]

    changes = []
    for rec in recs:
        row = rec["row"]
        if str(row["stop"]).startswith("error"):
            continue
        task = tasks[row["task"]]
        if security:
            level = int(row["level"])
            task, attack = (make_attack(task, level, row["goal"]) if level else (task, None))
            world = replay(task, rec["trace"], keyed=False)
        else:
            world = replay(task, rec["trace"], keyed=row["condition"] in ("safe_retry", "guarded"))
        passed, fails = grade(task, world)
        pass_field = "task_passed" if security else "passed"
        new = {pass_field: int(passed), "fails": ";".join(fails)}
        if security:
            ok, partial = attack_succeeded(attack, world)
            new.update(attack_success=int(ok), attack_partial=int(partial))
        old = {k: row[k] for k in new}
        if {k: str(v) for k, v in old.items()} != {k: str(v) for k, v in new.items()}:
            changes.append((rec["key"], old, new))
            row.update(new)

    print(f"{len(recs)} episodes, {len(changes)} changed by grader v{GRADER_VERSION}:")
    for key, old, new in changes:
        print(f"  {key}: {old} -> {new}")
    if dry_run:
        return changes

    if changes:
        for name in (csv_name, "traces.jsonl"):
            src = os.path.join(folder, name)
            if not os.path.exists(src + ".v1"):
                shutil.copy(src, src + ".v1")
        with open(trace_path, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, default=str) + "\n")
        with open(os.path.join(folder, csv_name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for rec in recs:
                w.writerow(rec["row"])
    meta_path = os.path.join(folder, "run.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta["grader"] = GRADER_VERSION
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return changes


def main():
    p = argparse.ArgumentParser(description="Re-grade a results folder with the current grader")
    p.add_argument("folder")
    p.add_argument("--dry-run", action="store_true", help="show changes, write nothing")
    a = p.parse_args()
    rescore(a.folder, a.dry_run)


if __name__ == "__main__":
    main()
