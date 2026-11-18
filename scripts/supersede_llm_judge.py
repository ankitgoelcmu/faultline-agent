"""Move llm_judge rows collected before the judge fix out of a security folder.

  .venv/bin/python scripts/supersede_llm_judge.py results/sec-full-sonnet [...]

Rows for `llm_judge` and `llm_judge_redact` are appended to
security.llm_judge_v1.csv / traces.llm_judge_v1.jsonl (kept, not deleted)
and removed from the live files, so rerunning the same security_run command
on the folder redoes exactly those episodes with the fixed judge.
See DEVIATIONS.md (2026-11-06, llm_judge incident).
"""

import csv
import json
import os
import sys

DEFS = {"llm_judge", "llm_judge_redact"}

for folder in sys.argv[1:]:
    csv_path, tr_path = os.path.join(folder, "security.csv"), os.path.join(folder, "traces.jsonl")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    keep = [r for r in rows if r["defense"] not in DEFS]
    old = [r for r in rows if r["defense"] in DEFS]
    recs = [json.loads(line) for line in open(tr_path) if line.strip()]
    keep_t = [r for r in recs if r["row"]["defense"] not in DEFS]
    old_t = [r for r in recs if r["row"]["defense"] in DEFS]
    if not old:
        print(f"{folder}: no llm_judge rows")
        continue
    sup_csv = os.path.join(folder, "security.llm_judge_v1.csv")
    new_sup = not os.path.exists(sup_csv)
    with open(sup_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_sup:
            w.writeheader()
        w.writerows(old)
    with open(os.path.join(folder, "traces.llm_judge_v1.jsonl"), "a") as f:
        for r in old_t:
            f.write(json.dumps(r, default=str) + "\n")
    with open(csv_path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    with open(tr_path + ".tmp", "w") as f:
        for r in keep_t:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(csv_path + ".tmp", csv_path)
    os.replace(tr_path + ".tmp", tr_path)
    print(f"{folder}: moved {len(old)} rows ({len(old_t)} traces) to the superseded files")
