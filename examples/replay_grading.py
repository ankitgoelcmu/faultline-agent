"""Replay-based grading reconstructs a different state than the live run.

  python examples/replay_grading.py

Standalone (standard library only). Benchmarks such as tau2-bench grade by
replaying the logged tool calls into a fresh database. If a call reported an
error *after* committing (a lost response), a replay that skips errored calls
misses that write; one that re-runs every call would also apply a transient
failure that never committed. Replaying each call according to what really
happened reproduces the live state. Expected output ends with "OK".
"""

log = [  # (call, what the client saw, what actually happened)
    ({"op": "book", "seat": "12A"}, "error: timeout", "committed"),      # lost response
    ({"op": "book", "seat": "14C"}, "error: unavailable", "not committed"),  # transient
    ({"op": "book", "seat": "15F"}, "ok", "committed"),
]

live = {call["seat"] for call, _, happened in log if happened == "committed"}
skip_errors = {call["seat"] for call, seen, _ in log if seen == "ok"}
rerun_all = {call["seat"] for call, _, _ in log}
by_outcome = {call["seat"] for call, _, happened in log if happened == "committed"}

print(f"live state:                 {sorted(live)}")
print(f"replay, skip errored calls: {sorted(skip_errors)}  (misses the lost response)")
print(f"replay, re-run every call:  {sorted(rerun_all)}  (applies the transient failure)")
print(f"replay by recorded outcome: {sorted(by_outcome)}")
assert skip_errors != live and rerun_all != live and by_outcome == live
print("OK")
