#!/bin/bash
# Pre-registered tau2-bench grid (PREREGISTRATION.md, "τ²-bench experiment").
# Usage: scripts/run_tau2.sh <litellm agent model> <trials> <name> [extra tau2_faults args]
# e.g.   scripts/run_tau2.sh anthropic/claude-sonnet-5-5 2 sonnet
#        scripts/run_tau2.sh openai/gemma4:26b 3 gemma --agent-api-base http://localhost:11434/v1
# Rerunning the same command resumes, and retries infrastructure errors.
set -u
cd "$(dirname "$0")/.."
model=$1; trials=$2; name=$3; shift 3
domains="airline retail"
if [ "${1:-}" = "--domain-only" ]; then domains=$2; shift 2; fi
ids() { python3 -c "import json; print(','.join(json.load(open('tau2_tasks.json'))['$1']))"; }
for domain in $domains; do
  for rate in 0 0.2; do
    for cond in baseline idempotent; do
      out=results/tau2/$name-$domain-$cond-p$rate
      echo "### $(date +%H:%M) $out"
      .venv-tau2/bin/python -m lab.tau2_faults --domain $domain --condition $cond --fault-rate $rate \
        --lost-share 0.5 --agent-llm "$model" --task-ids "$(ids $domain)" --num-trials $trials \
        --seed 300 --concurrency 4 --out $out "$@" 2>&1 | grep -E "graded simulations|WARNING|Traceback"
    done
  done
done
echo "### $(date +%H:%M) done"
