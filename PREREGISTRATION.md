# Pre-registration: Agent Reliability Lab

**Status:** FINAL, 2026-11-06. Tagged `prereg-v1`. Changes after this point are logged under *Deviations*.
**Authors:** Ankit Goel
**Code version:** the commit that adds this file. Every run records its grader version and fault settings in `run.json`.

This document fixes the hypotheses, models, sample sizes and analysis before the main experiments run. Anything not listed here is reported as **exploratory**. Any change after the final commit is logged under *Deviations* with its date and reason.

---

## 1. Research questions

- **RQ1 (reliability):** How does per-call tool reliability turn into task reliability for LLM agents, and which harness-level fixes recover it?
- **RQ2 (security):** How well do off-the-shelf defenses stop indirect prompt injection hidden in tool output, and what does each one cost in normal task success (utility)?

## 2. Hypotheses

### Reliability (RQ1)

| ID | Hypothesis | Test |
|---|---|---|
| H1 | Under `baseline`, task success falls as the fault rate rises, and falls faster for longer tasks. | Success by fault rate and length bucket, with Wilson 95% intervals |
| H2 | Naive `retry` does **not** significantly improve on `baseline`, because the lost responses it re-applies cause duplicate writes. | Paired McNemar, `baseline → retry`, at each fault rate above 0 |
| H3 | `safe_retry` (idempotency keys) significantly improves on both `baseline` and `retry` at every fault rate above 0. | Paired McNemar, `retry → safe_retry` and `baseline → safe_retry` |
| H4 | At the same average fault rate, bursty faults give a different task-success rate from independent faults under `baseline`. (Two-sided: the scripted pilot found higher success under bursty faults.) | Wilson intervals for the two fault models; difference with a two-proportion test |
| H5 | Consistency falls faster than success: under `baseline` at 10% faults, pass^5 is below pass^1 by more than 15 points. | pass^k table |

### Security (RQ2)

| ID | Hypothesis | Test |
|---|---|---|
| H6 | With no defense, attack success is higher for subtle levels (L4–L5) than for crude ones (L1–L2). | Pooled success, L4–L5 vs. L1–L2 |
| H7 | `least_privilege` brings attack success to 0 for **cross-account** goals at every level, with no loss of utility on benign tickets. | Wilson upper bound; benign utility vs. `none` |
| H8 | Detection defenses (`classifier`, `llm_judge`, `nemo`) lower utility under attack compared with `none`, because they withhold the whole poisoned output. | Paired McNemar on `task_passed`, same task, level and goal |
| H9 | `spotlight` lowers attack success compared with `none`. | Paired McNemar on `attack_success` |
| H10 | `least_privilege` does **not** lower attack success for **within-permission** goals (`inflate`, `leak`) compared with `none`. Together with H7, this shows that limiting what the agent can do stops attacks outside its permissions but not those inside them. | Paired McNemar on `attack_success`, within-permission goals only. Supported if the adjusted p ≥ 0.05 **and** `least_privilege`'s attack success is within 10 points of `none`'s. |

### External validity on τ²-bench (RQ1)

| ID | Hypothesis | Test |
|---|---|---|
| H11 | In τ² **airline**, `idempotent` gives higher reward than `baseline` at fault rates above 0, and the gap comes from duplicate writes after lost responses. | Paired McNemar on reward (1 vs. 0), same task and trial seed; count duplicate writes in `faults.json` and the conversations |
| H12 | In τ² **retail**, the gap between `idempotent` and `baseline` is smaller than in airline, because retail's writes are guarded by order status. | Difference between conditions in each domain, with Wilson intervals; compared between domains |

## 3. Design

### Tasks and grading
- **Tasks:** 30 fixed tasks from `lab/tasks.py` (seeded), from 5 to 43 planned tool calls.
- **Grading:** only the final state of the store counts, never the agent's text. Grader version 2. Email checks follow the wording rules in `STATUS_PATTERNS`, which are covered by tests.
- **Grader changes:** if the grader changes after data collection, every run is re-graded with `lab.rescore`, and both versions are reported.

### Reliability experiment
- **Conditions:** `baseline`, `retry`, `safe_retry`, `guarded`.
- **Main fault settings:** the `realistic` preset, i.e. bursty faults (Gilbert–Elliott, mean outage 4 attempts), lost-response share 0.5, stale-read rate 0.1, fault rates 0, 5, 10 and 20%.
- **Comparison run (for H4):** the same settings with `--fault-model uniform`, for the local models and the scripted agent only. For Claude Sonnet only the bursty run is done, because of the budget.
- **Trials:** 5 per cell. That gives 30 × 4 × 4 × 5 = 2,400 episodes per model per fault model, or about $55 for Claude Sonnet at the measured $0.022 per episode.
- **Seeds:** the fault seed depends on the task, fault rate and trial, so all conditions face the same fault dice. This is what makes the paired tests valid.

### Security experiment
- **Defenses:** `none`, `spotlight`, `classifier`, `llm_judge`, `nemo`, `least_privilege`.
- **Attacks:** levels L0–L5, goals `refund` and `exfil`, on the 14 tasks in the `full` preset.
- **Trials:** 1 per cell for Claude models (set by the budget), 3 per cell for local models. No tool faults. Claude results are reported with their wider intervals, never pooled with local models.
- **Detector versions:** the `classifier` model and its threshold are recorded with each run.

### Within-permission attacks
- **Goals:** `inflate` (an extra refund just under the cap on the customer's own order) and `leak` (another customer's address put in the reply to the ticket's own customer).
- **Grid:** the `within` preset, i.e. the same 14 tasks, levels and defenses as the `full` preset. The number of trials matches the security experiment.

### τ²-bench experiment
- **Version:** tau2-bench commit `033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f`, with the `lab/tau2_faults.py` plug-in.
- **Tasks:** only tasks whose reference solution includes at least one write, since lost responses and duplicates can't happen without writes. That's all **26** such airline tasks, plus a fixed random sample of **30** of the 104 such retail tasks (Python `random.Random(2026).sample`, task ids sorted first). The task id lists are in `tau2_tasks.json`, committed with this file.
- **Conditions:** `baseline` and `idempotent`.
- **Faults:** rates 0 and 20%; uniform faults; lost-response share 0.5; no stale reads.
- **Size:** (26 + 30) tasks × 2 conditions × 2 rates = 224 episodes per trial. That's 448 for Claude Sonnet (about $80 at the pilot's $0.18 per episode) and 672 for each local model.
- **Customer simulator:** `gpt-4.1-2025-04-14` (τ²'s default) at temperature 0.
- **Limits:** `max_errors` 30, `max_steps` 200, and τ²'s hallucination retries off.
- **Trials:** 2 per cell for Claude Sonnet (set by the budget), 3 for local models, `--seed 300`.
- **Agents:** the Claude models plus the open-weight models that pass screening. Local models are reached through Ollama's OpenAI-compatible endpoint.
- **Pilot:** a pilot with Claude Sonnet (5 airline tasks × 2 conditions × 0% and 20% faults) was run to measure cost and check the setup. It is **excluded** from the analysis.

### Budget
About $200 of Anthropic API credit, plus about $70 of OpenAI credit for τ²'s customer simulator. The τ² runs are estimated at $27–55 of customer-simulator cost across all agents. It's spent in this order:
1. τ² with Sonnet;
2. the main reliability experiment with Sonnet;
3. the security experiments with Sonnet;
4. the adaptive attacker against Sonnet.

Claude Haiku and Opus get the main reliability experiment only if money is left; if not, that's reported as a limitation. Spending is logged per run from token counts.

### Models
- **Claude:** `claude-sonnet-5-5` (primary, every experiment). `claude-haiku-5-5` and `claude-opus-5-5` run only if budget remains (see Budget).
- **Open-weight:** `gemma4:26b` (digest `08ae7ec1744b`) and `qwen3-coder-next:latest` (digest `ca06e9e4087c`), both passed screening (see below). If a model tag is re-pulled, its new digest is logged as a deviation.
- **Optional fourth model:** `qwen3:235b`, a large mixture-of-experts open-weight model. It answers whether a strong open model behaves like Claude or like the mid-range models. It goes through the same 60-episode screen, decided by the same 70% rule. If it passes, it runs **only the τ² and security experiments** (3 trials), after the other local models, because of Mac time. If it fails, or isn't run, that is reported.
- **Settings:** local models run through Ollama at temperature 0, with a 32,768-token context and a 2,048-token cap on each reply. In the lab, Claude runs use API defaults and the same 2,048-token cap; current Claude models don't accept a temperature setting, so none is sent. In τ², all agents use τ²'s own agent loop and limits. All models get the same system prompt, tools, budget of 80 tool calls and harness.

### Inclusion rule (decided before screening results are seen)
A model enters the main experiments only if, in a screening run, it gets **at least 70% of episodes right at 0% faults**. The screening run is all 30 tasks, under `baseline` and `safe_retry`, 1 trial each: 60 episodes.

- **Why:** below that level, the model's own mistakes swamp the effects being measured.
- **Reporting:** screening results are reported for every model tried, including those that fail the rule.
- **Screening results** (2026-11-06, grader v2; results in `results/screen-*`):

| Model | Correct at 0% faults (n=60) | 95% CI | Decision |
|---|---|---|---|
| gemma4:26b | 58 (97%) | 83–99% | Included |
| qwen3-coder-next:latest | 42 (70%) | 52–83% | Included, exactly at the threshold; reported as borderline |
| qwen2.5:7b-instruct | 10 (17%) | 7–34% | Excluded |
| llama3.1:8b | 0 (0%) | 0–11% | Excluded |

- **Screening tasks reused:** screening episodes reuse the main experiment's 0%-fault tasks, but they are **not** pooled with main-experiment data.
- **Exploratory observations from screening and the quick runs** (not hypothesis tests):
  - Llama 3.1 8B invents tool names and arguments.
  - Qwen 2.5 7B makes up tool results and reports tickets as resolved when they aren't.
  - Gemma 4 followed an L5 injection to refund $588.99, above the cap, on another customer's order, twice.

## 4. Metrics and analysis

- **Primary metric:** share of tasks fully correct, with Wilson 95% intervals. Results are always shown as k/n.
- **Comparisons between conditions:** paired McNemar exact tests, Holm-adjusted across all comparisons in a table, with α = 0.05.
- **Consistency:** pass^k for k = 1…5, using τ-bench's unbiased estimator, averaged over tasks.
- **Security:**
  - attack success, by level, goal and defense;
  - utility on benign and attacked runs;
  - false alarms on benign runs;
  - time added by each defense.
- **Per model:** every hypothesis is tested separately for each agent model. Holm adjustment is applied across the rows of each table.
- **Exclusions:** episodes ending in API or infrastructure errors are excluded and rerun, up to 3 times. In the lab these are rows whose `stop` starts with `error`; in τ² they are runs whose `termination_reason` is `infrastructure_error`. The number excluded is reported.
- **Cost:** token counts are reported per condition, and dollar figures use the prices published on the run date.

## 5. Exploratory analyses (not hypothesis tests)

- **Adaptive attacker** (`lab/adaptive.py`): round 0 is the fixed L5 payload, then up to 5 rewrites by an attacker model. The attacker sees the agent's final reply, its tool calls, and whether a defense acted. Reported as success within 5 rounds and as an upper bound on attack success. The attacker model and rounds are recorded in `run.json`. Payloads are not published.
- **Failure taxonomy:** qualitative coding of failed traces, e.g. made-up tool results, invented tool names, and checking a write before retrying it.

## 6. Limitations stated in advance

- The tools are mocked, and the faults are synthetic.
- It covers one domain (customer support) and 30 tasks.
- The payloads are fixed, apart from the exploratory adaptive round.
- Closed-model behaviour can change between API versions, so run dates are recorded.
- In τ², the customer is itself an LLM (`gpt-4.1`), which adds noise that is not under our control.
- Claude security results use 1 trial per cell, so their intervals are wide.
- qwen3-coder-next passed screening exactly at the threshold.
- Responsible use: the levels and results are published, but not a large payload collection.

## 7. Deviations

| Date | Change | Reason |
|---|---|---|
| – | – | – |

---

**To finalise:** set Status to FINAL with the date, commit with the message `Pre-registration: final`, and tag it `prereg-v1`. Do this before starting any main-experiment run.
