# Reproducing every number in the paper

Every number in `paper/main.tex` comes from a script run on the saved results in `results/`. Running a script needs **no API calls and costs nothing**: the analyses read the stored episodes and traces, and the silent-failure judge reads its cached verdicts. To check the paper, run each command below and compare its output with the place in the paper listed next to it.

Environments: `.venv` (Python 3.9, lab and analysis), `.venv-sec` (Python 3.12, security), `.venv-tau2` (Python 3.12, τ²-bench plug-in).

## Tests

| Check | Command | Expected |
|---|---|---|
| Lab, grading, idempotency key collisions and the too-broad-key case | `.venv/bin/python tests/test_lab.py` | 13 `ok` |
| Security harness and defenses (no models, no API) | `.venv-sec/bin/python tests/test_security.py` | 12 `ok` |
| τ²-bench plug-in: replay is exact under faults | `.venv-tau2/bin/python tests/test_tau2_faults.py` | 4 `ok` |

## Confirmatory results

| Paper location | What | Command |
|---|---|---|
| Table I | Screening (60 episodes per model) | `.venv/bin/python -m lab.analyze results/screen-<model>` for `gemma4-26b`, `qwen3-coder-next-latest`, `qwen3-235b`, `qwen2-5-7b-instruct`, `llama3-1-8b` |
| Table III, Fig. 2, §IV-A (H1 long tasks), §IV-B (H2, H3 paired tests, Holm p) | Main reliability grids | `.venv/bin/python -m lab.analyze results/main-realistic-<model>` for `sonnet`, `haiku`, `gemma4-26b` |
| Table IV, §IV-C (pass^5 vs. p^5) | Task-clustered bootstrap; pass^k independence check | `.venv/bin/python scripts/clustered_stats.py` |
| Fig. 3 | pass^k curves | `.venv/bin/python scripts/make_figures.py` |
| §IV-E, Fig. 4 (H11, H12) | τ²-bench paired differences, H12 interaction, 0% noise floor | `.venv/bin/python scripts/tau2_paired.py` |
| §IV-E | Lost responses and duplicates in τ²-bench | `results/tau2/<run>/faults.json` (per-simulation fault log) |
| §V, Fig. 5 (H8) | Claude security: attack success, utility, paired tests vs. none | `.venv-sec/bin/python -m lab.security_analyze results/sec-full-<model>` and `results/sec-within-<model>` for `sonnet`, `haiku` |

## Exploratory results (§VI)

| Paper location | What | Command |
|---|---|---|
| §VI-A | Exactly-once instruction (+6.7) and read-back tools (+17.3) vs. baseline | `.venv/bin/python scripts/exploratory_paired.py` |
| §VI-A | Failed and duplicate episodes per condition | `lab.analyze` on `results/instruct-sonnet`, `results/obs-sonnet`, `results/main-realistic-sonnet` |
| §VI-B, Fig. `fig_keys` | Agent-supplied keys: sent with 99%/89% of writes; reused on 96/96 and 92/98 retries of the same operation; Gemma 0 of 3,973; success vs. no key and harness keys | `lab.analyze` on `results/gen-sonnet55-keys`, `results/gen-sonnet45`, `results/main-realistic-gemma4-26b` ("Key adoption" table); `.venv/bin/python scripts/exploratory_paired.py` (rows "agent keys"); figure: `scripts/make_figures.py` |
| §VI-C | Sonnet 4.5 (+18.0) and Opus 5.5 (+13.3) | `lab.analyze` on `results/gen-sonnet45`, `results/main-slice-opus` |
| §VI-C | Sonnet 4.5 security | `lab.security_analyze` on `results/sec-full-sonnet45`, `results/sec-within-sonnet45` |
| §VI-D | Redacting detectors (100% utility; flagged 112 vs. 83 of 140) | `lab.security_analyze` on `results/sec-full-redact-sonnet`, `results/sec-within-redact-sonnet` |
| §VI-E | Error-channel injection (−10.8; per goal; averaged per task and trial) | `.venv/bin/python scripts/exploratory_paired.py` (rows `defense=none`) and `.venv/bin/python -m lab.error_injection` summary in `results/errinj-sonnet/summary.md` |
| §VI-F | Silent failures (19–25% Sonnet, 27–53% Haiku) | `.venv/bin/python -m lab.silent_failures results/main-realistic-sonnet results/main-realistic-haiku`; add `--show 20` to print judged cases for the manual check |

## Appendix

| Paper location | What | Command |
|---|---|---|
| Table VI, "Deduplication audit" | Deduplicated writes by first result; identical arguments; episodes | `.venv/bin/python scripts/dedup_audit.py` |
| "Realised fault exposure" | Faults, lost responses, logical calls per episode | `lab.analyze` on the main grids ("Faults actually injected" table) |
| Table V | Deviations and incidents | `DEVIATIONS.md` |
| Fig. 1 | Study setup diagram | `.venv/bin/python scripts/make_overview.py` |

## Machine-readable results and comparing your own model

`.venv/bin/python scripts/export_tables.py` writes `results/summary/{reliability,paired,security,tau2,keys}.csv`. To set a run of your own model (same layout as `results/main-realistic-*`, from `python -m lab.run --preset realistic`) next to the published ones: `.venv/bin/python scripts/export_tables.py --compare <your-folder>`.

## Standalone examples

`examples/` has three scripts that use only the Python standard library, each printing `OK` when it runs as described: a lost response duplicated by a naive retry, a too-broad idempotency key merging two intended writes, and replay grading that reconstructs a different state from the live run. Copy them to check your own tool wrapper or evaluator.

## Rerunning the experiments (not needed to check the paper)

The run commands are in `README.md` and `scripts/run_local_rest.sh`. Claude runs use the Anthropic API, and τ²-bench runs use the OpenAI API for the customer simulator, so both cost money. Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) retires from the API on 2026-11-30, so its runs can't be repeated after that date.
