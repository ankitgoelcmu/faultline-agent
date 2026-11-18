"""Rebuild every paper figure from the results folders.

  .venv/bin/python scripts/make_figures.py            # all models found
  .venv/bin/python scripts/make_figures.py --test     # plumbing check on non-study data

Writes paper/figures/*.pdf (for LaTeX) and *.png (for slides and posts).
Missing folders are skipped, so this runs at any point. Results folder names:
  results/main-realistic-<model>           main reliability grid (bursty)
  results/main-uniform-<model>             uniform comparison (H4)
  results/sec-full-<model>                 security, cross-account goals
  results/sec-within-<model>               security, within-permission goals
  results/tau2/<model>-<domain>-<condition>-p<rate>/
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab.stats import pass_hat_k, wilson   # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures")

MODELS = {   # results-folder suffix -> label
    "sonnet": "Claude Sonnet 5.5",
    "haiku": "Claude Haiku 5.5",
    "gemma4-26b": "Gemma 4 26B",
    "qwen3-coder-next-latest": "qwen3-coder-next",
    "qwen3-235b": "Qwen3 235B",
}
TAU2_NAMES = {"sonnet": "sonnet", "haiku": "haiku", "gemma4-26b": "gemma", "qwen3-coder-next-latest": "qwen3coder",
              "qwen3-235b": "qwen235"}
COND_COLORS = {"baseline": "#d1495b", "retry": "#edae49", "safe_retry": "#00798c",
               "guarded": "#30638e", "idempotent": "#00798c"}
DEF_ORDER = ["none", "spotlight", "classifier", "llm_judge", "nemo", "least_privilege"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 160})


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f) if not str(r.get("stop", "")).startswith("error")]


def ci_bar(flags):
    k, n = sum(flags), len(flags)
    lo, hi = wilson(k, n)
    p = k / n if n else 0
    return 100 * p, 100 * (p - lo), 100 * (hi - p)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_reliability(models):
    """Task success vs. fault rate, one panel per model (H1-H3)."""
    have = [(m, rows(os.path.join(RES, f"main-realistic-{m}", "episodes.csv"))) for m in models]
    have = [(m, r) for m, r in have if r]
    opus = rows(os.path.join(RES, "main-slice-opus", "episodes.csv")) if "opus" in models else []
    if opus:   # exploratory slice: baseline and safe_retry at 0% and 20% only
        have.append(("opus", opus))
        MODELS.setdefault("opus", "Claude Opus 5.5 (slice)")
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(2.4 * len(have), 2.2), sharey=True, squeeze=False)
    for ax, (m, rs) in zip(axes[0], have):
        by = defaultdict(list)
        for r in rs:
            by[(r["condition"], float(r["fault_rate"]))].append(int(r["passed"]))
        rates = sorted({k[1] for k in by})
        for i, c in enumerate(["baseline", "retry", "safe_retry", "guarded"]):
            pts = [ci_bar(by[(c, x)]) for x in rates if by[(c, x)]]
            if not pts:
                continue
            # Small horizontal offsets so identical curves stay visible.
            ax.errorbar([100 * x + 0.35 * (i - 1.5) for x in rates], [p[0] for p in pts],
                        ls=["-", "--", "-.", ":"][i],
                        yerr=[[p[1] for p in pts], [p[2] for p in pts]],
                        marker="osD^"[i], ms=3, lw=1.2, capsize=2, label=c, color=COND_COLORS[c])
        ax.set_title(MODELS.get(m, m))
        ax.set_xlabel("Per-call fault rate (%)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel("Tasks fully correct (%)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), ncol=4, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_reliability")


def fig_passk(models, rate=0.20):
    """pass^k for baseline vs. safe_retry at one fault rate (H5)."""
    have = []
    for m in models:
        rs = [r for r in rows(os.path.join(RES, f"main-realistic-{m}", "episodes.csv"))
              if abs(float(r["fault_rate"]) - rate) < 1e-9]
        if rs:
            have.append((m, rs))
    if not have:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    palette = ["#1f4e79", "#2a9d8f", "#c1666b", "#8d6a9f", "#e9c46a"]
    for mi, (m, rs) in enumerate(have):
        for c, ls in (("baseline", "-"), ("safe_retry", "--")):
            per_task = defaultdict(list)
            for r in rs:
                if r["condition"] == c:
                    per_task[r["task"]].append(int(r["passed"]))
            if not per_task:
                continue
            n = min(len(v) for v in per_task.values())
            ks = list(range(1, n + 1))
            ys = [100 * sum(pass_hat_k(sum(v), len(v), k) for v in per_task.values()) / len(per_task)
                  for k in ks]
            # One colour and marker per model, line style per condition, and a
            # small offset so models with near-identical values stay visible.
            ax.plot([k + 0.06 * (mi - (len(have) - 1) / 2) for k in ks], ys, ls,
                    marker="os^Dv"[mi % 5], ms=3, color=palette[mi % 5],
                    label=f"{MODELS.get(m, m)}, {c}")
    ax.set_xlabel("k (attempts that must all succeed)")
    ax.set_ylabel(rf"pass$^k$ at {100 * rate:.0f}% faults (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    save(fig, "fig_passk")


def tau2_cell(name, domain, cond, rate):
    d = os.path.join(RES, "tau2", f"{name}-{domain}-{cond}-p{rate}")
    p = os.path.join(d, "results.json")
    if not os.path.exists(p):
        return None
    sims = [s for s in json.load(open(p))["simulations"]
            if s.get("termination_reason") != "infrastructure_error" and s.get("reward_info")]
    rewards = [int(s["reward_info"]["reward"] >= 1.0) for s in sims]
    faults = json.load(open(os.path.join(d, "faults.json")))["faults"] if os.path.exists(
        os.path.join(d, "faults.json")) else {}
    kinds = Counter(e["kind"] for log in faults.values() for e in log)
    return {"rewards": rewards, "dup": duplicate_writes(sims), "kinds": kinds}


def duplicate_writes(sims):
    """Writes that ran more than once with identical arguments in one episode.
    Counted from executed, non-error writes plus lost responses (which ran)."""
    from_tools = {"book_reservation", "cancel_reservation", "send_certificate",
                  "update_reservation_baggages", "update_reservation_flights",
                  "update_reservation_passengers", "cancel_pending_order",
                  "exchange_delivered_order_items", "modify_pending_order_address",
                  "modify_pending_order_items", "modify_pending_order_payment",
                  "modify_user_address", "return_delivered_order_items"}
    total = 0
    for s in sims:
        calls = {}
        for m in s["messages"]:
            for t in m.get("tool_calls") or []:
                calls[t["id"]] = (t["name"], json.dumps(t["arguments"], sort_keys=True))
        ran = Counter()
        for m in s["messages"]:
            if m.get("role") != "tool" or m.get("id") not in calls:
                continue
            name, _ = calls[m["id"]]
            lost = (m.get("content") or "").startswith("Error: 504")
            if name in from_tools and m.get("execution_status") == "executed" and (not m.get("error") or lost):
                ran[calls[m["id"]]] += 1
        total += sum(v - 1 for v in ran.values() if v > 1)
    return total


def fig_tau2(models):
    """tau2 reward: baseline vs. idempotent, airline vs. retail, 0% and 20% faults (H11-H12)."""
    panels = []
    for m in models:
        name = TAU2_NAMES.get(m, m)
        cells = {(dom, c, r): tau2_cell(name, dom, c, r)
                 for dom in ("airline", "retail") for c in ("baseline", "idempotent") for r in ("0", "0.2")}
        if any(cells.values()):
            panels.append((m, cells))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.3), sharey=True, squeeze=False)
    for ax, (m, cells) in zip(axes[0], panels):
        xs, labels = 0, []
        for dom in ("airline", "retail"):
            for r in ("0", "0.2"):
                for j, c in enumerate(("baseline", "idempotent")):
                    cell = cells[(dom, c, r)]
                    if cell and cell["rewards"]:
                        p, lo, hi = ci_bar(cell["rewards"])
                        ax.bar(xs + 0.38 * j, p, 0.36, yerr=[[lo], [hi]], capsize=2,
                               error_kw={"lw": 0.8},
                               color=COND_COLORS[c], label=c if (dom, r) == ("airline", "0") else None)
                        if c == "baseline" and cell["dup"]:
                            ax.text(xs, p + hi + 2, f"{cell['dup']} dup", ha="center", fontsize=6)
                labels.append((xs + 0.19, f"{dom}\n{float(r) * 100:.0f}%"))
                xs += 1
        ax.set_xticks([x for x, _ in labels], [t for _, t in labels], fontsize=7)
        ax.set_title(MODELS.get(m, m))
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=.3)
    axes[0][0].set_ylabel(r"$\tau^2$-bench tasks solved (%)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), ncol=2, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_tau2")


def fig_attacks(models, kind="full"):
    """Attack success by level and defense (H6-H9; 'within' for H10)."""
    have = [(m, rows(os.path.join(RES, f"sec-{kind}-{m}", "security.csv"))) for m in models]
    have = [(m, r) for m, r in have if r]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(2.6 * len(have), 2.3), sharey=True, squeeze=False)
    for ax, (m, rs) in zip(axes[0], have):
        defs = sorted({r["defense"] for r in rs}, key=lambda d: DEF_ORDER.index(d) if d in DEF_ORDER else 99)
        levels = sorted({int(r["level"]) for r in rs if int(r["level"]) > 0})
        for d in defs:
            ys = []
            for lv in levels:
                g = [int(r["attack_success"]) for r in rs if r["defense"] == d and int(r["level"]) == lv]
                ys.append(100 * sum(g) / len(g) if g else float("nan"))
            ax.plot(levels, ys, marker="o", ms=3, lw=1.2, label=d)
        ax.set_xticks(levels, [f"L{lv}" for lv in levels])
        ax.set_title(MODELS.get(m, m))
        ax.set_ylim(-3, 103)
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel("Attack success (%)")
    axes[0][-1].legend(fontsize=6)
    save(fig, f"fig_attacks_{kind}")


def fig_tradeoff(models):
    """What each defense costs: real work done with and without attack, per defense,
    with attack success printed above the bars (H8; cross-account + within goals)."""
    have = [(m, rows(os.path.join(RES, f"sec-full-{m}", "security.csv"))
                + rows(os.path.join(RES, f"sec-within-{m}", "security.csv"))) for m in models]
    have = [(m, r) for m, r in have if r]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(3.4 * len(have), 2.4), sharey=True, squeeze=False)
    for ax, (m, rs) in zip(axes[0], have):
        defs = sorted({r["defense"] for r in rs}, key=lambda d: DEF_ORDER.index(d) if d in DEF_ORDER else 99)
        for i, d in enumerate(defs):
            ben = [int(r["task_passed"]) for r in rs if r["defense"] == d and int(r["level"]) == 0]
            att = [r for r in rs if r["defense"] == d and int(r["level"]) > 0]
            for j, (vals, color, lab) in enumerate(((ben, "#00798c", "no attack"),
                                                     ([int(r["task_passed"]) for r in att], "#d1495b",
                                                      "under attack"))):
                if vals:
                    p_, lo, hi = ci_bar(vals)
                    ax.bar(i + 0.38 * j, p_, 0.36, yerr=[[lo], [hi]], capsize=2, error_kw={"lw": 0.8},
                           color=color, label=lab if i == 0 else None)
            if att:
                a = 100 * sum(int(r["attack_success"]) for r in att) / len(att)
                ax.text(i + 0.19, 106, f"{a:.0f}%", ha="center", fontsize=6)
        ax.set_xticks([i + 0.19 for i in range(len(defs))], [d.replace("_", "\n") for d in defs], fontsize=6)
        ax.set_title(MODELS.get(m, m))
        ax.set_ylim(0, 114)
        ax.grid(axis="y", alpha=.3)
    axes[0][0].set_ylabel("Legitimate task completion (%)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), ncol=2, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_tradeoff")



KEY_RUNS = [   # label, folder with key_available, folder with baseline and safe_retry
    ("Sonnet 5.5", "gen-sonnet55-keys", "main-realistic-sonnet"),
    ("Sonnet 4.5", "gen-sonnet45", "gen-sonnet45"),
    ("Gemma 4 26B", "main-realistic-gemma4-26b", "main-realistic-gemma4-26b"),
]


def fig_keys():
    """Exploratory: agents given an optional idempotency_key. Left: what the
    agent's next write did after a lost response. Right: task success at 20%
    faults with no key, agent-supplied keys, and harness-computed keys."""
    from lab.analyze import key_adoption
    have = [(lab, k, b) for lab, k, b in KEY_RUNS
            if os.path.exists(os.path.join(RES, k, "traces.jsonl"))
            and os.path.exists(os.path.join(RES, b, "episodes.csv"))]
    if not have:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.2), gridspec_kw={"height_ratios": [1, 1.1]})
    parts = (("reused", "same key", "#00798c"), ("new_key", "new key", "#edae49"), ("no_key", "no key", "#9a9a9a"))
    for i, (lab, kdir, _) in enumerate(have):
        c = key_adoption(os.path.join(RES, kdir, "traces.jsonl"))
        n = c["reused"] + c["new_key"] + c["no_key"]
        left = 0
        for key, name, color in parts:
            w = 100 * c[key] / n if n else 0
            a1.barh(i, w, left=left, color=color, label=name if i == 0 else None, height=0.6)
            if w >= 12:
                text = f"no key on {w:.0f}% of retries" if key == "no_key" else f"{w:.0f}%"
                a1.text(left + w / 2, i, text, ha="center", va="center", fontsize=7, color="white")
            left += w
        a1.text(102, i, f"keys on {100 * c['keyed'] / max(c['writes'], 1):.0f}%\nof writes",
                va="center", fontsize=6.5)
    a1.set_yticks(range(len(have)), [h[0] for h in have])
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.set_xlabel("Retries after a lost response (%)")
    a1.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    conds = (("baseline", "no key", "#d1495b"), ("key_available", "agent keys", "#edae49"),
             ("safe_retry", "harness keys", "#00798c"))
    for i, (lab, kdir, bdir) in enumerate(have):
        pool = rows(os.path.join(RES, bdir, "episodes.csv")) + (
            rows(os.path.join(RES, kdir, "episodes.csv")) if kdir != bdir else [])
        for j, (cond, name, color) in enumerate(conds):
            vals = [int(r["passed"]) for r in pool if r["condition"] == cond and float(r["fault_rate"]) == 0.2]
            if vals:
                p_, lo, hi = ci_bar(vals)
                a2.bar(i + 0.27 * (j - 1), p_, 0.25, yerr=[[lo], [hi]], capsize=2, error_kw={"lw": 0.8},
                       color=color, label=name if i == 0 else None)
    a2.set_xticks(range(len(have)), [h[0].replace(" ", "\n", 1) for h in have], fontsize=8)
    a2.set_ylim(0, 105)
    a2.set_ylabel("Tasks correct,\n20% faults (%)")
    a2.grid(axis="y", alpha=.3)
    a2.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, "fig_keys")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=None,
                   help="comma-separated results-folder suffixes (default: all known models); "
                        "use it to leave out runs that are still in progress")
    p.add_argument("--test", action="store_true",
                   help="draw from non-study data (scripted, gullible, pilot) to check the plumbing")
    a = p.parse_args()
    global OUT
    if a.test:
        # Map non-study folders onto the expected names in a temporary results tree.
        import tempfile
        tmp = tempfile.mkdtemp()
        global RES
        links = {"main-realistic-scripted": "scripted-bursty", "sec-full-scripted": "sec-gullible",
                 "sec-within-scripted": "sec-gullible-within"}
        os.makedirs(os.path.join(tmp, "tau2"))
        for new, old in links.items():
            if os.path.exists(os.path.join(RES, old)):
                os.symlink(os.path.join(RES, old), os.path.join(tmp, new))
        for c in ("baseline", "idempotent"):
            for r in ("0", "0.2"):
                src = os.path.join(RES, "tau2", f"pilot-sonnet-airline-{c}-p{r}")
                if os.path.exists(src):
                    os.symlink(src, os.path.join(tmp, "tau2", f"scripted-airline-{c}-p{r}"))
        RES, OUT = tmp, os.path.join(ROOT, "paper", "figures-test")
        MODELS["scripted"] = "scripted (test)"
        TAU2_NAMES["scripted"] = "scripted"
        models = ["scripted"]
    else:
        models = a.models.split(",") if a.models else list(MODELS) + ["opus"]
    fig_reliability(models)
    fig_passk(models)
    fig_tau2(models)
    fig_attacks(models, "full")
    fig_attacks(models, "within")
    fig_tradeoff(models)
    fig_keys()


if __name__ == "__main__":
    main()
