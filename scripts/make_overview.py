"""Figure 1: where faults, fixes and defenses sit between an agent and its tools.

  .venv/bin/python scripts/make_overview.py   ->  paper/figures/fig_overview.{pdf,png}
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import FancyBboxPatch   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
INK, FAULT, FIX, DEF, ATK = "#222222", "#d1495b", "#00798c", "#30638e", "#edae49"


def box(ax, x, y, w, h, title, lines, color, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=8.5, weight="bold", color=color)
    for i, line in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.48 - 0.27 * i, line, ha="left", va="top", fontsize=6.6, color=INK)


def arrow(ax, x1, y1, x2, y2, label, color=INK, dy=0.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, shrinkA=0, shrinkB=0))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", va="bottom", fontsize=6.4, color=color)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.35)
    ax.axis("off")

    box(ax, 0.1, 1.6, 2.3, 2.2, "Agent (LLM)",
        ["reads tickets, decides,", "calls tools", "", "Claude (Sonnet, Haiku,", "Opus); Gemma 4, Qwen"], INK)

    box(ax, 3.2, 0.25, 4.9, 4.95, "Harness", [], INK, fill="#fafafa")
    box(ax, 3.45, 3.05, 4.4, 1.55, "Fault injection",
        ["transient error  (nothing happened)",
         "lost response  (write applied, reply lost)",
         "stale read  (replica one write behind)",
         "uniform or bursty (Gilbert-Elliott)"], FAULT)
    box(ax, 3.45, 1.6, 4.4, 1.25, "Recovery on tool calls",
        ["retry  |  retry + idempotency keys",
         "policy gate  |  least privilege (task scope)",
         "agent-supplied keys (key_available)"], FIX)
    box(ax, 3.45, 0.45, 4.4, 0.95, "Defenses on tool output",
        ["spotlighting  |  classifier, LLM judge, NeMo",
         "(withhold record or redact passage)"], DEF)

    box(ax, 9.0, 1.6, 2.9, 2.2, "Tools and store",
        ["6 reads (+2 optional),", "5 writes (refunds, emails...)", "", "graded on final state;", "also tau2 airline, retail"], INK)

    ax.add_patch(FancyBboxPatch((9.0, 4.15), 2.9, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#fff6e5", ec=ATK, lw=1.4))
    ax.text(10.45, 4.55, "Attacker text in the data\n(ticket body, account note)", ha="center", va="center",
            fontsize=6.6, color=INK)
    arrow(ax, 10.45, 4.15, 10.45, 3.8, "", ATK)

    arrow(ax, 2.4, 3.2, 3.2, 3.2, "tool call")
    arrow(ax, 8.1, 3.2, 9.0, 3.2, "")
    arrow(ax, 9.0, 2.0, 8.1, 2.0, "result")
    arrow(ax, 3.2, 2.0, 2.4, 2.0, "")
    ax.text(2.8, 1.62, "result /\nerror", ha="center", va="top", fontsize=6.4, color=INK)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_overview.{ext}"), bbox_inches="tight", dpi=200)
    print("wrote fig_overview")


if __name__ == "__main__":
    main()
