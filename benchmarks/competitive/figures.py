"""Static figures for the Phase D write-up (requires matplotlib).

Colour follows the system, never its rank: a system keeps its colour in every
figure. Budget-exempt controls are drawn in neutral ink as reference lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {
    "evicortex": ("EviCortex", "#2a78d6", "o"),
    "hybrid_rag": ("Hybrid RAG", "#eb6834", "s"),
    "dense_rag": ("Dense RAG", "#1baf7a", "^"),
    "bm25_chunks": ("BM25 chunks", "#eda100", "D"),
    "mem0": ("Mem0", "#e87ba4", "v"),
    "evicortex_claims": ("EviCortex + claims", "#6250d6", "P"),
}
CONTROLS = {
    "oracle": ("Oracle evidence", (0, (1, 1))),
    "full_history": ("Full history", (0, (5, 2))),
    "no_memory": ("No memory", (0, (2, 3))),
}


def _slug(name: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in name.lower()).strip("-").split("--")[0]


def _style(axis: Any, *, ylabel: str, title: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    axis.set_ylabel(ylabel, color=SECONDARY, fontsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)


def _budget_axis(axis: Any, budgets: list[int]) -> None:
    axis.set_xscale("log")
    axis.set_xticks(budgets)
    axis.set_xticklabels([f"{budget:,}" for budget in budgets])
    axis.minorticks_off()
    axis.set_xlabel("Evidence budget (estimated tokens)", color=SECONDARY, fontsize=9)
    axis.set_xlim(budgets[0] * 0.85, budgets[-1] * 1.9)


def _lines(axis: Any, series: dict[str, dict[int, float]], *, percent: bool = True) -> None:
    ends: list[tuple[float, str, str]] = []
    for system, (label, colour, marker) in SERIES.items():
        points = series.get(system)
        if not points:
            continue
        budgets = sorted(points)
        values = [points[budget] * (100 if percent else 1) for budget in budgets]
        axis.plot(budgets, values, color=colour, linewidth=2, marker=marker, markersize=7,
                  markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ends.append((values[-1], f"{label}  {values[-1]:.0f}%", colour))
    # Direct labels at the right end, nudged apart so they never overlap.
    ends.sort()
    low, high = axis.get_ylim()
    gap = (high - low) * 0.045
    placed: list[float] = []
    for value, text, colour in ends:
        position = value if not placed else max(value, placed[-1] + gap)
        placed.append(position)
        axis.annotate(text, xy=(1.0, position), xycoords=("axes fraction", "data"),
                      xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                      fontsize=8.5, color=INK,
                      bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0})
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=6, frameon=False,
                fontsize=8.5, labelcolor=SECONDARY, handlelength=1.6, columnspacing=1.2)


def write_figures(payload: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    for name, run in payload["runs"].items():
        slug = "locomo" if "locomo" in name.lower() else "longmemeval"
        answers = run["answers"]["cells"]
        if answers:
            accuracy: dict[str, dict[int, float]] = {}
            controls: dict[str, float] = {}
            for cell in answers:
                if cell["budget"] is None:
                    controls[cell["system"]] = cell["accuracy"]["mean"]
                else:
                    accuracy.setdefault(cell["system"], {})[cell["budget"]] = cell["accuracy"]["mean"]
            budgets = sorted({budget for points in accuracy.values() for budget in points})
            figure, axis = plt.subplots(figsize=(8.6, 5.2), facecolor=SURFACE)
            _style(axis, ylabel="Answers judged correct (%)",
                   title=f"{name}: answer accuracy by evidence budget")
            axis.set_ylim(0, 100)
            _budget_axis(axis, budgets)
            for system, (label, dashes) in CONTROLS.items():
                if system in controls:
                    value = controls[system] * 100
                    axis.axhline(value, color=MUTED, linewidth=1.2, linestyle=dashes, zorder=2)
                    axis.annotate(f"{label}  {value:.0f}%", xy=(0.0, value),
                                  xycoords=("axes fraction", "data"), xytext=(4, 4),
                                  textcoords="offset points", fontsize=8.5, color=SECONDARY)
            _lines(axis, accuracy)
            figure.text(0.01, 0.01,
                        f"n = {run['answers']['questions']} questions; reader {run['answers']['reader']}; "
                        "grey lines are controls that ignore the budget.",
                        fontsize=7.5, color=MUTED)
            figure.tight_layout(rect=(0, 0.03, 1, 1))
            figure.savefig(out / f"{slug}-accuracy-by-budget.png", dpi=160, facecolor=SURFACE)
            plt.close(figure)

            top = max(budgets)
            categories = sorted({category for cell in answers for category in cell["accuracy_by_category"]})
            figure, axis = plt.subplots(figsize=(8.6, 0.62 * len(categories) + 2.2), facecolor=SURFACE)
            _style(axis, ylabel="", title=f"{name}: accuracy by question type at {top:,} tokens")
            axis.grid(axis="y", visible=False)
            axis.grid(axis="x", color=GRID, linewidth=0.8)
            for row, category in enumerate(categories):
                axis.plot([0, 100], [row, row], color=GRID, linewidth=0.8, zorder=1)
                for cell in answers:
                    if cell["budget"] != top or cell["system"] not in SERIES:
                        continue
                    entry = cell["accuracy_by_category"].get(category)
                    if not entry or entry["mean"] is None:
                        continue
                    label, colour, marker = SERIES[cell["system"]]
                    axis.scatter(entry["mean"] * 100, row, color=colour, marker=marker, s=70,
                                 edgecolors=SURFACE, linewidths=1.5, zorder=3,
                                 label=label if row == 0 else None)
            counts = run["answers"]["category_counts"]
            axis.set_yticks(range(len(categories)))
            axis.set_yticklabels([f"{category}  (n={counts.get(category, 0)})" for category in categories],
                                 color=INK)
            axis.set_xlim(0, 100)
            axis.set_xlabel("Answers judged correct (%)", color=SECONDARY, fontsize=9)
            axis.invert_yaxis()
            axis.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=6, frameon=False,
                        fontsize=8.5, labelcolor=SECONDARY, columnspacing=1.2)
            figure.tight_layout()
            figure.savefig(out / f"{slug}-accuracy-by-type.png", dpi=160, facecolor=SURFACE)
            plt.close(figure)

        metric = "unit_recall" if slug == "locomo" else "session_recall"
        recall: dict[str, dict[int, float]] = {}
        for cell in run["retrieval"]["cells"]:
            if cell["budget"] is not None and cell[metric]["mean"] is not None:
                recall.setdefault(cell["system"], {})[cell["budget"]] = cell[metric]["mean"]
        if recall:
            budgets = sorted({budget for points in recall.values() for budget in points})
            figure, axis = plt.subplots(figsize=(8.6, 5.0), facecolor=SURFACE)
            unit = "turns" if slug == "locomo" else "sessions"
            _style(axis, ylabel=f"Gold evidence {unit} in packed context (%)",
                   title=f"{name}: evidence recall by budget")
            axis.set_ylim(0, 100)
            _budget_axis(axis, budgets)
            _lines(axis, recall)
            figure.text(0.01, 0.01,
                        f"n = {run['retrieval']['questions']} questions (answerable ones scored); "
                        "measured after hard-budget packing.",
                        fontsize=7.5, color=MUTED)
            figure.tight_layout(rect=(0, 0.03, 1, 1))
            figure.savefig(out / f"{slug}-evidence-recall.png", dpi=160, facecolor=SURFACE)
            plt.close(figure)
