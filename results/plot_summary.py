"""Headline figure: classical classifiers on small-LM embeddings vs. a zero-shot Claude baseline.

Datasets are on the X axis. For each one the figure shows, top to bottom, accuracy of the
best classifier per embedder next to Claude, per-text latency (log scale) with the speedup
over Claude, and Claude's (notional) cost per 1M texts. Everything is discovered from the
files present, so adding a dataset, embedder or Claude run needs no code change.

Reads results/<dataset>_<embedder>.{csv,json} (experiment.py) and
results/claude/<dataset>_<model>_claude-code_predictions.csv (experiment-claude.py).
Claude numbers come only from the predictions file; a row with an empty prediction is
unanswered and is excluded (the figure then marks the run as partial).

    uv run results/plot_summary.py

Writes results/plots/summary.png (light) and results/plots/summary-dark.png (dark).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent
PLOTS = RESULTS / "plots"

# Same family names as plot.py: rows of the CSV that are classifiers (not embedding/baseline).
CLASSIFIER_FAMILIES = {"k-means", "logistic regression", "SVM", "gradient boosting"}
EMBEDDER_ORDER = ["minilm", "gemma2"]  # fixed colour slots; unknown embedders follow, sorted
EMBEDDER_NAMES = {"minilm": "MiniLM", "gemma2": "Gemma 2"}
TIE = 0.01  # accuracy gap (absolute) below which the headline calls it a tie (about noise level here)
CLAUDE_CODE_COST_FACTOR_NOTE = "roughly 2x a direct API call"

# Light uses plot.py's palette. Dark steps come from the dataviz palette (same hues, dark surface).
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], claude="#52514e"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#97968d", grid="#333330",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"], claude="#6f6e67"),
}


@dataclass
class Bar:
    key: str  # "minilm", "gemma2" or "claude"
    label: str  # two-line x tick label
    accuracy: float
    latency_ms: float
    cost_per_1m: float | None  # None = no per-call cost
    partial: bool = False


@dataclass
class Group:
    title: str
    n_test: int
    majority: float
    bars: list[Bar]
    claude_note: str | None = None  # footnote text when the Claude run is partial


def sig2(x: float) -> str:
    """Two significant figures for speedups: 320, 52, 7.5."""
    if x >= 100:
        return f"{round(x, -1):,.0f}"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def fmt_ms(ms: float) -> str:
    return f"{ms:,.0f} ms" if ms >= 100 else f"{ms:.1f} ms"


def claude_name(stem: str) -> str:
    m = re.search(r"claude-(\w+?)-(\d+)-(\d+)", stem)
    return f"Claude {m[1].title()} {m[2]}.{m[3]}" if m else "Claude"


def load_claude(dataset: str) -> tuple[str, dict] | None:
    """Zero-shot Claude stats from the predictions file, answered texts only."""
    files = sorted((RESULTS / "claude").glob(f"{dataset}_claude-*_predictions.csv"))
    if not files:
        return None
    preds = pd.read_csv(files[0])
    answered = preds[preds["pred"].notna()]
    if answered.empty:
        return None
    correct = answered["pred"].astype(str).str.strip().str.lower() == answered["label"].astype(str).str.strip().str.lower()
    return claude_name(files[0].name), dict(
        accuracy=float(correct.mean()),
        latency_ms=float(answered["latency_ms"].median()),
        cost_per_1m=float(answered["cost_usd"].mean() * 1e6),
        n_answered=len(answered),
        n_total=len(preds),
    )


def load_groups() -> tuple[list[Group], list[str], str, int]:
    """Returns groups, embedder keys (colour order), Claude's display name, classifiers per embedder."""
    metas: dict[str, dict[str, dict]] = {}
    for path in sorted(RESULTS.glob("*.json")):
        meta = json.loads(path.read_text())
        if "embedder" in meta and "dataset" in meta:
            metas.setdefault(meta["dataset"], {})[meta["embedder"]] = meta | {"stem": path.stem}
    all_embedders = {e for d in metas.values() for e in d}
    embedders = [e for e in EMBEDDER_ORDER if e in all_embedders] + sorted(all_embedders - set(EMBEDDER_ORDER))

    groups, n_clf, cname = [], 0, "Claude"
    for dataset, by_emb in metas.items():
        bars, majority, title, n_test = [], 0.0, dataset, 0
        for emb in embedders:
            if emb not in by_emb:
                continue
            meta = by_emb[emb]
            df = pd.read_csv(RESULTS / f"{meta['stem']}.csv", index_col="classifier")
            clfs = df[df["family"].isin(CLASSIFIER_FAMILIES)]
            n_clf = max(n_clf, len(clfs))
            best = clfs["accuracy"].idxmax()  # selection on the test split
            embed_ms = df.loc["embedding", "latency_ms_mean"]
            name = EMBEDDER_NAMES.get(emb, emb.title())
            bars.append(Bar(emb, f"{name} +\n{best}", float(clfs.loc[best, "accuracy"]),
                            float(embed_ms + clfs.loc[best, "latency_ms_mean"]), None))
            majority, title, n_test = float(df.loc["majority class", "accuracy"]), meta["dataset_title"], meta["n_test"]
        note = None
        claude = load_claude(dataset)
        if claude:
            cname, c = claude
            partial = c["n_answered"] < c["n_total"]
            bars.append(Bar("claude", cname.replace("Claude ", "Claude\n"), c["accuracy"], c["latency_ms"],
                            c["cost_per_1m"], partial))
            if partial:
                note = (f"* Claude on {title}: {c['n_answered']:,} of {c['n_total']:,} test texts answered; "
                        f"accuracy, latency and cost use the answered texts only.")
        groups.append(Group(title, n_test, majority, bars, note))
    return groups, embedders, cname, n_clf


def headline(groups: list[Group], cname: str) -> tuple[str, str]:
    speedups, verdicts = [], []
    for g in groups:
        claude = next((b for b in g.bars if b.key == "claude"), None)
        if not claude:
            continue
        local = [b for b in g.bars if b.key != "claude"]
        speedups += [claude.latency_ms / b.latency_ms for b in local]
        gap = max(b.accuracy for b in local) - claude.accuracy
        name = g.title.removesuffix(" dataset").removesuffix(" Collection")
        if abs(gap) < TIE:
            verdicts.append(f"tie on {name}")
        else:
            verdicts.append(f"{'+' if gap > 0 else '−'}{abs(gap) * 100:.0f} points on {name}")
    first = f"Trained classifiers on small-LM embeddings vs. zero-shot {cname}"
    if not verdicts:
        return first, ""
    lo, hi = min(speedups), max(speedups)
    return first, f"{', '.join(verdicts)}, at {sig2(lo)}–{sig2(hi)}× lower latency per text"


def render(groups: list[Group], embedders: list[str], cname: str, n_clf: int, theme: str, path: Path) -> None:
    t = THEMES[theme]
    color = {e: t["series"][i % len(t["series"])] for i, e in enumerate(embedders)} | {"claude": t["claude"]}

    # x layout: bars at consecutive integers inside a group, a wider gap between groups.
    xs, x0, spans = [], 0.0, []
    for g in groups:
        pos = [x0 + i for i in range(len(g.bars))]
        xs.append(pos)
        spans.append((pos[0], pos[-1]))
        x0 = pos[-1] + 2.3
    xlim = (spans[0][0] - 0.8, spans[-1][1] + 1.75)

    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["hatch.linewidth"] = 1.2
    W, H = 10.0, 11.0  # inches; vertical layout below is in inches from the top
    fig = plt.figure(figsize=(W, H), facecolor=t["surface"])
    left, right = 0.075, 0.975
    def rect(top: float, height: float) -> tuple[float, float, float, float]:
        return (left, 1 - (top + height) / H, right - left, height / H)

    ax_acc = fig.add_axes(rect(2.9, 2.6))
    ax_lat = fig.add_axes(rect(6.65, 1.5), sharex=ax_acc)
    ax_cost = fig.add_axes(rect(8.8, 0.55), sharex=ax_acc)
    for ax in (ax_acc, ax_lat, ax_cost):
        ax.set_facecolor(t["surface"])
        ax.set_xlim(*xlim)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    for ax, title, dy in ((ax_acc, "Accuracy on the test split  ·  higher is better", 0.95),
                          (ax_lat, "Time per text, log scale  ·  lower is better", 0.22),
                          (ax_cost, "Cost of 1M texts", 0.12)):
        pos = ax.get_position()
        fig.text(0.03, pos.y1 + dy / H, title, ha="left", va="bottom", fontsize=11.5, fontweight="bold", color=t["ink"])

    # ---- accuracy
    ax = ax_acc
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"], color=t["muted"], fontsize=10)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    tick_x, tick_l = [], []
    for g, pos in zip(groups, xs):
        for b, x in zip(g.bars, pos):
            c = color[b.key]
            if b.partial:  # hollow + hatched = partial run
                ax.bar(x, b.accuracy, 0.74, facecolor=t["surface"], edgecolor=c, linewidth=1.6, hatch="////", zorder=2)
            else:
                ax.bar(x, b.accuracy, 0.74, color=c, zorder=2)
            ax.text(x, b.accuracy + 0.015, f"{b.accuracy:.1%}" + ("*" if b.partial else ""), ha="center", va="bottom",
                    fontsize=12.5, fontweight="bold", color=t["ink"])
            tick_x.append(x)
            tick_l.append(b.label)
        a, z = pos[0], pos[-1]
        ax.plot([a - 0.5, z + 0.5], [g.majority] * 2, color=t["ink"], linewidth=1.8, zorder=4, solid_capstyle="butt")
        ax.text(z + 0.62, g.majority, f"majority\nclass {g.majority:.0%}", ha="left", va="center", fontsize=10,
                color=t["ink2"], linespacing=1.15)
        # dataset header over the group
        ax.text((a + z) / 2, 1.17, g.title, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=t["ink"])
        ax.text((a + z) / 2, 1.115, f"{g.n_test:,} test texts", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=10, color=t["muted"])
    ax.set_xticks(tick_x, tick_l, fontsize=10.5, color=t["ink2"], linespacing=1.2)
    ax.tick_params(axis="x", pad=6)

    # ---- latency (lollipops on a log axis: a bar would imply a zero baseline)
    ax = ax_lat
    ax.set_yscale("log")
    ax.set_ylim(1, 2.5e4)
    ax.set_yticks([1, 10, 100, 1000, 10000], ["1 ms", "10", "100", "1,000", "10,000"], color=t["muted"], fontsize=10)
    ax.minorticks_off()
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), visible=False)
    for g, pos in zip(groups, xs):
        claude = next((b for b in g.bars if b.key == "claude"), None)
        for b, x in zip(g.bars, pos):
            c = color[b.key]
            ax.plot([x, x], [1, b.latency_ms], color=c, linewidth=2.2, zorder=2, solid_capstyle="butt")
            ax.scatter([x], [b.latency_ms], s=110, facecolor=t["surface"] if b.partial else c, edgecolor=c,
                       linewidth=2.2 if b.partial else 2, zorder=3)
            if b.key == "claude" or not claude:
                ax.annotate(fmt_ms(b.latency_ms) + ("*" if b.partial else ""), (x, b.latency_ms), xytext=(0, 11),
                            textcoords="offset points", ha="center", va="bottom", fontsize=12.5,
                            fontweight="bold", color=t["ink"])
            else:
                ax.annotate(f"{sig2(claude.latency_ms / b.latency_ms)}× faster", (x, b.latency_ms), xytext=(0, 11),
                            textcoords="offset points", ha="center", va="bottom", fontsize=12.5,
                            fontweight="bold", color=t["ink"])
                ax.annotate(fmt_ms(b.latency_ms), (x, b.latency_ms), xytext=(0, 29),
                            textcoords="offset points", ha="center", va="bottom", fontsize=10.5, color=t["ink2"])

    # ---- cost (text only: the local pipeline has no per-call price to plot)
    ax = ax_cost
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    plt.setp(ax.get_xticklabels(), visible=False)
    for g, pos in zip(groups, xs):
        for b, x in zip(g.bars, pos):
            if b.cost_per_1m is not None:
                ax.text(x, 0.62, f"${b.cost_per_1m:,.0f}" + ("*" if b.partial else ""), ha="center", va="center", fontsize=13,
                        fontweight="bold", color=t["ink"])
                ax.text(x, 0.05, "Claude Code est.", ha="center", va="center", fontsize=9.5, color=t["muted"])
            else:  # no per-call price: show the laptop compute time for 1M texts, one at a time
                ax.text(x, 0.62, f"{b.latency_ms * 1e6 / 3.6e6:.1f} h", ha="center", va="center", fontsize=13,
                        fontweight="bold", color=t["ink"])
                ax.text(x, 0.05, "laptop, no API", ha="center", va="center", fontsize=9.5, color=t["muted"])
    ax.axhline(1, color=t["grid"], linewidth=0.8)

    # ---- title, legend, footnotes
    first, second = headline(groups, cname)
    fig.text(0.03, 1 - 0.25 / H, first, ha="left", va="top", fontsize=16.5, fontweight="bold", color=t["ink"])
    fig.text(0.03, 1 - 0.62 / H, second, ha="left", va="top", fontsize=16.5, fontweight="bold", color=t["ink2"])
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10, color=color[e],
                          label=f"{EMBEDDER_NAMES.get(e, e.title())} embeddings + best classifier") for e in embedders]
    handles.append(plt.Line2D([], [], marker="s", linestyle="", markersize=10, color=t["claude"], label=f"{cname}, zero-shot"))
    handles.append(plt.Line2D([], [], color=t["ink"], linewidth=1.8, label="majority-class baseline"))
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=10.5, labelcolor=t["ink2"],
               bbox_to_anchor=(0.022, 1 - 1.0 / H), columnspacing=2.0, handletextpad=0.5, labelspacing=0.5)

    notes = [
        "Classifiers are trained on each dataset's labeled train split; Claude gets only the label names (low effort).",
        f"Best classifier per embedder = highest test accuracy among {n_clf} classifiers (selected on the test split itself).",
        "Local time = embedding + classifier, mean per text, on an Apple M2 Pro laptop (embedder picks its device).",
        f"Claude time = median model time per request reported by Claude Code (excludes ~1.5 s CLI startup).",
        f"Claude cost = mean cost_usd x 1M texts: a Claude Code estimate, {CLAUDE_CODE_COST_FACTOR_NOTE}. Local = time per text x 1M, one at a time, training excluded.",
    ] + [g.claude_note for g in groups if g.claude_note]
    fig.text(0.03, 1 - 9.6 / H, "\n".join(notes), ha="left", va="top", fontsize=9.5, color=t["ink2"], linespacing=1.5)
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    groups, embedders, cname, n_clf = load_groups()
    PLOTS.mkdir(exist_ok=True)
    for theme, name in (("light", "summary.png"), ("dark", "summary-dark.png")):
        render(groups, embedders, cname, n_clf, theme, PLOTS / name)
        print(f"wrote {(PLOTS / name).relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
