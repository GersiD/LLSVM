"""Headline figures: classical classifiers on small-LM embeddings vs. a zero-shot Claude baseline.

Writes to results/plots/, each with a -dark variant for GitHub's dark mode:
    pareto.png     accuracy vs. time per text per dataset, with the Pareto frontier (the README's chart)
    accuracy.png   test accuracy, datasets on the X axis, one bar per method (the full breakdown)

The code lives in results/figures/ (data.py loads results, style.py holds the gruvbox look,
bars.py and pareto.py draw). The caveats live in the README, not in the figures.

    uv run results/plot_summary.py
"""

from pathlib import Path

from figures import bars, pareto, style
from figures.data import load

PLOTS = Path(__file__).resolve().parent / "plots"
FIGURES = {"pareto": pareto.pareto, "accuracy": bars.accuracy}


def main() -> None:
    summary = load()
    PLOTS.mkdir(exist_ok=True)
    for theme in style.THEMES:
        for name, draw in FIGURES.items():
            path = PLOTS / f"{name}{style.THEMES[theme]['suffix']}.png"
            draw(summary, theme, path)
            print(f"wrote {path.relative_to(PLOTS.parent.parent)}")


if __name__ == "__main__":
    main()
