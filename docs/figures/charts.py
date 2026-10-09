"""README chart: v1 pilot results as a dot-and-whisker plot of ROC-AUC.

Run from the repo root:  .venv/Scripts/python.exe docs/figures/charts.py
Writes docs/assets/pilot-results-{light,dark}.svg
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

# Stratified 5-fold CV ROC-AUC, mean and SD across folds (archived v1 study).
# Sorted best first; None marks a model that was not completed.
RESULTS = [
    ("Augmented two-tower", "Waveform + tabular", 0.66, 0.043),
    ("Two-tower 1-D CNN + MLP", "Waveform + tabular", 0.64, 0.063),
    ("1-D CNN", "Waveform", 0.63, 0.06),
    ("Random Forest, 200 trees", "Tabular", 0.61, 0.04),
    ("1-D CNN on 2 s windows", "Waveform", None, None),
]
X_MIN, X_MAX = 0.45, 0.80


def pilot_results(theme):
    c = style.apply_theme(theme)
    W, row = 8.8, 0.56
    left, right, top, bottom = 2.25, 0.25, 0.62, 0.62
    H = top + row * len(RESULTS) + bottom

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, (W - left - right) / W, row * len(RESULTS) / H])
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(len(RESULTS) - 0.5, -0.5)

    # Recessive hairline grid and a dashed chance reference
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="x", color=c["border"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=6, labelsize=8.5, labelcolor=c["muted"])
    for lab in ax.get_xticklabels():
        lab.set_family("monospace")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(0.5, color=c["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.5, -0.5, "chance", ha="center", va="bottom", fontsize=8.5, color=c["muted"],
            transform=ax.transData, clip_on=False)
    ax.set_xlabel("ROC-AUC, mean ± SD across folds", fontsize=9, color=c["ink2"], labelpad=8)

    for i, (name, tag, mean, sd) in enumerate(RESULTS):
        # Row label and input-type tag in the left margin
        ax.text(-0.02, i - 0.08, name, transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=9.5, color=c["ink"])
        ax.text(-0.02, i + 0.14, tag, transform=ax.get_yaxis_transform(), ha="right",
                va="top", fontsize=8.5, color=c["muted"],
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.5", fc="none", ec=c["border"], lw=0.7))
        if mean is None:
            ax.text(0.506, i, "not completed", ha="left", va="center", fontsize=8.5,
                    style="italic", color=c["muted"])
            continue
        color = c["pr"] if i == 0 else c["ink2"]
        ax.plot([mean - sd, mean + sd], [i, i], color=color, lw=1.5, solid_capstyle="round", zorder=2)
        ax.plot(mean, i, "o", ms=8, color=color, mec=c["surface"], mew=1.5, zorder=3)
        ax.text(mean + sd + 0.007, i, f"{mean:.2f} ± {sd:.2f}", ha="left", va="center",
                fontsize=9, family="monospace", color=c["ink"])

    fig.text(0.02 / W, (H - 0.12) / H, "Stratified 5-fold CV · 86 CTRCD vs 632 control ECGs",
             ha="left", va="top", fontsize=9.5, color=c["ink2"])
    return fig


if __name__ == "__main__":
    for theme in ("light", "dark"):
        print(style.save(pilot_results(theme), "pilot-results", theme))
