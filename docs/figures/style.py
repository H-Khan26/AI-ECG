"""Shared palette, ECG-paper rendering and PTB-XL loading for the README figures.

Every figure is rendered twice, once per GitHub theme, and saved as
``<name>-light.<ext>`` and ``<name>-dark.<ext>`` under ``docs/assets/``.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.signal import decimate

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
CACHE = Path(__file__).resolve().parent / ".cache"

FS = 250                     # v1 input contract: 10 s at 250 Hz
N_SAMPLES = 2500
LEADS_8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# Interval colours are validated all-pairs for CVD and normal vision on both
# surfaces. Rhythm/RR annotations use ink, never a fourth hue.
THEMES = {
    "light": {
        "surface": "#ffffff",
        "card": "#f6f8fa",
        "border": "#d0d7de",
        "ink": "#1f2328",
        "ink2": "#59636e",
        "muted": "#818b98",
        "trace": "#1f2328",
        "grid_minor": (0.89, 0.29, 0.28, 0.10),
        "grid_major": (0.89, 0.29, 0.28, 0.26),
        "pr": "#2a78d6",
        "qrs": "#eb6834",
        "qt": "#1baf7a",
        "accent": "#e34948",
    },
    "dark": {
        "surface": "#0d1117",
        "card": "#161b22",
        "border": "#30363d",
        "ink": "#e6edf3",
        "ink2": "#9198a1",
        "muted": "#6e7681",
        "trace": "#e6edf3",
        "grid_minor": (0.90, 0.40, 0.40, 0.09),
        "grid_major": (0.90, 0.40, 0.40, 0.22),
        "pr": "#3987e5",
        "qrs": "#d95926",
        "qt": "#199e70",
        "accent": "#e66767",
    },
}

SANS = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
MONO = ["Consolas", "Menlo", "DejaVu Sans Mono"]

# Clean, human-validated PTB-XL records (no noise/drift flags) per pattern.
# Several candidates each; figure scripts pick the cleanest and pin it.
CANDIDATES = {
    "NORM": [3, 7, 10],
    "AFIB": [4117, 8215, 12542],
    "PVC": [219, 628],
    "CLBBB": [286, 338, 618],
    "1AVB": [218, 209],
    "LNGQT": [320, 6310],
}


def apply_theme(theme: str) -> dict:
    """Set matplotlib rcParams for a theme and return its colour dict."""
    c = THEMES[theme]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "font.monospace": MONO,
        "font.size": 10,
        "text.color": c["ink"],
        "axes.labelcolor": c["ink2"],
        "axes.edgecolor": c["border"],
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "figure.facecolor": c["surface"],
        "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        "svg.fonttype": "none",
    })
    return c


def ecg_paper(ax, c: dict, t_max: float, y_min: float, y_max: float, labels: bool = False):
    """Draw standard ECG paper: 0.04 s / 0.1 mV minor, 0.2 s / 0.5 mV major, square boxes."""
    ax.set_xlim(0, t_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(which="minor", color=c["grid_minor"], linewidth=0.5)
    ax.grid(which="major", color=c["grid_major"], linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_aspect(0.4)
    ax.tick_params(which="both", length=0)
    if not labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for s in ax.spines.values():
        s.set_visible(False)


def load_record(ecg_id: int, leads=LEADS_8) -> dict:
    """Stream one PTB-XL 500 Hz record from PhysioNet, cache it, and return
    {lead: mV array of 2500 samples} decimated to 250 Hz like the v1 pipeline."""
    import wfdb

    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{ecg_id:05d}.npz"
    if cached.exists():
        data = np.load(cached)
        return {k: data[k] for k in leads}

    folder = f"{(ecg_id // 1000) * 1000:05d}"
    rec = wfdb.rdrecord(f"{ecg_id:05d}_hr", pn_dir=f"ptb-xl/1.0.3/records500/{folder}")
    sig = {name: rec.p_signal[:, i] for i, name in enumerate(rec.sig_name)}
    out = {name: decimate(x, 2, ftype="fir") for name, x in sig.items()}
    np.savez_compressed(cached, **out)
    return {k: out[k] for k in leads}


def save(fig, name: str, theme: str, ext: str = "svg", **kw):
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"{name}-{theme}.{ext}"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15, **kw)
    plt.close(fig)
    return path
