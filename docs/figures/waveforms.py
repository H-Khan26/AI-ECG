"""README waveform figures: the v1 8-lead model input and a rhythm-pattern gallery.

Run from the repo root:  .venv/Scripts/python.exe docs/figures/waveforms.py
Writes docs/assets/{ecg-8lead,rhythm-gallery}-{light,dark}.svg
"""

import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from style import FS, LEADS_8, N_SAMPLES  # noqa: E402

warnings.filterwarnings("ignore", module="neurokit2")

NORM_ID = 10          # cleanest of CANDIDATES["NORM"]: stable baseline, clear P and T waves
W = 8.8               # figure width in inches (README column ~880 px)


def inch_axes(fig, H, x, y, w, h):
    """Add axes positioned in inches from the figure's bottom-left corner."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


# --------------------------------------------------------------------------
# Figure 1: the exact v1 input, 8 leads x 2500 samples on ECG paper
# --------------------------------------------------------------------------

def ecg_8lead(theme):
    c = style.apply_theme(theme)
    rec = style.load_record(NORM_ID)

    pitch = 2.0                       # mV between lead baselines
    t_lo, t_hi = -0.4, 10.0           # 0.4 s of paper before t = 0 for the calibration pulse
    y_lo, y_hi = -pitch * (len(LEADS_8) - 1) - 1.0, 1.5
    gutter, paper_w = 0.5, W - 0.6
    k = paper_w / (t_hi - t_lo)       # inches per second; 0.5 mV spans the same as 0.2 s
    paper_h = k * 0.4 * (y_hi - y_lo)
    top, bottom = 0.34, 0.5
    H = top + paper_h + bottom

    fig = plt.figure(figsize=(W, H))
    ax = inch_axes(fig, H, gutter, bottom, paper_w, paper_h)
    style.ecg_paper(ax, c, t_hi, y_lo, y_hi)
    ax.set_xlim(t_lo, t_hi)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(c["border"])
        s.set_linewidth(0.8)

    t = np.arange(N_SAMPLES) / FS
    for i, lead in enumerate(LEADS_8):
        base = -pitch * i
        x = rec[lead] - np.median(rec[lead])          # display offset only
        ax.plot(t, x + base, color=c["trace"], lw=0.65, solid_joinstyle="round")
        # 1 mV x 0.2 s calibration pulse
        cal_t = [-0.36, -0.30, -0.30, -0.10, -0.10, -0.04]
        cal_y = [0, 0, 1, 1, 0, 0]
        ax.plot(cal_t, np.add(cal_y, base), color=c["trace"], lw=0.65)
        ax.text(t_lo - 0.08, base, lead, ha="right", va="center", fontsize=9,
                color=c["ink"], fontweight="semibold", clip_on=False)

    for s in range(0, 11, 2):
        ax.text(s, y_lo - 0.25, f"{s} s", ha="center", va="top", fontsize=8.5,
                family="monospace", color=c["muted"], clip_on=False)

    fig.text(gutter / W, (H - 0.12) / H,
             "v1 input · 8 leads × 2500 samples · 250 Hz · 10 s",
             ha="left", va="top", family="monospace", fontsize=8.5, color=c["ink"])
    fig.text((gutter + paper_w) / W, (H - 0.12) / H,
             f"PTB-XL record #{NORM_ID} (CC BY 4.0)",
             ha="right", va="top", family="monospace", fontsize=8.5, color=c["ink2"])
    fig.text(gutter / W, 0.08 / H,
             "Grid 0.04 s × 0.1 mV small, 0.2 s × 0.5 mV large · calibration pulse 1 mV",
             ha="left", va="bottom", family="monospace", fontsize=8.5, color=c["muted"])
    return fig


# --------------------------------------------------------------------------
# Figure 2: rhythm gallery
# --------------------------------------------------------------------------

STRIP_S = 4.5         # seconds per strip

# One real PTB-XL record per pattern, chosen after rendering every candidate.
# `beat` is the approximate R-peak sample (250 Hz) of the annotated beat.
# Interval boundaries come from NeuroKit2 DWT delineation of the shown lead.
# At 250 Hz DWT often puts QRS onset on the P wave (or on the LBBB notch) and
# misses PVC boundaries, so those fiducials were read off the trace and are
# pinned in `fix` (sample index). Every value is asserted against its label.
PANELS = [
    dict(key="NORM", title="Normal sinus rhythm",
         sig="Regular RR, P before every QRS, PR 120–200 ms, QRS < 120 ms",
         ecg_id=NORM_ID, lead="II", t0=5.3, y=(-0.5, 1.5),
         beat=1661, fix=dict(qrs_on=1655, qrs_off=1670), show=("PR", "QRS", "QT")),
    dict(key="1AVB", title="First-degree AV block",
         sig="PR > 200 ms",
         ecg_id=209, lead="II", t0=5.2, y=(-0.5, 1.5),
         beat=1589, fix=dict(qrs_on=1581), show=("PR",)),
    dict(key="CLBBB", title="Left bundle branch block",
         sig="QRS ≥ 120 ms, broad notched R in I/V5–V6",
         ecg_id=338, lead="I", t0=3.8, y=(-0.5, 1.5),
         beat=1106, fix=dict(qrs_on=1081, qrs_off=1116), show=("QRS",)),
    dict(key="PVC", title="Premature ventricular complex",
         sig="Early, wide QRS with no preceding P, followed by a compensatory pause",
         ecg_id=219, lead="V5", t0=5.5, y=(-1.0, 2.0),
         beat=2088, fix=dict(qrs_on=2067, qrs_off=2097), show=("QRS",), rr=True),
    dict(key="AFIB", title="Atrial fibrillation",
         sig="Irregularly irregular RR, no discernible P waves",
         ecg_id=12542, lead="II", t0=3.2, y=(-0.5, 1.5), show=(), rr=True),
    dict(key="LNGQT", title="Long QT",
         sig="QTc ≥ 450 ms (men) / ≥ 460 ms (women)",
         ecg_id=320, lead="II", t0=3.0, y=(-0.5, 1.0),
         beat=1083, fix=dict(qrs_on=1074), show=("QT",)),
]

# Sanity checks: each printed number must agree with its pattern's criterion.
CHECKS = {
    "NORM": lambda m: 120 <= m["PR"] <= 200 and m["QRS"] < 120 and m["QTc"] < 450,
    "1AVB": lambda m: m["PR"] > 200,
    "CLBBB": lambda m: m["QRS"] >= 120,
    "PVC": lambda m: m["QRS"] >= 120 and m["RR"][-2] < m["RR"][-3] < m["RR"][-1],
    "AFIB": lambda m: np.ptp(m["RR"]) > 160,
    "LNGQT": lambda m: m["QTc"] >= 460,
}


def fiducials(x, beat, fix):
    """NeuroKit2 DWT fiducials (sample index) for the beat nearest `beat`, plus pinned fixes."""
    clean = nk.ecg_clean(x, sampling_rate=FS)
    _, info = nk.ecg_peaks(clean, sampling_rate=FS)
    r = info["ECG_R_Peaks"]
    _, d = nk.ecg_delineate(clean, r, sampling_rate=FS, method="dwt")
    b = int(np.argmin(np.abs(r - beat)))
    f = dict(p_on=d["ECG_P_Onsets"][b], qrs_on=d["ECG_R_Onsets"][b],
             qrs_off=d["ECG_R_Offsets"][b], t_off=d["ECG_T_Offsets"][b],
             rr_prev=r[b] - r[b - 1])
    f.update(fix)
    return f


def r_peaks(x):
    """R peaks of a strip: dominant deflections at least 0.3 s apart."""
    pk, _ = find_peaks(x, prominence=0.5 * np.ptp(x), distance=int(0.3 * FS))
    return pk


def measure(p):
    """Return the displayed strip and the interval measurements for one panel."""
    x = style.load_record(p["ecg_id"], leads=[p["lead"]])[p["lead"]]
    i0 = int(p["t0"] * FS)
    seg = x[i0:i0 + int(STRIP_S * FS)]
    m = dict(seg=seg - np.median(seg), i0=i0)
    if "beat" in p:
        f = fiducials(x, p["beat"], p["fix"])
        ms = lambda a, b: (f[b] - f[a]) * 1000 / FS
        m.update(f=f, PR=ms("p_on", "qrs_on"), QRS=ms("qrs_on", "qrs_off"),
                 QT=ms("qrs_on", "t_off"))
        m["QTc"] = m["QT"] / np.sqrt(f["rr_prev"] / FS)          # Bazett
    if p.get("rr"):
        pk = r_peaks(m["seg"])
        m.update(peaks=pk, RR=np.diff(pk) * 1000 / FS)
    return m


def bracket(ax, t0, t1, y, color, lw=1.6, tick=0.07):
    ax.plot([t0, t0, t1, t1], [y - tick, y, y, y - tick], color=color, lw=lw,
            solid_capstyle="butt", solid_joinstyle="miter", zorder=4)


def label(ax, t, y, text, c, ha="center", va="bottom"):
    ax.text(t, y, text, ha=ha, va=va, fontsize=8.5, family="monospace", color=c["ink"],
            zorder=5, bbox=dict(boxstyle="square,pad=0.12", fc=c["surface"], ec="none", alpha=0.85))


def annotate(ax, p, m, c):
    """Shade and label the intervals of interest on the strip."""
    if "f" in m:
        f = {k: (v - m["i0"]) / FS for k, v in m["f"].items() if k != "rr_prev"}
    top = p["y"][1]
    gap = 0.04

    if "PR" in p["show"]:
        ax.axvspan(f["p_on"], f["qrs_on"], color=c["pr"], alpha=0.16, lw=0, zorder=1)
        bracket(ax, f["p_on"], f["qrs_on"], top - 0.3, c["pr"])
        label(ax, f["p_on"] - gap, top - 0.3, f"PR {m['PR']:.0f} ms", c, ha="right", va="center")
    if "QRS" in p["show"]:
        ax.axvspan(f["qrs_on"], f["qrs_off"], color=c["qrs"], alpha=0.20, lw=0, zorder=1)
        y = top - 0.3 if p["key"] != "PVC" else top - 0.62
        bracket(ax, f["qrs_on"], f["qrs_off"], y, c["qrs"])
        prefix = "PVC · " if p["key"] == "PVC" else ""
        label(ax, f["qrs_off"] + gap, y, f"{prefix}QRS {m['QRS']:.0f} ms", c, ha="left", va="center")
    if "QT" in p["show"]:
        # QT starts at QRS onset; where QRS is already shaded, shade only the remainder
        start = f["qrs_off"] if "QRS" in p["show"] else f["qrs_on"]
        ax.axvspan(start, f["t_off"], color=c["qt"], alpha=0.14, lw=0, zorder=1)
        # below the trace when the top row is taken by PR/QRS, else along the top
        y, tick = (p["y"][0] + 0.18, -0.07) if "QRS" in p["show"] else (top - 0.3, 0.07)
        bracket(ax, f["qrs_on"], f["t_off"], y, c["qt"], tick=tick)
        label(ax, f["t_off"] + gap, y, f"QT {m['QT']:.0f} ms · QTc {m['QTc']:.0f} ms", c,
              ha="left", va="center")

    if p.get("rr"):
        t = m["peaks"] / FS
        y = top - 0.18
        rr = list(zip(t[:-1], t[1:], m["RR"]))
        if p["key"] == "PVC":
            rr = rr[-3:]      # last normal RR, coupling interval, compensatory pause
        for a, b, v in rr:
            ax.plot([a, a, b, b], [y - 0.07, y, y, y - 0.07], color=c["ink2"], lw=1.0, zorder=4)
            label(ax, (a + b) / 2, y + 0.04, f"RR {v:.0f}", c)


def rhythm_gallery(theme):
    c = style.apply_theme(theme)
    results = [measure(p) for p in PANELS]

    strip_x = 2.45
    strip_w = W - strip_x - 0.05
    k = strip_w / STRIP_S
    gap, footer = 0.28, 0.55
    heights = [k * 0.4 * (p["y"][1] - p["y"][0]) for p in PANELS]
    H = sum(heights) + gap * (len(PANELS) - 1) + footer + 0.05

    fig = plt.figure(figsize=(W, H))
    y_top = H - 0.05
    for p, m, h in zip(PANELS, results, heights):
        ax = inch_axes(fig, H, strip_x, y_top - h, strip_w, h)
        style.ecg_paper(ax, c, STRIP_S, *p["y"])
        t = np.arange(len(m["seg"])) / FS
        ax.plot(t, m["seg"], color=c["trace"], lw=0.8, zorder=3, solid_joinstyle="round")
        annotate(ax, p, m, c)

        fig.text(0.02 / W, (y_top - 0.02) / H, p["title"], ha="left", va="top",
                 fontsize=11, fontweight="semibold", color=c["ink"])
        lines = textwrap.wrap(p["sig"], 34)
        for j, line in enumerate(lines):
            fig.text(0.02 / W, (y_top - 0.29 - 0.175 * j) / H, line, ha="left", va="top",
                     fontsize=9, color=c["ink2"])
        fig.text(0.02 / W, (y_top - 0.33 - 0.175 * len(lines)) / H,
                 f"Lead {p['lead']} · record #{p['ecg_id']}", ha="left", va="top",
                 fontsize=8.5, family="monospace", color=c["muted"])
        y_top -= h + gap

    foot = ("Real PTB-XL records (Wagner et al., 2020; CC BY 4.0), resampled to 250 Hz as in the "
            "v1 pipeline.\nIntervals measured on the marked beat: NeuroKit2 DWT fiducials, "
            "QRS boundaries verified by eye. QTc by Bazett.")
    fig.text(0.02 / W, 0.06 / H, foot, ha="left", va="bottom", fontsize=8.5,
             color=c["ink2"], linespacing=1.5)
    return fig, results


def report(results):
    for p, m in zip(PANELS, results):
        shown = list(p["show"]) + (["QTc"] if "QT" in p["show"] else [])
        vals = {k: round(float(m[k])) for k in shown}
        if p.get("rr"):
            vals["RR"] = [round(float(v)) for v in m["RR"]]
        ok = CHECKS[p["key"]](m)
        print(f"{p['key']:6} #{p['ecg_id']:<6} lead {p['lead']:3} {vals}  {'ok' if ok else 'FAILED'}")
        assert ok, f"{p['key']}: measured values contradict the pattern label"


if __name__ == "__main__":
    for theme in ("light", "dark"):
        print(style.save(ecg_8lead(theme), "ecg-8lead", theme))
        fig, results = rhythm_gallery(theme)
        print(style.save(fig, "rhythm-gallery", theme))
    report(results)
