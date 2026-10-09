"""Animated SVG hero banner and v1 data-pipeline diagram for the README.

Writes docs/assets/{hero,pipeline}-{light,dark}.svg from one template per figure.
Run from the repo root:  .venv/Scripts/python.exe docs/figures/diagrams_hero_pipeline.py
"""

import math
import sys
from xml.sax.saxutils import escape
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

W = 1200
MM = 8.0                     # px per mm of ECG paper
PX_PER_S = 25 * MM           # 25 mm/s
PX_PER_MV = 10 * MM          # 10 mm/mV
HERO_RECORD = 3


def rgba(c):
    if isinstance(c, str):
        return c
    r, g, b, a = c
    return f"rgba({round(r * 255)},{round(g * 255)},{round(b * 255)},{a})"


def theme(name):
    return {k: rgba(v) for k, v in style.THEMES[name].items()}


def f1(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def rdp(pts, eps):
    """Ramer-Douglas-Peucker polyline simplification."""
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = pts[a], pts[b]
        d = q - p
        seg = pts[a + 1:b] - p
        dist = np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / (np.hypot(*d) or 1)
        i = int(np.argmax(dist))
        if dist[i] > eps:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return pts[keep]


def grid_pattern(c, pid, x0=0.0, y0=0.0):
    """ECG paper: 1 mm minor and 5 mm major lines, as a userSpace pattern."""
    major = 5 * MM
    minor = "".join(f"M{f1(i * MM)} 0V{f1(major)}M0 {f1(i * MM)}H{f1(major)}" for i in range(1, 5))
    return (
        f'<pattern id="{pid}" width="{f1(major)}" height="{f1(major)}" patternUnits="userSpaceOnUse" '
        f'x="{f1(x0)}" y="{f1(y0)}">'
        f'<path d="{minor}" stroke="{c["grid_minor"]}" stroke-width="0.8" fill="none"/>'
        f'<path d="M0.5 0V{f1(major)}M0 0.5H{f1(major)}" stroke="{c["grid_major"]}" stroke-width="1" fill="none"/>'
        f'</pattern>'
    )


# ─── HERO ─────────────────────────────────────────────────────────────────────

def hero_trace():
    """Real PTB-XL lead II, baseline-corrected and 30 Hz low-passed, 6 s window starting in a TP segment."""
    x = style.load_record(HERO_RECORD)["II"]
    base = median_filter(median_filter(x, 51, mode="nearest"), 151, mode="nearest")
    x = x - base
    b, a = butter(4, 30, fs=style.FS)                # monitor-mode smoothing
    x = filtfilt(b, a, x)
    fs = style.FS
    n = int(W / PX_PER_S * fs)                       # 6 s
    r = [261, 505, 746, 985, 1213, 1444, 1676, 1910, 2143, 2378]
    best = None
    for rk in r:
        for off in np.arange(0.30, 0.50, 0.01):
            s = int(rk - off * fs)
            if s < 0 or s + n >= len(x):
                continue
            score = abs(x[s]) + abs(x[s + n]) + abs(x[s] - x[s + n])
            if best is None or score < best[0]:
                best = (score, s)
    s = best[1]
    seg = x[s:s + n + 1]
    seg = seg - np.median(seg)
    return seg


def hero(tn):
    c = theme(tn)
    H = 320
    base_y = 272.0
    period = 6.0                                     # real time: 6 s across 1200 px
    seg = hero_trace()
    t = np.arange(len(seg)) / style.FS
    pts = np.column_stack([t * PX_PER_S, base_y - seg * PX_PER_MV])
    pts = rdp(pts, 0.22)
    d = "M" + " ".join(f"{f1(px)} {f1(py)}" for px, py in pts)

    # write-head keyframes: x is linear in time, so each vertex sits at x/W of the loop
    kf = []
    for px, py in pts:
        kf.append(f"{px / W * 100:.3f}%{{transform:translate({f1(px)}px,{f1(py)}px)}}")
    kf = "".join(kf)

    gap = 44
    fresh = 170
    css = f"""
.sans{{font-family:{SANS}}}
.mono{{font-family:{MONO}}}
.sweep{{animation:sweep {period}s linear infinite}}
.head{{animation:head {period}s linear infinite}}
@keyframes sweep{{from{{transform:translateX(0px)}}to{{transform:translateX({W}px)}}}}
@keyframes head{{{kf}}}
@media (prefers-reduced-motion: reduce){{
.sweep,.head{{animation:none}}
.anim{{display:none}}
}}
"""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
<title id="t">AI-ECG</title>
<desc id="d">AI-ECG: deep learning on 12-lead ECG waveforms to predict cardiac dysfunction. A real lead II ECG (PTB-XL record {HERO_RECORD}) sweeps across ECG paper like a bedside monitor.</desc>
<style>{css}</style>
<defs>
{grid_pattern(c, "grid")}
<clipPath id="card"><rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="12"/></clipPath>
<linearGradient id="fade" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff"/></linearGradient>
<linearGradient id="erase" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#000"/><stop offset="0.75" stop-color="#000"/><stop offset="1" stop-color="#fff"/></linearGradient>
<mask id="gap" maskUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">
<rect width="{W}" height="{H}" fill="#fff"/>
<g class="sweep anim"><rect x="3" y="0" width="{gap}" height="{H}" fill="url(#erase)"/><rect x="{3 - W}" y="0" width="{gap}" height="{H}" fill="url(#erase)"/></g>
</mask>
<mask id="fresh" maskUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">
<g class="sweep"><rect x="{-fresh}" y="0" width="{fresh + 1}" height="{H}" fill="url(#fade)"/><rect x="{W - fresh}" y="0" width="{fresh + 1}" height="{H}" fill="url(#fade)"/></g>
</mask>
<filter id="glow" x="-2" y="-2" width="5" height="5"><feGaussianBlur stdDeviation="4"/></filter>
</defs>
<g clip-path="url(#card)">
<rect width="{W}" height="{H}" fill="{c["surface"]}"/>
<rect width="{W}" height="{H}" fill="url(#grid)"/>
<g mask="url(#gap)">
<path d="{d}" fill="none" stroke="{c["trace"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
<path class="anim" mask="url(#fresh)" d="{d}" fill="none" stroke="{c["pr"]}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>
</g>
<g class="head anim">
<circle r="10" fill="{c["pr"]}" opacity="0.55" filter="url(#glow)"/>
<circle r="3.6" fill="{c["pr"]}"/>
<circle r="1.6" fill="{c["surface"]}"/>
</g>
</g>
<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="none" stroke="{c["border"]}"/>
<text class="mono" x="48" y="62" font-size="15" letter-spacing="1.6" fill="{c["ink2"]}">UCLA HEALTH · CARDIO-ONCOLOGY RESEARCH</text>
<text class="sans" x="45" y="128" font-size="62" font-weight="700" letter-spacing="-1" fill="{c["ink"]}">AI-ECG</text>
<text class="sans" x="48" y="168" font-size="22" fill="{c["ink2"]}">Deep learning on 12-lead ECG waveforms to predict cardiac dysfunction</text>
<text class="mono" x="{W - 48}" y="62" font-size="15" text-anchor="end" fill="{c["muted"]}">II · 25 mm/s · 10 mm/mV</text>
</svg>
"""


# ─── PIPELINE ─────────────────────────────────────────────────────────────────

STAGES = [
    ("01", "Archive", ["GE MUSE-style", "XML ECG files"], ["30,000+", "records"]),
    ("02", "Decode", ["Base64 → int16;", "8 leads × 2 sets", "(median, rhythm)"], ["20,000+", "XMLs decoded"]),
    ("03", "Quality control", ["Byte and sample", "checks, clamp,", "paced removed"], ["byte · sample", "limit · paced"]),
    ("04", "Normalise", ["FIR-decimate", "500 → 250 Hz;", "I, II, V1–V6"], ["8 × 2500", "@ 250 Hz"]),
    ("05", "Cohort window", ["ECGs around ICI", "start, linked by", "patient ID"], ["−4 y … +30 d", "ICI start = 0"]),
    ("06", "Features", ["NeuroKit2 on", "lead II, joined", "with clinical"], ["HR · QRS", "QT · T axis"]),
    ("07", "Model-ready", ["86 CTRCD ECGs", "632 control ECGs"], ["718 ECGs", "960 patients"]),
]

BARS = [
    # (stage tag, label, value, display, token)
    ("01", "ECG records in archive", 30000, "30,000+", "muted"),
    ("02", "Encrypted XMLs decoded", 20000, "20,000+", "muted"),
    ("07", "Model-ready ECGs", 718, "718", "muted"),
    ("", "CTRCD", 86, "86 · 126 patients", "ink"),
    ("", "ICI control", 632, "632 · 834 patients", "ink2"),
]


def pipeline(tn):
    c = theme(tn)
    n = len(STAGES)
    gutter = 24
    cw = (W - (n - 1) * gutter) / n
    top = 46
    ch = 214
    out = []

    out.append(f'<text class="mono" x="0" y="20" font-size="15" letter-spacing="1.6" fill="{c["ink2"]}">V1 DATA PIPELINE · UCLA HEALTH</text>')
    out.append(f'<text class="mono" x="{W - 2}" y="20" font-size="15" text-anchor="end" fill="{c["muted"]}">XML archive → model-ready cohort</text>')

    for i, (tag, name, desc, metric) in enumerate(STAGES):
        x = i * (cw + gutter)
        out.append(f'<rect x="{f1(x + 0.5)}" y="{top + 0.5}" width="{f1(cw - 1)}" height="{ch - 1}" rx="8" fill="{c["card"]}" stroke="{c["border"]}"/>')
        tx = x + 13
        out.append(f'<text class="mono" x="{f1(tx)}" y="{top + 30}" font-size="15" fill="{c["muted"]}">{tag}</text>')
        out.append(f'<text class="sans" x="{f1(tx)}" y="{top + 56}" font-size="17" font-weight="600" fill="{c["ink"]}">{name}</text>')
        for j, line in enumerate(desc):
            out.append(f'<text class="sans" x="{f1(tx)}" y="{top + 86 + j * 20}" font-size="15" fill="{c["ink2"]}">{escape(line)}</text>')
        out.append(f'<path d="M{f1(x + 14)} {top + 146.5}H{f1(x + cw - 14)}" stroke="{c["border"]}"/>')
        for j, line in enumerate(metric):
            col = c["ink"] if j == 0 else c["ink2"]
            fw = ' font-weight="600"' if j == 0 else ""
            out.append(f'<text class="mono" x="{f1(tx)}" y="{top + 172 + j * 20}" font-size="15"{fw} fill="{col}">{line}</text>')
        if i < n - 1:
            x0, x1 = x + cw + 3, x + cw + gutter - 3
            y = top + ch / 2
            out.append(f'<path d="M{f1(x0)} {y}H{f1(x1 - 4)}" stroke="{c["border"]}" stroke-width="2"/>')
            out.append(f'<path class="flow" d="M{f1(x0)} {y}H{f1(x1 - 4)}" stroke="{c["pr"]}" stroke-width="2" stroke-dasharray="3 7"/>')
            out.append(f'<path d="M{f1(x1 - 6)} {y - 4.5}L{f1(x1)} {y}L{f1(x1 - 6)} {y + 4.5}" fill="none" stroke="{c["ink2"]}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>')

    # scale strip: log10 bars on ECG-paper grid
    sy = top + ch + 50
    out.append(f'<text class="mono" x="0" y="{sy}" font-size="15" letter-spacing="1.6" fill="{c["ink2"]}">DATA VOLUME · LOG SCALE</text>')
    lx = 270                                        # label column width
    px0, px1 = lx, W - 90                           # plot extent (value labels to the right)
    dmin, dmax = 1, 5                               # 10 … 100,000
    def X(v):
        return px0 + (math.log10(v) - dmin) / (dmax - dmin) * (px1 - px0)
    rh, rgap = 14, 18
    py0 = sy + 22
    ph = len(BARS) * (rh + rgap) - rgap + 24
    # grid: decades major, 2..9 minor
    for dec in range(dmin, dmax + 1):
        xx = X(10 ** dec)
        out.append(f'<path d="M{f1(xx)} {py0}V{py0 + ph}" stroke="{c["grid_major"]}" stroke-width="1"/>')
        lbl = f"{10 ** dec:,}"
        out.append(f'<text class="mono" x="{f1(xx)}" y="{py0 + ph + 22}" font-size="15" text-anchor="middle" fill="{c["muted"]}">{lbl}</text>')
        if dec < dmax:
            for k in range(2, 10):
                xx = X(k * 10 ** dec)
                out.append(f'<path d="M{f1(xx)} {py0}V{py0 + ph}" stroke="{c["grid_minor"]}" stroke-width="1"/>')
    out.append(f'<path d="M{px0} {py0 + ph + 0.5}H{px1}" stroke="{c["grid_major"]}" stroke-width="1"/>')

    for i, (tag, label, v, disp, tok) in enumerate(BARS):
        y = py0 + 12 + i * (rh + rgap)
        indent = 28 if not tag else 0
        ty = y + rh / 2 + 5.5
        if tag:
            out.append(f'<text class="mono" x="0" y="{ty}" font-size="15" fill="{c["muted"]}">{tag}</text>')
        out.append(f'<text class="sans" x="{34 + indent}" y="{ty}" font-size="15" fill="{c["ink"] if tag else c["ink2"]}">{label}</text>')
        if not tag:
            out.append(f'<path d="M40 {y - 12}V{y + rh / 2}H{34 + indent - 6}" fill="none" stroke="{c["border"]}" stroke-width="1.2"/>')
        bx1 = X(v)
        out.append(f'<rect x="{px0}" y="{y}" width="{f1(bx1 - px0)}" height="{rh}" rx="2" fill="{c[tok]}"/>')
        num, _, rest = disp.partition(" · ")
        extra = f'<tspan font-weight="400" fill="{c["ink2"]}">  ·  {rest}</tspan>' if rest else ""
        out.append(f'<text class="mono" x="{f1(bx1 + 10)}" y="{ty}" font-size="15" font-weight="600" fill="{c["ink"]}" xml:space="preserve">{num}{extra}</text>')

    H = py0 + ph + 34
    css = f"""
.sans{{font-family:{SANS}}}
.mono{{font-family:{MONO}}}
.flow{{animation:flow 1.6s linear infinite}}
@keyframes flow{{to{{stroke-dashoffset:-20}}}}
@media (prefers-reduced-motion: reduce){{.flow{{animation:none}}}}
"""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
<title id="t">AI-ECG v1 data pipeline</title>
<desc id="d">Seven stages from a 30,000+ record ECG archive to a model-ready cohort of 718 ECGs from 960 patients (86 CTRCD, 632 control), with a log-scale strip of data volume.</desc>
<style>{css}</style>
<rect width="{W}" height="{H}" fill="{c["surface"]}"/>
{chr(10).join(out)}
</svg>
"""


def main():
    style.ASSETS.mkdir(parents=True, exist_ok=True)
    for tn in ("light", "dark"):
        for name, fn in (("hero", hero), ("pipeline", pipeline)):
            p = style.ASSETS / f"{name}-{tn}.svg"
            p.write_text(fn(tn), encoding="utf-8")
            print(f"{p.relative_to(style.ROOT)}  {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
