"""Hand-built SVG diagrams for the README: record anatomy, the v1 two-tower
model and the paired (v1 vs v2) study design.

Each diagram is written once per GitHub theme as ``<name>-light.svg`` and
``<name>-dark.svg`` in ``docs/assets/`` with identical layout.

    .venv/Scripts/python.exe docs/figures/diagrams_model_design.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"
FS = style.FS
RECORD = 3  # PTB-XL NORM record used for every waveform thumbnail


# ─── primitives ──────────────────────────────────────────────────────────────

def n(v: float) -> str:
    """Compact number: one decimal, trailing zeros stripped."""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rgba_to_hex(t):
    r, g, b, a = t
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255)), a


class Svg:
    def __init__(self, w: int, h: int, title: str, label: str, theme: str):
        self.w, self.h, self.title, self.label = w, h, title, label
        self.c = style.THEMES[theme]
        self.el: list[str] = []
        self.defs: list[str] = []
        self.css = ""

    def col(self, k: str) -> str:
        return self.c.get(k, k) if not isinstance(self.c.get(k), tuple) else rgba_to_hex(self.c[k])[0]

    def add(self, s: str):
        self.el.append(s)

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1, rx=0, fo=None, so=None, dash=None, extra=""):
        a = f'<rect x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}"'
        if rx:
            a += f' rx="{n(rx)}"'
        a += f' fill="{self.col(fill)}"'
        if fo is not None:
            a += f' fill-opacity="{fo}"'
        if stroke:
            a += f' stroke="{self.col(stroke)}" stroke-width="{n(sw)}"'
            if so is not None:
                a += f' stroke-opacity="{so}"'
            if dash:
                a += f' stroke-dasharray="{dash}"'
        self.add(a + extra + "/>")

    def line(self, x1, y1, x2, y2, stroke="border", sw=1, dash=None, so=None, marker=None, extra=""):
        a = (f'<line x1="{n(x1)}" y1="{n(y1)}" x2="{n(x2)}" y2="{n(y2)}" '
             f'stroke="{self.col(stroke)}" stroke-width="{n(sw)}"')
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if so is not None:
            a += f' stroke-opacity="{so}"'
        if marker:
            a += f' marker-end="url(#{marker})"'
        self.add(a + extra + "/>")

    def path(self, d, stroke=None, sw=1, fill="none", so=None, fo=None, marker=None, extra=""):
        a = f'<path d="{d}" fill="{self.col(fill)}"'
        if fo is not None:
            a += f' fill-opacity="{fo}"'
        if stroke:
            a += f' stroke="{self.col(stroke)}" stroke-width="{n(sw)}"'
            if so is not None:
                a += f' stroke-opacity="{so}"'
        if marker:
            a += f' marker-end="url(#{marker})"'
        self.add(a + extra + "/>")

    def text(self, x, y, s, size=15, fill="ink", mono=False, weight=None, anchor=None,
             ls=None, extra=""):
        if mono:
            s = s.replace(" ", " ")
        a = f'<text x="{n(x)}" y="{n(y)}" class="{"m" if mono else "s"}"'
        if size != 15:
            a += f' font-size="{size}"'
        a += f' fill="{self.col(fill)}"'
        if weight:
            a += f' font-weight="{weight}"'
        if anchor:
            a += f' text-anchor="{anchor}"'
        if ls:
            a += f' letter-spacing="{ls}"'
        self.add(f"{a}{extra}>{esc(s)}</text>")

    def eyebrow(self, x, y, s, fill="ink2", anchor=None, sub=None):
        """Upper-case label, optionally followed by a plain-case description."""
        a = f' text-anchor="{anchor}"' if anchor else ""
        t = (f'<text x="{n(x)}" y="{n(y)}" class="s"{a}><tspan fill="{self.col(fill)}" '
             f'font-weight="600" letter-spacing="0.08em">{esc(s.upper())}</tspan>')
        if sub:
            t += f'<tspan dx="14" fill="{self.col("ink2")}">{esc(sub)}</tspan>'
        self.add(t + "</text>")

    def arrow_marker(self, mid: str, color: str):
        self.defs.append(
            f'<marker id="{mid}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" '
            f'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">'
            f'<path d="M0,0.5 L7.5,4 L0,7.5 Z" fill="{self.col(color)}"/></marker>')

    def grid(self, x, y, w, h, minor):
        """ECG-paper grid: minor every `minor` px, major every 5 minor."""
        mh, ma = rgba_to_hex(self.c["grid_minor"])
        Mh, Ma = rgba_to_hex(self.c["grid_major"])
        dmin, dmaj = [], []
        k = 0
        while k * minor <= w + 0.01:
            (dmaj if k % 5 == 0 else dmin).append(f"M{n(x + k * minor)} {n(y)}V{n(y + h)}")
            k += 1
        k = 0
        while k * minor <= h + 0.01:
            (dmaj if k % 5 == 0 else dmin).append(f"M{n(x)} {n(y + k * minor)}H{n(x + w)}")
            k += 1
        self.add(f'<path d="{"".join(dmin)}" stroke="{mh}" stroke-opacity="{ma}" stroke-width="0.6" fill="none"/>')
        self.add(f'<path d="{"".join(dmaj)}" stroke="{Mh}" stroke-opacity="{Ma}" stroke-width="1" fill="none"/>')

    def render(self) -> str:
        css = (f".s{{font-family:{SANS};font-size:15px}}"
               f".m{{font-family:{MONO};font-size:15px}}" + self.css)
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(self.label)}">\n'
                f"<title>{esc(self.title)}</title>\n<style>{css}</style>\n")
        defs = f"<defs>{''.join(self.defs)}</defs>\n" if self.defs else ""
        bg = f'<rect width="{self.w}" height="{self.h}" fill="{self.c["surface"]}"/>\n'
        return head + defs + bg + "\n".join(self.el) + "\n</svg>\n"


def trace_d(sig, x0, ymid, pxs, pxmv, t0=0.0, t1=None, step=2):
    """SVG path for a lead segment: pxs px per second, pxmv px per mV."""
    i0 = int(t0 * FS)
    i1 = len(sig) if t1 is None else int(t1 * FS)
    seg = sig[i0:i1:step]
    pts = [f"{n(x0 + i * step / FS * pxs)} {n(ymid - v * pxmv)}" for i, v in enumerate(seg)]
    return "M" + "L".join(pts)


def load_leads():
    rec = style.load_record(RECORD)
    return {k: v - np.median(v) for k, v in rec.items()}


def median_beat(sig, pre=0.35, post=0.65):
    peaks, _ = find_peaks(sig, distance=int(0.4 * FS), height=0.5 * sig.max())
    a, b = int(pre * FS), int(post * FS)
    beats = [sig[p - a:p + b] for p in peaks if p - a >= 0 and p + b <= len(sig)]
    m = np.median(np.stack(beats), axis=0)
    return m - np.median(m[:int(0.1 * FS)])


def write(svg: Svg, name: str, theme: str):
    style.ASSETS.mkdir(parents=True, exist_ok=True)
    p = style.ASSETS / f"{name}-{theme}.svg"
    p.write_text(svg.render(), encoding="utf-8")
    print(f"{p.relative_to(style.ROOT)}  {p.stat().st_size / 1024:.1f} KB")


REDUCED = "@media (prefers-reduced-motion: reduce){.fx{animation:none!important}.flow{display:none}}"


# ─── 1. record anatomy ───────────────────────────────────────────────────────

def record_anatomy(theme: str, leads: dict):
    CX, CW = 24, 500          # XML card
    BX = 556                  # "becomes" column
    DX, DW = 914, 262         # destinations
    LH = 22
    rows = [
        dict(tag="<Waveform> × 2", dest="tensor",
             kids=["Median beat · Rhythm strip (10 s)",
                   "LeadData × 8: LeadID, WaveFormData",
                   "LeadByteCountTotal, LeadSampleCountTotal",
                   "LeadAmplitudeUnitsPerBit, LeadHigh/LowLimit"],
             bec=["Rhythm strip → 8 leads × 2500 samples",
                  "Base64 → little-endian int16 × units-per-bit",
                  "checked against byte and sample counts,",
                  "clamped to the lead high/low limits",
                  "III, aVR, aVL, aVF: derived from I and II"]),
        dict(tag="<TestDemographics>", dest=None,
             kids=["AcquisitionDate"],
             bec=["Acquisition date places the ECG",
                  "relative to ICI start",
                  "selection window only, not a feature"]),
        dict(tag="<PatientDemographics>", dest="tab", item="age · sex",
             kids=["IP_PATIENT_ID  PatientAge  Gender",
                   "DateofBirth  Race"],
             bec=["Patient ID (IPPAT) links records",
                  "across hospital databases",
                  "Age · sex · date of birth",
                  "Race dropped: too many missing values"]),
        dict(tag="<RestingECGMeasurements>", dest="tab", item="ECG intervals & rates",
             kids=["VentricularRate  AtrialRate",
                   "QTInterval  QRSDuration  TAxis",
                   "ECGSampleBase"],
             bec=["Ventricular & atrial rate, QT interval,",
                  "QRS duration, T axis",
                  "sampling base (Hz)"]),
        dict(tag="<Diagnosis>", dest="tab", item="abnormal flag",
             kids=["DiagnosisStatement / StmtText"],
             bec=["Abnormal flag if any statement reads:",
                  "1° AV block, AF, RBBB, LBBB, ST-T change,",
                  "LVH, PAC, PVC, low voltage",
                  "Paced rhythm → ECG excluded"]),
    ]
    linked = dict(dest="tab", item="clinical covariates",
                  kids=["AGE_AT_FIRST_ICI  ICI_GROUP  ef_prior  CR",
                        "SMOKE  htn  hld  dm",
                        "*_BEFORE_ICI: acs, arrhythmia, cad, hf,",
                        "              cardiacarrest, stroke"],
                  bec=["Age at first ICI, ICI group, baseline EF,",
                       "creatinine, smoking, hypertension,",
                       "hyperlipidemia, diabetes",
                       "History before ICI: ACS, arrhythmia, CAD,",
                       "heart failure, cardiac arrest, stroke"])

    TOP = 52
    HEAD = 46
    y = TOP + HEAD
    for r in rows:
        nl = max(1 + len(r["kids"]), len(r["bec"]))
        r["h"] = LH * nl + 34
        if r["dest"] == "tensor":
            r["h"] += 66 + 46
        r["y"] = y
        y += r["h"]
    card_bot = y
    linked["y"] = card_bot + 20
    linked["h"] = LH * (2 + len(linked["kids"])) + 34
    H = linked["y"] + linked["h"] + 24

    s = Svg(1200, H, "What one ECG record contains",
            "Diagram: the sections of one MUSE-style resting ECG XML record (Waveform, "
            "TestDemographics, PatientDemographics, RestingECGMeasurements, Diagnosis) plus "
            "linked clinical tables, and how each becomes either the 8 by 2500 waveform "
            "tensor at 250 Hz or the tabular feature vector.", theme)
    s.arrow_marker("ah", "ink2")

    s.eyebrow(CX, 34, "One MUSE-style XML record")
    s.eyebrow(BX, 34, "What it becomes")
    s.eyebrow(DX, 34, "In the dataset")

    # document card with folded corner
    fold = 22
    s.path(f"M{CX} {TOP}H{CX + CW - fold}L{CX + CW} {TOP + fold}V{card_bot}H{CX}Z",
           stroke="border", fill="card")
    s.path(f"M{CX + CW - fold} {TOP}V{TOP + fold}H{CX + CW}", stroke="border", fill="surface")
    s.text(CX + 20, TOP + 29, "<RestingECG>", mono=True, fill="muted")
    s.line(CX, TOP + HEAD, CX + CW, TOP + HEAD, "border")

    def bec(r, x):
        for i, ln in enumerate(r["bec"]):
            muted = ln.startswith(("Race", "selection")) or ln.startswith("Paced")
            s.text(x, r["y"] + 30 + LH * i, ln,
                   fill="ink" if i == 0 else ("muted" if muted else "ink2"),
                   weight=600 if i == 0 else None)

    ay_list = []
    for i, r in enumerate(rows):
        yy = r["y"]
        if i:
            s.line(CX, yy, CX + CW, yy, "border")
        s.text(CX + 20, yy + 30, r["tag"], 16, mono=True, weight=600)
        for k, ln in enumerate(r["kids"]):
            s.text(CX + 20, yy + 30 + LH * (k + 1), ln, mono=True, fill="ink2")
        bec(r, BX)
        s.line(CX + CW + 6, yy + 25, BX - 12, yy + 25, "border", 1.2)
        if r["dest"]:
            s.line(BX + 322, yy + 25, DX - 2, yy + 25, "ink2", 1.2, marker="ah")
            ay_list.append((yy + 25, r))

    # waveform thumbnails inside the Waveform row
    w = rows[0]
    ty = w["y"] + 30 + LH * 4 + 18
    th = 66
    pxs = 82.5                       # 0.2 s = 16.5 px = 0.5 mV square boxes
    pxmv = 33
    mb = median_beat(leads["II"], 0.45, 0.75)
    mx = CX + 20
    mw = pxs * 1.2
    s.rect(mx, ty, mw, th, "surface")
    s.grid(mx, ty, mw, th, pxs * 0.04)
    s.path(trace_d(mb, mx, ty + th * 0.68, pxs, pxmv, step=1), "trace", 1.3,
           extra=' stroke-linejoin="round"')
    rx0 = mx + mw + 20
    rw = pxs * 4
    s.rect(rx0, ty, rw, th, "surface")
    s.grid(rx0, ty, rw, th, pxs * 0.04)
    s.path(trace_d(leads["II"], rx0, ty + th * 0.68, pxs, pxmv, 0, 4, step=1), "trace", 1.3,
           extra=' stroke-linejoin="round"')
    s.text(mx, ty + th + 22, "median beat", mono=True, fill="muted")
    s.text(rx0, ty + th + 22, "rhythm · lead II · first 4 of 10 s", mono=True, fill="muted")

    # linked tables card (outside the XML)
    ly = linked["y"]
    s.rect(CX, ly, CW, linked["h"], "surface", "ink2", 1.2, rx=6, dash="6 5", so=0.7)
    s.text(CX + 20, ly + 30, "Linked clinical tables", 16, weight=600)
    s.text(CX + 20, ly + 30 + LH, "outside the XML · joined on IPPAT", fill="ink2")
    for k, ln in enumerate(linked["kids"]):
        s.text(CX + 20, ly + 30 + LH * (k + 2), ln, mono=True, fill="ink2")
    bec(linked, BX)
    s.line(CX + CW + 6, ly + 25, BX - 12, ly + 25, "border", 1.2)
    s.line(BX + 322, ly + 25, DX - 2, ly + 25, "ink2", 1.2, marker="ah")
    ay_list.append((ly + 25, linked))

    # destination: waveform tensor
    ty0, th0 = w["y"] + 10, w["h"] - 20
    s.rect(DX, ty0, DW, th0, "card", "border", rx=6)
    s.rect(DX, ty0 + 10, 3, th0 - 20, "pr")
    s.text(DX + 20, ty0 + 30, "Waveform tensor", 17, weight=600)
    s.text(DX + 20, ty0 + 54, "[8 × 2500] @ 250 Hz", mono=True, fill="pr")
    gy0, gy1 = ty0 + 70, ty0 + th0 - 12
    gx0, gx1 = DX + 54, DX + DW - 14
    s.grid(gx0, gy0, gx1 - gx0, gy1 - gy0, (gx1 - gx0) / 50)
    step = (gy1 - gy0) / 8
    for i, ld in enumerate(style.LEADS_8):
        yc = gy0 + step * (i + 0.62)
        s.text(DX + 18, yc + 5, ld, mono=True, fill="muted")
        s.path(trace_d(leads[ld], gx0, yc, (gx1 - gx0) / 2.0, 9, 0, 2.0, step=2), "trace", 0.9,
               extra=' stroke-linejoin="round"')

    # destination: tabular features
    tb0 = rows[1]["y"] + 10
    tb1 = ly + linked["h"]
    s.rect(DX, tb0, DW, tb1 - tb0, "card", "border", rx=6)
    s.rect(DX, tb0 + 10, 3, tb1 - tb0 - 20, "qt")
    s.text(DX + 20, tb0 + 30, "Tabular features", 17, weight=600)
    s.text(DX + 20, tb0 + 54, "intervals + demographics", fill="ink2")
    s.text(DX + 20, tb0 + 54 + LH, "+ covariates", fill="ink2")
    for yy, r in ay_list[1:]:
        s.rect(DX + 20, yy - 5, 9, 9, "qt", rx=1.5)
        s.text(DX + 38, yy + 5, r["item"], fill="ink")
    write(s, "record-anatomy", theme)


# ─── 2. two-tower model ──────────────────────────────────────────────────────

def two_tower(theme: str, leads: dict):
    H = 660
    s = Svg(1200, H, "v1 two-tower model",
            "Diagram of the v1 two-tower network. Waveform tower: 8-lead ECG [B, 8, 2500] "
            "through three Conv1d blocks (8 to 32 kernel 7, 32 to 64 kernel 5, 64 to 128 "
            "kernel 3, each with BatchNorm, ReLU, MaxPool 2, Dropout 0.2) giving 32 by 1250, "
            "64 by 625 and 128 by 312, then adaptive average pooling to a 128-d embedding. "
            "Tabular tower: Linear 64 and Linear 32 with BatchNorm, ReLU, Dropout 0.3 giving a "
            "32-d embedding. Fusion head: concatenate to 160, Linear 64, Linear 16, Linear 1 "
            "with sigmoid giving P(CTRCD).", theme)
    s.arrow_marker("ah", "ink2")
    s.css = (".flow{stroke-dasharray:3 15;animation:fl 1.8s linear infinite}"
             "@keyframes fl{to{stroke-dashoffset:-36}}" + REDUCED)

    WA, TA = 206, 470                   # tower axes
    FA = (WA + TA) // 2                 # fusion axis
    LAB1, LAB2 = WA - 112, WA - 92      # label baselines above waveform blocks
    SHP = WA + 64 + 30                  # shape baseline below waveform blocks

    def flow(d, color):
        s.add(f'<path class="flow fx" d="{d}" fill="none" stroke="{s.col(color)}" '
              f'stroke-width="2" stroke-linecap="round" stroke-opacity="0.9"/>')

    def arrow(x1, y1, x2, y2, color):
        s.line(x1, y1, x2 - 2, y2, "ink2", 1.3, marker="ah")
        flow(f"M{n(x1)} {n(y1)}H{n(x2 - 10)}", color)

    # ── waveform tower
    s.eyebrow(24, 44, "Waveform tower", fill="pr",
              sub="1-D CNN · each block: Conv1d → BatchNorm → ReLU → MaxPool 2 → Dropout 0.2")

    ix0, iy0, iw, ih = 54, WA - 76, 132, 152
    s.rect(ix0, iy0, iw, ih, "surface", "border", rx=3)
    s.grid(ix0, iy0, iw, ih, iw / 40)
    rh = ih / 8
    for i, ld in enumerate(style.LEADS_8):
        yc = iy0 + rh * (i + 0.6)
        s.text(ix0 - 8, yc + 5, ld, mono=True, fill="muted", anchor="end")
        s.path(trace_d(leads[ld], ix0, yc, iw / 2.4, 10, 0, 2.4, step=2), "trace", 0.85,
               extra=' stroke-linejoin="round"')
    s.text(ix0 + iw / 2, LAB1, "8-lead ECG", anchor="middle", weight=600)
    s.text(ix0 + iw / 2, LAB2, "10 s · 250 Hz", anchor="middle", fill="ink2")
    s.text(ix0 + iw / 2, SHP, "[B, 8, 2500]", mono=True, fill="ink2", anchor="middle")

    k = 0.13                                       # px per time step
    blocks = [  # centre x, channels, T, op label, kernel
        (316, 32, 1250, "Conv1d 8→32", "kernel 7"),
        (474, 64, 625, "Conv1d 32→64", "kernel 5"),
        (604, 128, 312, "Conv1d 64→128", "kernel 3"),
    ]
    prev_r = ix0 + iw
    for cx, ch, T, op, ker in blocks:
        bw, bh = T * k, ch
        x0 = cx - bw / 2
        arrow(prev_r + 8, WA, x0 - 6, WA, "pr")
        s.rect(x0, WA - bh / 2, bw, bh, "pr", "pr", 1.2, rx=3, fo=0.16)
        for j in range(1, ch // 16):               # channel strata
            yy = WA - bh / 2 + j * 16
            s.line(x0 + 1, yy, x0 + bw - 1, yy, "pr", 0.6, so=0.28)
        s.text(cx, LAB1, op, mono=True, anchor="middle", weight=600)
        s.text(cx, LAB2, ker, mono=True, anchor="middle", fill="ink2")
        s.text(cx, SHP, f"[{ch} × {T}]", mono=True, anchor="middle", fill="ink2")
        prev_r = x0 + bw

    ex = 740
    arrow(prev_r + 8, WA, ex - 7 - 6, WA, "pr")
    s.rect(ex - 7, WA - 64, 14, 128, "pr", rx=2)
    s.text(ex, LAB1, "AdaptiveAvgPool", mono=True, anchor="middle", weight=600)
    s.text(ex, LAB2, "→ embedding", mono=True, anchor="middle", fill="ink2")
    s.text(ex, SHP, "128-d", mono=True, anchor="middle", fill="pr", weight=600)

    # ── tabular tower
    s.eyebrow(24, TA - 104, "Tabular tower", fill="qt",
              sub="MLP · each layer: Linear → BatchNorm → ReLU → Dropout 0.3")
    bx0, bw0, bh0 = 24, 162, 116
    s.rect(bx0, TA - bh0 / 2, bw0, bh0, "card", "border", rx=6)
    s.rect(bx0, TA - bh0 / 2 + 10, 3, bh0 - 20, "qt")
    s.text(bx0 + 18, TA - bh0 / 2 + 28, "Tabular features", weight=600)
    for i, t in enumerate(["ECG intervals", "demographics", "clinical covariates"]):
        s.text(bx0 + 18, TA - bh0 / 2 + 54 + 22 * i, t, fill="ink2")
    s.text(bx0 + bw0 / 2, TA + 88, "[B, tab_in_dim]", mono=True, anchor="middle",
           fill="ink2")

    tabs = [(474, 64, "Linear → 64", "[64]"), (ex, 32, "Linear → 32", "32-d")]
    prev_r = bx0 + bw0
    for cx, u, op, shp in tabs:
        w_ = 14
        arrow(prev_r + 8, TA, cx - w_ / 2 - 6, TA, "qt")
        last = cx == ex
        s.rect(cx - w_ / 2, TA - u / 2, w_, u, "qt", "qt", 1.2, rx=2, fo=1 if last else 0.16)
        s.text(cx, TA - 58, op, mono=True, anchor="middle", weight=600)
        s.text(cx, TA + 88, shp, mono=True, anchor="middle",
               fill="qt" if last else "ink2", weight=600 if last else None)
        prev_r = cx + w_ / 2

    # ── fusion head
    fx = [846, 934, 1022, 1110]
    LABF = FA - 98
    s.eyebrow(fx[0] - 8, LABF - 34, "Fusion head", fill="ink")
    cbx = fx[0] - 8
    ctop = FA - 80
    s.rect(cbx, ctop, 16, 128, "pr", rx=2)
    s.rect(cbx, ctop + 128, 16, 32, "qt", rx=2)
    s.line(cbx, ctop + 128, cbx + 16, ctop + 128, "surface", 2)

    r = 10
    jx = 796
    yw = ctop + 64
    yt = ctop + 144
    d1 = (f"M{ex + 7 + 8} {WA}H{jx - r}Q{jx} {WA} {jx} {WA + r}V{yw - r}"
          f"Q{jx} {yw} {jx + r} {yw}H{cbx - 2}")
    d2 = (f"M{ex + 7 + 8} {TA}H{jx - r}Q{jx} {TA} {jx} {TA - r}V{yt + r}"
          f"Q{jx} {yt} {jx + r} {yt}H{cbx - 2}")
    s.path(d1, "ink2", 1.3, marker="ah")
    s.path(d2, "ink2", 1.3, marker="ah")
    flow(d1.replace(f"H{cbx - 2}", f"H{cbx - 10}"), "pr")
    flow(d2.replace(f"H{cbx - 2}", f"H{cbx - 10}"), "qt")

    DIM = FA + 108
    s.text(fx[0], LABF, "concat", anchor="middle", weight=600)
    s.text(fx[0], DIM, "160", mono=True, anchor="middle", fill="ink2")
    prev_r = cbx + 16
    for cx, u, op in [(fx[1], 64, "Linear 64"), (fx[2], 16, "Linear 16")]:
        arrow(prev_r + 8, FA, cx - 7 - 6, FA, "ink")
        s.rect(cx - 7, FA - u / 2, 14, u, "ink", "ink", 1.2, rx=2, fo=0.12)
        s.text(cx, LABF, op, anchor="middle", weight=600)
        s.text(cx, DIM, str(u), mono=True, anchor="middle", fill="ink2")
        prev_r = cx + 7
    arrow(prev_r + 8, FA, fx[3] - 9 - 6, FA, "ink")
    s.add(f'<circle cx="{fx[3]}" cy="{FA}" r="9" fill="{s.col("ink")}"/>')
    s.text(fx[3], LABF, "Linear 1", anchor="middle", weight=600)
    s.text(fx[3], DIM, "1", mono=True, anchor="middle", fill="ink2")
    s.text(fx[3], FA + 38, "P(CTRCD)", mono=True, anchor="middle", weight=600)
    s.text(cbx, DIM + 34, "ReLU + Dropout (0.3, 0.2) between layers", fill="muted")
    s.text(cbx, DIM + 56, "sigmoid on the output", fill="muted")

    # ── training note
    ny = H - 48
    s.line(24, ny - 30, 1176, ny - 30, "border")
    s.text(24, ny, "Trained with BCE · class weights · stratified 5-fold CV", mono=True,
           fill="ink2")
    s.text(24, ny + 24, "Augmentations: noise · shift · crop · time-warp · baseline wander · "
           "lead dropout", mono=True, fill="ink2")
    write(s, "two-tower", theme)


# ─── 3. paired design ────────────────────────────────────────────────────────

def paired_design(theme: str, leads: dict):
    H = 690
    s = Svg(1200, H, "v1 and v2 study design",
            "Timeline diagram. v1 UCLA ICI cohort: ECGs from 4 years before to 30 days after "
            "ICI start, so the window crosses the index; outcome is the CTRCD registry label. "
            "v2 planned open-data replication: baseline window 365 days to 0 days before the "
            "index containing a baseline echo with LVEF at least 50 percent and the last ECG "
            "before the index as model input; index event at t = 0 (drug start, surgery or "
            "admission); outcome window 30 to 365 days after with a later echo showing LVEF "
            "at most 40 percent or a drop of at least 10 points. Data: MIMIC-IV-ECG, MIMIC-IV "
            "hosp and MIMIC-IV-ECHO.", theme)
    s.css = (".fx{animation:fi .7s ease-out both}"
             ".d1{animation-delay:.15s}.d2{animation-delay:.75s}.d3{animation-delay:1.35s}"
             ".d4{animation-delay:1.95s}"
             "@keyframes fi{from{opacity:0}to{opacity:1}}" + REDUCED)
    s.defs.append(
        f'<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><rect width="6" height="6" fill="{s.col("pr")}" '
        f'fill-opacity="0.16"/><line x1="0" y1="0" x2="0" y2="6" stroke="{s.col("pr")}" '
        f'stroke-width="2.2" stroke-opacity="0.75"/></pattern>')

    # time → x: compressed −4 y … −1 y, break, then linear −365 d … +365 d
    A0, A1 = 48, 186                     # −4 y … −1 y
    L0, X0, L1 = 232, 640, 1048          # −365 d, 0, +365 d
    ppd = (X0 - L0) / 365

    def xd(d):
        return X0 + d * ppd

    def xy(y):
        return A0 + (y + 4) / 3 * (A1 - A0)

    BRK = (A1 + L0) / 2
    AXY = 412

    def brk(x, y0, y1, color="ink2"):
        """Two slanted strokes marking an axis break."""
        for dx in (-5, 5):
            s.line(x + dx - 4, y1, x + dx + 4, y0, color, 1.5)

    # shared guides
    for d in (-365, 30, 365):
        s.line(xd(d), 96, xd(d), AXY, "border", 1, dash="2 4")

    # ── v1 row
    s.add('<g class="fx d1">')
    s.text(24, 82, "v1 · UCLA ICI cohort", 17, weight=600)
    s.text(206, 82, "index event = ICI start", fill="ink2")
    by, bh = 112, 40
    s.rect(A0, by, A1 - A0 + 8, bh, "pr", "pr", 1.2, fo=0.16, extra=' rx="3"')
    s.rect(L0 - 8, by, xd(30) - L0 + 8, bh, "pr", "pr", 1.2, fo=0.16, extra=' rx="3"')
    s.rect(BRK - 16, by - 2, 32, bh + 4, "surface")
    brk(BRK, by - 4, by + bh + 4, "pr")
    s.rect(X0, by, xd(30) - X0, bh, "url(#hatch)")
    s.line(xd(30), by, xd(30), by + bh, "pr", 1.2)
    s.text(L0 + 10, by + 26, "ECG window  −4 y … +30 d around ICI start", fill="ink")
    s.path(f"M{n(xd(15))} {by + bh + 4}V{by + bh + 22}H{n(xd(15) + 22)}", "ink2", 1.1)
    s.text(xd(15) + 28, by + bh + 27, "window crosses the index", fill="ink2")
    ox = 770
    s.rect(ox, by, 1176 - ox, bh, "surface", "qt", 1.3, rx=3, dash="5 4")
    s.text(ox + 16, by + 26, "Outcome", fill="qt", weight=600)
    s.text(ox + 88, by + 26, "CTRCD registry label", fill="ink")
    s.add("</g>")

    # ── index line
    s.add('<g class="fx d2">')
    s.line(X0, 50, X0, AXY, "accent", 1.6, dash="6 4")
    s.text(X0, 36, "t = 0 · index event", anchor="middle", fill="accent", weight=600)
    s.add("</g>")

    # ── v2 row
    vy = 252
    s.add('<g class="fx d3">')
    s.text(24, vy - 30, "v2 · open-data replication (planned)", 17, weight=600)
    s.text(326, vy - 30, "index event = drug start · surgery · admission", fill="ink2")
    s.rect(L0, vy, X0 - L0, bh, "pr", "pr", 1.2, fo=0.16, rx=3)
    s.text(L0 + 12, vy + 26, "Baseline window  −365 d … 0 d", fill="ink")
    # (a) echo marker
    ex_ = xd(-150)
    s.add(f'<rect x="{n(ex_ - 6)}" y="{vy + bh / 2 - 6}" width="12" height="12" '
          f'transform="rotate(45 {n(ex_)} {vy + bh / 2})" fill="{s.col("pr")}"/>')
    s.path(f"M{n(ex_)} {vy + bh + 2}V{vy + bh + 22}", "ink2", 1.1)
    s.text(ex_ + 6, vy + bh + 40, "(a) baseline echo · LVEF ≥ 50 %", fill="ink",
           weight=600, anchor="end")
    # (b) last ECG: real median beat glyph
    gx = xd(-24)
    mb = median_beat(leads["II"], 0.25, 0.45)
    gw = 54
    s.rect(gx - gw / 2, vy + 5, gw, bh - 10, "surface", "pr", 1, rx=2)
    s.path(trace_d(mb, gx - gw / 2 + 2, vy + bh - 11, (gw - 4) / 0.7, 18, step=1), "trace",
           1.2, extra=' stroke-linejoin="round"')
    s.path(f"M{n(gx)} {vy + bh + 2}V{vy + bh + 46}", "ink2", 1.1)
    s.text(gx + 6, vy + bh + 64, "(b) last ECG before the index", fill="ink", weight=600,
           anchor="end")
    s.text(gx + 6, vy + bh + 86, "= model input", fill="ink2", anchor="end")
    s.add("</g>")

    s.add('<g class="fx d4">')
    s.rect(xd(30), vy, L1 - xd(30), bh, "qt", "qt", 1.2, fo=0.16, rx=3)
    s.text(xd(30) + 12, vy + 26, "Outcome window  +30 d … +365 d", fill="ink")
    oe = xd(300)
    s.add(f'<rect x="{n(oe - 6)}" y="{vy + bh / 2 - 6}" width="12" height="12" '
          f'transform="rotate(45 {n(oe)} {vy + bh / 2})" fill="{s.col("qt")}"/>')
    s.path(f"M{n(oe)} {vy + bh + 2}V{vy + bh + 22}", "ink2", 1.1)
    s.text(oe - 4, vy + bh + 40, "later echo · LVEF ≤ 40 %", fill="ink", weight=600,
           anchor="end")
    s.text(oe - 4, vy + bh + 62, "or a ≥ 10-point drop", fill="ink2", anchor="end")
    s.add("</g>")

    # ── axis
    s.line(A0, AXY, BRK - 9, AXY, "ink2", 1.2)
    s.line(BRK + 9, AXY, L1 + 20, AXY, "ink2", 1.2)
    brk(BRK, AXY - 8, AXY + 8)
    ticks = [(xy(-4), "−4 y"), (xy(-3), "−3 y"), (xy(-2), "−2 y"), (xd(-365), "−365 d"),
             (xd(-180), "−180 d"), (X0, "0"), (xd(30), "+30 d"), (xd(180), "+180 d"),
             (xd(365), "+365 d")]
    for x, lab in ticks:
        s.line(x, AXY, x, AXY + 6, "ink2", 1.2)
        s.text(x, AXY + 26, lab, mono=True, anchor="middle",
               fill="accent" if lab == "0" else "muted")
    s.text((A0 + A1) / 2, AXY + 50, "compressed", anchor="middle", fill="muted")

    s.text(X0 + 14, AXY + 50, "Everything the model sees is fixed at t = 0.", fill="ink",
           weight=600)
    s.text(X0 + 14, AXY + 72, "Patients without a follow-up echo are excluded,", fill="ink2")
    s.text(X0 + 14, AXY + 94, "not labelled negative.", fill="ink2")

    # ── data panel
    py = AXY + 124
    s.rect(24, py, 1152, H - py - 16, "card", "border", rx=6)
    s.eyebrow(44, py + 34, "v2 data")
    cols = [(170, "MIMIC-IV-ECG", [("800,035", 1), (" ECGs · ", 0), ("161,352", 1),
                                   (" patients", 0)]),
            (520, "MIMIC-IV hosp", [("drug, procedure, diagnosis events", 0)]),
            (860, "MIMIC-IV-ECHO", [("structured LVEF", 0)])]
    for i, (x, name, parts) in enumerate(cols):
        if i:
            s.text(x - 26, py + 46, "+", 18, fill="muted", anchor="middle")
        s.text(x, py + 34, name, mono=True, weight=600)
        spans = "".join(f'<tspan class="{"m" if m else "s"}">{esc(t)}</tspan>' for t, m in parts)
        s.add(f'<text x="{x}" y="{py + 58}" class="s" fill="{s.col("ink2")}">{spans}</text>')
    s.line(44, py + 80, 1156, py + 80, "border")
    s.add(f'<text x="44" y="{py + 108}" class="s" fill="{s.col("ink2")}">'
          f'<tspan font-weight="600" fill="{s.col("ink")}">Fallback endpoint</tspan>'
          f'  new-onset AF after cardiac surgery, predicted from the pre-op ECG</text>')
    write(s, "paired-design", theme)


if __name__ == "__main__":
    leads = load_leads()
    for th in ("light", "dark"):
        record_anatomy(th, leads)
        two_tower(th, leads)
        paired_design(th, leads)
