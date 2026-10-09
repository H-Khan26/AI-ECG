"""Animated README figures: interval delineation sweep and training-time augmentations.

Writes docs/assets/{delineation,augmentations}-{light,dark}.gif.
Run from the repo root:

    .venv/Scripts/python.exe docs/figures/animations.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from style import ASSETS, CANDIDATES, FS, THEMES, load_record  # noqa: E402

SS = 3                      # supersampling factor for antialiased traces
W = 940                     # output width (px)
FRAME_MS = 50               # 20 fps

RECORD = CANDIDATES["NORM"][2]   # PTB-XL #10: every beat yields ordered DWT fiducials
SEED = 32

# ---------------------------------------------------------------- drawing utils

FONT_DIR = Path("C:/Windows/Fonts")


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    files = {"sans": "segoeui.ttf", "semi": "seguisb.ttf", "mono": "consola.ttf"}
    fallback = {"sans": "DejaVuSans.ttf", "semi": "DejaVuSans-Bold.ttf", "mono": "DejaVuSansMono.ttf"}
    path = FONT_DIR / files[kind]
    if not path.exists():
        from matplotlib import font_manager
        path = Path(font_manager.findfont(fallback[kind]))
    return ImageFont.truetype(str(path), size)


def rgb(c) -> tuple:
    if isinstance(c, str):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    if all(isinstance(v, int) for v in c[:3]):
        return tuple(c[:3])
    return tuple(int(round(255 * v)) for v in c[:3])


def blend(fg, bg, a: float) -> tuple:
    """Solid colour of `fg` at opacity `a` over `bg`."""
    f, b = rgb(fg), rgb(bg)
    return tuple(int(round(a * x + (1 - a) * y)) for x, y in zip(f, b))


def palette(c: dict) -> dict:
    """Theme tokens as RGB tuples, with RGBA grid tokens flattened onto the surface."""
    p = {k: rgb(v) for k, v in c.items() if isinstance(v, str)}
    p["grid_minor"] = blend(c["grid_minor"], c["surface"], c["grid_minor"][3])
    p["grid_major"] = blend(c["grid_major"], c["surface"], c["grid_major"][3])
    return p


def polyline(d: ImageDraw.ImageDraw, xs, ys, fill, width: float):
    """Draw a polyline given in output pixels onto a supersampled canvas."""
    if len(xs) < 2:
        return
    pts = [(x * SS, y * SS) for x, y in zip(xs, ys)]
    d.line(pts, fill=fill, width=max(1, round(width * SS)), joint="curve")


def disc(d: ImageDraw.ImageDraw, x, y, r, fill):
    d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], fill=fill)


def text_runs(d: ImageDraw.ImageDraw, x, y, runs, anchor="ls"):
    """Draw consecutive (text, font, colour) runs on one baseline; return end x."""
    for s, f, col in runs:
        d.text((x, y), s, font=f, fill=col, anchor=anchor)
        x += f.getlength(s)
    return x


def runs_width(runs) -> float:
    return sum(f.getlength(s) for s, f, _ in runs)


def write_gif(frames, durations, name: str, theme: str, colors: int = 96):
    """Quantise all frames to one adaptive palette and write a looping GIF."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    picks = sorted({0, len(frames) // 3, len(frames) // 2, 2 * len(frames) // 3, len(frames) - 1})
    sheet = Image.new("RGB", (W, sum(frames[i].height for i in picks)))
    y = 0
    for i in picks:
        sheet.paste(frames[i], (0, y))
        y += frames[i].height
    pal = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    path = ASSETS / f"{name}-{theme}.gif"
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations,
              loop=0, optimize=True, disposal=1)
    return path


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- 1. delineation

T0 = 2.2                    # window start (s); four complete beats follow
WIN_S = 4.0
FEATURED_R = 1160           # R peak (sample) of the annotated beat, 3rd in the window
# At 250 Hz DWT places this beat's QRS onset 5 samples early (1149, on the PR
# segment), so the QRS boundaries were read off the raw lead II trace and are
# pinned here (sample index). DWT's P onset (1111) and T offset (1232, which
# matches the tangent method) hold up on the same zoom and are kept.
FEATURED_FIX = {"qrs_on": 1154, "qrs_off": 1168}
PX_PER_S = 225              # 9 px per 0.04 s box
PX_PER_MV = 90              # 9 px per 0.1 mV box (square boxes)
V_TOP, V_BOT = 1.3, -0.7
FIDS = ["ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets", "ECG_R_Onsets", "R",
        "ECG_R_Offsets", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets"]


def delineate(sig: np.ndarray):
    """NeuroKit2 R-peak detection and DWT delineation of lead II at 250 Hz.

    The signal is band-passed with NeuroKit2's zero-phase 0.67-45 Hz FIR
    ("biosppy") filter before delineation; on this record it gives ordered
    fiducials on every beat. The annotated beat's QRS boundaries are pinned
    (FEATURED_FIX) because DWT can be off by a few samples at 250 Hz.
    With the default ecg_clean used by the archived feature extractor, DWT puts
    the QRS onset near the P peak on 8 of 10 beats here (QRS ~200 ms).
    """
    import neurokit2 as nk

    band = nk.ecg_clean(sig, sampling_rate=FS, method="biosppy")
    _, info = nk.ecg_peaks(band, sampling_rate=FS)
    r = np.asarray(info["ECG_R_Peaks"])
    _, waves = nk.ecg_delineate(band, r, sampling_rate=FS, method="dwt")
    waves["R"] = list(r)
    fid = np.array([np.array(waves[k], float) for k in FIDS]).T      # beats x 9
    ordered = np.array([not np.isnan(b).any() and np.all(np.diff(b) > 0) for b in fid])
    return band, r, fid, ordered


def featured_beat(r, fid) -> np.ndarray:
    """Fiducials of the annotated beat (FIDS order), with the pinned QRS boundaries."""
    b = fid[int(np.argmin(np.abs(r - FEATURED_R)))].copy()
    b[3], b[5] = FEATURED_FIX["qrs_on"], FEATURED_FIX["qrs_off"]
    assert np.all(np.diff(b) > 0), "featured-beat fiducials out of order"
    return b.astype(int)


def interval_stats(r, fb) -> dict:
    """HR from the median RR of the record; PR, QRS, QT and QTc (Bazett, preceding RR)
    of the annotated beat."""
    ms = 1000 / FS
    rr_prev = (fb[4] - r[r < fb[4]][-1]) * ms
    s = {
        "HR": 60000 / (float(np.median(np.diff(r))) * ms),
        "PR": (fb[3] - fb[0]) * ms,
        "QRS": (fb[5] - fb[3]) * ms,
        "QT": (fb[8] - fb[3]) * ms,
    }
    s["QTc"] = s["QT"] / np.sqrt(rr_prev / 1000)
    # Sanity bounds for a normal sinus beat; refuse to print otherwise.
    for k, (lo, hi) in {"HR": (50, 100), "PR": (120, 200), "QRS": (40, 120),
                        "QT": (280, 460), "QTc": (300, 450)}.items():
        assert lo <= s[k] <= hi, f"{k} = {s[k]:.0f} outside plausible range"
    return s


def render_delineation(theme: str, band, r, fid, ordered, fb, stats):
    c = palette(THEMES[theme])
    s0, s1 = int(T0 * FS), int((T0 + WIN_S) * FS)
    x0, y0 = 20, 62
    pw, ph = int(WIN_S * PX_PER_S), int((V_TOP - V_BOT) * PX_PER_MV)
    H = y0 + ph + 34

    def X(s):
        return x0 + (np.asarray(s, float) - s0) * PX_PER_S / FS

    def Y(v):
        return y0 + (V_TOP - np.asarray(v, float)) * PX_PER_MV

    f_title, f_lab = font("semi", 16), font("semi", 12)
    f_num, f_small = font("mono", 13), font("sans", 11)
    f_cap, f_wave = font("sans", 12), font("semi", 12)

    # Beats inside the window and their coloured spans (onset, offset, colour).
    beats = [i for i in range(len(r)) if ordered[i] and fid[i, 0] >= s0 and fid[i, 8] < s1]
    spans = []
    for i in beats:
        b = fb if r[i] == FEATURED_R else fid[i].astype(int)
        spans += [(b[0], b[2], c["pr"]), (b[3], b[5], c["qrs"]), (b[6], b[8], c["qt"])]
    hr_at = r[(r > s0)][1]           # HR is known once a second R peak is on screen

    # Static background: surface, ECG paper and readout card, supersampled.
    bg = Image.new("RGB", (W * SS, H * SS), c["surface"])
    d = ImageDraw.Draw(bg)
    for k in range(0, pw // 9 + 1):
        if k % 5:
            d.rectangle([(x0 + 9 * k) * SS, y0 * SS, (x0 + 9 * k) * SS + SS - 1, (y0 + ph) * SS + SS - 1], fill=c["grid_minor"])
    for k in range(0, ph // 9 + 1):
        if k % 5:
            d.rectangle([x0 * SS, (y0 + 9 * k) * SS, (x0 + pw) * SS + SS - 1, (y0 + 9 * k) * SS + SS - 1], fill=c["grid_minor"])
    for k in range(0, pw // 9 + 1, 5):
        d.rectangle([(x0 + 9 * k) * SS, y0 * SS, (x0 + 9 * k) * SS + SS - 1, (y0 + ph) * SS + SS - 1], fill=c["grid_major"])
    for k in range(0, ph // 9 + 1, 5):
        d.rectangle([x0 * SS, (y0 + 9 * k) * SS, (x0 + pw) * SS + SS - 1, (y0 + 9 * k) * SS + SS - 1], fill=c["grid_major"])

    # Readout layout: fixed-width numeric fields so the line never shifts.
    def readout_runs(vals):
        runs = []
        for j, (lab, key, unit) in enumerate([("HR", "HR", "bpm"), ("PR", "PR", "ms"), ("QRS", "QRS", "ms"),
                                               ("QT", "QT", "ms"), ("QTc", "QTc", "ms")]):
            if j:
                runs.append(("  ·  ", f_small, c["muted"]))
            v = vals.get(key)
            runs += [(lab + " ", f_lab, c["ink2"]),
                     (f"{v:3.0f}" if v is not None else " --", f_num, c["ink"] if v is not None else c["muted"]),
                     (" " + unit, f_small, c["ink2"])]
        return runs

    full = {k: stats[k] for k in ("HR", "PR", "QRS", "QT", "QTc")}
    sublabel = "HR: median RR  ·  intervals: marked beat  ·  QTc by Bazett"
    card_w = int(max(runs_width(readout_runs(full)), f_small.getlength(sublabel))) + 24
    card = (x0 + pw - card_w, 6, x0 + pw, 52)
    d.rounded_rectangle([v * SS for v in card], radius=6 * SS, fill=c["card"], outline=c["border"], width=SS)

    guide = {k: blend(c[k], c["surface"], 0.45) for k in ("pr", "qrs", "qt")}
    cursor_col = blend(c["muted"], c["surface"], 0.55)
    glow = [blend(c["trace"], c["surface"], a) for a in (0.10, 0.22)]

    def frame(head: float, done: bool):
        """Render the sweep with the trace drawn up to sample `head`."""
        im = bg.copy()
        d = ImageDraw.Draw(im)
        n = int(np.floor(head))
        idx = np.arange(s0, n + 1)
        xs, ys = list(X(idx)), list(Y(band[idx]))
        if not done and n < s1:                      # interpolate the head between samples
            frac = head - n
            xs.append(float(X(head)))
            ys.append(float(Y(band[n] + frac * (band[n + 1] - band[n]))))
        polyline(d, xs, ys, c["trace"], 1.6)
        for a, b, col in spans:                      # re-ink each wave once the sweep has passed it
            if head >= b:
                seg = np.arange(a, b + 1)
                polyline(d, X(seg), Y(band[seg]), col, 2.0)
        if not done:
            hx, hy = xs[-1], ys[-1]
            d.rectangle([round(hx) * SS, y0 * SS, round(hx) * SS + SS - 1, (y0 + ph) * SS], fill=cursor_col)
            disc(d, hx, hy, 7, glow[0])
            disc(d, hx, hy, 4.5, glow[1])
            disc(d, hx, hy, 2.6, c["trace"])
        im = im.reduce(SS)
        d = ImageDraw.Draw(im)

        # Header: title and readout (fields fill in as the featured beat is measured).
        d.text((x0, 30), "Finding the intervals of interest", font=f_title, fill=c["ink"], anchor="ls")
        vals = {}
        if head >= hr_at:
            vals["HR"] = stats["HR"]
        if head >= fb[3]:
            vals["PR"] = stats["PR"]
        if head >= fb[5]:
            vals["QRS"] = stats["QRS"]
        if head >= fb[8]:
            vals["QT"], vals["QTc"] = stats["QT"], stats["QTc"]
        text_runs(d, card[0] + 12, 26, readout_runs(vals))
        d.text((card[2] - 12, 44), sublabel,
               font=f_small, fill=c["muted"], anchor="rs")

        # Featured beat: wave labels, brackets and per-beat values.
        def bracket(xa, xb, y, col, up):
            t = 6 if up else -6
            d.line([(xa, y), (xb, y)], fill=col, width=2)
            d.line([(xa, y - 0), (xa, y + t)], fill=col, width=2)
            d.line([(xb, y - 0), (xb, y + t)], fill=col, width=2)

        def guide_line(s, y_to, col):
            xg = round(float(X(s)))
            ya = float(Y(band[s]))
            lo, hi = sorted((ya, y_to))
            for yy in range(int(lo) + 3, int(hi) - 2, 4):
                d.line([(xg, yy), (xg, min(yy + 1, hi))], fill=col, width=1)

        y_pr, y_qt, y_qrs = round(float(Y(-0.40))), round(float(Y(-0.58))), round(float(Y(1.08)))
        if head >= fb[2]:
            d.text((float(X(fb[1])), float(Y(band[fb[1]] + 0.12))), "P", font=f_wave, fill=c["ink2"], anchor="ms")
        if head >= fb[3]:
            xa, xb = round(float(X(fb[0]))), round(float(X(fb[3])))
            guide_line(fb[0], y_pr, guide["pr"])
            guide_line(fb[3], y_pr, guide["pr"])
            bracket(xa, xb, y_pr, c["pr"], up=False)
            runs = [("PR ", f_lab, c["ink"]), (f"{stats['PR']:.0f} ms", f_num, c["ink"])]
            text_runs(d, xa - 8 - runs_width(runs), y_pr + 4, runs)
        if head >= fb[5]:
            xa, xb = round(float(X(fb[3]))), round(float(X(fb[5])))
            bracket(xa, xb, y_qrs, c["qrs"], up=True)
            runs = [("QRS ", f_lab, c["ink"]), (f"{stats['QRS']:.0f} ms", f_num, c["ink"])]
            text_runs(d, (xa + xb) / 2 - runs_width(runs) / 2, y_qrs - 6, runs)
        if head >= fb[8]:
            d.text((float(X(fb[7])), float(Y(band[fb[7]] + 0.12))), "T", font=f_wave, fill=c["ink2"], anchor="ms")
            xa, xb = round(float(X(fb[3]))), round(float(X(fb[8])))
            guide_line(fb[8], y_qt, guide["qt"])
            bracket(xa, xb, y_qt, c["qt"], up=False)
            runs = [("QT ", f_lab, c["ink"]), (f"{stats['QT']:.0f} ms", f_num, c["ink"])]
            text_runs(d, xb + 8, y_qt + 4, runs)

        # Footer caption and paper scale.
        d.text((x0, H - 12), f"Lead II  ·  PTB-XL #{RECORD}  ·  NeuroKit2 DWT fiducials, boundaries verified by eye",
               font=f_cap, fill=c["muted"], anchor="ls")
        d.text((x0 + pw, H - 12), "25 mm/s  ·  10 mm/mV", font=f_cap, fill=c["muted"], anchor="rs")
        return im

    n_sweep = int(WIN_S * 1000 / FRAME_MS)
    frames, durations = [frame(s0, False)], [500]
    for k in range(1, n_sweep + 1):
        frames.append(frame(s0 + k * (s1 - s0 - 1) / n_sweep, False))
        durations.append(FRAME_MS)
    frames.append(frame(s1 - 1, True))
    durations.append(2000)
    return frames, durations


# ---------------------------------------------------------------- 2. augmentations

# Each transform reproduces one step of archive/ctrcd-ici-study/ecg_augmentations.py
# in isolation, with the archived parameters.
NOISE_STD = 0.01            # mV
MAX_SHIFT = int(0.2 * FS)   # samples
CROP = int(2.0 * FS)        # samples
WARP_PCT = 0.05
WANDER_AMP = 0.1            # mV
DROP_P = 0.1
N = 2500


def augment_draws(sig: np.ndarray, rng: np.random.Generator):
    """Return [(chip name, chip parameter, draw note, fn)] for lead II, where fn(a)
    blends the original (a = 0) into this draw (a = 1) by scaling the drawn parameter."""
    t = np.arange(N)
    out = []

    noise = rng.standard_normal(N) * NOISE_STD
    out.append(("Gaussian noise", "σ 0.01 mV", "subtle at true scale; inset magnified 5×",
                lambda a: sig + a * noise))

    shift = int(rng.integers(-MAX_SHIFT, MAX_SHIFT))
    out.append(("Time shift", "±0.2 s (circular)", f"this draw: {shift / FS:+.2f} s",
                lambda a: np.interp(t - a * shift, t, sig, period=N)))    # a = 1: np.roll

    start = int(rng.integers(0, N - CROP + 1))
    lo_end, hi_end = start, start + CROP           # the window closes in from both ends
    out.append(("Random crop", "2 s window, zero-padded",
                f"this draw: {start / FS:.2f}–{(start + CROP) / FS:.2f} s kept",
                lambda a: np.where((t >= a * lo_end) & (t < N - a * (N - hi_end)), sig, 0.0)))

    warp = 1.0 + rng.uniform(-WARP_PCT, WARP_PCT)
    out.append(("Time warp", "±5 %", f"this draw: {100 * (warp - 1):+.1f} %",
                lambda a: np.interp(np.clip(t * (1 + a * (warp - 1)), 0, N - 1), t, sig)))

    freq = rng.uniform(0.1, 0.5)
    drift = WANDER_AMP * np.sin(2 * np.pi * freq * t / FS)
    out.append(("Baseline wander", "0.1–0.5 Hz, 0.1 mV", f"this draw: {freq:.2f} Hz",
                lambda a: sig + a * drift))
    return out


def render_augmentations(theme: str, leads: dict, draws):
    c = palette(THEMES[theme])
    H = 304
    x0, pw = 44, 880
    py0, ph = 84, 176
    px_s = pw / 10
    f_title, f_chip, f_num = font("semi", 16), font("semi", 13), font("mono", 13)
    f_small, f_cap, f_lead = font("sans", 11), font("sans", 12), font("semi", 12)
    f_tick = font("mono", 11)
    ghost = blend(c["muted"], c["surface"], 0.45)
    t = np.arange(N)
    xs = x0 + t * px_s / FS

    # Single strip (lead II) and three-lead stack (I, II, V1) share one plot area.
    single = [("II", 1.2, -0.4)]
    stack = [("I", 0.55, -0.2), ("II", 1.05, -0.3), ("V1", 0.3, -0.95)]
    gap = 8

    def rows(spec):
        total = sum(top - bot for _, top, bot in spec)
        scale = (ph - gap * (len(spec) - 1)) / total
        out, y = [], py0
        for name, top, bot in spec:
            out.append((name, y, top, bot, scale))
            y += (top - bot) * scale + gap
        return out

    # Noise inset: 5× magnification of a 0.4 s stretch of the T-P segment.
    ins_s0, ins_s1 = int(4.0 * FS), int(4.4 * FS)
    ins_box = (700, 8, 920, 74)
    zoom = 5

    def base():
        im = Image.new("RGB", (W * SS, H * SS), c["surface"])
        d = ImageDraw.Draw(im)
        for k in range(11):
            xk = round(x0 + k * px_s)
            d.rectangle([xk * SS, py0 * SS, xk * SS + SS - 1, (py0 + ph) * SS], fill=c["grid_minor"])
        return im

    bg = base()

    def strip(d, spec, sigs, aug_sigs):
        for (name, ytop, top, bot, scale), sig, aug in zip(rows(spec), sigs, aug_sigs):
            def Y(v):
                return ytop + (top - np.clip(v, bot, top)) * scale
            yb = float(Y(0.0))
            d.rectangle([x0 * SS, round(yb) * SS, (x0 + pw) * SS, round(yb) * SS + SS - 1], fill=c["grid_minor"])
            polyline(d, xs, Y(sig), ghost, 1.3)
            polyline(d, xs, Y(aug), c["trace"], 1.5)

    def frame(k: int, alpha: float, layout: str):
        """k: active augmentation, alpha: morph amount, layout: 'single' or 'stack'."""
        im = bg.copy()
        d = ImageDraw.Draw(im)
        sig2 = leads["II"]
        if layout == "single":
            strip(d, single, [sig2], [draws[k][3](alpha) if k < 5 else sig2])
        else:
            sigs = [leads[n] for n in ("I", "II", "V1")]
            strip(d, stack, sigs, [sigs[0], sigs[1], (1 - alpha) * sigs[2]])
        im = im.reduce(SS)
        d = ImageDraw.Draw(im)

        spec = single if layout == "single" else stack
        for name, ytop, top, _, scale in rows(spec):
            d.text((x0 - 10, ytop + top * scale + 4), name, font=f_lead, fill=c["ink2"], anchor="rs")
        for k_s in range(0, 11, 2):
            lab = f"{k_s} s" if k_s == 10 else str(k_s)
            d.text((x0 + k_s * px_s, py0 + ph + 16), lab, font=f_tick, fill=c["muted"],
                   anchor="rs" if k_s == 10 else "ms")

        d.text((20, 26), "Training-time augmentation", font=f_title, fill=c["ink"], anchor="ls")
        if k == 5:
            name, param, note = "Lead dropout", f"p = {DROP_P} per lead", "example draw: V1 zeroed"
        else:
            name, param, note = draws[k][:3]
        runs = [(name, f_chip, c["ink"]), ("  ·  ", f_small, c["muted"]), (param, f_num, c["ink2"])]
        cw = runs_width(runs) + 22
        d.rounded_rectangle([20, 38, 20 + cw, 66], radius=6, fill=c["card"], outline=c["border"])
        text_runs(d, 31, 57, runs)
        d.text((20 + cw + 12, 57), note, font=f_small, fill=c["muted"], anchor="ls")

        if k == 0 and layout == "single":
            # Inset: same samples, magnified 5× in time and amplitude.
            bx0, by0, bx1, by1 = ins_box
            d.rounded_rectangle(ins_box, radius=6, fill=c["surface"], outline=c["border"])
            seg = np.arange(ins_s0, ins_s1)
            mid = float(np.median(sig2[seg]))
            sx = bx0 + 8 + (seg - ins_s0) * (bx1 - bx0 - 16) / (ins_s1 - ins_s0 - 1)
            ym = (by0 + by1) / 2 + 6
            inset_scale = zoom * ph / 1.6
            sub = Image.new("RGB", ((bx1 - bx0) * SS, (by1 - by0) * SS), c["surface"])
            sd = ImageDraw.Draw(sub)
            polyline(sd, sx - bx0, ym - by0 - (sig2[seg] - mid) * inset_scale, ghost, 1.3)
            aug = draws[0][3](alpha)
            polyline(sd, sx - bx0, ym - by0 - (aug[seg] - mid) * inset_scale, c["trace"], 1.4)
            sub = sub.reduce(SS)
            im.paste(sub.crop((2, 16, sub.width - 2, sub.height - 2)), (bx0 + 2, by0 + 16))
            d.text((bx0 + 9, by0 + 14), "5× zoom of boxed region", font=f_small, fill=c["muted"], anchor="ls")
            # Source box on the strip.
            name, ytop, top, _, scale = rows(single)[0]
            yc = ytop + (top - mid) * scale
            hh = (by1 - by0 - 18) / zoom / 2
            d.rectangle([float(xs[ins_s0]), yc - hh, float(xs[ins_s1]), yc + hh], outline=c["ink2"])

        # Footer: legend, source and progress dots.
        y = H - 12
        d.line([(20, y - 4), (40, y - 4)], fill=ghost, width=2)
        xx = text_runs(d, 46, y, [("original", f_cap, c["muted"])])
        d.line([(xx + 16, y - 4), (xx + 36, y - 4)], fill=c["trace"], width=2)
        xx = text_runs(d, xx + 42, y, [("augmented", f_cap, c["muted"])])
        d.text((xx + 16, y), f"·   PTB-XL #{RECORD}  ·  seeded example draws", font=f_cap, fill=c["muted"], anchor="ls")
        for j in range(6):
            cx, cy = W - 20 - (5 - j) * 16 - 4, y - 4
            if j == k:
                d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=c["ink"])
            else:
                d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=c["muted"], width=1)
        return im

    frames, durations = [], []
    n_in, n_out = 6, 4
    for k in range(6):                               # layouts switch with a clean cut
        layout = "stack" if k == 5 else "single"
        frames.append(frame(k, 0.0, layout))
        durations.append(150)
        for i in range(1, n_in + 1):
            frames.append(frame(k, smooth(i / n_in), layout))
            durations.append(FRAME_MS)
        durations[-1] = 1000
        for i in range(1, n_out + 1):
            frames.append(frame(k, smooth(1 - i / n_out), layout))
            durations.append(FRAME_MS)
    return frames, durations


# ---------------------------------------------------------------- main

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    leads = load_record(RECORD, leads=["I", "II", "V1"])

    band, r, fid, ordered = delineate(leads["II"])
    fb = featured_beat(r, fid)
    stats = interval_stats(r, fb)
    for theme in THEMES:
        frames, durations = render_delineation(theme, band, r, fid, ordered, fb, stats)
        path = write_gif(frames, durations, "delineation", theme)
        print(f"{path.name}: {len(frames)} frames, {sum(durations) / 1000:.1f} s, "
              f"{path.stat().st_size / 1024:.0f} KB")
    print("printed values:", {k: round(v) for k, v in stats.items()})

    draws = augment_draws(leads["II"], np.random.default_rng(SEED))
    for theme in THEMES:
        frames, durations = render_augmentations(theme, leads, draws)
        path = write_gif(frames, durations, "augmentations", theme)
        print(f"{path.name}: {len(frames)} frames, {sum(durations) / 1000:.1f} s, "
              f"{path.stat().st_size / 1024:.0f} KB")
    print("augmentation draws:", [d[2] for d in draws])


if __name__ == "__main__":
    main()
