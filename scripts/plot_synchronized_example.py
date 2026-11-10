"""Plot synchronized speech, EEG and facial EMG around one utterance (as Figure 1e).

Top: the voice-masked speech waveform (second microphone monitor channel of the EDF, on the EEG
time line) with the aligned words (red dashed boundaries, text) and phones (grey dotted
boundaries, labels). Middle: good scalp EEG channels, band-passed 1-40 Hz for display. Bottom:
facial EMG, bipolar (positive minus negative input) and high-passed at 20 Hz. Speech events are
shaded. Use a continuous overt-speech run with voice-masked audio (readings of public-domain
texts).

Usage
-----
uv run python scripts/plot_synchronized_example.py --edf <run>_eeg.edf --event 10 --out figures/fig1e_example.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from japaneeg import bids  # noqa: E402
from japaneeg.emg import EMG_TYPES  # noqa: E402

CJK_FONTS = ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Yu Gothic", "MS Gothic", "Hiragino Sans", "Noto Sans JP")
PHONE_FONT = "DejaVu Sans"  # bundled with matplotlib; has the IPA symbols of the phone labels


def installed(names: tuple[str, ...]) -> list[str]:
    """The installed font families among ``names`` (in that order), then DejaVu Sans."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in names if n in have] + [PHONE_FONT]


def bipolar_emg(channels: list[dict[str, str]]) -> list[tuple[str, str, str | None]]:
    """EMG channels as (label, positive input, negative input or None), in file order."""
    emg = [c for c in channels if c.get("type") in EMG_TYPES]
    pos = [c for c in emg if "positive" in (c.get("description") or "").lower()]
    neg = [c for c in emg if "negative" in (c.get("description") or "").lower()]
    if pos and len(pos) == len(neg):
        return [
            ((p.get("description") or p["name"]).split(" electrode")[0], p["name"], n["name"]) for p, n in zip(pos, neg)
        ]
    return [(c["name"], c["name"], None) for c in emg]


def main() -> None:
    """Draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf", required=True, help="continuous overt-speech run")
    ap.add_argument("--event", type=int, default=0, help="index of the overt event in the run")
    ap.add_argument("--pre", type=float, default=0.2, help="seconds before the event onset")
    ap.add_argument("--post", type=float, default=0.5, help="seconds after the event end")
    ap.add_argument(
        "--eeg-rows", type=int, nargs="*", default=[0, 16, 32, 64, 96], help="indices among the good scalp EEG channels"
    )
    ap.add_argument("--out", default="figures/fig1e_example.png")
    args = ap.parse_args()
    plt.rcParams["font.family"] = installed(CJK_FONTS)

    edf = Path(args.edf)
    rows = bids.read_tsv(bids.sidecar(edf, "events.tsv"))
    channels = bids.read_tsv(bids.sidecar(edf, "channels.tsv"))
    overt = sorted((float(r["onset"]), float(r["duration"])) for r in rows if r.get("trial_type") == "overt")
    onset, dur = overt[args.event]
    t0, t1 = onset - args.pre, onset + dur + args.post

    raw = bids.load_raw(edf, preload=False, annotate=False)
    sf = float(raw.info["sfreq"])
    a, b = int(t0 * sf), int(t1 * sf)
    bad = {c["name"] for c in channels if c.get("status") == "bad"}
    good = [c for c in bids.scalp_eeg(channels) if c not in bad]
    eeg_rows = [good[i] for i in args.eeg_rows if i < len(good)]
    mic = bids.microphone_channels(channels)
    emg = bipolar_emg(channels)
    emg_names = sorted({n for _, p, q in emg for n in (p, q) if n})
    names = eeg_rows + mic[1:2] + emg_names
    pad = int(2 * sf)  # filter on a longer stretch, then cut (no edge effects in the plot)
    lo = max(0, a - pad)
    x = raw.get_data(picks=names, start=lo, stop=min(raw.n_times, b + pad))
    data = dict(zip(names, x))
    cut = slice(a - lo, a - lo + (b - a))
    t = np.arange(b - a) / sf - (onset - t0)

    from scipy.signal import butter, sosfiltfilt

    band = butter(4, [1.0, 40.0], "bandpass", fs=sf, output="sos")
    hp_emg = butter(4, 20.0, "highpass", fs=sf, output="sos")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 2, 1]})
    for ax in axes:
        for on, d in overt:
            if on + d > t0 and on < t1:
                ax.axvspan(on - onset, on + d - onset, color="tab:orange", alpha=0.08, lw=0)
    speech = data[mic[1]][cut] if mic[1:2] else np.zeros(b - a)
    axes[0].plot(t, speech, lw=0.4, color="k")
    for r in rows:
        kind = r.get("trial_type")
        if kind not in ("word", "phone"):
            continue
        on = float(r["onset"]) - onset
        if not (-args.pre <= on <= dur + args.post):
            continue
        color, ls, lw = {"word": ("red", "--", 0.9), "phone": ("0.6", ":", 0.7)}[kind]
        axes[0].axvline(on, color=color, ls=ls, lw=lw)
        axes[0].text(on, 0.95 if kind == "word" else 0.05, r.get("value", ""), transform=axes[0].get_xaxis_transform(),
                     fontsize=7 if kind == "word" else 5, color=color, va="top" if kind == "word" else "bottom",
                     fontfamily=None if kind == "word" else PHONE_FONT)  # fmt: skip
    axes[0].set_ylabel("speech\n(voice-masked)")
    sig = np.array([sosfiltfilt(band, data[n])[cut] for n in eeg_rows]) * 1e6
    step = 5 * np.median(np.std(sig, axis=1)) if len(sig) else 1.0
    for k, (name, s) in enumerate(zip(eeg_rows, sig)):
        axes[1].plot(t, s - k * step, lw=0.4, label=name)
    axes[1].legend(fontsize=6, ncol=max(1, len(eeg_rows)), loc="upper right")
    axes[1].set_ylabel("EEG 1-40 Hz (µV, offset)")
    for label, p, q in emg:
        s = data[p] - data[q] if q else data[p]
        axes[2].plot(t, sosfiltfilt(hp_emg, s)[cut] * 1e6, lw=0.4, label=label)
    if emg:
        axes[2].legend(fontsize=6, ncol=len(emg), loc="upper right")
    axes[2].set_ylabel("EMG (µV)")
    axes[2].set_xlabel("Time from event onset (s)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
