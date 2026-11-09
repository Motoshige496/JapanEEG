"""Plot synchronized speech, EEG and facial EMG around one utterance (as Figure 1e).

Top: the voice-masked speech waveform (second microphone monitor channel of the EDF, on the EEG
time line) with the aligned words (red dashed boundaries, text) and phones (grey dashed
boundaries, labels). Middle: scalp EEG channels (high-pass 1 Hz). Bottom: facial EMG. Speech
events are shaded. Use a continuous overt-speech run with voice-masked audio (readings of
public-domain texts).

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

from japaneeg import bids  # noqa: E402
from japaneeg.emg import EMG_TYPES  # noqa: E402

plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "IPAexGothic", "Yu Gothic", "MS Gothic", "DejaVu Sans"]


def main() -> None:
    """Draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf", required=True, help="continuous overt-speech run")
    ap.add_argument("--event", type=int, default=0, help="index of the overt event in the run")
    ap.add_argument("--pre", type=float, default=0.2, help="seconds before the event onset")
    ap.add_argument("--post", type=float, default=0.5, help="seconds after the event end")
    ap.add_argument("--eeg-rows", type=int, nargs="*", default=[0, 16, 32, 64, 96], help="scalp EEG channel indices")
    ap.add_argument("--out", default="figures/fig1e_example.png")
    args = ap.parse_args()

    edf = Path(args.edf)
    rows = bids.read_tsv(bids.sidecar(edf, "events.tsv"))
    channels = bids.read_tsv(bids.sidecar(edf, "channels.tsv"))
    overt = sorted((float(r["onset"]), float(r["duration"])) for r in rows if r.get("trial_type") == "overt")
    onset, dur = overt[args.event]
    t0, t1 = onset - args.pre, onset + dur + args.post

    raw = bids.load_raw(edf, preload=False, annotate=False)
    sf = raw.info["sfreq"]
    a, b = int(t0 * sf), int(t1 * sf)
    eeg = bids.scalp_eeg(channels)
    eeg_rows = [eeg[i] for i in args.eeg_rows if i < len(eeg)]
    mic = bids.microphone_channels(channels)
    emg = [c["name"] for c in channels if c.get("type") in EMG_TYPES]
    x = raw.get_data(picks=eeg_rows + mic[1:2] + emg, start=a, stop=b)
    t = np.arange(x.shape[1]) / sf - (onset - t0)

    from scipy.signal import butter, sosfiltfilt

    hp = butter(2, 1.0, "highpass", fs=sf, output="sos")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 2, 1]})
    for ax in axes:
        for on, d in overt:
            if on + d > t0 and on < t1:
                ax.axvspan(on - onset, on + d - onset, color="tab:orange", alpha=0.08, lw=0)
    speech = x[len(eeg_rows)] if mic[1:2] else np.zeros(x.shape[1])
    axes[0].plot(t, speech, lw=0.4, color="k")
    for r in rows:
        kind = r.get("trial_type")
        if kind not in ("word", "phone"):
            continue
        on = float(r["onset"]) - onset
        if not (-args.pre <= on <= dur + args.post):
            continue
        style = {"word": ("red", "--", 0.9), "phone": ("0.6", ":", 0.7)}[kind]
        axes[0].axvline(on, color=style[0], ls=style[1], lw=style[2])
        y = 0.95 if kind == "word" else 0.05
        axes[0].text(on, y, r.get("value", ""), transform=axes[0].get_xaxis_transform(), fontsize=7 if kind == "word" else 5,
                     color=style[0], va="top" if kind == "word" else "bottom")  # fmt: skip
    axes[0].set_ylabel("speech\n(voice-masked)")
    sig = sosfiltfilt(hp, x[: len(eeg_rows)], axis=1) * 1e6
    step = 4 * np.median(np.std(sig, axis=1)) if len(sig) else 1.0
    for k, (name, s) in enumerate(zip(eeg_rows, sig)):
        axes[1].plot(t, s - k * step, lw=0.4, label=name)
    axes[1].legend(fontsize=6, ncol=len(eeg_rows), loc="upper right")
    axes[1].set_ylabel("EEG (µV, offset)")
    if emg:
        e = sosfiltfilt(hp, x[len(eeg_rows) + len(mic[1:2]) :], axis=1) * 1e6
        for name, s in zip(emg, e):
            axes[2].plot(t, s, lw=0.4, label=name)
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
