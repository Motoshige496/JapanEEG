"""Numbers of the Data Descriptor computed from the released files.

Subcommands (each writes a TSV to ``--out`` and prints a summary):

``tables``        Tables 4 and 5: events, event hours, recording hours, duration statistics.
``articulation``  articulation rate per participant from the released phone intervals
                  (``--morae`` adds morae per second; needs ``pip install pyopenjtalk``).
``onset``         latency of the sound after the speech-event onset (overt speech after >= 0.5 s
                  of silence), on the voice-masked speech channel with the envelope of the paper.
``speech-pause``  60-118 Hz slopes during speech and pauses with the recording reference and the
                  common average reference, per file; Wilcoxon signed-rank test speech vs pause.
``coherence``     EEG-EMG coherence (60-118 Hz) with the recording reference and the common
                  average reference, with chance levels, per file.
``bad-channels``  bad channels per continuous run by device, and the share of speech and task
                  events overlapping a ``BAD_high_amplitude`` stretch.

Exact or approximate: Tables 4-5, the articulation rate and the share of events overlapping
``BAD_high_amplitude`` use only the released events and EDF headers and reproduce the paper's
values. The bad-channel counts cover the runs released as continuous runs (the segments of a
``ses-shuffled`` file carry the flags of their source run, but not the run). The paper computed
the onset latency on the original microphone signal and the slopes and coherence on the original
runs. Here the voice-masked speech channel and the released pieces are used (a continuous run as
one piece, the segments of a ``ses-shuffled`` file pooled per file), so these values are close
but not identical.

Usage
-----
uv run python scripts/paper_numbers.py tables --bids-root data/JapanEEG/bids
uv run python scripts/paper_numbers.py articulation --bids-root data/JapanEEG/bids --morae
uv run python scripts/paper_numbers.py onset --bids-root data/JapanEEG/bids --subject 01 --acq pangolin
uv run python scripts/paper_numbers.py speech-pause --bids-root data/JapanEEG/bids --subject 02
uv run python scripts/paper_numbers.py coherence --bids-root data/JapanEEG/bids --subject 02
uv run python scripts/paper_numbers.py bad-channels --bids-root data/JapanEEG/bids
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from japaneeg import bids, emg, erp, speech, tables


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows as TSV (nothing if empty)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def median_iqr(values: list[float]) -> str:
    """``median (q25-q75)``."""
    q25, med, q75 = np.nanpercentile(np.asarray(values, np.float64), [25, 50, 75])
    return f"{med:.2f} ({q25:.2f}-{q75:.2f})"


def cmd_tables(args: argparse.Namespace) -> None:
    """Tables 4 and 5."""
    rows, hours, dates = tables.count(args.bids_root)
    t4, t5 = tables.table4(rows, hours, dates), tables.table5(rows)
    write_tsv(args.out / "table4.tsv", t4)
    write_tsv(args.out / "table5.tsv", t5)
    for t in (t4, t5):
        if t:
            print("\t".join(t[0]))
            for r in t:
                print("\t".join(v or "-" for v in r.values()))
            print()
    print(f"total recording hours: {sum(hours.values()):.1f}")
    print("days_continuous and period_continuous count the sessions released as continuous runs only")


def cmd_articulation(args: argparse.Namespace) -> None:
    """Articulation rate per participant."""
    files = sorted(Path(args.bids_root).rglob("*task-speechopen*_events.tsv"))
    rows: list[dict[str, str]] = []
    if args.morae:
        speech.count_morae("あ")  # pyopenjtalk downloads its dictionary once here, not in every worker
    with Pool(args.n_workers) as pool:
        for part in pool.imap_unordered(partial(speech.utterance_rates, morae=args.morae), files, chunksize=4):
            rows.extend(part)
    write_tsv(args.out / "utterance_rates.tsv", rows)
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[r["participant"]].append(r)
    for pid in sorted(by):
        sub = [r for r in by[pid] if r["mora_rate"]] if args.morae else by[pid]
        line = f"{pid}: {len(sub)} utterances, phones/s {median_iqr([float(r['rate']) for r in sub])}"
        if args.morae and sub:
            line += f", morae/s {median_iqr([float(r['mora_rate']) for r in sub])}"
        print(line)


def _onsets(edf: str, min_silence: float) -> list[float]:
    """Acoustic onset latencies (ms) of the selected overt events of one file (worker)."""
    path = Path(edf)
    channels = bids.read_tsv(bids.sidecar(path, "channels.tsv"))
    mic = bids.microphone_channels(channels)
    if not mic:
        return []
    channel = mic[1] if len(mic) > 1 else mic[0]  # voice-masked speech; else the released envelope
    raw = bids.load_raw(path, annotate=False)
    sf = float(raw.info["sfreq"])
    out: list[float] = []
    for p in emg.pieces(path):
        evs = [erp.Event(a, b - a) for a, b in emg.spans_of(p.rows, ("overt",))]
        for on in erp.select_onsets(evs, p.start, min_silence):
            if on - 0.5 < p.start or on + 0.5 > p.stop:
                continue
            a = int(max(p.start, on - 1.0) * sf)  # extra signal before the baseline for the filter
            x = raw.get_data(picks=[channel], start=a, stop=int((on + 0.5) * sf))[0]
            if not np.any(x):
                continue  # no voice-masked audio here
            env = speech.sound_envelope(x, sf) if len(mic) > 1 else x
            lat = speech.onset_latency(env, sf, on - a / sf)
            if np.isfinite(lat):
                out.append(1000.0 * lat)
    return out


def cmd_onset(args: argparse.Namespace) -> None:
    """Acoustic onset after the speech-event onset."""
    files = [str(f) for f in bids.find_runs(args.bids_root, args.subject, "speechopen", args.acq)]
    lat: list[float] = []
    with Pool(args.n_workers) as pool:
        for part in pool.imap_unordered(partial(_onsets, min_silence=0.5), files):
            lat.extend(part)
    write_tsv(args.out / "acoustic_onsets.tsv", [{"latency_ms": f"{v:.1f}"} for v in lat])
    print(f"{len(lat)} utterances, acoustic onset after the event onset: median (IQR) {median_iqr(lat)} ms")


def _speech_pause(edf: str) -> dict[str, str] | None:
    """Slopes of one file (worker)."""
    freqs, power = emg.speech_pause_power(Path(edf))
    if min(power["speech"].n, power["pause"].n) < 20:
        return None
    row = {"file": Path(edf).name, "speech_windows": str(power["speech"].n), "pause_windows": str(power["pause"].n)}
    for cls in ("speech", "pause"):
        for ref, v in emg.class_slopes(freqs, power[cls]).items():
            row[f"{cls}_{ref}"] = f"{v:.2f}"
    return row


def cmd_speech_pause(args: argparse.Namespace) -> None:
    """Speech vs pause slopes."""
    from scipy.stats import wilcoxon

    files = [str(f) for f in bids.find_runs(args.bids_root, args.subject, "speechopen", args.acq)]
    rows = []
    with Pool(args.n_workers) as pool:
        for r in pool.imap_unordered(_speech_pause, files):
            if r is not None:
                rows.append(r)
    write_tsv(args.out / f"speech_pause_sub-{args.subject}_{args.acq}.tsv", rows)
    if not rows:
        print("no file with enough speech and pause windows")
        return
    for ref in ("recording", "car"):
        s = np.array([float(r[f"speech_{ref}"]) for r in rows])
        p = np.array([float(r[f"pause_{ref}"]) for r in rows])
        stat = wilcoxon(s, p) if len(rows) > 5 else None
        print(f"{ref}: {len(rows)} files, median slope speech {np.median(s):+.1f}, pause {np.median(p):+.1f} dB/decade; "
              f"speech higher in {int((s > p).sum())} of {len(rows)}"
              + (f"; Wilcoxon p = {stat.pvalue:.2g}" if stat is not None else ""))  # fmt: skip


def _coherence(edf: str) -> dict[str, str] | None:
    """Coherence of one file (worker)."""
    c = emg.coherence(Path(edf))
    return {"file": Path(edf).name, **{k: f"{v:.4f}" for k, v in c.items()}} if c else None


def cmd_coherence(args: argparse.Namespace) -> None:
    """EEG-EMG coherence."""
    files = [str(f) for f in bids.find_runs(args.bids_root, args.subject, "speechopen", args.acq)]
    rows = []
    with Pool(args.n_workers) as pool:
        for r in pool.imap_unordered(_coherence, files):
            if r is not None:
                rows.append(r)
    write_tsv(args.out / f"coherence_sub-{args.subject}_{args.acq}.tsv", rows)
    for key in ("recording", "recording_chance", "car", "car_chance"):
        print(f"{key}: median over {len(rows)} files {np.median([float(r[key]) for r in rows]):.4f}")


def cmd_bad_channels(args: argparse.Namespace) -> None:
    """Bad channels per continuous run and events overlapping BAD_high_amplitude."""
    per_device: dict[str, list[int]] = defaultdict(list)
    n_events = n_overlap = 0
    for edf in bids.find_runs(args.bids_root):
        rows = bids.read_tsv(bids.sidecar(edf, "events.tsv"))
        if "_ses-shuffled_" not in edf.name:
            channels = bids.read_tsv(bids.sidecar(edf, "channels.tsv"))
            scalp = set(bids.scalp_eeg(channels))
            acq = edf.name.split("_acq-")[1].split("_")[0]
            per_device[acq].append(sum(1 for c in channels if c["name"] in scalp and c.get("status") == "bad"))
        bad = [(a, b - a) for a, b in emg.spans_of(rows, emg.MARKS)]
        for a, b in emg.spans_of(rows, tables.EVENT_TYPES):
            n_events += 1
            n_overlap += erp.overlaps(a, b, bad)
    out = []
    for acq, counts in sorted(per_device.items()):
        c = np.asarray(counts)
        q25, med, q75 = np.percentile(c, [25, 50, 75])
        out.append({"device": acq, "continuous_runs": str(c.size), "median_bad": f"{med:g}",
                    "iqr": f"{q25:g}-{q75:g}", "pct_ten_or_more": f"{100 * np.mean(c >= 10):.0f}"})  # fmt: skip
        print("  ".join(f"{k}={v}" for k, v in out[-1].items()))
    write_tsv(args.out / "bad_channels.tsv", out)
    print(f"{n_overlap} of {n_events} events ({100 * n_overlap / max(n_events, 1):.1f}%) overlap BAD_high_amplitude")


def main() -> None:
    """Parse the subcommand and run it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["tables", "articulation", "onset", "speech-pause", "coherence", "bad-channels"])
    ap.add_argument("--bids-root", required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/paper_numbers"))
    ap.add_argument("--subject", default="01")
    ap.add_argument("--acq", default="pangolin")
    ap.add_argument("--morae", action="store_true")
    ap.add_argument("--n-workers", type=int, default=-1, help="-1: all CPUs but one")
    args = ap.parse_args()
    args.n_workers = max(1, (os.cpu_count() or 2) - 1) if args.n_workers < 0 else args.n_workers
    {"tables": cmd_tables, "articulation": cmd_articulation, "onset": cmd_onset,
     "speech-pause": cmd_speech_pause, "coherence": cmd_coherence, "bad-channels": cmd_bad_channels}[args.command](args)  # fmt: skip


if __name__ == "__main__":
    main()
