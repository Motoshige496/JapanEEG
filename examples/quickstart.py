"""Usage examples: load a run with MNE, cut a ses-shuffled segment, read utterances.

Run after downloading at least one session and the sidecars, for example::

    uv run japaneeg-download --out data/JapanEEG --include "sidecars/*" "archives/sub-01/sub-01_ses-20230829*"
    uv run python examples/quickstart.py --bids-root data/JapanEEG/bids
"""

from __future__ import annotations

import argparse

import mne

from japaneeg import bids


def main() -> None:
    """Print what a few typical calls return."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids-root", default="data/JapanEEG/bids")
    args = ap.parse_args()

    # 1. Continuous runs: one EDF per recording run, with channel types, bad channels and events
    runs = [r for r in bids.find_runs(args.bids_root, subject="01", task="speechopen") if "ses-shuffled" not in r.name]
    print(f"{len(runs)} continuous overt-speech runs of sub-01 found")
    if runs:
        raw = bids.load_raw(runs[0])
        print(raw)
        print("bad channels:", raw.info["bads"])
        print("event types:", sorted(set(raw.annotations.description)))
        # epochs around speech onsets; MNE skips epochs overlapping BAD_high_amplitude stretches
        events, ids = mne.events_from_annotations(raw, event_id={"overt": 1}, verbose="error")
        raw.load_data().pick("eeg", exclude="bads").filter(1.0, 40.0, verbose="error")
        epochs = mne.Epochs(raw, events, ids, tmin=-0.3, tmax=0.8, baseline=(-0.3, -0.2), verbose="error")
        print(epochs)

    # 2. ses-shuffled files: cut each segment before filtering it
    shuffled = bids.find_runs(args.bids_root, subject="01", session="shuffled")
    if shuffled:
        segs = bids.segments(shuffled[0])
        print(f"{shuffled[0].name}: {len(segs)} segments")
        s = segs[0]
        raw = bids.load_raw(shuffled[0], annotate=False)
        piece = raw.copy().crop(s.onset, s.onset + s.duration - 1.0 / raw.info["sfreq"]).load_data()
        print(f"segment {s.frag_id}: {s.duration:.1f} s, bad channels of the source run: {list(s.bad_channels)}")
        print(piece)

    # 3. Overt utterances with EEG, voice-masked audio and text
    for k, u in enumerate(bids.iter_utterances(args.bids_root, subject="01", pre=0.5, post=0.5)):
        print(u["source"], u["file"], u["eeg"].shape, len(u["audio"]), str(u["text"])[:30])
        if k == 2:
            break


if __name__ == "__main__":
    main()
