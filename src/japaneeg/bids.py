"""Read runs, events, segments and utterances of the JapanEEG BIDS tree.

The release has two kinds of recordings:

* continuous runs, ``sub-XX/ses-YYYYMMDD/eeg/*_eeg.edf``, one per recording run;
* ``sub-XX/ses-shuffled/eeg/*_eeg.edf``: readings of copyrighted texts, the covert repetition task
  and audiobook listening, cut into segments (``frag_id``, ``segment_onset``,
  ``segment_duration`` in ``events.tsv``) that are concatenated in random order. Never filter
  across segment boundaries: cut a segment first (:func:`segments`), then filter it.

Main entry points:

* :func:`find_runs` lists the EDF files of a subject, task, device or session;
* :func:`load_raw` reads one EDF file as an MNE ``Raw`` object with the channel types of
  ``channels.tsv``, the bad channels of continuous runs and the events as annotations
  (``BAD_high_amplitude`` stretches are skipped by MNE when epoching);
* :func:`segments` lists the segments of a ``ses-shuffled`` file with their bad channels;
* :func:`iter_utterances` yields overt utterances (EEG, voice-masked audio, text) from both kinds.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

AUDIO_SR = 16000
CHANNEL_TYPES: dict[str, str] = {
    "EEG": "eeg",
    "EOG": "eog",
    "EMG": "emg",
    "EMG_UPPER_LIP": "emg",
    "EMG_LOWER_LIP": "emg",
    "LOWER_LIP": "emg",
    "UPPER_LIP": "emg",
    "MIC": "misc",
    "AUDIO": "misc",
    "TRIG": "stim",
    "MISC": "misc",
}
"""MNE channel type of each ``channels.tsv`` type (others: ``misc``)."""
REFERENCE_SITES: frozenset[str] = frozenset({"M1", "M2", "EEG063", "EEG064"})
"""EEG-typed reference-site electrodes (eego mastoids, g.SCARABEO earlobes); not scalp EEG."""


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    """Rows of a BIDS TSV file.

    Args:
        path: The file.

    Returns:
        One dictionary per row.
    """
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sidecar(edf: str | Path, suffix: str) -> Path:
    """Path of a sidecar of an EDF file, e.g. ``sidecar(edf, "events.tsv")``.

    Args:
        edf: ``*_eeg.edf`` file.
        suffix: ``events.tsv``, ``channels.tsv`` or ``eeg.json``.

    Returns:
        The sidecar path.
    """
    edf = Path(edf)
    return edf.with_name(edf.name[: -len("eeg.edf")] + suffix)


def find_runs(
    root: str | Path,
    subject: str | None = None,
    task: str | None = None,
    acq: str | None = None,
    session: str | None = None,
) -> list[Path]:
    """EDF files of the release, filtered by entity.

    Args:
        root: BIDS root.
        subject: e.g. ``"01"`` or ``"sub-01"`` (None: all).
        task: ``"speechopen"``, ``"listeningcovert"`` or ``"listening"`` (None: all).
        acq: ``"pangolin"``, ``"scarabeo"`` or ``"eego"`` (None: all).
        session: e.g. ``"20230829"`` or ``"shuffled"`` (None: all).

    Returns:
        Sorted EDF paths.
    """
    sub = "sub-" + subject.removeprefix("sub-") if subject else "sub-*"
    ses = "ses-" + session.removeprefix("ses-") if session else "ses-*"
    name = f"{sub}_{ses}_task-{task or '*'}_acq-{acq or '*'}_run-*_eeg.edf"
    return sorted(Path(root).glob(f"{sub}/{ses}/eeg/{name}"))


def scalp_eeg(channels: list[dict[str, str]]) -> list[str]:
    """Names of the scalp EEG channels (type ``EEG`` without the reference-site electrodes).

    Args:
        channels: Rows of ``channels.tsv``.

    Returns:
        Channel names in file order.
    """
    return [c["name"] for c in channels if c.get("type") == "EEG" and c["name"] not in REFERENCE_SITES
            and not is_audio(c)]  # fmt: skip


def is_audio(channel: dict[str, str]) -> bool:
    """Whether a ``channels.tsv`` row is a microphone or PC-audio channel (by type or description)."""
    desc = (channel.get("description") or "").lower()
    return channel.get("type") in ("MIC", "AUDIO") or any(
        k in desc for k in ("microphone", "audio monitor", "pc audio")
    )


def microphone_channels(channels: list[dict[str, str]]) -> list[str]:
    """Names of the microphone channels in file order.

    In the release, the first is the 30-Hz amplitude envelope of the microphone signal and the
    second the voice-masked speech on the EEG time line (zero where no masked audio exists).
    This holds for the ``ses-shuffled`` files too, whose ``channels.tsv`` keeps the original
    description ("Speaking microphone monitor channel 1/2").

    Args:
        channels: Rows of ``channels.tsv``.

    Returns:
        Channel names.
    """
    return [c["name"] for c in channels
            if c.get("type") == "MIC" or "microphone" in (c.get("description") or "").lower()]  # fmt: skip


def load_raw(edf: str | Path, preload: bool = False, annotate: bool = True):  # noqa: ANN201 (mne.io.Raw)
    """One EDF file as an MNE ``Raw`` object.

    Channel types follow ``channels.tsv``; ``status == "bad"`` channels (continuous runs) are set
    as ``raw.info["bads"]``; with ``annotate`` the rows of ``events.tsv`` become annotations
    named after their ``trial_type`` (``BAD_high_amplitude`` keeps its name, so that
    ``mne.Epochs`` rejects epochs that overlap it).

    Args:
        edf: ``*_eeg.edf`` file.
        preload: Load the data into memory.
        annotate: Add the events as annotations.

    Returns:
        ``mne.io.Raw``.
    """
    import mne

    edf = Path(edf)
    raw = mne.io.read_raw_edf(edf, preload=preload, verbose="error")
    channels = read_tsv(sidecar(edf, "channels.tsv"))
    types = {c["name"]: CHANNEL_TYPES.get(c.get("type", ""), "misc") for c in channels if c["name"] in raw.ch_names}
    raw.set_channel_types(types, verbose="error")
    raw.info["bads"] = [c["name"] for c in channels if c.get("status") == "bad" and c["name"] in raw.ch_names]
    if annotate:
        rows = read_tsv(sidecar(edf, "events.tsv"))
        onsets, durations, names = [], [], []
        for r in rows:
            if r.get("trial_type") in (None, "", "n/a", "word", "phone", "word_ref", "phone_ref"):
                continue
            onsets.append(float(r["onset"]))
            d = r.get("duration", "")
            durations.append(float(d) if d not in ("", "n/a") else 0.0)
            names.append(r["trial_type"])
        raw.set_annotations(mne.Annotations(onsets, durations, names))
    return raw


@dataclass(frozen=True)
class Segment:
    """One segment of a ``ses-shuffled`` file.

    Attributes:
        frag_id: Segment identifier.
        onset: Start in the file (s).
        duration: Length (s).
        bad_channels: Scalp EEG channels flagged bad in the source run (empty: none or unknown).
        events: Rows of ``events.tsv`` that belong to the segment (onsets in file time).
    """

    frag_id: str
    onset: float
    duration: float
    bad_channels: tuple[str, ...]
    events: tuple[dict[str, str], ...]


def segments(edf: str | Path) -> list[Segment]:
    """Segments of a ``ses-shuffled`` file in file order.

    Args:
        edf: ``*_eeg.edf`` file of ``ses-shuffled``.

    Returns:
        The segments.
    """
    rows = read_tsv(sidecar(edf, "events.tsv"))
    frags: dict[str, tuple[float, float, tuple[str, ...]]] = {}
    events: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        fid = r.get("frag_id", "")
        if not fid or fid == "n/a":
            continue
        if fid not in frags:
            bad = r.get("source_run_bad_channels", "n/a")
            names = tuple(json.loads(bad)) if bad not in ("", "n/a") else ()
            frags[fid] = (float(r["segment_onset"]), float(r["segment_duration"]), names)
        events.setdefault(fid, []).append(r)
    return [
        Segment(fid, on, dur, bad, tuple(events[fid]))
        for fid, (on, dur, bad) in sorted(frags.items(), key=lambda x: x[1][0])
    ]


def _audio(wav: Path, start: float, dur: float) -> NDArray[np.float32]:
    """Audio window (16 kHz), zero-padded outside the file."""
    import soundfile as sf

    n = int(round(dur * AUDIO_SR))
    out = np.zeros(n, dtype=np.float32)
    with sf.SoundFile(wav) as f:
        a = int(round(start * f.samplerate))
        lo = max(0, a)
        f.seek(lo)
        x = f.read(frames=max(0, n - (lo - a)), dtype="float32", always_2d=True)[:, 0]
    out[lo - a : lo - a + len(x)] = x
    return out


def iter_utterances(
    root: str | Path,
    subject: str | None = None,
    acq: str | None = None,
    pre: float = 0.0,
    post: float = 0.0,
    include: tuple[str, ...] = ("continuous", "shuffled"),
) -> Iterator[dict[str, object]]:
    """Yield one dictionary per overt utterance with EEG, voice-masked audio and text.

    Args:
        root: BIDS root.
        subject: e.g. ``"01"`` (None: all).
        acq: Device, e.g. ``"pangolin"`` (None: all).
        pre: Seconds before the speech onset (continuous runs; in ``ses-shuffled`` the released
            margins of at most 1 s are always included).
        post: Seconds after the speech offset (continuous runs).
        include: ``"continuous"`` and/or ``"shuffled"``.

    Yields:
        ``eeg`` (channels x samples, volts), ``sfreq``, ``ch_names``, ``audio`` (16 kHz, aligned
        with ``eeg``; empty without audio), ``text``, ``speech_onset`` and ``speech_offset``
        (s within the window), ``source`` and ``file``.
    """
    import mne

    for edf in find_runs(root, subject, "speechopen", acq):
        shuffled = "_ses-shuffled_" in edf.name
        if ("shuffled" if shuffled else "continuous") not in include:
            continue
        rows = read_tsv(sidecar(edf, "events.tsv"))
        if not rows or "value" not in rows[0]:
            continue
        wav = edf.parents[1] / "beh" / edf.name.replace("_eeg.edf", "_recording-vocal_beh.wav")
        raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
        sfreq = float(raw.info["sfreq"])
        for r in rows:
            if r.get("trial_type") != "overt":
                continue
            onset, dur = float(r["onset"]), float(r["duration"])
            if shuffled:
                t0 = float(r["segment_onset"])
                t1 = t0 + float(r["segment_duration"])
                a0 = t0  # the WAV of ses-shuffled shares the EDF time line
            else:
                wav_on = r.get("wav_onset", "n/a")
                if wav_on in ("", "n/a"):
                    continue
                t0, t1 = onset - pre, onset + dur + post
                a0 = float(wav_on) - pre  # time line of the run's audio file
            a, b = int(round(t0 * sfreq)), int(round(t1 * sfreq))
            if a < 0 or b > raw.n_times:
                continue
            yield {
                "eeg": raw.get_data(start=a, stop=b),
                "sfreq": sfreq,
                "ch_names": list(raw.ch_names),
                "audio": _audio(wav, a0, t1 - t0) if wav.exists() else np.zeros(0, np.float32),
                "text": r.get("value", ""),
                "speech_onset": onset - t0,
                "speech_offset": onset + dur - t0,
                "source": "shuffled" if shuffled else "continuous",
                "file": edf.name,
            }
