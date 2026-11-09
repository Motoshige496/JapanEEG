"""Facial muscle activity in the EEG: high-frequency slopes during speech and pauses, EEG-EMG coherence.

Both measures compare the recording reference with the common average of the good scalp EEG
channels (CAR), on the raw EEG without temporal filtering (the 60-118 Hz band lies far from the
high-pass, and 97-103 Hz around the line harmonic is left out).

* :func:`speech_pause_power`: within one run, all speech intervals (``overt`` events) are pooled
  into one spectrum and all pauses between consecutive events (at least ``min_pause`` s, after
  removing short vocalisations, coughs and cue sounds, ``trim`` s removed at both ends) into
  another, from Hann windows of ``win_sec`` with 50% overlap. ``BAD_high_amplitude``
  stretches are removed first. Day, device and electrode fit are identical for both classes.
* :func:`coherence`: magnitude-squared coherence between every scalp EEG channel and every
  facial EMG channel, averaged over 60-118 Hz without 97-103 Hz; windows of 4 s inside
  4-8 s intervals centred on each overt event. The chance level pairs every EEG window with
  the EMG of the previous window (circularly).

A continuous run is one piece. In a ``ses-shuffled`` file, every segment is a piece, and the
pieces of the file are pooled (the run of a segment is not released).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from japaneeg import bids
from japaneeg.psd import HF_BAND, HF_EXCLUDE, hf_slope

PAUSE_EXCLUDE = ("short_speech", "cough", "cue")
MARKS = ("BAD_high_amplitude",)
EMG_TYPES = ("EMG", "EMG_UPPER_LIP", "EMG_LOWER_LIP", "UPPER_LIP", "LOWER_LIP")


def spans_of(
    rows: list[dict[str, str]] | tuple[dict[str, str], ...], types: tuple[str, ...]
) -> list[tuple[float, float]]:
    """Sorted ``(start, end)`` of the rows of the given trial types.

    Args:
        rows: Event rows.
        types: Trial types.

    Returns:
        Intervals (s).
    """
    out = []
    for r in rows:
        if r.get("trial_type") in types:
            d = r.get("duration", "")
            on = float(r["onset"])
            out.append((on, on + (float(d) if d not in ("", "n/a") else 0.0)))
    return sorted(out)


def subtract(spans: list[tuple[float, float]], cut: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Parts of ``spans`` outside every interval of ``cut`` (sorted by start).

    Args:
        spans: Intervals.
        cut: Intervals to remove.

    Returns:
        The remaining pieces.
    """
    out: list[tuple[float, float]] = []
    for a, b in spans:
        start = a
        for c, d in cut:
            if d <= start or c >= b:
                continue
            if c > start:
                out.append((start, c))
            start = max(start, d)
        if start < b:
            out.append((start, b))
    return out


def speech_pause_intervals(rows: list[dict[str, str]] | tuple[dict[str, str], ...], min_pause: float = 1.5,
                           trim: float = 0.3) -> dict[str, list[tuple[float, float]]]:  # fmt: skip
    """Speech and pause intervals of one piece.

    Args:
        rows: Event rows of the piece.
        min_pause: Shortest pause (s) after removing the excluded events.
        trim: Seconds removed at both ends of every pause.

    Returns:
        ``{"speech": [...], "pause": [...]}``.
    """
    overt = spans_of(rows, ("overt",))
    marks = spans_of(rows, MARKS)
    gaps = [(a[1], b[0]) for a, b in zip(overt, overt[1:])]
    gaps = subtract(subtract(gaps, spans_of(rows, PAUSE_EXCLUDE)), marks)
    return {"speech": subtract(overt, marks), "pause": [(a + trim, b - trim) for a, b in gaps if b - a >= min_pause]}


@dataclass
class Piece:
    """A continuous run, or one segment of a shuffled file, with its good scalp EEG channels."""

    start: float
    stop: float
    eeg: list[str]
    rows: tuple[dict[str, str], ...]


@dataclass
class ClassPower:
    """Summed window power of one interval class."""

    recording: NDArray[np.float64]
    car: NDArray[np.float64]
    n: int = 0
    names: list[str] = field(default_factory=list)


def pieces(edf: Path) -> list[Piece]:
    """Pieces of one file with their good scalp EEG channels and events.

    Args:
        edf: EDF file.

    Returns:
        The pieces.
    """
    channels = bids.read_tsv(bids.sidecar(edf, "channels.tsv"))
    eeg = bids.scalp_eeg(channels)
    if "_ses-shuffled_" in edf.name:
        return [Piece(s.onset, s.onset + s.duration, [c for c in eeg if c not in s.bad_channels], s.events)
                for s in bids.segments(edf)]  # fmt: skip
    import mne

    raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
    bad = {c["name"] for c in channels if c.get("status") == "bad"}
    rows = tuple(bids.read_tsv(bids.sidecar(edf, "events.tsv")))
    return [Piece(0.0, raw.n_times / raw.info["sfreq"], [c for c in eeg if c not in bad], rows)]


def speech_pause_power(edf: Path, win_sec: float = 0.5, min_pause: float = 1.5,
                       trim: float = 0.3) -> tuple[NDArray[np.float64], dict[str, ClassPower]]:  # fmt: skip
    """Pooled window power of speech and pauses for one file (all channels in its scalp EEG order).

    Channels that are bad in a piece are left out of that piece's CAR and power (their power
    is summed over the pieces where they are good; the counts are per channel).

    Args:
        edf: EDF file.
        win_sec: Window length (s).
        min_pause: Shortest pause (s).
        trim: Seconds removed at both ends of every pause.

    Returns:
        Frequencies and the power of each class: ``recording`` and ``car`` sums ``(n_ch, n_freqs)``
        and ``n`` windows per channel in ``names`` order (the count is stored in ``n`` as the
        maximum; use :func:`class_slopes`).
    """
    import mne

    raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
    sf = float(raw.info["sfreq"])
    names = bids.scalp_eeg(bids.read_tsv(bids.sidecar(edf, "channels.tsv")))
    nwin = int(round(win_sec * sf))
    hann = np.hanning(nwin)
    freqs = np.fft.rfftfreq(nwin, 1 / sf)
    out = {c: ClassPower(np.zeros((len(names), len(freqs))), np.zeros((len(names), len(freqs))), 0, names)
           for c in ("speech", "pause")}  # fmt: skip
    counts = {c: np.zeros(len(names)) for c in out}
    for p in pieces(edf):
        rows = tuple(r for r in p.rows if p.start <= float(r["onset"]) < p.stop)
        idx = [names.index(c) for c in p.eeg]
        if len(idx) < 2:
            continue
        for cls, spans in speech_pause_intervals(rows, min_pause, trim).items():
            for a, b in spans:
                a, b = max(a, p.start), min(b, p.stop)
                if b - a < win_sec:
                    continue
                data = raw.get_data(picks=p.eeg, start=int(a * sf), stop=int(b * sf))
                for s in range(0, data.shape[1] - nwin + 1, nwin // 2):
                    w = data[:, s : s + nwin]
                    w = w - w.mean(axis=1, keepdims=True)
                    out[cls].recording[idx] += np.abs(np.fft.rfft(w * hann, axis=1)) ** 2
                    out[cls].car[idx] += np.abs(np.fft.rfft((w - w.mean(axis=0, keepdims=True)) * hann, axis=1)) ** 2
                    counts[cls][idx] += 1
    for cls in out:
        out[cls].n = int(counts[cls].max())
        with np.errstate(invalid="ignore", divide="ignore"):
            out[cls].recording /= counts[cls][:, None]
            out[cls].car /= counts[cls][:, None]
    return freqs, out


def class_slopes(freqs: NDArray[np.float64], power: ClassPower) -> dict[str, float]:
    """Median channel slope (dB per decade, 60-118 Hz without 97-103 Hz) for both references.

    Args:
        freqs: Frequencies (Hz).
        power: Mean window power of one class (NaN rows: channel never good).

    Returns:
        ``{"recording": ..., "car": ...}``.
    """
    out = {}
    for ref in ("recording", "car"):
        p = getattr(power, ref)
        ok = np.all(np.isfinite(p[:, 1:]) & (p[:, 1:] > 0), axis=1)
        out[ref] = float(np.median(hf_slope(freqs, p[ok], HF_BAND, HF_EXCLUDE))) if ok.any() else float("nan")
    return out


def event_intervals(rows: list[dict[str, str]] | tuple[dict[str, str], ...], start: float, stop: float,
                    min_sec: float = 4.0, max_sec: float = 8.0, win_sec: float = 4.0) -> list[tuple[float, float]]:  # fmt: skip
    """Merged intervals of 4-8 s centred on every overt event, without ``BAD_high_amplitude``.

    Args:
        rows: Event rows.
        start: Start of the piece (s).
        stop: End of the piece (s).
        min_sec: Shortest interval (s).
        max_sec: Longest interval (s).
        win_sec: Intervals shorter than one window are dropped.

    Returns:
        Non-overlapping intervals (s).
    """
    spans = []
    for on, end in spans_of(rows, ("overt",)):
        length = min(max(end - on, min_sec), max_sec)
        mid = (on + end) / 2
        spans.append((max(start, mid - length / 2), min(stop, mid + length / 2)))
    merged: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(a, b) for a, b in subtract(merged, spans_of(rows, MARKS)) if b - a >= win_sec]


def coherence(edf: Path, win_sec: float = 4.0, band: tuple[float, float] = HF_BAND,
              skip: tuple[float, float] = HF_EXCLUDE[0]) -> dict[str, float]:  # fmt: skip
    """EEG-EMG magnitude-squared coherence of one file, recording reference and CAR, with chance levels.

    Args:
        edf: EDF file of overt speech.
        win_sec: Window length (s).
        band: Frequency range (Hz).
        skip: Range left out (Hz).

    Returns:
        Median over channel pairs of the band-averaged coherence: ``recording``, ``car``,
        ``recording_chance``, ``car_chance``; and ``n_windows``. Empty if fewer than 3 windows.
    """
    import mne

    raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
    sf = float(raw.info["sfreq"])
    channels = bids.read_tsv(bids.sidecar(edf, "channels.tsv"))
    emg = [c["name"] for c in channels if c.get("type") in EMG_TYPES]
    nwin = int(round(win_sec * sf))
    hann = np.hanning(nwin)
    freqs = np.fft.rfftfreq(nwin, 1 / sf)
    sel = (freqs >= band[0]) & (freqs <= band[1]) & ~((freqs >= skip[0]) & (freqs <= skip[1]))
    names = bids.scalp_eeg(channels)
    sums: dict[str, NDArray[np.complex128] | NDArray[np.float64]] = {}

    def add(key: str, value: NDArray[np.complex128] | NDArray[np.float64]) -> None:
        sums[key] = sums[key] + value if key in sums else value

    y_prev: NDArray[np.complex128] | None = None
    x_first: dict[str, NDArray[np.complex128]] = {}
    n = 0
    for p in pieces(edf):
        good = [c for c in names if c in p.eeg]
        if len(good) < 2 or not emg:
            continue
        mask = np.array([c in good for c in names])
        for a, b in event_intervals(tuple(r for r in p.rows if p.start <= float(r["onset"]) < p.stop),
                                    p.start, p.stop, win_sec=win_sec):  # fmt: skip
            data = raw.get_data(picks=names + emg, start=int(a * sf), stop=int(b * sf))
            for s in range(0, data.shape[1] - nwin + 1, nwin // 2):
                w = data[:, s : s + nwin]
                spec = np.fft.rfft((w - w.mean(axis=1, keepdims=True)) * hann, axis=1)[:, sel]
                x0 = np.where(mask[:, None], spec[: len(names)], 0)
                car = x0 - np.where(mask[:, None], spec[: len(names)][mask].mean(axis=0, keepdims=True), 0)
                y = spec[len(names) :]
                for ref, x in (("recording", x0), ("car", car)):
                    add(f"{ref}_sxy", np.einsum("cf,ef->cef", x, np.conj(y)))
                    add(f"{ref}_sxx", np.abs(x) ** 2)
                    if y_prev is not None:
                        add(f"{ref}_chance_sxy", np.einsum("cf,ef->cef", x, np.conj(y_prev)))
                    x_first.setdefault(ref, x)
                add("syy", np.abs(y) ** 2)
                y_prev = y
                n += 1
    if n < 3 or y_prev is None:
        return {}
    out: dict[str, float] = {"n_windows": float(n)}
    for ref in ("recording", "car"):
        add(f"{ref}_chance_sxy", np.einsum("cf,ef->cef", x_first[ref], np.conj(y_prev)))
        sxx = np.real(sums[f"{ref}_sxx"])
        used = sxx.sum(axis=1) > 0
        for label in ("", "_chance"):
            coh = np.abs(sums[f"{ref}{label}_sxy"]) ** 2 / np.maximum(
                sxx[:, None, :] * np.real(sums["syy"])[None], 1e-300
            )
            out[f"{ref}{label}"] = float(np.median(coh[used].mean(axis=2)))
    return out
