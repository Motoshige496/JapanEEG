"""Grand-average ERPs of Figures 3 and 4 recomputed from the released files.

The paper's figures were computed on the original continuous recordings. Part of the release
is cut into segments (``ses-shuffled``), so a user processes every continuous run as one piece
and every segment of a shuffled file on its own:

1. per piece: notch, average reference, band-pass and resampling (:func:`filter_piece`);
   bad channels are the released flags (``channels.tsv`` ``status`` of continuous runs,
   ``source_run_bad_channels`` of segments) and channels that are flat or have an SD above
   5 x the median in the piece; common average reference over the good channels;
2. per file: pieces whose amplitude (median channel SD) exceeds 5 x the file median are left
   out; every channel is standardised by its median and 1.4826 x MAD over the remaining pieces;
3. NLMS adaptive filter with the weights carried from one piece to the next in file order
   (no signal is filtered across a segment boundary);
4. epochs around the selected onsets that fit into their piece and do not overlap a
   ``BAD_high_amplitude`` stretch; baseline correction;
5. rejection over all trials of the condition, epoch normalisation, average per file and then
   over files (:func:`japaneeg.erp.grand_average`).

The run-wise standardisation and the adaptive-filter weight history of the original continuous
recordings cannot be restored from the segments, so the result approximates the paper's
figures; :func:`compare` reports how closely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from japaneeg import bids, erp
from japaneeg.preprocessing import SFREQ_OUT, detect_bad, filter_piece, initial_weights, nlms

REF_TYPES = ("EMG", "EOG", "EMG_UPPER_LIP", "EMG_LOWER_LIP", "UPPER_LIP", "LOWER_LIP")


@dataclass(frozen=True)
class Condition:
    """One figure condition and its released files."""

    subject: str
    task: str
    acq: str
    trial_type: str


CONDITIONS: dict[str, Condition] = {
    "P01-overt-pangolin": Condition("01", "speechopen", "pangolin", "overt"),
    "P02-overt-pangolin": Condition("02", "speechopen", "pangolin", "overt"),
    "P03-overt-pangolin": Condition("03", "speechopen", "pangolin", "overt"),
    "P01-overt-scarabeo": Condition("01", "speechopen", "scarabeo", "overt"),
    "P01-overt-eego": Condition("01", "speechopen", "eego", "overt"),
    "P01-covert-pangolin": Condition("01", "listeningcovert", "pangolin", "covert"),
    "P01-listening-pangolin": Condition("01", "listening", "pangolin", "listening"),
}
"""Figure conditions; the listening panel uses the continuous listening task only."""


@dataclass(frozen=True)
class Piece:
    """A continuous run or one segment of a shuffled file."""

    start: float
    duration: float
    bad: frozenset[str]
    events: tuple[erp.Event, ...]


def reference_rows(channels: list[dict[str, str]], names: list[str]) -> tuple[list[int], list[int]]:
    """Rows of the three EMG/EOG references (bipolar: positive and negative inputs).

    Args:
        channels: Rows of ``channels.tsv``.
        names: Channel names of the EDF file.

    Returns:
        Rows of the (positive) references and rows of the negative inputs (empty if the
        references are already single channels).
    """
    refs = [c for c in channels if c.get("type") in REF_TYPES]
    pos = [names.index(c["name"]) for c in refs if "positive" in c.get("description", "").lower()]
    neg = [names.index(c["name"]) for c in refs if "negative" in c.get("description", "").lower()]
    if len(pos) == 3 and len(neg) == 3:
        return pos, neg
    if len(refs) == 3:
        return [names.index(c["name"]) for c in refs], []
    raise ValueError(f"expected 3 references or 3 bipolar pairs, found {len(refs)} channels")


def pieces(edf: Path, trial_type: str) -> list[Piece]:
    """Pieces of one file with the events of ``trial_type`` (file time).

    Args:
        edf: EDF file.
        trial_type: ``overt``, ``covert`` or ``listening``.

    Returns:
        One piece for a continuous run, one per segment for a shuffled file.
    """

    def evs(rows: list[dict[str, str]] | tuple[dict[str, str], ...]) -> tuple[erp.Event, ...]:
        out = []
        for r in rows:
            if r.get("trial_type") == trial_type:
                d = r.get("duration", "")
                out.append(erp.Event(float(r["onset"]), float(d) if d not in ("", "n/a") else 0.0))
        return tuple(out)

    if "_ses-shuffled_" in edf.name:
        return [Piece(s.onset, s.duration, frozenset(s.bad_channels), evs(s.events)) for s in bids.segments(edf)]
    channels = bids.read_tsv(bids.sidecar(edf, "channels.tsv"))
    bad = frozenset(c["name"] for c in channels if c.get("status") == "bad")
    import mne

    raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
    duration = raw.n_times / float(raw.info["sfreq"])
    return [Piece(0.0, duration, bad, evs(bids.read_tsv(bids.sidecar(edf, "events.tsv"))))]


def bad_spans(edf: Path) -> list[tuple[float, float]]:
    """``(onset, duration)`` of the ``BAD_high_amplitude`` rows of a file.

    Args:
        edf: EDF file.

    Returns:
        The stretches (file time).
    """
    return [(float(r["onset"]), float(r["duration"])) for r in bids.read_tsv(bids.sidecar(edf, "events.tsv"))
            if r.get("trial_type") == "BAD_high_amplitude"]  # fmt: skip


def process_file(
    edf: Path,
    trial_type: str,
    sd_factor: float = 5.0,
    outlier_factor: float = 5.0,
    mu: float = 0.1,
    eps: float = 1e-3,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Steps 1-4 of the module docstring for one file.

    Args:
        edf: EDF file.
        trial_type: ``overt``, ``covert`` or ``listening``.
        sd_factor: Bad-channel amplitude criterion within a piece.
        outlier_factor: Pieces louder than this x the file median are left out.
        mu: NLMS step size.
        eps: NLMS regularisation.
        seed: Seed of the initial NLMS weights.

    Returns:
        Baseline-corrected epochs ``(n_trials, n_ch, n_samples)`` in robust units (bad: NaN).
    """
    import mne

    raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
    sfreq = float(raw.info["sfreq"])
    channels = bids.read_tsv(bids.sidecar(edf, "channels.tsv"))
    eeg_names = bids.scalp_eeg(channels)
    names = list(raw.ch_names)
    eeg_rows = [names.index(n) for n in eeg_names]
    pos, neg = reference_rows(channels, names)
    spans = bad_spans(edf)
    baseline = erp.BASELINES_MS[trial_type]
    min_sil = erp.MIN_SILENCE_SEC[trial_type]

    eeg_p, ref_p, bad_p, keep_p = [], [], [], []
    for p in pieces(edf, trial_type):
        a, b = int(round(p.start * sfreq)), int(round((p.start + p.duration) * sfreq))
        x = raw.get_data(start=a, stop=b)
        refs = x[pos] - x[neg] if neg else x[pos]
        e, m = filter_piece(x[eeg_rows], refs, sfreq)
        bad = np.array([n in p.bad for n in eeg_names]) | detect_bad(e, sd_factor)
        if (~bad).sum() < 2:
            continue
        e = e - e[~bad].mean(axis=0, keepdims=True)
        eeg_p.append(e)
        ref_p.append(m)
        bad_p.append(bad)
        keep_p.append(p)
    if not eeg_p:
        return np.zeros((0, len(eeg_names), len(erp.epoch_times(SFREQ_OUT))))
    amp = np.array([float(np.median(e[~b].std(axis=1))) for e, b in zip(eeg_p, bad_p)])
    ok = amp <= outlier_factor * float(np.median(amp))
    cols = []
    for e, b, o in zip(eeg_p, bad_p, ok):
        if o:
            y = e.copy()
            y[b] = np.nan
            cols.append(y)
    cat = np.concatenate(cols, axis=1)
    med = np.nanmedian(cat, axis=1)
    scale = 1.4826 * np.nanmedian(np.abs(cat - med[:, None]), axis=1) + np.finfo(float).eps
    med = np.where(np.isfinite(med), med, 0.0)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    del cat, cols
    m_all = np.concatenate([m for m, o in zip(ref_p, ok) if o], axis=1)
    m_mean, m_sd = m_all.mean(axis=1), m_all.std(axis=1, ddof=1) + np.finfo(float).eps
    w = initial_weights(len(pos), len(eeg_names), seed)
    out = []
    for e, m, b, p, o in zip(eeg_p, ref_p, bad_p, keep_p, ok):
        if not o:
            continue
        d, w = nlms((e - med[:, None]) / scale[:, None], (m - m_mean[:, None]) / m_sd[:, None], mu, eps, w)
        onsets = [t - p.start for t in erp.select_onsets(list(p.events), p.start, min_sil)
                  if not erp.overlaps(t + erp.EPOCH_SEC[0], t + erp.EPOCH_SEC[1], spans)]  # fmt: skip
        ep = erp.epochs(d, onsets, SFREQ_OUT, baseline_ms=baseline, bad=b)
        if len(ep):
            out.append(ep)
    if not out:
        return np.zeros((0, len(eeg_names), len(erp.epoch_times(SFREQ_OUT))))
    return np.concatenate(out)


def files_of(root: str | Path, cond: Condition) -> list[Path]:
    """Released files of a condition: shuffled files and continuous runs.

    Args:
        root: BIDS root.
        cond: Condition.

    Returns:
        EDF paths.
    """
    return bids.find_runs(root, cond.subject, cond.task, cond.acq)


def compare(a: NDArray[np.floating], b: NDArray[np.floating], times_ms: NDArray[np.floating],
            window_ms: tuple[float, float] = (-200.0, 800.0)) -> float:  # fmt: skip
    """Pearson correlation of two grand averages over all channels in ``window_ms``.

    Args:
        a: Grand average ``(n_ch, n_samples)``.
        b: Grand average of the same shape.
        times_ms: Time axis (ms).
        window_ms: Window (ms).

    Returns:
        The correlation (NaN if fewer than three finite pairs).
    """
    t = np.asarray(times_ms)
    m = (t >= window_ms[0]) & (t <= window_ms[1])
    x, y = np.asarray(a)[:, m].ravel(), np.asarray(b)[:, m].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else math.nan
