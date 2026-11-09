"""Event-related potentials of the technical validation.

Methods of the Data Descriptor, as implemented here:

* onsets: overt speech and listening events preceded by at least 0.5 s of silence (since the end
  of the previous event of the same type in the same run or segment); all covert trial onsets;
* epochs from -300 to 800 ms; baseline -300 to -200 ms for overt speech (before the pre-onset
  deflection) and -200 to 0 ms for covert speech and listening;
* trials overlapping a ``BAD_high_amplitude`` stretch are excluded;
* trial rejection: the maximum absolute value over channels and samples above 3 x its median in
  the condition;
* each accepted epoch and channel divided by its SD over the epoch; averaged over the trials of
  each run (or file) and then over the runs in which the channel was good.

Summary statistics: global field power (GFP) and its peaks, the ratio of the post-onset RMS to
the baseline SD, and the split-half reliability across runs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

EPOCH_SEC = (-0.3, 0.8)
BASELINES_MS: dict[str, tuple[float, float]] = {
    "overt": (-300.0, -200.0),
    "covert": (-200.0, 0.0),
    "listening": (-200.0, 0.0),
}
MIN_SILENCE_SEC: dict[str, float | None] = {"overt": 0.5, "covert": None, "listening": 0.5}
REJECT_FACTOR = 3.0


@dataclass(frozen=True)
class Event:
    """One event with its duration (s, in the time line of the piece)."""

    onset: float
    duration: float


def select_onsets(events: list[Event], start: float = 0.0, min_silence: float | None = 0.5) -> list[float]:
    """Onsets preceded by at least ``min_silence`` s without an event of the same type.

    The silence before an event runs from the end of the previous event (from ``start`` for
    the first one).

    Args:
        events: Events of one type in one run or segment.
        start: Start of the run or segment (s).
        min_silence: Required silence (None: keep every onset).

    Returns:
        Selected onsets (s).
    """
    out: list[float] = []
    prev_end = start
    for ev in sorted(events, key=lambda e: e.onset):
        if min_silence is None or ev.onset - prev_end >= min_silence:
            out.append(ev.onset)
        prev_end = max(prev_end, ev.onset + ev.duration)
    return out


def overlaps(t0: float, t1: float, spans: list[tuple[float, float]]) -> bool:
    """Whether ``[t0, t1]`` overlaps any ``(onset, duration)`` span.

    Args:
        t0: Start (s).
        t1: End (s).
        spans: ``(onset, duration)`` pairs, e.g. the ``BAD_high_amplitude`` rows.

    Returns:
        True if any span overlaps, even in part.
    """
    return any(on < t1 and on + dur > t0 for on, dur in spans)


def epochs(
    data: NDArray[np.floating],
    onsets: list[float],
    sfreq: float,
    tmin: float = EPOCH_SEC[0],
    tmax: float = EPOCH_SEC[1],
    baseline_ms: tuple[float, float] = BASELINES_MS["overt"],
    bad: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Baseline-corrected epochs of one piece.

    Onsets whose epoch does not fit into the piece are skipped. Bad channels become NaN.

    Args:
        data: Preprocessed EEG, ``(n_ch, n_times)``.
        onsets: Onsets in seconds from the start of ``data``.
        sfreq: Sampling rate (Hz).
        tmin: Epoch start (s).
        tmax: Epoch end (s).
        baseline_ms: Baseline window (ms).
        bad: Bad-channel flags (None: none).

    Returns:
        Epochs, ``(n_trials, n_ch, n_samples)``.
    """
    pre = int(round(-tmin * sfreq))
    length = int(round((tmax - tmin) * sfreq)) + 1
    b0 = pre + int(round(baseline_ms[0] * sfreq / 1000))
    b1 = pre + int(round(baseline_ms[1] * sfreq / 1000))
    out = []
    for t in onsets:
        a = int(t * sfreq) - pre
        if a < 0 or a + length > data.shape[1]:
            continue
        ep = np.asarray(data[:, a : a + length], np.float64)
        ep = ep - ep[:, b0:b1].mean(axis=1, keepdims=True)
        if bad is not None:
            ep[bad] = np.nan
        out.append(ep)
    return np.stack(out) if out else np.zeros((0, data.shape[0], length))


def epoch_times(sfreq: float, tmin: float = EPOCH_SEC[0], tmax: float = EPOCH_SEC[1]) -> NDArray[np.float64]:
    """Time axis of :func:`epochs` (s).

    Args:
        sfreq: Sampling rate (Hz).
        tmin: Epoch start (s).
        tmax: Epoch end (s).

    Returns:
        Times of the samples.
    """
    pre = int(round(-tmin * sfreq))
    length = int(round((tmax - tmin) * sfreq)) + 1
    return (np.arange(length) - pre) / sfreq


def grand_average(
    per_file: list[NDArray[np.float64]], reject_factor: float = REJECT_FACTOR
) -> tuple[NDArray[np.float64], dict[str, int], NDArray[np.int32]]:
    """Rejection, epoch normalisation and the two-stage average of one condition.

    Args:
        per_file: Epochs of each run or file, ``(n_trials, n_ch, n_samples)`` (NaN rows: bad).
        reject_factor: Trials with a maximum absolute value above this x the median are rejected.

    Returns:
        Grand average ``(n_ch, n_samples)``, trial counts (``selected``, ``accepted``) and the
        number of files contributing to each channel.
    """
    maxabs = [np.nanmax(np.abs(e), axis=(1, 2)) if len(e) else np.zeros(0) for e in per_file]
    allmax = np.concatenate(maxabs) if maxabs else np.zeros(0)
    if allmax.size == 0:
        raise ValueError("no trials")
    thr = reject_factor * float(np.median(allmax))
    means = []
    n_acc = 0
    with warnings.catch_warnings():  # channels that are bad in a whole file are all NaN there
        warnings.simplefilter("ignore", RuntimeWarning)
        for e, mx in zip(per_file, maxabs):
            keep = mx <= thr
            if not keep.any():
                continue
            acc = e[keep]
            acc = acc / (np.nanstd(acc, axis=2, keepdims=True) + 1e-12)
            means.append(np.nanmean(acc, axis=0))
            n_acc += int(keep.sum())
        stack = np.stack(means)
        grand = np.nanmean(stack, axis=0)
    n_files = np.sum(np.isfinite(stack[:, :, 0]), axis=0).astype(np.int32)
    return grand, {"selected": int(allmax.size), "accepted": n_acc}, n_files


def gfp(erp: NDArray[np.floating]) -> NDArray[np.float64]:
    """Global field power: SD across channels at each time point.

    Args:
        erp: ``(n_ch, n_samples)``.

    Returns:
        GFP, ``(n_samples,)``.
    """
    return np.nanstd(np.asarray(erp, np.float64), axis=0)


def gfp_peaks(
    erp: NDArray[np.floating],
    times_ms: NDArray[np.floating],
    window_ms: tuple[float, float] = (-200.0, 600.0),
    n: int = 2,
    min_sep_ms: float = 100.0,
) -> list[float]:
    """Latencies of the ``n`` largest GFP local maxima at least ``min_sep_ms`` apart.

    Args:
        erp: Grand-average ERP, ``(n_ch, n_samples)``.
        times_ms: Time axis (ms).
        window_ms: Search range (ms).
        n: Number of peaks.
        min_sep_ms: Minimum separation (ms); 40 ms for the listening response.

    Returns:
        Peak latencies in increasing order (ms).
    """
    g = gfp(erp)
    t = np.asarray(times_ms, np.float64)
    idx = [
        i for i in range(1, len(g) - 1) if window_ms[0] <= t[i] <= window_ms[1] and g[i] >= g[i - 1] and g[i] > g[i + 1]
    ]
    best: list[int] = []
    for i in sorted(idx, key=lambda i: g[i], reverse=True):
        if all(abs(t[i] - t[j]) >= min_sep_ms for j in best):
            best.append(i)
        if len(best) == n:
            break
    return sorted(float(t[i]) for i in best)


def baseline_ratio(
    erp: NDArray[np.floating],
    times_ms: NDArray[np.floating],
    baseline_ms: tuple[float, float],
    post_ms: tuple[float, float] = (0.0, 400.0),
) -> float:
    """Mean over channels of the post-onset RMS divided by the mean over channels of the baseline SD.

    Args:
        erp: Grand-average ERP, ``(n_ch, n_samples)``.
        times_ms: Time axis (ms).
        baseline_ms: Baseline window (ms).
        post_ms: Post-onset window (ms).

    Returns:
        The ratio.
    """
    t = np.asarray(times_ms)
    base = (t >= baseline_ms[0]) & (t < baseline_ms[1])
    post = (t >= post_ms[0]) & (t < post_ms[1])
    x = np.asarray(erp, np.float64)
    x = x[np.isfinite(x).any(axis=1)]  # channels never good
    base_sd = float(np.nanmean(np.nanstd(x[:, base], axis=1)))
    post_rms = float(np.nanmean(np.sqrt(np.nanmean(x[:, post] ** 2, axis=1))))
    return post_rms / base_sd


def split_half_reliability(
    per_run: NDArray[np.floating],
    times_ms: NDArray[np.floating],
    window_ms: tuple[float, float] = (-200.0, 800.0),
    n_splits: int = 200,
    seed: int = 0,
) -> float:
    """Spearman-Brown corrected split-half reliability across runs.

    The runs are split at random into two halves ``n_splits`` times; each half is averaged over
    runs (NaN-aware) and the halves are correlated over all channels and samples in
    ``window_ms``. Returns ``2 r / (1 + r)`` for the mean r.

    Args:
        per_run: Per-run ERPs, ``(n_runs, n_ch, n_samples)``.
        times_ms: Time axis (ms).
        window_ms: Window of the correlation (ms).
        n_splits: Number of random splits.
        seed: Random seed.

    Returns:
        The reliability.
    """
    x = np.asarray(per_run, np.float64)
    t = np.asarray(times_ms)
    mask = (t >= window_ms[0]) & (t <= window_ms[1])
    rng = np.random.default_rng(seed)
    rs = []
    n = x.shape[0]
    for _ in range(n_splits):
        perm = rng.permutation(n)
        with warnings.catch_warnings():  # channels bad in all runs of one half
            warnings.simplefilter("ignore", RuntimeWarning)
            a = np.nanmean(x[perm[: n // 2]][:, :, mask], axis=0)
            b = np.nanmean(x[perm[n // 2 :]][:, :, mask], axis=0)
        ok = np.isfinite(a) & np.isfinite(b)
        rs.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    r = float(np.mean(rs))
    return 2 * r / (1 + r)
