"""Power spectral density of the technical validation.

* Welch's method (4-s Hann windows, 50% overlap) within each task segment, averaged over the
  segments of a run;
* per channel, the median over runs (a mean would be dominated by a few runs with very high
  power) and its approximate standard error in dB,
  ``(10 / ln 10) * sqrt(pi / 2) * 1.4826 * MAD / (median * sqrt(N))``;
* the high-frequency slope: a line fitted to ``10 log10(PSD)`` against ``log10(f)`` per
  channel over 60-118 Hz without 97-103 Hz, reported as the median over channels in dB per
  decade.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

HF_BAND = (60.0, 118.0)
HF_EXCLUDE = ((97.0, 103.0),)


def welch(
    data: NDArray[np.floating], sfreq: float, window_sec: float = 4.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Welch PSD per channel (Hann window, 50% overlap).

    Args:
        data: ``(n_ch, n_times)``.
        sfreq: Sampling rate (Hz).
        window_sec: Window length (s).

    Returns:
        Frequencies and PSD ``(n_ch, n_freqs)``.
    """
    from scipy.signal import welch as _welch

    nperseg = int(round(window_sec * sfreq))
    if data.shape[1] < nperseg:
        raise ValueError("segment shorter than one window")
    f, p = _welch(
        np.asarray(data, np.float64), fs=sfreq, window="hann", nperseg=nperseg, noverlap=nperseg // 2, axis=-1
    )
    return f, p


def run_median(per_run: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Median and median absolute deviation over runs (NaN-aware).

    Args:
        per_run: Linear PSD per run, ``(n_runs, n_ch, n_freqs)``.

    Returns:
        Median and MAD, ``(n_ch, n_freqs)`` each.
    """
    x = np.asarray(per_run, np.float64)
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    return med, mad


def median_curve_db(
    median: NDArray[np.floating], mad: NDArray[np.floating], n_runs: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Channel average of the median PSD in dB and of its approximate standard error.

    Args:
        median: Median PSD over runs, ``(n_ch, n_freqs)``.
        mad: Median absolute deviation over runs, ``(n_ch, n_freqs)``.
        n_runs: Number of runs.

    Returns:
        Curve and band half-width (dB), ``(n_freqs,)`` each.
    """
    med = np.asarray(median, np.float64)
    curve = (10.0 * np.log10(med + 1e-30)).mean(axis=0)
    se = np.sqrt(np.pi / 2.0) * 1.4826 * np.asarray(mad, np.float64) / np.sqrt(max(n_runs, 1))
    band = np.nanmean((10.0 / np.log(10.0)) * se / (med + 1e-30), axis=0)
    return curve, band


def hf_slope(freqs: NDArray[np.floating], psd: NDArray[np.floating], band: tuple[float, float] = HF_BAND,
             exclude: tuple[tuple[float, float], ...] = HF_EXCLUDE) -> NDArray[np.float64]:  # fmt: skip
    """Slope of ``10 log10(PSD)`` against ``log10(f)`` per channel (dB per decade).

    Args:
        freqs: Frequencies (Hz).
        psd: Linear PSD, ``(n_ch, n_freqs)``.
        band: Fit range (Hz).
        exclude: Ranges left out (Hz), e.g. the 100-Hz line-noise peak.

    Returns:
        Slopes, ``(n_ch,)``. Report their median.
    """
    f = np.asarray(freqs, np.float64)
    keep = (f >= band[0]) & (f <= band[1])
    for lo, hi in exclude:
        keep &= ~((f >= lo) & (f <= hi))
    x = np.log10(f[keep])
    y = 10.0 * np.log10(np.asarray(psd, np.float64)[:, keep] + 1e-30)
    xc = x - x.mean()
    return ((y - y.mean(axis=1, keepdims=True)) @ xc) / float(xc @ xc)
