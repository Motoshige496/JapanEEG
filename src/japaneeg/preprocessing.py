"""Preprocessing of the JapanEEG technical validation.

The steps follow the Methods of the Data Descriptor:

1. notch filter at 50 Hz and its harmonics, average reference, band-pass 2-120 Hz (EEG) and
   20-120 Hz (EMG and EOG references), resampling to 240 Hz (:func:`filter_piece`);
2. bad channels excluded and the common average reference recomputed over the good channels
   (:func:`good_channel_reference`);
3. every channel standardised over its file or run, by mean and SD (PSD analysis) or by the
   median and 1.4826 x the median absolute deviation (ERP analyses; :func:`robust_zscore`);
4. a normalised least-mean-squares (NLMS) adaptive filter that removes from each EEG channel the
   part predicted from the three EMG/EOG references (:func:`nlms`; step size 0.1,
   regularisation 0.001, initial weights from N(0, 0.5) with seed 0).

All functions work on one piece of signal: a continuous run or one segment of a ``ses-shuffled``
file. Never filter across segment boundaries.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

SFREQ_OUT = 240.0
EEG_BAND = (2.0, 120.0)
REF_BAND = (20.0, 120.0)


def filter_piece(
    eeg: NDArray[np.floating],
    refs: NDArray[np.floating],
    sfreq: float,
    sfreq_out: float = SFREQ_OUT,
    eeg_band: tuple[float, float] = EEG_BAND,
    ref_band: tuple[float, float] = REF_BAND,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Notch, average reference, band-pass and resampling of one piece (MNE calls).

    Args:
        eeg: Scalp EEG, ``(n_eeg, n_times)``, volts.
        refs: The three references (EOG, upper lip, lower lip), ``(3, n_times)``.
        sfreq: Sampling rate (Hz).
        sfreq_out: Output sampling rate (Hz).
        eeg_band: EEG pass band (Hz).
        ref_band: Pass band of the references (Hz).

    Returns:
        Filtered EEG and references at ``sfreq_out``.
    """
    import mne

    n_eeg = eeg.shape[0]
    x = np.concatenate([np.asarray(eeg, np.float64), np.asarray(refs, np.float64)])
    info = mne.create_info(x.shape[0], sfreq, ["eeg"] * n_eeg + ["eog", "emg", "emg"])
    raw = mne.io.RawArray(x, info, verbose=False)
    with warnings.catch_warnings():  # short segments: filters longer than the signal
        warnings.filterwarnings("ignore", message="filter_length", category=RuntimeWarning)
        raw.notch_filter(np.arange(50, sfreq / 2, 50), fir_design="firwin2", picks="all", verbose=False)
        raw.set_eeg_reference("average", ch_type="eeg", verbose=False)
        raw.filter(eeg_band[0], eeg_band[1], picks="eeg", verbose=False)
        raw.filter(ref_band[0], ref_band[1], picks=["emg", "eog"], verbose=False)
        raw.resample(sfreq_out, verbose=False)
    y = raw.get_data()
    return y[:n_eeg], y[n_eeg:]


def detect_bad(eeg: NDArray[np.floating], sd_factor: float = 5.0) -> NDArray[np.bool_]:
    """Simple bad-channel flags of one piece: flat, or SD above ``sd_factor`` x the median SD.

    The release marks bad channels with the PREP criteria (``channels.tsv`` ``status`` of
    continuous runs, ``source_run_bad_channels`` of ``ses-shuffled`` segments). This check adds
    channels that fail within a short segment.

    Args:
        eeg: EEG of one piece, ``(n_eeg, n_times)``.
        sd_factor: Amplitude criterion.

    Returns:
        Boolean flags, ``(n_eeg,)``.
    """
    sd = np.asarray(eeg, np.float64).std(axis=1)
    med = float(np.median(sd))
    return (sd <= 1e-6 * med) | (sd > sd_factor * med)


def good_channel_reference(eeg: NDArray[np.floating], bad: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Common average reference over the good channels.

    Args:
        eeg: EEG, ``(n_eeg, n_times)``.
        bad: Bad-channel flags.

    Returns:
        Re-referenced EEG (bad rows are re-referenced too and must be excluded later).
    """
    x = np.asarray(eeg, np.float64)
    if (~bad).sum() < 2:
        raise ValueError("fewer than two good channels")
    return x - x[~bad].mean(axis=0, keepdims=True)


def zscore(x: NDArray[np.floating]) -> NDArray[np.float64]:
    """Per-row z-score (mean, SD with ddof=1).

    Args:
        x: ``(n_rows, n_times)``.

    Returns:
        Standardised rows.
    """
    x = np.asarray(x, np.float64)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, ddof=1, keepdims=True) + np.finfo(float).eps)


def robust_zscore(x: NDArray[np.floating]) -> NDArray[np.float64]:
    """Per-row robust z-score: median and 1.4826 x the median absolute deviation.

    The scale equals the SD for normally distributed data, but rare large artifacts do not
    inflate it.

    Args:
        x: ``(n_rows, n_times)``.

    Returns:
        Standardised rows.
    """
    x = np.asarray(x, np.float64)
    med = np.median(x, axis=1, keepdims=True)
    return (x - med) / (1.4826 * np.median(np.abs(x - med), axis=1, keepdims=True) + np.finfo(float).eps)


def initial_weights(n_refs: int, n_eeg: int, seed: int = 0) -> NDArray[np.float64]:
    """Initial NLMS weights drawn from N(0, 0.5).

    Args:
        n_refs: Number of reference channels.
        n_eeg: Number of EEG channels.
        seed: Random seed.

    Returns:
        Weights, ``(n_refs, n_eeg)``.
    """
    return np.random.default_rng(seed).normal(0.0, 0.5, (n_refs, n_eeg))


def nlms(
    eeg: NDArray[np.floating],
    refs: NDArray[np.floating],
    mu: float = 0.1,
    eps: float = 1e-3,
    w0: NDArray[np.float64] | None = None,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalised LMS filter: removes from each EEG channel the part predicted from the references.

    For every sample ``k``: ``e_k = d_k - x_k W``, ``W += mu / (eps + x_k x_k^T) * x_k^T e_k``.

    Args:
        eeg: Standardised EEG, ``(n_eeg, n_times)``.
        refs: Standardised references, ``(n_refs, n_times)``.
        mu: Step size.
        eps: Regularisation of the input power.
        w0: Initial weights ``(n_refs, n_eeg)`` (None: :func:`initial_weights` with ``seed``).
        seed: Seed of the initial weights.

    Returns:
        Filtered EEG ``(n_eeg, n_times)`` and the weights after the last sample (to carry them
        to the next segment of the same file).
    """
    d = np.asarray(eeg, np.float64).T
    x = np.asarray(refs, np.float64).T
    w = np.array(w0, np.float64) if w0 is not None else initial_weights(x.shape[1], d.shape[1], seed)
    e = np.empty_like(d)
    for k in range(d.shape[0]):
        xk = x[k]
        ek = d[k] - xk @ w
        e[k] = ek
        w += (mu / (eps + xk @ xk)) * np.outer(xk, ek)
    return e.T, w
