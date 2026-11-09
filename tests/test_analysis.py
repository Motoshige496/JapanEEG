"""Tests of the preprocessing, ERP and PSD functions on synthetic data."""

from __future__ import annotations

import numpy as np

from japaneeg import erp, psd
from japaneeg.preprocessing import good_channel_reference, nlms, robust_zscore, zscore


def test_robust_zscore_ignores_rare_artifacts() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, (2, 20000))
    x[0, :20] = 1e4  # rare large artifact
    z = robust_zscore(x)
    assert abs(float(np.median(z[0, 100:])) ) < 0.05
    assert 0.9 < float(np.std(z[1])) < 1.1
    assert float(np.std(zscore(x)[0, 100:])) < 0.2  # the plain z-score is shrunk by the artifact


def test_nlms_removes_reference_component() -> None:
    rng = np.random.default_rng(1)
    refs = rng.normal(size=(3, 20000))
    brain = rng.normal(size=(4, 20000))
    mix = np.array([[1.0, 0.5, 0.0, 2.0], [0.0, 1.0, 0.3, 0.0], [0.5, 0.0, 1.0, 1.0]])
    eeg = brain + mix.T @ refs
    out, w = nlms(eeg, refs, mu=0.1)
    tail = slice(10000, None)
    before = np.corrcoef(eeg[0, tail], brain[0, tail])[0, 1]
    after = np.corrcoef(out[0, tail], brain[0, tail])[0, 1]
    assert after > 0.9 and after > before + 0.2
    assert np.allclose(w, mix, atol=0.5)  # step size 0.1: the weights fluctuate around the mixing


def test_good_channel_reference_excludes_bad() -> None:
    x = np.zeros((4, 10))
    x[3] = 1e3
    bad = np.array([False, False, False, True])
    assert np.allclose(good_channel_reference(x, bad)[:3], 0.0)


def test_select_onsets_silence_rule() -> None:
    ev = [erp.Event(1.0, 2.0), erp.Event(3.2, 1.0), erp.Event(5.0, 1.0)]
    assert erp.select_onsets(ev, 0.0, 0.5) == [1.0, 5.0]
    assert erp.select_onsets(ev, 0.0, None) == [1.0, 3.2, 5.0]


def test_epochs_baseline_and_bad() -> None:
    sfreq = 240.0
    data = np.ones((3, 2400))
    bad = np.array([False, True, False])
    ep = erp.epochs(data, [2.0, 9.9], sfreq, baseline_ms=(-300, -200), bad=bad)
    assert ep.shape == (1, 3, len(erp.epoch_times(sfreq)))  # the second onset does not fit
    assert np.allclose(ep[0, 0], 0.0) and np.all(np.isnan(ep[0, 1]))


def test_grand_average_rejects_and_counts() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(20, 2, 50))
    b = rng.normal(size=(20, 2, 50))
    b[0] *= 100.0
    b[:, 1] = np.nan  # channel 1 bad in file b
    grand, counts, n_files = erp.grand_average([a, b])
    assert counts == {"selected": 40, "accepted": 39}
    assert list(n_files) == [2, 1]
    assert grand.shape == (2, 50)


def test_gfp_peaks_and_ratio() -> None:
    t = np.arange(-300, 801, 1000 / 240)
    topo = np.linspace(-1, 1, 16)[:, None]
    erp_ = topo * (np.exp(-(((t - 100) / 30) ** 2)) + 0.5 * np.exp(-(((t - 300) / 40) ** 2)))
    assert [round(p) for p in erp.gfp_peaks(erp_, t)] == [100, 300]
    rng = np.random.default_rng(3)
    noisy = erp_ + 0.01 * rng.normal(size=erp_.shape)
    assert erp.baseline_ratio(noisy, t, (-300, -200)) > 10


def test_split_half_reliability_high_for_consistent_runs() -> None:
    rng = np.random.default_rng(4)
    t = np.arange(-300, 801, 1000 / 240)
    signal = np.sin(t / 50.0)[None, :] * np.ones((8, 1))
    runs = signal[None] + 0.3 * rng.normal(size=(30, 8, len(t)))
    assert erp.split_half_reliability(runs, t, n_splits=20) > 0.95


def test_hf_slope_of_power_law() -> None:
    f = np.linspace(1, 120, 500)
    p = np.vstack([f ** -2.0, f ** 1.0])  # -20 and +10 dB per decade
    s = psd.hf_slope(f, p)
    assert np.allclose(s, [-20.0, 10.0], atol=1e-6)


def test_welch_and_median_curve() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3, 240 * 20))
    f, p = psd.welch(x, 240.0)
    med, mad = psd.run_median(np.stack([p, 2 * p, 3 * p]))
    assert np.allclose(med, 2 * p)
    curve, band = psd.median_curve_db(med, mad, 3)
    assert curve.shape == f.shape and np.all(band > 0)
