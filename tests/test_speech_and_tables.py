"""Tests of the speech measures, the speech/pause intervals and the table counts on synthetic files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from japaneeg import emg, speech, tables


def write_events(path: Path, rows: list[dict[str, str]]) -> None:
    """Write an events.tsv with the columns used by the release."""
    cols = ["onset", "duration", "trial_type", "value", "alignment_flag", "frag_id"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", restval="n/a")
        w.writeheader()
        w.writerows(rows)


def ev(onset: float, duration: float, kind: str, value: str = "n/a", flag: str = "n/a") -> dict[str, str]:
    """One event row."""
    return {"onset": f"{onset}", "duration": f"{duration}", "trial_type": kind, "value": value, "alignment_flag": flag}


def test_utterance_rates_counts_phones_inside_events(tmp_path: Path) -> None:
    rows = [ev(1.0, 1.0, "overt"), ev(5.0, 1.0, "overt")]
    rows += [ev(1.0 + 0.1 * k, 0.1, "phone", "a") for k in range(8)]  # 8 phones, 0.8 s
    rows += [ev(1.8, 0.2, "phone", "sp")]  # pause: not counted
    rows += [ev(5.0 + 0.1 * k, 0.1, "phone", "i", "1" if k == 0 else "n/a") for k in range(5)]  # flagged
    path = tmp_path / "sub-07_ses-x_task-speechopen_acq-y_run-1_events.tsv"
    write_events(path, rows)
    out = speech.utterance_rates(path)
    assert len(out) == 1
    assert out[0]["participant"] == "P07" and out[0]["n_phones"] == "8"
    assert abs(float(out[0]["rate"]) - 10.0) < 1e-6


def test_onset_latency_finds_step() -> None:
    sf = 1000.0
    rng = np.random.default_rng(0)
    env = rng.normal(0.0, 0.01, 2000)
    env[1040:] += 1.0  # sound 40 ms after the onset at 1.0 s
    lat = speech.onset_latency(env, sf, 1.0)
    assert abs(lat - 0.040) < 0.002
    assert np.isnan(speech.onset_latency(rng.normal(0.0, 0.01, 2000), sf, 1.0))


def test_subtract_and_speech_pause_intervals() -> None:
    assert emg.subtract([(0.0, 10.0)], [(2.0, 3.0), (5.0, 6.0)]) == [(0.0, 2.0), (3.0, 5.0), (6.0, 10.0)]
    assert emg.subtract([(0.0, 1.0)], [(-1.0, 2.0)]) == []
    rows = [ev(0.0, 1.0, "overt"), ev(4.0, 1.0, "overt"), ev(5.5, 1.0, "overt"), ev(2.0, 0.2, "cough")]
    iv = emg.speech_pause_intervals(rows, min_pause=1.0, trim=0.3)
    assert iv["speech"] == [(0.0, 1.0), (4.0, 5.0), (5.5, 6.5)]
    # gap 1-4 s is split by the cough into 1-2 and 2.2-4; the 0.5 s gap 5-5.5 is too short
    assert [(round(a, 2), round(b, 2)) for a, b in iv["pause"]] == [(1.3, 1.7), (2.5, 3.7)]


def test_event_intervals_merge_and_drop_marks() -> None:
    rows = [ev(10.0, 1.0, "overt"), ev(11.5, 1.0, "overt"), ev(30.0, 1.0, "overt"), ev(29.0, 3.0, "BAD_high_amplitude")]
    iv = emg.event_intervals(rows, 0.0, 100.0)
    assert iv == [(8.5, 14.0)]


def write_edf_header(path: Path, n_records: int, duration: float) -> None:
    """Write the first 256 bytes of an EDF header (enough for :func:`tables.edf_hours`)."""
    head = bytearray(b" " * 256)
    head[236:244] = f"{n_records:<8d}".encode("ascii")
    head[244:252] = f"{duration:<8g}".encode("ascii")
    path.write_bytes(bytes(head))


def test_tables_count_events_and_hours(tmp_path: Path) -> None:
    for ses, n in (("20230101", 2), ("20230301", 1), ("shuffled", 1)):
        d = tmp_path / "sub-01" / f"ses-{ses}" / "eeg"
        d.mkdir(parents=True)
        stem = f"sub-01_ses-{ses}_task-speechopen_acq-pangolin_run-01"
        write_edf_header(d / f"{stem}_eeg.edf", 1800 * n, 1.0)
        write_events(d / f"{stem}_events.tsv", [ev(1.0, 2.0, "overt"), ev(5.0, 4.0, "overt"), ev(9.0, 1.0, "cue")])
    rows, hours, dates = tables.count(tmp_path)
    key = ("P01", "speechopen", "pangolin")
    assert abs(hours[key] - 2.0) < 1e-9
    assert rows[(*key, "overt")].events == 6
    t4 = tables.table4(rows, hours, dates)
    assert t4[0]["days_continuous"] == "2" and t4[0]["period_continuous"] == "2023-01 -- 2023-03"
    t5 = tables.table5(rows)
    assert t5[0]["events"] == "6" and t5[0]["mean_s"] == "3.00" and t5[0]["max_s"] == "4.00"
