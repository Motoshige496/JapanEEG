"""End-to-end run of the command-line scripts on a small synthetic BIDS tree (EDF files written here)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SFREQ = 500
N_EEG = 10


def write_edf(path: Path, names: list[str], units: list[str], data: np.ndarray, sfreq: int) -> None:
    """Minimal EDF writer: 1-s records, 16-bit samples, fixed physical range per channel."""
    ns, n = data.shape
    n_rec = n // sfreq
    phys = [3200.0 if u == "uV" else 10.0 for u in units]

    def field(values: list[str], width: int) -> bytes:
        return b"".join(v[:width].ljust(width).encode("ascii") for v in values)

    head = (field(["0"], 8) + field(["X X X X"], 80) + field(["Startdate X X X X"], 80) + field(["01.01.23"], 8)
            + field(["00.00.00"], 8) + field([str(256 * (ns + 1))], 8) + field([""], 44) + field([str(n_rec)], 8)
            + field(["1"], 8) + field([str(ns)], 4))  # fmt: skip
    head += field(names, 16) + field([""] * ns, 80) + field(units, 8)
    head += field([f"{-p:g}" for p in phys], 8) + field([f"{p:g}" for p in phys], 8)
    head += field(["-32768"] * ns, 8) + field(["32767"] * ns, 8) + field([""] * ns, 80)
    head += field([str(sfreq)] * ns, 8) + field([""] * ns, 32)
    scale = np.array([32767.0 / p for p in phys])[:, None]
    digital = np.clip(np.round(data[:, : n_rec * sfreq] * scale), -32768, 32767).astype("<i2")
    body = digital.reshape(ns, n_rec, sfreq).transpose(1, 0, 2).tobytes()
    path.write_bytes(head + body)


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    """TSV with the union of the keys as columns (missing: n/a)."""
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", restval="n/a")
        w.writeheader()
        w.writerows(rows)


def make_run(
    folder: Path, stem: str, seconds: int, seed: int, segments: list[tuple[float, float]] | None = None
) -> None:
    """One synthetic overt-speech run: EEG, three references, two microphone channels and events."""
    rng = np.random.default_rng(seed)
    n = seconds * SFREQ
    t = np.arange(n) / SFREQ
    onsets = np.arange(3.0, seconds - 3.0, 4.0)
    speaking = np.zeros(n, bool)
    for on in onsets:
        speaking[int(on * SFREQ) : int((on + 1.5) * SFREQ)] = True
    muscle = rng.normal(0, 30.0, n) * speaking
    erp = np.zeros(n)
    for on in onsets:
        k = (t >= on - 0.1) & (t < on + 0.4)
        erp[k] += 8.0 * np.sin(2 * np.pi * 4 * (t[k] - on + 0.1))
    eeg = rng.normal(0, 10.0, (N_EEG, n)) + erp + 0.8 * muscle + 5 * np.sin(2 * np.pi * 50 * t)
    refs = np.stack([rng.normal(0, 20.0, n), muscle + rng.normal(0, 5.0, n), muscle + rng.normal(0, 5.0, n)])
    env = np.zeros(n)
    for on in onsets:
        env[int((on + 0.04) * SFREQ) : int((on + 1.5) * SFREQ)] = 1.0
    mic = np.stack([env + rng.normal(0, 0.01, n), np.sin(2 * np.pi * 150 * t) * env + rng.normal(0, 0.01, n)])
    # microphone channels named like scalp channels and typed EEG, as on some caps: found by description
    names = [f"EEG{k:03d}" for k in range(1, N_EEG + 3)] + ["EOG", "UL", "LL"]
    names = names[:N_EEG] + names[N_EEG + 2 :] + names[N_EEG : N_EEG + 2]
    units = ["uV"] * (N_EEG + 3) + ["V", "V"]
    folder.mkdir(parents=True, exist_ok=True)
    write_edf(folder / f"{stem}_eeg.edf", names, units, np.concatenate([eeg, refs, mic]), SFREQ)
    types = ["EEG"] * N_EEG + ["EOG", "EMG_UPPER_LIP", "EMG_LOWER_LIP", "EEG", "EEG"]
    desc = [""] * (N_EEG + 3) + ["microphone; 30 Hz amplitude envelope of the microphone signal",
                                 "microphone; voice-masked speech on the EEG time line, see README"]  # fmt: skip
    write_tsv(folder / f"{stem}_channels.tsv",
              [{"name": a, "type": b, "units": u, "status": "bad" if a == "EEG003" and not segments else "good",
                "description": d} for a, b, u, d in zip(names, types, units, desc)])  # fmt: skip
    rows: list[dict[str, str]] = []
    for on in onsets:
        rows.append({"onset": f"{on:.3f}", "duration": "1.500", "trial_type": "overt", "value": "n/a"})
        rows.append({"onset": f"{on:.3f}", "duration": "1.500", "trial_type": "word", "value": "今日"})
        for k in range(15):
            rows.append({"onset": f"{on + 0.1 * k:.3f}", "duration": "0.100", "trial_type": "phone", "value": "a"})
    rows.append({"onset": "20.000", "duration": "0.500", "trial_type": "BAD_high_amplitude", "value": "n/a"})
    if segments:
        for r in rows:
            on = float(r["onset"])
            for k, (a, d) in enumerate(segments):
                if a <= on < a + d:
                    r.update({"frag_id": f"f{k}", "segment_onset": f"{a}", "segment_duration": f"{d}",
                              "source_run_bad_channels": json.dumps(["EEG003"])})  # fmt: skip
        rows = [r for r in rows if "frag_id" in r]
    write_tsv(folder / f"{stem}_events.tsv", sorted(rows, key=lambda r: float(r["onset"])))


@pytest.fixture(scope="module")
def bids_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bids")
    for k, ses in enumerate(("20230101", "20230102", "20230201")):
        make_run(root / "sub-01" / f"ses-{ses}" / "eeg", f"sub-01_ses-{ses}_task-speechopen_acq-pangolin_run-01", 90, k)
    make_run(root / "sub-01" / "ses-shuffled" / "eeg", "sub-01_ses-shuffled_task-speechopen_acq-pangolin_run-01",
             90, 9, segments=[(0.0, 44.0), (44.0, 46.0)])  # fmt: skip
    return root


def run(*args: str) -> str:
    """Run a script of ``scripts/`` and return its output."""
    res = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


def test_paper_numbers(bids_root: Path, tmp_path: Path) -> None:
    common = ["--bids-root", str(bids_root), "--out", str(tmp_path), "--n-workers", "2"]
    out = run("scripts/paper_numbers.py", "tables", *common)
    assert "overt" in out
    out = run("scripts/paper_numbers.py", "articulation", *common)
    assert "phones/s 10.00" in out
    out = run("scripts/paper_numbers.py", "onset", *common)
    # true onset 40 ms; the zero-phase filter and the centred smoothing detect it a little earlier
    assert 15.0 <= float(out.split("median (IQR) ")[1].split()[0]) <= 55.0, out
    out = run("scripts/paper_numbers.py", "speech-pause", *common)
    assert "recording:" in out and "car:" in out
    out = run("scripts/paper_numbers.py", "coherence", *common)
    assert "car_chance" in out
    out = run("scripts/paper_numbers.py", "bad-channels", *common)
    assert "median_bad=1" in out


def test_reproduce_and_statistics(bids_root: Path, tmp_path: Path) -> None:
    run("scripts/reproduce_from_release.py", "--bids-root", str(bids_root), "--out", str(tmp_path),
        "--conditions", "P01-overt-pangolin", "--without-adaptive-filter", "--paper", str(tmp_path / "none"),
        "--n-workers", "2")  # fmt: skip
    out = run("scripts/release_statistics.py", "--out", str(tmp_path), "--n-splits", "20")
    assert "split_half=" in out and "r_without_adaptive_filter=" in out


def test_synchronized_example(bids_root: Path, tmp_path: Path) -> None:
    edf = next((bids_root / "sub-01" / "ses-20230101" / "eeg").glob("*_eeg.edf"))
    run("scripts/plot_synchronized_example.py", "--edf", str(edf), "--event", "2", "--eeg-rows", "0", "4",
        "--out", str(tmp_path / "fig.png"))  # fmt: skip
    assert (tmp_path / "fig.png").stat().st_size > 10000
