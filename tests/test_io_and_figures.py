"""Tests of the release readers, the download helpers and the figures on synthetic files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from japaneeg import bids, download, figures
from japaneeg.aggregates import ERP_CONDITIONS, ErpAggregate, PsdAggregate, load_all, save_erp, save_psd


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    path.write_text("\t".join(keys) + "\n" + "".join("\t".join(r[k] for k in keys) + "\n" for r in rows),
                    encoding="utf-8")  # fmt: skip


def test_segments_of_shuffled_file(tmp_path: Path) -> None:
    edf = tmp_path / "sub-01/ses-shuffled/eeg/sub-01_ses-shuffled_task-speechopen_acq-pangolin_run-01_eeg.edf"
    rows = [
        {"onset": "10.5", "duration": "2.0", "trial_type": "overt", "frag_id": "b", "segment_onset": "10.0",
         "segment_duration": "4.0", "source_run_bad_channels": json.dumps(["E12"])},
        {"onset": "0.4", "duration": "1.5", "trial_type": "overt", "frag_id": "a", "segment_onset": "0.0",
         "segment_duration": "10.0", "source_run_bad_channels": "[]"},
        {"onset": "10.0", "duration": "0", "trial_type": "boundary", "frag_id": "n/a", "segment_onset": "n/a",
         "segment_duration": "n/a", "source_run_bad_channels": "n/a"},
    ]  # fmt: skip
    _write_tsv(bids.sidecar(edf, "events.tsv"), rows)
    segs = bids.segments(edf)
    assert [s.frag_id for s in segs] == ["a", "b"]
    assert segs[1].bad_channels == ("E12",) and segs[1].onset == 10.0 and len(segs[1].events) == 1
    assert bids.find_runs(tmp_path) == []  # no EDF written
    edf.write_bytes(b"")
    assert bids.find_runs(tmp_path, subject="01", session="shuffled") == [edf]


def test_scalp_eeg_excludes_reference_sites() -> None:
    chans = [{"name": n, "type": t} for n, t in [("Fp1", "EEG"), ("M1", "EEG"), ("EEG063", "EEG"), ("EOG", "EOG")]]
    assert bids.scalp_eeg(chans) == ["Fp1"]


def test_manifest_selection(tmp_path: Path) -> None:
    man = tmp_path / "archives_manifest.tsv"
    _write_tsv(man, [
        {"archive": "archives/sub-01/sub-01_ses-20230829.tar.xz", "n_files": "3", "raw_bytes": "9", "bytes": "5", "sha256": "x"},
        {"archive": "sidecars/sub-01_sidecars.tar.xz", "n_files": "3", "raw_bytes": "9", "bytes": "5", "sha256": "y"},
    ])  # fmt: skip
    arch = download.read_manifest(man)
    assert [a.path for a in download.select(arch, ["sidecars/*"])] == ["sidecars/sub-01_sidecars.tar.xz"]
    assert len(download.select(arch, None)) == 2


def _synthetic_aggregates(root: Path) -> None:
    rng = np.random.default_rng(0)
    t = np.arange(-72, 193) / 240.0
    for name in ERP_CONDITIONS:
        lateral = name.endswith("pangolin")
        n = 128 if lateral else 32
        if lateral:
            pos = rng.uniform(-0.06, 0.06, (n, 2))
            names = [f"E{i}" for i in range(n)]
        else:
            import mne

            mont = mne.channels.make_standard_montage("standard_1020")
            names = mont.ch_names[:n]
            pos = np.array([mont.get_positions()["ch_pos"][c] for c in names])
        save_erp(root / "erp" / f"{name}.npz", ErpAggregate(
            name, rng.normal(0, 0.05, (n, len(t))), t, np.full(n, 10, np.int32), names,
            rng.permutation(n).astype(np.int32), "lateral" if lateral else "top", pos, 0.09, 0.1,
            {"n_runs": 10, "n_trials": 1000}))  # fmt: skip
        f = np.linspace(0, 120, 481)
        for kind in ("raw", "preprocessed"):
            save_psd(root / "psd" / f"{name}_{kind}.npz",
                     PsdAggregate(name, kind, f, np.abs(rng.normal(1, 0.1, (n, len(f)))), np.full((n, len(f)), 0.1), 10))


def test_figures_from_synthetic_aggregates(tmp_path: Path) -> None:
    _synthetic_aggregates(tmp_path / "agg")
    erps, psds = load_all(tmp_path / "agg")
    assert set(erps) == set(ERP_CONDITIONS) and len(psds) == 2 * len(ERP_CONDITIONS)
    figures.save(figures.figure2(psds), tmp_path / "fig2.png")
    figures.save(figures.figure3(erps), tmp_path / "fig3.png")
    figures.save(figures.figure4(erps), tmp_path / "fig4.png")
    assert all((tmp_path / f"fig{k}.png").stat().st_size > 10000 for k in (2, 3, 4))
