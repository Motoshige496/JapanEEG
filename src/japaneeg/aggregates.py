"""Aggregate arrays behind Figures 2-4 and their file format.

``aggregates/erp/<condition>.npz`` (one per ERP panel):

* ``eeg_grand_avg`` ``(n_ch, n_samples)``: grand-average ERP in units of the SD of single epochs;
* ``times`` (s); ``n_runs_per_channel``: runs contributing to each channel;
* ``ch_names``; ``color_rank``: index of each channel on the viridis scale of the butterfly plot;
* ``layout``: ``"lateral"`` (g.Pangolin, left lateral view) or ``"top"`` (caps);
* ``pos``: ``(n_ch, 2)`` positions in the lateral view (m, anterior to the left) for
  ``"lateral"``, or ``(n_ch, 3)`` head positions (m) for ``"top"``;
* ``head_radius`` and ``nasion_dist`` (m, ``"lateral"`` only): outline of the lateral head.

``aggregates/erp/<condition>.json``: participant, device, task, trial type, run and trial counts,
baseline, rejection and the processing settings.

``aggregates/psd/<condition>_<kind>.npz`` (``kind`` = ``raw`` or ``preprocessed``): ``freqs``
(Hz), ``median`` and ``mad`` ``(n_ch, n_freqs)`` of the linear PSD over runs (raw: µV²/Hz;
preprocessed: units of the standardised signal), ``n_runs``.

Conditions use the participant labels of the paper (P01-P03).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ERP_CONDITIONS: tuple[str, ...] = (
    "P01-overt-pangolin",
    "P02-overt-pangolin",
    "P03-overt-pangolin",
    "P01-overt-scarabeo",
    "P01-overt-eego",
    "P01-covert-pangolin",
    "P01-listening-pangolin",
)
PSD_CONDITIONS: tuple[str, ...] = ERP_CONDITIONS
DEVICE_LABELS: dict[str, str] = {"pangolin": "g.Pangolin", "scarabeo": "g.SCARABEO", "eego": "eego™ sports"}


@dataclass
class ErpAggregate:
    """One ERP panel (see the module docstring)."""

    name: str
    eeg_grand_avg: NDArray[np.float64]
    times: NDArray[np.float64]
    n_runs_per_channel: NDArray[np.int32]
    ch_names: list[str]
    color_rank: NDArray[np.int32]
    layout: str
    pos: NDArray[np.float64]
    head_radius: float
    nasion_dist: float
    meta: dict[str, object]

    @property
    def device(self) -> str:
        """Device label of the condition name (``pangolin``, ``scarabeo`` or ``eego``)."""
        return self.name.split("-")[2]

    @property
    def times_ms(self) -> NDArray[np.float64]:
        """Time axis in ms."""
        return self.times * 1000.0


@dataclass
class PsdAggregate:
    """One PSD curve (see the module docstring)."""

    name: str
    kind: str
    freqs: NDArray[np.float64]
    median: NDArray[np.float64]
    mad: NDArray[np.float64]
    n_runs: int


def save_erp(path: str | Path, agg: ErpAggregate) -> None:
    """Write an ERP aggregate (``.npz`` and ``.json``).

    Args:
        path: ``<dir>/<condition>.npz``.
        agg: The aggregate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        eeg_grand_avg=agg.eeg_grand_avg.astype(np.float32),
        times=agg.times,
        n_runs_per_channel=agg.n_runs_per_channel,
        ch_names=np.array(agg.ch_names),
        color_rank=agg.color_rank,
        layout=np.array(agg.layout),
        pos=agg.pos,
        head_radius=np.array(agg.head_radius),
        nasion_dist=np.array(agg.nasion_dist),
    )
    path.with_suffix(".json").write_text(json.dumps(agg.meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_erp(path: str | Path) -> ErpAggregate:
    """Read an ERP aggregate.

    Args:
        path: ``<dir>/<condition>.npz``.

    Returns:
        The aggregate.
    """
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return ErpAggregate(
        name=path.stem,
        eeg_grand_avg=np.asarray(z["eeg_grand_avg"], np.float64),
        times=np.asarray(z["times"], np.float64),
        n_runs_per_channel=np.asarray(z["n_runs_per_channel"], np.int32),
        ch_names=[str(c) for c in z["ch_names"]],
        color_rank=np.asarray(z["color_rank"], np.int32),
        layout=str(z["layout"]),
        pos=np.asarray(z["pos"], np.float64),
        head_radius=float(z["head_radius"]),
        nasion_dist=float(z["nasion_dist"]),
        meta=meta,
    )


def save_psd(path: str | Path, agg: PsdAggregate) -> None:
    """Write a PSD aggregate.

    Args:
        path: ``<dir>/<condition>_<kind>.npz``.
        agg: The aggregate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, freqs=agg.freqs, median=agg.median.astype(np.float64), mad=agg.mad.astype(np.float64),
                        n_runs=np.array(agg.n_runs), kind=np.array(agg.kind))  # fmt: skip


def load_psd(path: str | Path) -> PsdAggregate:
    """Read a PSD aggregate.

    Args:
        path: ``<dir>/<condition>_<kind>.npz``.

    Returns:
        The aggregate.
    """
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    name, _, kind = path.stem.rpartition("_")
    return PsdAggregate(name, kind, np.asarray(z["freqs"]), np.asarray(z["median"]), np.asarray(z["mad"]), int(z["n_runs"]))


def load_all(root: str | Path) -> tuple[dict[str, ErpAggregate], dict[tuple[str, str], PsdAggregate]]:
    """All aggregates found under ``root`` (``erp/`` and ``psd/``).

    Args:
        root: Aggregates folder.

    Returns:
        ERP aggregates by condition and PSD aggregates by (condition, kind).
    """
    root = Path(root)
    erps = {p.stem: load_erp(p) for p in sorted((root / "erp").glob("*.npz"))}
    psds = {}
    for p in sorted((root / "psd").glob("*.npz")):
        a = load_psd(p)
        psds[(a.name, a.kind)] = a
    return erps, psds
