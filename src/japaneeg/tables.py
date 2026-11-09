"""Event counts and hours of Tables 4 and 5 from the released files.

Speech and task events are the rows of ``events.tsv`` whose ``trial_type`` is ``overt``,
``covert`` or ``listening`` (``covert_irregular`` marks five interrupted covert trials that the
tables do not count). Recording hours are summed from the EDF headers per participant, task and
device; the two trial types of the covert repetition task share the same recordings. The
recording days and periods of Table 4 come from the session dates of the continuous runs; the
``ses-shuffled`` files carry no dates, so they are reported for the continuous runs only.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from japaneeg.bids import find_runs, read_tsv, sidecar

EVENT_TYPES = ("overt", "covert", "listening")
NAME = re.compile(r"^sub-(\d+)_ses-([^_]+)_task-([^_]+)_acq-([^_]+)_run-")


def edf_hours(path: Path) -> float:
    """Recording length from the EDF header (records x record duration), in hours.

    Args:
        path: EDF file.

    Returns:
        Hours.
    """
    with open(path, "rb") as f:
        head = f.read(256)
    n_records = int(head[236:244].decode("ascii").strip())
    duration = float(head[244:252].decode("ascii").strip())
    return n_records * duration / 3600.0


@dataclass
class Row:
    """One configuration (participant, task, device, trial type)."""

    events: int = 0
    seconds: float = 0.0
    durations: list[float] = field(default_factory=list)


def count(root: str | Path) -> tuple[dict[tuple[str, str, str, str], Row], dict[tuple[str, str, str], float],
                                     dict[tuple[str, str, str], list[str]]]:  # fmt: skip
    """Events, event durations and recording hours of the whole release.

    Args:
        root: BIDS root.

    Returns:
        Rows by (participant, task, device, trial type); recording hours by (participant,
        task, device); session dates of the continuous runs by (participant, task, device).
    """
    rows: dict[tuple[str, str, str, str], Row] = defaultdict(Row)
    hours: dict[tuple[str, str, str], float] = defaultdict(float)
    dates: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for edf in find_runs(root):
        m = NAME.match(edf.name)
        if m is None:
            continue
        sub, ses, task, acq = m.groups()
        key = (f"P{sub}", task, acq)
        hours[key] += edf_hours(edf)
        if ses != "shuffled":
            dates[key].append(ses)
        for r in read_tsv(sidecar(edf, "events.tsv")):
            if r.get("trial_type") in EVENT_TYPES:
                d = float(r["duration"])
                row = rows[(*key, r["trial_type"])]
                row.events += 1
                row.seconds += d
                row.durations.append(d)
    return rows, hours, dates


def table4(rows: dict[tuple[str, str, str, str], Row], hours: dict[tuple[str, str, str], float],
           dates: dict[tuple[str, str, str], list[str]]) -> list[dict[str, str]]:  # fmt: skip
    """Rows of Table 4.

    Args:
        rows: From :func:`count`.
        hours: From :func:`count`.
        dates: From :func:`count`.

    Returns:
        One row per participant, task, device and trial type.
    """
    out = []
    for (pid, task, acq, trial), r in sorted(rows.items()):
        d = sorted(set(dates.get((pid, task, acq), [])))
        out.append({
            "participant": pid, "task": task, "device": acq, "trial_type": trial, "events": str(r.events),
            "event_hours": f"{r.seconds / 3600:.1f}", "rec_hours": f"{hours[(pid, task, acq)]:.1f}",
            "days_continuous": str(len(d)),
            "period_continuous": f"{d[0][:4]}-{d[0][4:6]} -- {d[-1][:4]}-{d[-1][4:6]}" if d else "",
        })  # fmt: skip
    return out


def table5(rows: dict[tuple[str, str, str, str], Row]) -> list[dict[str, str]]:
    """Rows of Table 5 (per trial type over all configurations).

    Args:
        rows: From :func:`count`.

    Returns:
        One row per trial type.
    """
    out = []
    for trial in EVENT_TYPES:
        d = np.concatenate([np.asarray(r.durations) for k, r in rows.items() if k[3] == trial] or [np.zeros(0)])
        if not d.size:
            continue
        out.append({"trial_type": trial, "events": str(d.size), "total_hours": f"{d.sum() / 3600:.1f}",
                    "min_s": f"{d.min():.2f}", "max_s": f"{d.max():.2f}", "mean_s": f"{d.mean():.2f}",
                    "sd_s": f"{d.std(ddof=1):.2f}"})  # fmt: skip
    return out
