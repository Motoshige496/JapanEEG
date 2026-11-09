"""ERP statistics of the paper from the output of ``scripts/reproduce_from_release.py``.

For every condition in ``<out>/erp``: runs (files), trials, the ratio of the post-onset RMS
(0-400 ms) to the baseline SD, the two largest GFP peaks and the split-half reliability (runs
split at random into two halves 200 times, halves correlated over all channels from -200 to
800 ms, Spearman-Brown corrected). If ``<out>/erp_without_adaptive_filter`` exists: the Pearson
correlation of the grand averages with and without the adaptive filter over all channels from 0
to 800 ms, and how much lower the RMS amplitude is with the filter.

The values are written to ``<out>/statistics.tsv``. They come from the released files and
therefore approximate the paper's values (see the README). The ratio and GFP peaks of the
published figures are printed exactly by ``scripts/redraw_figures.py``.

Usage
-----
uv run python scripts/release_statistics.py --out outputs/release_reproduction
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from japaneeg import erp
from japaneeg.aggregates import load_erp

WITHOUT = "_without_adaptive_filter"


def main() -> None:
    """Print and write the statistics."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/release_reproduction", help="output folder of reproduce_from_release.py")
    ap.add_argument("--n-splits", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    rows = []
    for path in sorted((out / "erp").glob("*.npz")):
        agg = load_erp(path)
        name = path.stem
        trial_type = name.split("-")[1]
        t = agg.times_ms
        row = {"condition": name, "files": str(agg.meta.get("n_runs", "")), "trials": str(agg.meta.get("n_trials", ""))}
        row["ratio"] = f"{erp.baseline_ratio(agg.eeg_grand_avg, t, erp.BASELINES_MS[trial_type]):.1f}"
        peaks = erp.gfp_peaks(agg.eeg_grand_avg, t, min_sep_ms=40.0 if trial_type == "listening" else 100.0)
        row["gfp_peaks_ms"] = ", ".join(f"{p:.0f}" for p in peaks)
        per_file = out / "per_file" / f"{name}.npy"
        row["split_half"] = (f"{erp.split_half_reliability(np.load(per_file), t, n_splits=args.n_splits):.3f}"
                             if per_file.is_file() else "")  # fmt: skip
        row["r_without_adaptive_filter"] = row["rms_lower_with_filter_pct"] = ""
        other = out / ("erp" + WITHOUT) / path.name
        if other.is_file():
            a, b = agg.eeg_grand_avg, load_erp(other).eeg_grand_avg
            m = (t >= 0) & (t <= 800)
            x, y = a[:, m].ravel(), b[:, m].ravel()
            ok = np.isfinite(x) & np.isfinite(y)
            row["r_without_adaptive_filter"] = f"{np.corrcoef(x[ok], y[ok])[0, 1]:.4f}"
            rms_with, rms_without = np.sqrt(np.mean(x[ok] ** 2)), np.sqrt(np.mean(y[ok] ** 2))
            row["rms_lower_with_filter_pct"] = f"{100 * (1 - rms_with / rms_without):.1f}"
        rows.append(row)
        print("  ".join(f"{k}={v}" for k, v in row.items() if v))
    if rows:
        with open(out / "statistics.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out / 'statistics.tsv'}")


if __name__ == "__main__":
    main()
