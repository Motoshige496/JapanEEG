"""Redraw Figures 2-4 of the Data Descriptor from the aggregate arrays and print their statistics.

Usage
-----
uv run python scripts/redraw_figures.py                       # aggregates/ -> figures/
uv run python scripts/redraw_figures.py --aggregates outputs/release_reproduction --out figures_release

Printed per ERP condition: runs, trials, the ratio of the post-onset RMS (0-400 ms) to the
baseline SD and the two largest GFP peaks; per PSD curve: the median 60-118 Hz slope.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from japaneeg import erp, figures, psd
from japaneeg.aggregates import load_all


def main() -> None:
    """Draw the figures and print the statistics."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregates", default="aggregates", help="folder with erp/ and psd/")
    ap.add_argument("--out", default="figures", help="output folder")
    args = ap.parse_args()
    erps, psds = load_all(args.aggregates)
    out = Path(args.out)
    if psds:
        figures.save(figures.figure2(psds), out / "fig2_psd.png")
    if erps:
        figures.save(figures.figure3(erps), out / "fig3_erp.png")
        figures.save(figures.figure4(erps), out / "fig4_topomap.png")
    print(f"{'condition':26s} {'runs':>5s} {'trials':>7s} {'ratio':>6s}  GFP peaks (ms)")
    for name, agg in erps.items():
        trial_type = name.split("-")[1]
        base = erp.BASELINES_MS["overt" if trial_type == "overt" else "covert"]
        ratio = erp.baseline_ratio(agg.eeg_grand_avg, agg.times_ms, base)
        sep = 40.0 if trial_type == "listening" else 100.0
        peaks = erp.gfp_peaks(agg.eeg_grand_avg, agg.times_ms, min_sep_ms=sep)
        print(f"{name:26s} {agg.meta.get('n_runs', '')!s:>5s} {agg.meta.get('n_trials', '')!s:>7s} "
              f"{ratio:6.1f}  {', '.join(f'{p:.0f}' for p in peaks)}")  # fmt: skip
    for (name, kind), agg in psds.items():
        slope = float(np.median(psd.hf_slope(agg.freqs, agg.median)))
        print(f"{name:26s} {kind:13s} 60-118 Hz slope {slope:+.1f} dB/decade")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
