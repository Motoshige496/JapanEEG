"""Recompute the ERPs of Figures 3 and 4 from the released files and compare them with the paper.

Every released file of a condition (continuous runs and ``ses-shuffled`` files) is processed as
described in ``japaneeg.pipeline``; files are processed in parallel. The grand averages are
written in the aggregate format (channel layout taken from ``aggregates/erp``), so that
``scripts/redraw_figures.py --aggregates <out>`` draws them like the paper's figures. The
correlation with the paper's grand average over all channels from -200 to 800 ms is printed and
written to ``<out>/comparison.tsv``. The epochs of every file are cached as float16 in
``<out>/cache`` (about 20 GB for all conditions; an interrupted run resumes from it).

Usage
-----
uv run python scripts/reproduce_from_release.py --bids-root data/JapanEEG/bids --out outputs/release_reproduction
uv run python scripts/reproduce_from_release.py --bids-root ... --conditions P01-overt-eego --max-files 3   # quick test
"""

from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from japaneeg import erp, pipeline
from japaneeg.aggregates import ErpAggregate, load_erp, save_erp
from japaneeg.preprocessing import SFREQ_OUT


def _job(args: tuple[str, str, str]) -> tuple[str, np.ndarray]:
    """Epochs of one file, cached as float16 (worker); returns the per-trial maxima."""
    edf, trial_type, cache = args
    path = Path(cache)
    if path.is_file():
        ep = np.load(path)
    else:
        ep = pipeline.process_file(Path(edf), trial_type).astype(np.float16)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, ep)
    maxabs = np.nanmax(np.abs(ep.astype(np.float32)), axis=(1, 2)) if len(ep) else np.zeros(0)
    return cache, maxabs


def average_cached(caches: list[str], maxabs: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Rejection over all cached trials, epoch normalisation, average per file and over files."""
    allmax = np.concatenate([maxabs[c] for c in caches])
    thr = erp.REJECT_FACTOR * float(np.median(allmax))
    means = []
    n_acc = 0
    for c in caches:
        keep = maxabs[c] <= thr
        if not keep.any():
            continue
        acc = np.load(c)[keep].astype(np.float64)
        acc /= np.nanstd(acc, axis=2, keepdims=True) + 1e-12
        means.append(np.nanmean(acc, axis=0))
        n_acc += int(keep.sum())
    stack = np.stack(means)
    with np.errstate(invalid="ignore"):
        grand = np.nanmean(stack, axis=0)
    return grand, {"selected": int(allmax.size), "accepted": n_acc}, np.sum(np.isfinite(stack[:, :, 0]), axis=0)


def main() -> None:
    """Process the conditions, write the aggregates and the comparison."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids-root", required=True, help="BIDS root of the release")
    ap.add_argument("--out", default="outputs/release_reproduction", help="output folder")
    ap.add_argument("--paper", default="aggregates/erp", help="paper aggregates (layout and comparison)")
    ap.add_argument("--conditions", nargs="*", default=list(pipeline.CONDITIONS), help="figure conditions")
    ap.add_argument("--max-files", type=int, default=0, help="process at most this many files per condition (test)")
    ap.add_argument("--n-workers", type=int, default=-1, help="-1: all CPUs but one")
    args = ap.parse_args()
    n_workers = max(1, (os.cpu_count() or 2) - 1) if args.n_workers < 0 else args.n_workers
    out = Path(args.out)
    rows = []
    times = erp.epoch_times(SFREQ_OUT)
    for name in args.conditions:
        cond = pipeline.CONDITIONS[name]
        files = pipeline.files_of(args.bids_root, cond)
        if args.max_files:
            files = files[: args.max_files]
        if not files:
            print(f"{name}: no files under {args.bids_root}")
            continue
        jobs = [(str(f), cond.trial_type, str(out / "cache" / name / (f.name[: -len("_eeg.edf")] + ".npy")))
                for f in files]  # fmt: skip
        maxabs: dict[str, np.ndarray] = {}
        with Pool(min(n_workers, len(files))) as pool:
            for k, (cache, mx) in enumerate(pool.imap_unordered(_job, jobs), 1):
                maxabs[cache] = mx
                print(f"{name}: [{k}/{len(files)}] {Path(cache).stem} {len(mx)} trials", flush=True)
        caches = [j[2] for j in jobs if len(maxabs[j[2]])]
        grand, counts, n_files = average_cached(caches, maxabs)
        paper_path = Path(args.paper) / f"{name}.npz"
        paper = load_erp(paper_path) if paper_path.is_file() else None
        n_ch = grand.shape[0]
        agg = ErpAggregate(
            name=name,
            eeg_grand_avg=grand,
            times=times,
            n_runs_per_channel=n_files.astype(np.int32),
            ch_names=paper.ch_names if paper else [f"ch{i}" for i in range(n_ch)],
            color_rank=paper.color_rank if paper else np.arange(n_ch, dtype=np.int32),
            layout=paper.layout if paper else "top",
            pos=paper.pos if paper else np.zeros((n_ch, 3)),
            head_radius=paper.head_radius if paper else 0.0,
            nasion_dist=paper.nasion_dist if paper else 0.0,
            meta={"source": "released files", "n_runs": len(files), "n_trials": counts["accepted"],
                  "n_selected": counts["selected"], "baseline_ms": list(erp.BASELINES_MS[cond.trial_type])},
        )  # fmt: skip
        save_erp(out / "erp" / f"{name}.npz", agg)
        r = pipeline.compare(grand, paper.eeg_grand_avg, times * 1000) if paper and paper.eeg_grand_avg.shape == grand.shape else float("nan")
        rejected = 100 * (1 - counts["accepted"] / max(counts["selected"], 1))
        rows.append({"condition": name, "files": len(files), "trials": counts["accepted"],
                     "rejected_pct": f"{rejected:.1f}", "r_vs_paper": f"{r:.3f}"})  # fmt: skip
        print(f"{name}: {len(files)} files, {counts['accepted']} trials, r vs paper {r:.3f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["condition", "files", "trials", "rejected_pct", "r_vs_paper"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
