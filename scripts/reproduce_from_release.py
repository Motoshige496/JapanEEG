"""Recompute the ERPs of Figures 3 and 4 from the released files and compare them with the paper.

Every released file of a condition (continuous runs and ``ses-shuffled`` files) is processed as
described in ``japaneeg.pipeline``; files are processed in parallel. The grand averages are
written in the aggregate format (channel layout taken from ``aggregates/erp``), so that
``scripts/redraw_figures.py --aggregates <out>`` draws them like the paper's figures. The
correlation with the paper's grand average over all channels from -200 to 800 ms is printed and
written to ``<out>/comparison.tsv``. The per-file averages are written to ``<out>/per_file`` for
the split-half reliability (``scripts/release_statistics.py``).

With ``--without-adaptive-filter``, the same epochs are also averaged without the NLMS filter
(same pieces, standardisation, onsets and accepted trials) into ``<out>/erp_without_adaptive_filter``,
as in the sensitivity check of the paper.

The epochs of every file are cached as float16 in ``<out>/cache`` (about 20 GB for all
conditions, twice that with ``--without-adaptive-filter``); an interrupted run resumes from it.

Usage
-----
uv run python scripts/reproduce_from_release.py --bids-root data/JapanEEG/bids --out outputs/release_reproduction
uv run python scripts/reproduce_from_release.py --bids-root data/JapanEEG/bids --out outputs/release_reproduction \\
    --conditions P01-overt-pangolin P01-overt-scarabeo P01-overt-eego P01-listening-pangolin --without-adaptive-filter
uv run python scripts/reproduce_from_release.py --bids-root ... --conditions P01-overt-eego --max-files 3   # quick test
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from japaneeg import erp, pipeline
from japaneeg.aggregates import ErpAggregate, load_erp, save_erp
from japaneeg.preprocessing import SFREQ_OUT

WITHOUT = "_without_adaptive_filter"


def _job(args: tuple[str, str, str, bool]) -> tuple[str, NDArray[np.float32]]:
    """Epochs of one file, cached as float16 (worker); returns the per-trial maxima."""
    edf, trial_type, cache, without = args
    path = Path(cache)
    raw_path = path.with_name(path.stem + WITHOUT + ".npy")
    if path.is_file() and (raw_path.is_file() or not without):
        ep = np.load(path)
    else:
        res = pipeline.process_file(Path(edf), trial_type, without_adaptive_filter=without)
        ep, ep_raw = res if isinstance(res, tuple) else (res, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, ep.astype(np.float16))
        if ep_raw is not None:
            np.save(raw_path, ep_raw.astype(np.float16))
        ep = ep.astype(np.float16)
    maxabs = np.nanmax(np.abs(ep.astype(np.float32)), axis=(1, 2)) if len(ep) else np.zeros(0, np.float32)
    return cache, maxabs


def average_cached(
    caches: list[str], maxabs: dict[str, NDArray[np.float32]], suffix: str = ""
) -> tuple[NDArray[np.float64], dict[str, int], NDArray[np.int32], NDArray[np.float32]]:
    """Rejection over all cached trials, epoch normalisation, average per file and over files.

    Args:
        caches: Cache files of the condition (epochs with the adaptive filter).
        maxabs: Per-trial maxima of the epochs with the adaptive filter; they decide the
            rejection for both arms, so the accepted trials are the same.
        suffix: ``""`` for the epochs with the adaptive filter, ``WITHOUT`` for those without.

    Returns:
        Grand average, trial counts, files per channel and the per-file averages.
    """
    allmax = np.concatenate([maxabs[c] for c in caches])
    thr = erp.REJECT_FACTOR * float(np.median(allmax))
    means = []
    n_acc = 0
    for c in caches:
        keep = maxabs[c] <= thr
        if not keep.any():
            continue
        path = Path(c)
        acc = np.load(path.with_name(path.stem + suffix + ".npy"))[keep].astype(np.float64)
        with warnings.catch_warnings():  # bad channels are NaN in every trial of a file
            warnings.simplefilter("ignore", RuntimeWarning)
            acc /= np.nanstd(acc, axis=2, keepdims=True) + 1e-12
            means.append(np.nanmean(acc, axis=0))
        n_acc += int(keep.sum())
    stack = np.stack(means)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grand = np.nanmean(stack, axis=0)
    n_files = np.sum(np.isfinite(stack[:, :, 0]), axis=0).astype(np.int32)
    return grand, {"selected": int(allmax.size), "accepted": n_acc}, n_files, stack.astype(np.float32)


def main() -> None:
    """Process the conditions, write the aggregates and the comparison."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids-root", required=True, help="BIDS root of the release")
    ap.add_argument("--out", default="outputs/release_reproduction", help="output folder")
    ap.add_argument("--paper", default="aggregates/erp", help="paper aggregates (layout and comparison)")
    ap.add_argument("--conditions", nargs="*", default=list(pipeline.CONDITIONS), help="figure conditions")
    ap.add_argument("--without-adaptive-filter", action="store_true", help="also average without the NLMS filter")
    ap.add_argument("--max-files", type=int, default=0, help="process at most this many files per condition (test)")
    ap.add_argument("--n-workers", type=int, default=-1, help="-1: all CPUs but one")
    args = ap.parse_args()
    n_workers = max(1, (os.cpu_count() or 2) - 1) if args.n_workers < 0 else args.n_workers
    out = Path(args.out)
    rows = []
    times = erp.epoch_times(SFREQ_OUT)
    arms = [("", "erp")] + ([(WITHOUT, "erp" + WITHOUT)] if args.without_adaptive_filter else [])
    for name in args.conditions:
        cond = pipeline.CONDITIONS[name]
        files = pipeline.files_of(args.bids_root, cond)
        if args.max_files:
            files = files[: args.max_files]
        if not files:
            print(f"{name}: no files under {args.bids_root}")
            continue
        jobs = [(str(f), cond.trial_type, str(out / "cache" / name / (f.name[: -len("_eeg.edf")] + ".npy")),
                 args.without_adaptive_filter) for f in files]  # fmt: skip
        maxabs: dict[str, NDArray[np.float32]] = {}
        with Pool(min(n_workers, len(files))) as pool:
            for k, (cache, mx) in enumerate(pool.imap_unordered(_job, jobs), 1):
                maxabs[cache] = mx
                print(f"{name}: [{k}/{len(files)}] {Path(cache).stem} {len(mx)} trials", flush=True)
        caches = [j[2] for j in jobs if len(maxabs[j[2]])]
        paper_path = Path(args.paper) / f"{name}.npz"
        paper = load_erp(paper_path) if paper_path.is_file() else None
        for suffix, folder in arms:
            grand, counts, n_files, per_file = average_cached(caches, maxabs, suffix)
            n_ch = grand.shape[0]
            agg = ErpAggregate(
                name=name,
                eeg_grand_avg=grand,
                times=times,
                n_runs_per_channel=n_files,
                ch_names=paper.ch_names if paper else [f"ch{i}" for i in range(n_ch)],
                color_rank=paper.color_rank if paper else np.arange(n_ch, dtype=np.int32),
                layout=paper.layout if paper else "top",
                pos=paper.pos if paper else np.zeros((n_ch, 3)),
                head_radius=paper.head_radius if paper else 0.0,
                nasion_dist=paper.nasion_dist if paper else 0.0,
                meta={"source": "released files", "n_runs": len(per_file), "n_trials": counts["accepted"],
                      "n_selected": counts["selected"], "baseline_ms": list(erp.BASELINES_MS[cond.trial_type]),
                      "adaptive_filter": suffix == ""},
            )  # fmt: skip
            save_erp(out / folder / f"{name}.npz", agg)
            (out / "per_file").mkdir(parents=True, exist_ok=True)
            np.save(out / "per_file" / f"{name}{suffix}.npy", per_file)
            if suffix:
                continue
            same = paper is not None and paper.eeg_grand_avg.shape == grand.shape
            r = pipeline.compare(grand, paper.eeg_grand_avg, times * 1000) if same and paper else float("nan")
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
