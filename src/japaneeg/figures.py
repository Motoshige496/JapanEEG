"""Figures 2-4 of the Data Descriptor from the aggregate arrays.

The panels are drawn with the settings of the paper (layout, colours, scale bars, time points).
The published figures were assembled from these panels with manual layout edits (labels,
spacing), so the redrawn figures match them in content but not in every typographic detail.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from japaneeg.aggregates import DEVICE_LABELS, ErpAggregate, PsdAggregate  # noqa: E402

CM = 1.0 / 2.54
AXIS_LW = 0.3 * 2.8346
TRACE_LW = 0.2 * 2.8346
LABEL_SIZE = 7
TICK_SIZE = 6
PANEL_LABEL_SIZE = 6
TOPO_SPHERE = (0.0, 0.0, -0.015, 0.1)

COLUMNS: dict[str, list[tuple[str, str]]] = {
    "Participant": [("P01-overt-pangolin", "P01"), ("P02-overt-pangolin", "P02"), ("P03-overt-pangolin", "P03")],
    "Device": [("P01-overt-pangolin", "g.Pangolin"), ("P01-overt-scarabeo", "g.SCARABEO"),
               ("P01-overt-eego", "eego™ sports")],
    "Task": [("P01-overt-pangolin", "overt"), ("P01-covert-pangolin", "covert"),
             ("P01-listening-pangolin", "listening")],
}  # fmt: skip
PSD_COLOURS: dict[str, list[str]] = {
    "Participant": ["#6BAED6", "#3182BD", "#08519C"],
    "Device": ["#1F77B4", "#756BB1", "#E7298A"],
    "Task": ["#1F77B4", "#E6550D", "#31A354"],
}
STACK_OFFSETS_DB = (25.0, 0.0, -25.0)


def channel_colors(agg: ErpAggregate) -> NDArray[np.float64]:
    """RGBA colour of every channel (viridis, ordered by ``color_rank``).

    Args:
        agg: ERP aggregate.

    Returns:
        ``(n_ch, 4)`` colours.
    """
    n = len(agg.color_rank)
    cmap = plt.get_cmap("viridis", n)
    base = np.array([cmap(i) for i in range(n)])
    return base[agg.color_rank]


def draw_lateral_head(ax: plt.Axes, nasion_dist: float, lw: float = 1.0) -> None:
    """Schematic head seen from the left, scaled by the nasion distance.

    Args:
        ax: Axes.
        nasion_dist: Distance from the head-frame origin to the nasion (m).
        lw: Line width.
    """
    a, b, zc = 1.08 * nasion_dist, 1.15 * nasion_dist, 0.20 * nasion_dist
    th = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(a * np.cos(th), zc + b * np.sin(th), color="k", lw=lw)
    x0 = -a * np.sqrt(1.0 - (zc / b) ** 2)
    ax.plot([x0, x0 - 0.10 * nasion_dist, x0], [0.10 * nasion_dist, 0.0, -0.10 * nasion_dist], color="k", lw=lw)
    ax.add_patch(Ellipse((0.0, 0.0), 0.10 * nasion_dist, 0.22 * nasion_dist, facecolor="0.8", edgecolor="none"))
    ax.set_xlim(-1.14 * a, 1.03 * a)
    ax.set_ylim(zc - 1.03 * b, zc + 1.03 * b)
    ax.set_aspect("equal")


def _cap_info(agg: ErpAggregate):  # noqa: ANN202 (mne.Info)
    """MNE Info with the head positions of a cap aggregate."""
    import mne

    info = mne.create_info(agg.ch_names, sfreq=1.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(dict(zip(agg.ch_names, agg.pos)), coord_frame="head")
    info.set_montage(montage, verbose="error")
    return info


def topomap(ax: plt.Axes, agg: ErpAggregate, values: NDArray[np.floating], vlim: tuple[float, float]) -> None:
    """2D scalp map: lateral view for g.Pangolin, standard top view for the caps.

    Args:
        ax: Axes.
        agg: Aggregate (positions and layout).
        values: One value per channel.
        vlim: Colour limits.
    """
    import mne

    ok = np.isfinite(values)
    if agg.layout == "lateral":
        mne.viz.plot_topomap(values[ok], agg.pos[ok], axes=ax, show=False, cmap="RdBu_r", vlim=vlim, contours=0,
                             sphere=(0.0, 0.0, 0.0, agg.head_radius), outlines=None, extrapolate="local",
                             sensors=False, res=96)  # fmt: skip
        draw_lateral_head(ax, agg.nasion_dist)
    else:
        info = _cap_info(agg)
        mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r", vlim=vlim, contours=0,
                             sphere=TOPO_SPHERE)  # fmt: skip


def sensor_inset(ax: plt.Axes, agg: ErpAggregate, colors: NDArray[np.float64]) -> None:
    """Electrode layout coloured like the butterfly traces, above the panel.

    Args:
        ax: ERP axes.
        agg: Aggregate.
        colors: Channel colours.
    """
    import mne

    ins = ax.inset_axes([0.82, 1.04, 0.18, 0.52])
    if agg.layout == "lateral":
        ins.scatter(agg.pos[:, 0], agg.pos[:, 1], s=0.6, c=colors, edgecolors="none")
        draw_lateral_head(ins, agg.nasion_dist, lw=0.4)
    else:
        mne.viz.plot_sensors(_cap_info(agg), axes=ins, show=False, sphere=TOPO_SPHERE, show_names=False,
                             kind="topomap", pointsize=0.6)  # fmt: skip
        for coll in ins.collections:
            coll.set_facecolor(colors)
            coll.set_edgecolor("none")
    ins.axis("off")


def figure3(erps: dict[str, ErpAggregate], xlim_ms: tuple[float, float] = (-200.0, 800.0),
            scalebar: float = 0.05, unit: str = "SD") -> Figure:  # fmt: skip
    """Figure 3: butterfly ERPs by participant, device and task.

    Args:
        erps: ERP aggregates by condition.
        xlim_ms: Displayed time range (ms).
        scalebar: Scale-bar length (SD of single epochs).
        unit: Scale-bar unit.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18.3 * CM, 10.2 * CM), sharex=True)
    for c, (group, rows) in enumerate(COLUMNS.items()):
        for r, (cond, label) in enumerate(rows):
            ax = axes[r, c]
            if cond not in erps:
                ax.axis("off")
                continue
            agg = erps[cond]
            t = agg.times_ms
            colors = channel_colors(agg)
            for ch in range(agg.eeg_grand_avg.shape[0]):
                ax.plot(t, agg.eeg_grand_avg[ch], lw=TRACE_LW, alpha=0.5, color=colors[ch])
            ax.axvline(0, color="red", lw=TRACE_LW * 2, alpha=0.6)
            vis = (t >= xlim_ms[0]) & (t <= xlim_ms[1])
            lo, hi = float(np.nanmin(agg.eeg_grand_avg[:, vis])), float(np.nanmax(agg.eeg_grand_avg[:, vis]))
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_xlim(*xlim_ms)
            if r == 0:
                ax.set_title(group, fontsize=LABEL_SIZE)
            ax.text(0.02, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=PANEL_LABEL_SIZE,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.85), zorder=10)  # fmt: skip
            if group == "Device":
                sensor_inset(ax, agg, colors)
            if r == 2:
                ax.set_xlabel("Time (ms)", fontsize=TICK_SIZE)
            ax.spines[["left", "top", "right"]].set_visible(False)
            ax.spines["bottom"].set_linewidth(AXIS_LW)
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.tick_params(axis="x", labelsize=TICK_SIZE, width=AXIS_LW, length=2, direction="out")
            left = cond == "P01-covert-pangolin"  # bar clear of the covert waveforms
            y0, y1 = ax.get_ylim()
            x0, x1 = ax.get_xlim()
            bx = x0 + 0.005 * (x1 - x0) if left else x1 - 0.03 * (x1 - x0)
            by = y0 + 0.05 * (y1 - y0)
            ax.plot([bx, bx], [by, by + scalebar], color="black", lw=AXIS_LW * 2, solid_capstyle="butt", clip_on=False)
            text = f" {scalebar:g} {unit}" if left else f"{scalebar:g} {unit} "
            ax.text(bx, by + scalebar / 2, text, ha="left" if left else "right", va="center", fontsize=PANEL_LABEL_SIZE)
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.10, top=0.83, hspace=0.72, wspace=0.13)
    return fig


def figure4(erps: dict[str, ErpAggregate], times_ms: tuple[int, ...] = tuple(range(-200, 251, 50))) -> Figure:
    """Figure 4: scalp maps at fixed latencies, one row per condition.

    Args:
        erps: ERP aggregates by condition.
        times_ms: Latencies of the columns (ms).

    Returns:
        The figure.
    """
    rows = [(group, cond, label) for group, items in COLUMNS.items() for cond, label in items]
    heights = []
    for _ in range(3):
        heights += [0.55] + [1.0, 0.22] * 3
    fig = plt.figure(figsize=(18.3 * CM, 25.5 * CM))
    grid = GridSpec(21, len(times_ms), figure=fig, left=0.055, right=0.995, top=0.985, bottom=0.015,
                    wspace=0.08, hspace=0.15, height_ratios=heights)  # fmt: skip
    data_row = {k: 1 + 7 * (k // 3) + 2 * (k % 3) for k in range(9)}
    for k, (group, cond, label) in enumerate(rows):
        if cond not in erps:
            continue
        agg = erps[cond]
        vmax = float(np.nanpercentile(np.abs(agg.eeg_grand_avg), 97)) or 0.01
        for col, t in enumerate(times_ms):
            ax = fig.add_subplot(grid[data_row[k], col])
            idx = int(np.argmin(np.abs(agg.times_ms - t)))
            topomap(ax, agg, agg.eeg_grand_avg[:, idx], (-vmax, vmax))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k == 0:
                ax.set_title(f"{t}", fontsize=TICK_SIZE - 1, pad=1)
            if col == 0:
                ax.text(-0.30, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=LABEL_SIZE)
        slot = fig.add_subplot(grid[data_row[k] + 1, -3:])
        slot.axis("off")
        cax = slot.inset_axes([0.0, 0.45, 1.0, 0.35])
        sm = cm.ScalarMappable(norm=mcolors.Normalize(-vmax, vmax), cmap="RdBu_r")
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([-vmax, vmax])
        cbar.set_ticklabels([f"{-vmax:.2g}", f"{vmax:.2g}"])
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.tick_params(labelsize=4, width=0.3, length=1.0, pad=0.5)
        cbar.outline.set_linewidth(0.3)
    for g, group in enumerate(COLUMNS):
        left, right = grid[7 * g, 0].get_position(fig), grid[7 * g, len(times_ms) - 1].get_position(fig)
        fig.text(0.5 * (left.x0 + right.x1), 0.5 * (left.y0 + left.y1), group, ha="center", va="center",
                 fontsize=LABEL_SIZE + 2, fontweight="bold")  # fmt: skip
    fig.text(0.995, 0.005, "Amplitude [SD]", ha="right", va="bottom", fontsize=TICK_SIZE)
    return fig


def figure2(psds: dict[tuple[str, str], PsdAggregate], fmin: float = 0.5, fmax: float = 120.0) -> Figure:
    """Figure 2: median PSD over runs (raw and preprocessed) by participant, device and task.

    Args:
        psds: PSD aggregates by (condition, kind).
        fmin: Lowest displayed frequency (Hz).
        fmax: Highest displayed frequency (Hz).

    Returns:
        The figure.
    """
    from japaneeg.psd import median_curve_db

    mm = 1.0 / 25.4
    kinds = [("raw", "raw\nPSD (dB, stacked)"), ("preprocessed", "preprocessed\nPSD (dB, stacked)")]
    fig, axes = plt.subplots(2, 3, figsize=(180 * mm, 100 * mm), sharex=True)
    for r, (kind, ylabel) in enumerate(kinds):
        for c, (group, items) in enumerate(COLUMNS.items()):
            ax = axes[r, c]
            for i, (cond, label) in enumerate(items):
                agg = psds.get((cond, kind))
                if agg is None:
                    continue
                keep = (agg.freqs >= fmin) & (agg.freqs <= fmax)
                curve, band = median_curve_db(agg.median[:, keep], agg.mad[:, keep], agg.n_runs)
                curve = curve - float(np.median(curve)) + STACK_OFFSETS_DB[i]
                colour = PSD_COLOURS[group][i]
                ax.fill_between(agg.freqs[keep], curve - band, curve + band, color=colour, alpha=0.25, lw=0)
                ax.plot(agg.freqs[keep], curve, color=colour, lw=TRACE_LW, alpha=0.95)
                if r == 1:
                    ax.text(1.0, STACK_OFFSETS_DB[i] + 5, label, transform=ax.get_yaxis_transform(), color=colour,
                            fontsize=LABEL_SIZE, ha="right", va="bottom")  # fmt: skip
            ax.set_xscale("log")
            ax.set_xlim(fmin, fmax)
            ax.set_yticks(list(STACK_OFFSETS_DB))
            ax.set_yticklabels([f"{v:+g}" for v in STACK_OFFSETS_DB])
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE, width=AXIS_LW, length=2, direction="out")
            if r == 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_SIZE)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if r == 0:
                ax.set_title(group, fontsize=LABEL_SIZE)
    fig.tight_layout(w_pad=2.0)
    return fig


def save(fig: Figure, path: str | Path) -> None:
    """Save a figure (300 dpi) and close it.

    Args:
        fig: Figure.
        path: Output file (``.png`` or ``.pdf``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


__all__ = ["DEVICE_LABELS", "figure2", "figure3", "figure4", "save"]
