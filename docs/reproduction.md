# Reproduce the paper

Run the commands below from the repository root after `uv sync`. To work from
the BIDS files, follow the [download guide](data.md).

## Reproducing the figures

### Redraw Figures 2–4 from the supplied arrays

[`aggregates/`](../aggregates) contains the ERP grand averages, raw and
preprocessed PSDs, and electrode layouts behind Figures 2–4. The ERP metadata
record the trial counts and analysis settings.

```bash
uv run python scripts/redraw_figures.py
```

This writes `figures/fig2_psd.png`, `figures/fig3_erp.png`, and
`figures/fig4_topomap.png`. It also prints the post-onset RMS to baseline SD
ratios, GFP peak latencies, and 60–118 Hz spectral slopes. The manuscript
figures were assembled from these panels with manual layout edits.

### Recompute Figures 3 and 4 from the release

```bash
uv run python scripts/reproduce_from_release.py --bids-root data/JapanEEG/bids --out outputs/release_reproduction
uv run python scripts/redraw_figures.py --aggregates outputs/release_reproduction --out figures_release
```

The ERP recomputation treats each shuffled segment separately. It cannot
recover the original run-wise standardisation or adaptive-filter history,
and some onsets have less than 0.5 s of preceding EEG within their segment.
The script writes correlations with the manuscript grand averages to
`outputs/release_reproduction/comparison.tsv`. Allow several hours and about
20 GB for the epoch cache. Use `--conditions` and `--max-files` for a smaller run.

### Plot one event as in Figure 1e

```bash
uv run python scripts/plot_synchronized_example.py \
    --edf path/to/continuous_speechopen_eeg.edf --event 10 --out figures/fig1e_example.png
```

Replace the EDF path with a continuous overt-speech run. The plot shows the
voice-masked audio, aligned words and phones, five scalp EEG channels, and facial EMG.

## Reproducing the numbers of the paper

Run scripts with `uv run python`. The `paper_numbers.py` commands require
`--bids-root data/JapanEEG/bids` and write TSVs under `outputs/paper_numbers/` by
default. Run `uv sync --extra speech` before asking for mora counts.

| Measure | Script or command | Expected result |
|---|---|---|
| Post-onset RMS to baseline SD ratios, GFP peak latencies, median PSD slopes | `scripts/redraw_figures.py` | Uses the manuscript aggregates |
| Split-half reliability and recomputed ERP measures | `scripts/reproduce_from_release.py`, then `scripts/release_statistics.py` | Approximate |
| ERP correlation and RMS change with and without adaptive filtering | Add `--without-adaptive-filter` to `scripts/reproduce_from_release.py`, then run `scripts/release_statistics.py` | Approximate |
| Tables 4 and 5 | `scripts/paper_numbers.py tables` | Event counts and durations match the release. Recording days and periods cover only continuous runs |
| Articulation rate in morae and phones per second | `scripts/paper_numbers.py articulation --morae` | Uses released word and phone intervals |
| Spectral slopes during speech and pauses and Wilcoxon tests | `scripts/paper_numbers.py speech-pause --subject 01` | Approximate |
| EEG–EMG coherence and chance levels | `scripts/paper_numbers.py coherence --subject 01` | Approximate |
| Bad channels and event overlap with `BAD_high_amplitude` | `scripts/paper_numbers.py bad-channels` | Event overlap uses released annotations. Per-run bad-channel counts cover continuous runs |

For speech-and-pause slopes and coherence, the scripts pool shuffled segments
by file because their original run membership is unavailable. Stretches between
utterances at segment edges count as pauses. The medians and test counts can
differ from those in the manuscript, including the sign of the near-zero median
speech slope for P01.

The acoustic onset latency of about 40 ms was measured using the original
microphone signal, which is unavailable in the release. The release provides
its 30 Hz amplitude envelope and voice-masked audio within speech events.

## Analyses of the technical validation

| Module | Contents |
|---|---|
| [`japaneeg.preprocessing`](../src/japaneeg/preprocessing.py) | notch (50 Hz harmonics), band-pass 2–120 Hz, resampling to 240 Hz; good-channel common average reference; z-score and robust z-score (median, 1.4826 × MAD); NLMS adaptive filter (step 0.1, regularisation 0.001, initial weights N(0, 0.5), seed 0) |
| [`japaneeg.erp`](../src/japaneeg/erp.py) | onset selection (≥ 0.5 s silence), epochs −300 to 800 ms, baselines (overt −300 to −200 ms; covert and listening −200 to 0 ms), rejection at 3 × the median per-trial maximum, epoch normalisation, two-stage average; GFP peaks, post-onset RMS / baseline SD ratio, split-half reliability |
| [`japaneeg.psd`](../src/japaneeg/psd.py) | Welch PSD (4-s Hann, 50 % overlap), median over runs and its standard error in dB, 60–118 Hz slope without 97–103 Hz |
| [`japaneeg.pipeline`](../src/japaneeg/pipeline.py) | the ERP pipeline applied to released files, with and without the adaptive filter |
| [`japaneeg.emg`](../src/japaneeg/emg.py) | 60–118 Hz slopes during speech and pauses, EEG–EMG coherence, each with the recording reference and the common average reference (CAR) |
| [`japaneeg.speech`](../src/japaneeg/speech.py) | articulation rate from the released word and phone intervals (morae and phones per second) |
| [`japaneeg.tables`](../src/japaneeg/tables.py) | event counts, event hours and recording hours of Tables 4 and 5 |

## Run the tests

```bash
uv sync --extra dev
uv run pytest
```

The tests use synthetic data and need no dataset download.
