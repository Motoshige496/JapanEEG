# JapanEEG

Code for **JapanEEG**, a 1020-hour dataset of simultaneously recorded scalp EEG, facial EMG and
voice-masked speech audio from three native Japanese speakers, recorded with three EEG systems
(g.Pangolin, g.SCARABEO, eego™ sports) during open-vocabulary overt speech, covert repetition and
speech listening.

- Dataset: Harvard Dataverse, https://doi.org/10.7910/DVN/5RM3OT (controlled access, BIDS)
- Data Descriptor: *A 1000-hour EEG–EMG–audio dataset of Japanese speech production* (Scientific Data)

This repository provides

1. **download** of the release (`japaneeg-download`), with checksum verification and extraction;
2. **loading** of continuous runs and `ses-shuffled` segments with MNE-Python (`japaneeg.bids`);
3. the **technical-validation analyses** of the paper (preprocessing, PSD, ERP, facial muscle
   activity, speech measures and their statistics);
4. **exact redrawing of Figures 2–4** from the aggregate arrays behind the published figures
   (`aggregates/`, `scripts/redraw_figures.py`);
5. a **recomputation of Figures 3 and 4 and of the numbers of the paper from the released files**
   (`scripts/reproduce_from_release.py`, `scripts/release_statistics.py`, `scripts/paper_numbers.py`);
6. a plot of **synchronized speech, aligned words and phones, EEG and EMG** around one utterance,
   as in Figure 1e (`scripts/plot_synchronized_example.py`).

Items 4 and 5 differ. The published figures were computed from the original continuous
recordings. Part of the release is cut into segments for copyright reasons, so a recomputation
of the EEG analyses from the released files approximates them (see
[Reproducing the figures](#reproducing-the-figures) and
[Reproducing the numbers of the paper](#reproducing-the-numbers-of-the-paper)).

## Installation

```bash
git clone https://github.com/Motoshige496/JapanEEG.git
cd JapanEEG
uv sync                 # or: pip install -e .
```

Python 3.11 or later. The dependencies are pinned in `uv.lock`. The mora count of the
articulation rate needs pyopenjtalk (`uv sync --extra speech`).

## Access and download

1. Request access on Harvard Dataverse (https://doi.org/10.7910/DVN/5RM3OT) and accept the
   JapanEEG Data Use Agreement. After approval you receive a user name and password for the data
   server, valid for 30 days.
2. Put them in a `.env` file in the repository folder. The credentials are personal under the
   Data Use Agreement, and `.env` is listed in `.gitignore`. The values below are placeholders:

   ```bash
   cp .env.example .env
   # .env
   JAPANEEG_USER=your-user-name
   JAPANEEG_PASSWORD=your-password
   ```

3. Download, verify and extract:

   ```bash
   # everything: 0.37 TB of archives, 1.2 TB after extraction
   uv run japaneeg-download --out data/JapanEEG

   # or one session and the sidecar files of sub-01 for a first look
   uv run japaneeg-download --out data/JapanEEG \
       --include "sidecars/*" "archives/sub-01/sub-01_ses-20230829*"
   ```

   The archives land in `data/JapanEEG/` with their release paths, are checked against the SHA-256
   values of `archives_manifest.tsv` and are unpacked into the BIDS tree `data/JapanEEG/bids/`
   (`--extract-to` changes it, `--no-extract` skips it). An interrupted download resumes when the
   command is run again.

The same files can also be fetched with standard tools, for example
`wget -r -np -nH -R "index.html*" -c --user your-user-name --ask-password https://eegdata.araya.org/JapanEEG/`,
and unpacked with `for f in archives/*/*.tar.xz sidecars/*.tar.xz; do tar -xJf "$f"; done`.

## Loading the data

```python
import mne
from japaneeg import bids

root = "data/JapanEEG/bids"

# Continuous runs: channel types from channels.tsv, bad channels in raw.info["bads"],
# events as annotations (overt, listening, covert, short_speech, cue, BAD_high_amplitude, ...)
runs = bids.find_runs(root, subject="01", task="speechopen", acq="pangolin", session="20230829")
raw = bids.load_raw(runs[0], preload=True)
raw.pick("eeg", exclude="bads").filter(1.0, 40.0)

# Epochs around speech onsets; MNE drops epochs overlapping BAD_high_amplitude stretches
events, event_id = mne.events_from_annotations(raw, event_id={"overt": 1})
epochs = mne.Epochs(raw, events, event_id, tmin=-0.3, tmax=0.8, baseline=(-0.3, -0.2))

# ses-shuffled files: cut each segment before filtering it (never filter across segments)
shuffled = bids.find_runs(root, subject="01", session="shuffled", task="speechopen")
for seg in bids.segments(shuffled[0])[:3]:
    print(seg.frag_id, seg.onset, seg.duration, seg.bad_channels, len(seg.events))

# Overt utterances with EEG, voice-masked audio (16 kHz) and text, from both kinds of files
for u in bids.iter_utterances(root, subject="01", pre=0.5, post=0.5):
    print(u["source"], u["eeg"].shape, len(u["audio"]), u["text"])
    break
```

`examples/quickstart.py` runs these steps on the downloaded data:

```bash
uv run python examples/quickstart.py --bids-root data/JapanEEG/bids
```

Notes on the release (details in the dataset README):

- Scalp EEG excludes the reference-site electrodes recorded as EEG (eego™ sports M1/M2,
  g.SCARABEO A1/A2 = `EEG063`/`EEG064`); `bids.scalp_eeg(channels)` lists the scalp channels.
- Recording reference: CPz for eego™ sports; an electrode on the mastoid, usually the left one,
  for g.Pangolin and g.SCARABEO.
- `BAD_high_amplitude` rows mark stretches 1000 times above the usual amplitude or saturated;
  `high_amplitude` rows mark 100–1000 times.
- Readings of copyrighted texts, the covert repetition task (`task-listeningcovert`) and
  audiobook listening (`task-listening`) are released as segments in random order
  (`ses-shuffled`); `frag_id`, `segment_onset` and `segment_duration` identify each segment.

## Analyses of the technical validation

| Module | Contents |
|---|---|
| `japaneeg.preprocessing` | notch (50 Hz harmonics), band-pass 2–120 Hz, resampling to 240 Hz; good-channel common average reference; z-score and robust z-score (median, 1.4826 × MAD); NLMS adaptive filter (step 0.1, regularisation 0.001, initial weights N(0, 0.5), seed 0) |
| `japaneeg.erp` | onset selection (≥ 0.5 s silence), epochs −300 to 800 ms, baselines (overt −300 to −200 ms; covert and listening −200 to 0 ms), rejection at 3 × the median per-trial maximum, epoch normalisation, two-stage average; GFP peaks, post-onset RMS / baseline SD ratio, split-half reliability |
| `japaneeg.psd` | Welch PSD (4-s Hann, 50 % overlap), median over runs and its standard error in dB, 60–118 Hz slope without 97–103 Hz |
| `japaneeg.pipeline` | the ERP pipeline applied to released files, with and without the adaptive filter |
| `japaneeg.emg` | 60–118 Hz slopes during speech and pauses, EEG–EMG coherence, each with the recording reference and the common average reference (CAR) |
| `japaneeg.speech` | articulation rate from the released word and phone intervals (morae and phones per second) |
| `japaneeg.tables` | event counts, event hours and recording hours of Tables 4 and 5 |

## Reproducing the figures

### Exact redrawing from the aggregate arrays

`aggregates/` holds the arrays behind Figures 2–4: the grand-average ERPs of the nine ERP panels,
the median PSDs over runs (raw and preprocessed) and the electrode layouts. The trial counts and
analysis settings are in JSON files next to the arrays (format in `japaneeg.aggregates`).

```bash
uv run python scripts/redraw_figures.py          # writes figures/fig2_psd.png, fig3_erp.png, fig4_topomap.png
```

The script also prints, per condition, the ratio of the post-onset RMS to the baseline SD, the
GFP peak latencies and the 60–118 Hz slopes reported in the paper. The published figures were
assembled from the same panels with manual layout edits, so the redrawn figures match them in
content but not in every typographic detail.

### Approximate recomputation from the released files

```bash
uv run python scripts/reproduce_from_release.py --bids-root data/JapanEEG/bids --out outputs/release_reproduction
uv run python scripts/redraw_figures.py --aggregates outputs/release_reproduction --out figures_release
```

Every continuous run is processed as one piece and every `ses-shuffled` segment on its own, with
the paper's filters, references, NLMS filter, onsets, baselines, rejection and normalisation
(`japaneeg.pipeline`). The run-wise standardisation and the adaptive-filter history of the
original recordings cannot be restored from the segments, and some speech onsets lack the
required 0.5 s of preceding EEG within their segment. The correlations of the recomputed
grand averages with the published ones are written to `outputs/release_reproduction/comparison.tsv`.
Processing all files takes several hours on a multi-core machine and caches about 20 GB of
epochs; `--conditions` and `--max-files` select a smaller test.

### Synchronized signals around one utterance (as Figure 1e)

```bash
uv run python scripts/plot_synchronized_example.py \
    --edf data/JapanEEG/bids/sub-01/ses-20230829/eeg/<run>_eeg.edf --event 10 --out figures/fig1e_example.png
```

The plot shows the voice-masked speech with the aligned words and phones of the release, five
scalp EEG channels and the facial EMG on one time line. Use a continuous overt-speech run.

## Reproducing the numbers of the paper

| Numbers in the paper | Command | From the released files |
|---|---|---|
| Ratio of post-onset RMS to baseline SD, GFP peak latencies | `scripts/redraw_figures.py` | exact (aggregates of the published figures) |
| 60–118 Hz slopes of the median spectra | `scripts/redraw_figures.py` | exact (aggregates) |
| Split-half reliability; ratio and GFP peaks of the recomputed ERPs | `scripts/reproduce_from_release.py`, then `scripts/release_statistics.py` | approximate |
| ERPs with and without the adaptive filter (correlation, RMS change) | `scripts/reproduce_from_release.py --without-adaptive-filter`, then `scripts/release_statistics.py` | approximate |
| Tables 4 and 5: events, event hours, recording hours, duration statistics | `scripts/paper_numbers.py tables` | event counts and durations exact; recording days from the continuous runs only |
| Articulation rate in morae and phones per second | `scripts/paper_numbers.py articulation --morae` | exact |
| Slopes during speech and pauses, Wilcoxon signed-rank test, recording reference vs CAR | `scripts/paper_numbers.py speech-pause --subject 01` | approximate (per file; in `ses-shuffled` files the stretches between utterances at the segment edges count as pauses) |
| EEG–EMG coherence with the recording reference and CAR, chance level | `scripts/paper_numbers.py coherence --subject 01` | approximate (per file) |
| Bad channels per run, events overlapping `BAD_high_amplitude` | `scripts/paper_numbers.py bad-channels` | overlap exact; bad channels of the continuous runs only |

All commands except `redraw_figures.py` and `release_statistics.py` take
`--bids-root data/JapanEEG/bids`, and all write TSV files under `outputs/`. The paper computed the
speech and pause slopes and the coherence per run. The run of a `ses-shuffled` segment is not
released, so the segments of one file are pooled, and the counts of the test differ from the
paper. The acoustic onset about 40 ms after the speech-event onset was measured on the original
microphone signal, which is not released, so it cannot be computed from the release. The
microphone channels of the release hold the 30 Hz amplitude envelope of that signal and the
voice-masked speech, which is kept only within the speech events. The speech measures use the
released word and phone intervals, which come from forced
alignment of the original audio.

## Tests

```bash
uv sync --extra dev
uv run pytest
```

The tests use synthetic data and need no download.

## Citation

Please cite the Data Descriptor and the dataset (https://doi.org/10.7910/DVN/5RM3OT).

## License

The code is released under the MIT License (`LICENSE`). The data are not covered by it: they are
distributed under controlled access and the JapanEEG Data Use Agreement.
