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
3. the **technical-validation analyses** of the paper (preprocessing, PSD, ERP and their statistics);
4. **exact redrawing of Figures 2–4** from the aggregate arrays behind the published figures
   (`aggregates/`, `scripts/redraw_figures.py`);
5. an **approximate recomputation of Figures 3 and 4 from the released files**
   (`scripts/reproduce_from_release.py`).

Items 4 and 5 differ. The published figures were computed from the original continuous
recordings. Part of the release is cut into segments for copyright reasons, so a recomputation
from the released files can only approximate them (see [Reproducing the figures](#reproducing-the-figures)).

## Installation

```bash
git clone https://github.com/Motoshige496/JapanEEG.git
cd JapanEEG
uv sync                 # or: pip install -e .
```

Python 3.11 or later. The dependencies are pinned in `uv.lock`.

## Access and download

1. Request access on Harvard Dataverse (https://doi.org/10.7910/DVN/5RM3OT) and accept the
   JapanEEG Data Use Agreement. After approval you receive a user name and password for the data
   server, valid for 30 days.
2. Put them in a `.env` file in the repository folder (never commit it; `.env.example` shows the
   format). The values below are placeholders:

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
| `japaneeg.pipeline` | the ERP pipeline applied to released files |

## Reproducing the figures

### Exact redrawing from the aggregate arrays

`aggregates/` holds the arrays behind Figures 2–4: the grand-average ERPs of the nine ERP panels,
the median PSDs over runs (raw and preprocessed) and the electrode layouts, with the counts and
settings in JSON files and the SHA-256 of every file in `aggregates/manifest.json`. They contain
no recordings, run lists or dates.

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
