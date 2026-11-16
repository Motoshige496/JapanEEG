# Download and load the dataset

Follow the [setup and access steps](../README.md#get-started) before downloading.
Keep the credentials you receive after approval in `.env`. The file is excluded
from Git. The Data Use Agreement requires users to keep their credentials personal.

## Download options

Run these commands from the repository root. You can also use `pip install -e .`
to install the package and run `japaneeg-download` directly.

```bash
# Full release: about 0.38 TB of archives and 1.2 TB after extraction
uv run japaneeg-download --out data/JapanEEG

# One session and the sidecar files for a first look
uv run japaneeg-download --out data/JapanEEG \
    --include "sidecars/*" "archives/sub-01/sub-01_ses-20230829*"
```

The downloader resumes interrupted downloads, verifies each archive against
`archives_manifest.tsv`, and extracts into `data/JapanEEG/bids/`. Use
`--extract-to` to change the BIDS output directory or `--no-extract` to only
download and verify the archives.

You can also use the download instructions provided with your Araya credentials
if you prefer standard command-line tools.

## Load with MNE-Python

The [quickstart script](../examples/quickstart.py) shows how to load continuous
runs, inspect shuffled segments and read overt speech events with EEG, audio and text.

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

# Cut each shuffled segment before filtering it.
shuffled = bids.find_runs(root, subject="01", session="shuffled", task="speechopen")
for seg in bids.segments(shuffled[0])[:3]:
    print(seg.frag_id, seg.onset, seg.duration, seg.bad_channels, len(seg.events))

# Overt utterances with EEG, voice-masked audio (16 kHz) and text, from both kinds of files
for u in bids.iter_utterances(root, subject="01", pre=0.5, post=0.5):
    print(u["source"], u["eeg"].shape, len(u["audio"]), u["text"])
    break
```

## Notes on the release

- Use `bids.scalp_eeg(channels)` to select scalp EEG. It excludes reference-site
  electrodes recorded as EEG: eego™ sports M1/M2 and g.SCARABEO `EEG063`/`EEG064` (A1/A2).
- The recording reference is CPz for eego™ sports and a mastoid electrode, usually
  the left one, for g.Pangolin and g.SCARABEO.
- `BAD_high_amplitude` marks stretches 1,000 times above the usual amplitude or
  saturated. `high_amplitude` marks stretches 100–1,000 times above usual amplitude.
- `ses-shuffled` files contain segments in random order, including readings of
  copyrighted texts, the listening/covert task (`task-listeningcovert`), and
  audiobook listening (`task-listening`). Use `frag_id`, `segment_onset` and
  `segment_duration` to locate segments and process each segment separately.

See the dataset README on [Harvard Dataverse](https://doi.org/10.7910/DVN/5RM3OT)
for the BIDS layout and full release documentation.
