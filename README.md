# JapanEEG

Open-Vocabulary Japanese Sentence Reading EEG-EMG Dataset from OpenNeuro ([ds007600](https://openneuro.org/datasets/ds007600)).

- **3 long-term participants** with 600+ hours of EEG recorded during overt Japanese speech production
- **Open vocabulary** -- participants read sentences from novels, text-based TV games, and the JSUT corpus at natural speed, producing long-form speech
- **Multiple devices** -- g.PANGOLIN (128 ch), g.SCARABEO (64 ch), eego™ sports (61 ch), each with EOG + 2x EMG channels

## Quick Start

### 1. Setup

```bash
git clone https://github.com/Motoshige496/JapanEEG.git
cd JapanEEG
uv sync
```

### 2. Download the Dataset

```bash
# Download all data
uv run python -m japaneeg.download

# Or download a single subject/session for quick testing
uv run python -c "
from japaneeg.download import download
download(include=['sub-01/ses-20230829/eeg/*run-01*'])
"
```

### 3. Load and Explore EEG Data

```python
import mne
import mne_bids
import pandas as pd

# --- Load via MNE-BIDS ---
bids_path = mne_bids.BIDSPath(
    subject="01",
    session="20230829",
    task="speechopen",
    acquisition="pangolin",
    run="01",
    suffix="eeg",
    datatype="eeg",
    root="data/ds007600",
)
raw = mne_bids.read_raw_bids(bids_path, verbose=False)

print(raw.info)
# <Info | 139 channels, 1200.0 Hz, 1448.0 sec>

# --- Read events ---
events_tsv = bids_path.copy().update(suffix="events", extension=".tsv")
events = pd.read_csv(str(events_tsv), sep="\t")
print(events.head())
#    onset  duration      trial_type  sample            value
# 0   20.0       5.0  speech_window   20480       ルーシーは伝統の
# 1   25.0       5.0  speech_window   25600       スイッチをつけた

# --- Plot raw EEG (first 10 seconds) ---
raw.load_data()
raw.pick("eeg")
raw.plot(duration=10, n_channels=20, title="JapanEEG - sub-01")
```

## Dataset Structure

```
data/ds007600/
├── dataset_description.json
├── participants.tsv
├── README
├── sub-{01,02,03,04}/
│   └── ses-YYYYMMDD/
│       └── eeg/
│           ├── *_eeg.edf          # EEG recording (EDF format)
│           ├── *_eeg.json         # EEG metadata
│           ├── *_channels.tsv     # Channel information
│           └── *_events.tsv       # Speech events with Japanese text
```

## License

- **Code**: CC0-1.0
- **Dataset**: CC0-1.0
