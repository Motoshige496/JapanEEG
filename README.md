# JapanEEG

Code for JapanEEG, a 1,020-hour dataset of scalp EEG, facial EMG, and speech from
three native Japanese speakers. Recordings cover open-vocabulary overt speech,
covert repetition, and listening. The release includes voice-masked audio where
available and word and phone timings.

- [Dataset and Data Use Agreement (Harvard Dataverse)](https://doi.org/10.7910/DVN/5RM3OT)
- [Data Descriptor (arXiv preprint)](https://arxiv.org/abs/2606.01264)

## Get started

1. Sign in to Harvard Dataverse, open the [archive manifest](https://dataverse.harvard.edu/file.xhtml?fileId=14427160),
   and choose **Access File → Request Access**. Complete the form and agree to
   the Data Use Agreement. Araya sends download credentials after approval.
2. Install the package with Python 3.11 or later and `uv`.

   ```bash
   git clone https://github.com/Motoshige496/JapanEEG.git
   cd JapanEEG
   uv sync
   cp .env.example .env
   ```

3. Fill in your credentials in `.env`. Download one session and run the loading example.

   ```bash
   uv run japaneeg-download --out data/JapanEEG \
       --include "sidecars/*" "archives/sub-01/sub-01_ses-20230829*"
   uv run python examples/quickstart.py --bids-root data/JapanEEG/bids
   ```

   The downloader verifies SHA-256 checksums and extracts the archives into a BIDS tree.
   See [Download and load the dataset](docs/data.md) for the full download command,
   MNE examples, and notes on shuffled segments.

## Reproduce the paper

[Reproduction guide](docs/reproduction.md) covers the [figures](docs/reproduction.md#reproducing-the-figures),
[reported numbers](docs/reproduction.md#reproducing-the-numbers-of-the-paper), and
[technical validation code](docs/reproduction.md#analyses-of-the-technical-validation).
Redraw Figures 2–4 from the included aggregate arrays without downloading the dataset.
Recomputing the ERP figures from the released BIDS files gives approximate results.

## Citation and license

Please cite the [JapanEEG Data Descriptor](https://arxiv.org/abs/2606.01264) when referencing this work.

```bibtex
@misc{sato2026japaneeg,
  title         = {A 1000-hour {EEG-EMG}-audio dataset of Japanese speech production},
  author        = {Sato, Motoshige and Horiguchi, Ilya and Inoue, Masakazu and
                   Tomeoka, Kenichi and Hatakeyama, Eri and Kita, Yuya and
                   Yamamoto, Atsushi and Fujisawa, Ippei and Sasai, Shuntaro},
  year          = {2026},
  eprint        = {2606.01264},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.NC},
  url           = {https://arxiv.org/abs/2606.01264}
}
```

If you use the data, please also cite the [JapanEEG dataset on Harvard Dataverse](https://doi.org/10.7910/DVN/5RM3OT).

The code uses the [MIT License](LICENSE). For the data, see the terms on Harvard Dataverse.
