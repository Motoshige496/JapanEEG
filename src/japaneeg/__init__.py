"""Tools for the JapanEEG dataset: download, loading, the technical-validation analyses and figures.

Modules
-------
download        download, verify and extract the release (controlled access)
bids            find runs, load them with MNE, list ses-shuffled segments, iterate utterances
preprocessing   filtering, good-channel reference, standardisation, NLMS adaptive filter
erp, psd        analyses and statistics of the Data Descriptor's technical validation
pipeline        Figures 3 and 4 recomputed from the released files
aggregates      the arrays behind Figures 2-4 (in ``aggregates/``)
figures         Figures 2-4 drawn from the aggregates
"""

__version__ = "1.0.0"
