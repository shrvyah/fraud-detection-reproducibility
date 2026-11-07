# Reproducible Hybrid Interpretable Fraud Detection

Code and accompanying results for:

*A Reproducible Hybrid Interpretable Framework for Credit-Card Fraud Detection*

This repository contains the public-data rerun, implementation, result files,
figures, and final manuscript PDF. The raw credit-card CSV is not redistributed;
the acquisition script downloads it and verifies the expected SHA-256 checksum.

## arXiv files

- `fraud_detection_final.pdf` is the final PDF.
- `fraud_detection_final.docx` is an editable copy.

After this repository is published, add its permanent public URL to Section 7
of the manuscript before submitting to arXiv.

## Reproduce the reported public-data results

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt
    python data/download_creditcard.py
    python code/run_experiments.py data results ulb

The result files in `results/` were generated from the same public ULB file,
chronological split, five seeds, and model settings used in the manuscript.
The raw dataset is not redistributed; the downloader verifies its SHA-256.

## Author metadata

- Shravya Reddy Nalla — PhD Researcher in Artificial Intelligence,
  University of the Cumberlands
- Atanu Mukhopadhyay — Independent Researcher

No corresponding-author email was supplied, so none is invented in the
manuscript package.

## Citation

See `CITATION.cff` for machine-readable citation metadata.
