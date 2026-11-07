# Data and provenance

The paper uses the public anonymized credit-card benchmark distributed through
the TensorFlow imbalanced-classification tutorial and associated with the
Worldline/ULB dataset.

Download and verify the exact expected file:

    python data/download_creditcard.py

URL: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
SHA-256: 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89

The raw CSV is intentionally not included in the ZIP. The acquisition script
verifies that a downloaded copy is byte-for-byte the same dataset used for
the reported rerun. Review the host's terms before redistributing the raw file.
