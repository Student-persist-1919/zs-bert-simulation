# Dataset snapshots used by this project

These gzip files preserve the exact JSON bytes supplied in BTL/data/raw. They
are project inputs, not newly downloaded or independently verified copies of
the authors' Google Drive files. The original author's reference links are
retained in manifest.json for comparison and attribution.

- FewRel: 56,000 instances, 80 relations, 700 instances per relation.
- Wiki-ZSL: 93,383 instances, 113 relations. Table 1 reports 94,383 instances.
  This discrepancy is unresolved; no samples were added or dropped to match it.

The format has dict entities with half-open token spans [start, end).
manifest.json records compressed/raw SHA256, raw byte count and statistics.
The runner verifies both hashes before normalization. A smoke test still
validates the entire input before selecting a small subset for training.

Why these files exist: the first Kaggle smoke test reached the GPU import check
but gdown could not resolve the authors' public Drive link. Keeping versioned
inputs with code removes that network dependency and avoids manual Kaggle Inputs.
The author Drive download remains an explicit optional source; it is never used
as an unrecorded replacement for the project snapshot.

References:
- https://github.com/dinobby/ZS-BERT#dataset
- https://aclanthology.org/2021.naacl-main.272/
