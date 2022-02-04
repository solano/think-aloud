# think-aloud
Analysis of "Think Aloud" experience through phrase embedding.

Data (in CSV) are processed in pandas (Python). Phrase embeddings are done by the LASER project (laserembeddings). Statistical analysis is done with package lme4 in R.

- `preprocess.py`: remove oral markers from transcriptions, segment data in sub-row, row and probe levels. Exports `text_probes.csv`, `text_rows.csv` and `text_subrows.csv`.
- `embeddings.py`: embed phrases and analyse trajectory profiles in semantic space. Exports the files `*_as_embedding_transitions.csv`.
- `*_as_embedding_transitions.R`: statistical analysis of trajectory data.
