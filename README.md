# ML-Prefetching

Memory Access Delta Predictor

A two-stage LightGBM model developed as part of a research project at Systopia Lab (UBC) to predict future memory access deltas from HPC memory traces.


What it does:

- Given a program counter and previous memory access delta, the model predicts the next memory access delta.


Why two stages:

- Memory access deltas are sparse. The majority of values are zero. A single regressor performs poorly on sparse targets, so the model uses:

  - Stage 1 (Classifier): Predicts whether the next delta is zero or nonzero.
  - Stage 2 (Regressor): Predicts the actual delta value for nonzero cases.

- This approach improves exact match accuracy compared to a single model.


Scale:

- Trained and validated on up to 6 million memory access traces on Compute Canada HPC clusters using Slurm.
