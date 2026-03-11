"""
Module to randomly sample rows from a CC3M-format TSV and save the sample to a new TSV.

Behavior:
- Loads a tab-separated file "Train_GCC-training.tsv" without header and assigns columns
  ["caption", "url"].
- Uses pandas.DataFrame.sample(n=2000, random_state=42) to select 2000 rows at random.
  - Yes: df.sample(n=2000, random_state=42) selects 2000 random rows.
  - The random_state=42 argument makes the selection deterministic/reproducible.
- Writes the sampled rows to "cc3m_sample_700.tsv" (note: filename suggests 700 but code samples 2000).

Usage:
- Ensure the input file exists in the working directory.
- Adjust n and random_state as needed for sample size and determinism.
randomly sample 700 rows from the CC3M dataset and save to a new file
"""

import pandas as pd

# load dataset
df = pd.read_csv("Train_GCC-training.tsv", sep="\t", header=None)
df.columns = ["caption", "url"]

# randomly sample 700 rows
sample_df = df.sample(n=2000, random_state=42)

sample_df.to_csv("cc3m_sample_700.tsv", sep="\t", index=False)
