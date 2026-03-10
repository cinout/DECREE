"""
randomly sample 700 rows from the CC3M dataset and save to a new file
"""

import pandas as pd

# load dataset
df = pd.read_csv("Train_GCC-training.tsv", sep="\t", header=None)
df.columns = ["caption", "url"]

# randomly sample 700 rows
sample_df = df.sample(n=2000, random_state=42)

sample_df.to_csv("cc3m_sample_700.tsv", sep="\t", index=False)
