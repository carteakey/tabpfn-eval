# Generate a summary of the results of the experiments
import os
import pandas as pd

from .. import BASE_DIR

# Read all csv 
df_baseline = pd.read_csv(os.path.join(BASE_DIR, 'evals/openml_baseline_scores.csv'))
df_baseline_tabpfn = pd.read_csv(os.path.join(BASE_DIR, 'evals/openml_baseline_tabpfn.csv'))
df_modified_tabpfn = pd.read_csv(os.path.join(BASE_DIR, 'evals/openml_modified_tabpfn.csv'))

