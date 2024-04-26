# Generate a summary of the results of the experiments
import os
import pandas as pd

from .. import BASE_DIR

# Read all csv 
df_baseline = pd.read_json(os.path.join(BASE_DIR, 'data/openml_baseline_scores.json'))
df_baseline_tabpfn = pd.read_csv(os.path.join(BASE_DIR, 'data/openml_baseline_tabpfn.csv'))
df_modified_tabpfn = pd.read_csv(os.path.join(BASE_DIR, 'data/openml_modified_tabpfn.csv'))
openml_list = pd.read_csv(os.path.join(BASE_DIR, 'data/openml_list.csv'))

df_baseline = df_baseline.T
df_baseline = df_baseline.reset_index()
# df_baseline['Classifier'] = df_baseline['Classifier'].str.split('_').str[-1]
df_baseline.columns = [
    "index", "classifier", "Name", "roc", "cross_entropy", "accuracy",
    "pred_time", "train_time"
]
openml_list = openml_list.reset_index()
df_baseline = pd.merge(openml_list, df_baseline, on="Name")


# drop index_x and index_y
df_baseline = df_baseline.drop(columns=['index_x', 'index_y'])

df_baseline.head()

# add classifier name to the dataframes
df_baseline_tabpfn['classifier'] = 'baseline_tabpfn'
df_modified_tabpfn['classifier'] = 'modified_tabpfn'

# Move classifier column to the 9th column
cols = df_baseline_tabpfn.columns.tolist()
cols = cols[:9] + cols[-1:] + cols[9:-1]
df_baseline_tabpfn = df_baseline_tabpfn[cols]

# Move classifier column to the 9th column
cols = df_modified_tabpfn.columns.tolist()
cols = cols[:9] + cols[-1:] + cols[9:-1]
df_modified_tabpfn = df_modified_tabpfn[cols]


# drop index
df_baseline_tabpfn = df_baseline_tabpfn.drop(columns=['index'])
df_modified_tabpfn = df_modified_tabpfn.drop(columns=['index'])

# Join all dataframes
df = pd.concat([df_baseline, df_baseline_tabpfn, df_modified_tabpfn])

# sort by id
df = df.sort_values(by=['id'])

# combine train_time and pred_time
df['time'] = df['train_time'].fillna(0) + df['pred_time'].fillna(0)

# assign rank to each classifier based on roc
df['roc_rank'] = df.groupby(['id'])['roc'].rank(ascending=False)
df['accuracy_rank'] = df.groupby(['id'])['accuracy'].rank(ascending=False)
df['cross_entropy_rank'] = df.groupby(['id'])['cross_entropy'].rank(ascending=True)
df['time_rank'] = df.groupby(['id'])['time'].rank(ascending=False)


# group by classifier and report the mean of the ranks
df_grouped_rank = df.groupby(['classifier'])[["roc_rank", "accuracy_rank", "cross_entropy_rank", "time_rank"]].mean()
df_grouped_rank = df_grouped_rank.T
df_grouped_rank.T.reset_index()
# highlight the maximum value in each column
df_grouped_rank.style.highlight_min(axis=1)


# group by classifier and report the mean of the ranks
df_grouped = df.groupby(['classifier'])[["roc", "accuracy", "cross_entropy", "time"]].mean()
df_grouped = df_grouped.T
df_grouped.T.reset_index()
# highlight the maximum value in each column
df_grouped.style.highlight_max(axis=1)

df_grouped = df_grouped.round(3)
df_grouped_rank = df_grouped_rank.round(3)

# save the results
df_grouped.to_csv(os.path.join(BASE_DIR, 'data/openml_summary.csv'))
df_grouped_rank.to_csv(os.path.join(BASE_DIR, 'data/openml_summary_rank.csv'))


# df_baseline_tabpfn vs df_modified_tabpfn
df_baseline_tabpfn = df_baseline_tabpfn.round(3)
df_modified_tabpfn = df_modified_tabpfn.round(3)

df_baseline_grouped = df_baseline_tabpfn.groupby(['classifier'])[['roc', 'accuracy', 'cross_entropy', 'pred_time']].mean()
df_baseline_grouped = df_baseline_grouped.T
df_baseline_grouped.T.reset_index()
df_baseline_grouped.style.highlight_max(axis=1)

df_modified_grouped = df_modified_tabpfn.groupby(['classifier'])[['roc', 'accuracy', 'cross_entropy', 'pred_time']].mean()
df_modified_grouped = df_modified_grouped.T
df_modified_grouped.T.reset_index()
df_modified_grouped.style.highlight_max(axis=1)

# save the results
df_baseline_grouped.to_csv(os.path.join(BASE_DIR, 'data/openml_baseline_tabpfn_summary.csv'))
df_modified_grouped.to_csv(os.path.join(BASE_DIR, 'data/openml_modified_tabpfn_summary.csv'))
