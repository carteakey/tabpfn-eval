# %%
import pandas as pd


# Read all csv 
df_baseline = pd.read_csv('openml_baseline_scores.csv')
df_baseline_tabpfn = pd.read_csv('openml_baseline_tabpfn.csv')
df_modified_tabpfn = pd.read_csv('openml_modified_tabpfn.csv')




# drop index_x and index_y
df_baseline = df_baseline.drop(columns=['index_x', 'index_y'])

df_baseline.head()

# add classifier name to the dataframes
df_baseline_tabpfn['classifier'] = 'baseline_tabpfn'
df_modified_tabpfn['classifier'] = 'modified_tabpfn'

df_baseline_tabpfn.head()

# Move classifier column to the 9th column
cols = df_baseline_tabpfn.columns.tolist()
cols = cols[:9] + cols[-1:] + cols[9:-1]
df_baseline_tabpfn = df_baseline_tabpfn[cols]

# Move classifier column to the 9th column
cols = df_modified_tabpfn.columns.tolist()
cols = cols[:9] + cols[-1:] + cols[9:-1]
df_modified_tabpfn = df_modified_tabpfn[cols]

# Join all dataframes
df = pd.concat([df_baseline, df_baseline_tabpfn, df_modified_tabpfn])
df.head()

# sort by id
df = df.sort_values(by=['id'])

# combine train_time and pred_time
df['time'] = df['train_time'].fillna(0) + df['pred_time'].fillna(0)

# assign rank to each classifier based on roc
df['roc_rank'] = df.groupby(['id'])['roc'].rank(ascending=False)
df['accuracy_rank'] = df.groupby(['id'])['accuracy'].rank(ascending=False)
df['cross_entropy_rank'] = df.groupby(['id'])['cross_entropy'].rank(ascending=True)
df['time_rank'] = df.groupby(['id'])['time'].rank(ascending=False)

# sort by largest instances
df = df.sort_values(by=['# Instances','classifier'], ascending=False)


# groupby id and classifier and report roc
df_grouped = df.groupby(['Name', 'classifier'])['roc'].mean().unstack()
df_grouped = df_grouped[['lr','knn','rf','svm', 'mlp', 'baseline_tabpfn', 'modified_tabpfn']]
df_grouped.head(20)

# round to 3 decimal places
df_grouped = df_grouped.round(3)

# Highlight the best classifier for each dataset
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

df_grouped.style.apply(highlight_max, axis=1)



# # convert classfier to columns
# df = df.pivot(index='id', columns='classifier', values=['roc', 'accuracy', 'cross_entropy', 'time', 'roc_rank', 'accuracy_rank', 'cross_entropy_rank', 'time_rank'])


# df.T.head()

# group by classifier and report the mean of the ranks
df_grouped = df.groupby(['classifier'])[["roc_rank", "accuracy_rank", "cross_entropy_rank", "time_rank"]].mean()



df_grouped = df_grouped.T

# Sort columns by classifier
df_grouped = df_grouped[['lr','knn','rf','svm', 'mlp', 'baseline_tabpfn', 'modified_tabpfn']]



df_grouped.T.reset_index()

# highlight the maximum value in each column
df_grouped.style.highlight_min(axis=1)


