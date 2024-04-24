from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from utils import get_openml_classification, preprocess_impute
from tabpfn import TabPFNClassifier
from tqdm import tqdm
import torch as th
import time
import pandas as pd

# ignore warnings
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

openml_list = pd.read_csv('data/openml_list.csv')

# filter with instances less than 2000 and features less than 100
openml_list = openml_list[openml_list['# Instances'] <= 4000]
openml_list = openml_list[openml_list['# Features'] <= 100]
openml_list = openml_list[openml_list['# Classes'] <= 10]
openml_list = openml_list[openml_list["# Instances"] >= 100]

# Skip certain datasets because they are not working
openml_list = openml_list[openml_list['id'] != 278]
openml_list = openml_list[openml_list['id'] != 480]
openml_list = openml_list[openml_list['id'] != 462]

classifier = TabPFNClassifier(
    device='cuda',
    base_path='/home/kchauhan/repos/mds-tmu-dl/src/models/tabpfn/modified',
    model_string='kc',
    N_ensemble_configurations=32,
    seed=42)

scores = {}

for did in tqdm(openml_list.index):
    entry = openml_list.loc[did]
    print(entry)
    try:
        X, y, categorical_feats, attribute_names = get_openml_classification(
            int(entry.id), max_samples=2000, multiclass=True, shuffled=True)
    except:
        continue

    with th.no_grad():

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            train_size=0.5,
                                                            test_size=0.5)
        # preprocess_impute
        X_train, y_train, X_test, y_test = preprocess_impute(
            X_train,
            y_train,
            X_test,
            y_test,
            impute=True,
            one_hot=True,
            standardize=False,
            cat_features=categorical_feats)
        try:
            start = time.time()
            classifier.fit(X_train, y_train)
            y_eval = classifier.predict(X_test)
            y_prob = classifier.predict_proba(X_test)
            pred_time = time.time() - start
        except ValueError as ve:
            print(ve)
            print("ve", did)
            continue
        except TypeError as te:
            print(te)
            print("te", did)
            continue

        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        cross_entropy = log_loss(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_eval)

        scores[entry['Name']] = {
            "roc": roc_auc,
            "pred_time": pred_time,
            "cross_entropy": cross_entropy,
            "accuracy": accuracy
        }

for n, score in scores.items():
    print(n, score)

# Join scores and openml_list on name
scores_df = pd.DataFrame(scores).T
scores_df = scores_df.reset_index()
scores_df.columns = ['Name', 'roc', 'pred_time', 'cross_entropy', 'accuracy']
openml_list = openml_list.reset_index()
result = pd.merge(openml_list, scores_df, on='Name')
result.to_csv('data/openml_modified_tabpfn.csv', index=False)

roc = sum(s["roc"] for _, s in scores.items()) / len(scores)
# only calculate cross entropy for binary classification
cross_entropy_list = [
    s["cross_entropy"] for _, s in scores.items()
    if s["cross_entropy"] is not None
]
cross_entropy = sum(cross_entropy_list) / len(cross_entropy_list)
accuracy = sum(s["accuracy"] for _, s in scores.items()) / len(scores)
pred_time = sum(s["pred_time"] for _, s in scores.items()) / len(scores)

print(f"No of datasets: {len(scores)}")
print(f"Mean ROC: {round(roc,3)}")
print(f"Mean Cross Entropy: {round(cross_entropy,3)}")
print(f"Mean Accuracy: {round(accuracy,3)}")
print(f"Mean Prediction Time: {round(pred_time,3)}s")
