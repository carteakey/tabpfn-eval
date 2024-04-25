import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import autosklearn.classification
from autogluon.tabular import TabularPredictor

from utils import get_openml_classification, preprocess_impute

# ignore warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

openml_list = pd.read_csv("data/openml_list.csv")


classifier_dict = {
    "auto_sklearn": autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30),
    "auto_gluon": TabularPredictor(label='target').set_learner_type('default'),
}

# Set random seed
for key in classifier_dict:
    classifier_dict[key].random_state = 42

scores = {}

for did in tqdm(openml_list.index):
    entry = openml_list.loc[did]
    print(entry)
    try:
        X, y, categorical_feats, attribute_names = get_openml_classification(
            int(entry.id), max_samples=2000, multiclass=True, shuffled=True
        )
    except:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5)

    # preprocess_impute
    X_train, y_train, X_test, y_test = preprocess_impute(
        X_train,
        y_train,
        X_test,
        y_test,
        impute=True,
        one_hot=True,
        standardize=True,
        cat_features=categorical_feats,
    )

    # Replace the training and prediction code with the appropriate methods for Auto-Sklearn and AutoGluon
    for key in classifier_dict:
        try:
            start = time.time()
            classifier_dict[key].fit(X_train, y_train)
            train_time = time.time() - start
            y_pred = classifier_dict[key].predict(X_test)
            y_prob = classifier_dict[key].predict_proba(X_test)

            pred_time = time.time() - start

            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]

            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
            cross_entropy = log_loss(y_test, y_prob)
            accuracy = accuracy_score(y_test, y_pred)

        except Exception as e:
            print(e)
            continue

        print(
            f"Dataset: {entry['Name']}, Classifier: {key}, ROC: {roc_auc}, Cross-Entropy: {cross_entropy}, Accuracy: {accuracy}, Prediction Time: {pred_time}, Train Time: {train_time}"
        )

        scores[(entry["Name"], key)] = {
            "roc": roc_auc,
            "cross_entropy": cross_entropy,
            "accuracy": accuracy,
            "pred_time": pred_time,
            "train_time": train_time,
        }

# Join scores and openml_list on name
scores_df = pd.DataFrame(scores, index=["score"]).T
scores_df = scores_df.reset_index()

scores_df.to_csv('data/openml_baseline_scores.csv', index=False)

scores_df.columns = [
    "Name",
    "Classifier",
    "ROC",
    "Cross-Entropy",
    "Accuracy",
    "Prediction Time",
    "Train Time",
]
openml_list = openml_list.reset_index()
result = pd.merge(openml_list, scores_df, on="Name")
result.to_csv("data/openml_baseline_automl.csv", index=False)

print(f"No of datasets: {len(scores)}")

# Calculate mean scores for each classifier
for key in classifier_dict:
    roc = sum(s["roc"] for _, s in scores.items() if s["Classifier"] == key) / len(scores)
    cross_entropy_list = [s["cross_entropy"] for _, s in scores.items() if s["Classifier"] == key]
    cross_entropy = sum(cross_entropy_list) / len(cross_entropy_list)
    accuracy = sum(s["accuracy"] for _, s in scores.items() if s["Classifier"] == key) / len(scores)
    pred_time = sum(s["pred_time"] for _, s in scores.items() if s["Classifier"] == key) / len(scores)
    train_time = sum(s["train_time"] for _, s in scores.items() if s["Classifier"] == key) / len(scores)

    print(
        f"Classifier: {key}, Mean ROC: {round(roc,3)}, Mean Cross Entropy: {round(cross_entropy,3)}, Mean Accuracy: {round(accuracy,3)}, Mean Prediction Time: {round(pred_time,3)}, Mean Train Time: {round(train_time,3)}s"
    )
