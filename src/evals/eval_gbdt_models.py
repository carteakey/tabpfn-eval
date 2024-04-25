import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from utils import get_openml_classification, preprocess_impute
from src import BASE_DIR
import os


# ignore warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

openml_list = pd.read_csv(os.path.join(BASE_DIR, "data/openml_list.csv"))


classifier_dict = {
    "xgb": xgb.XGBClassifier(),
    "lgb": lgb.LGBMClassifier(),
    "cat": CatBoostClassifier(verbose=0),
}

# Hyperparameters space for the classifiers
hyperparameters = {
    "xgb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [6, 10, 15, 20],
        "learning_rate": [0.001, 0.01, 0.1],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "lgb": {
        "n_estimators": [50, 100, 200],
        "max_depth": [6, 10, 15, 20],
        "learning_rate": [0.001, 0.01, 0.1],
        "num_leaves": [31, 60, 120, 240, 480, 960],
        "min_child_samples": [10, 20, 30, 40, 50],
    },
    "cat": {
        "iterations": [50, 100, 200],
        "depth": [6, 8, 10],
        "learning_rate": [0.001, 0.01, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
    },
}

# Set random seed
for key in classifier_dict:
    classifier_dict[key].random_state = 42

# Add cross-validation

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

    for key in classifier_dict:

        try:

            avg_pred_time = 0
            avg_train_time = 0
            avg_roc = 0
            avg_cross_entropy = 0
            avg_accuracy = 0

            values = []

            # 20 random hyperparameters
            for _ in range(20):

                # Set hyperparameters randomly
                for param in hyperparameters[key]:
                    setattr(
                        classifier_dict[key],
                        param,
                        np.random.choice(hyperparameters[key][param]),
                    )

                start = time.time()

                # try if the classifier can be trained on the en

                try:
                    # predict on test data
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

                values.append((roc_auc, cross_entropy, accuracy, pred_time, train_time))

            roc_auc, cross_entropy, accuracy, pred_time, train_time = max(values, key=lambda x: x[0])

        except ValueError as ve:
            print(ve)
            print("ve", did)
            continue
        except TypeError as te:
            print(te)
            print("te", did)
            continue
        except Exception as e:
            print(e)
            print("e", did)
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
result.to_csv("data/openml_baseline_gbdt.csv", index=False)

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
