import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from func_timeout import func_timeout
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from tabpfn import TabPFNClassifier

from .. import BASE_DIR, OPENML_LIST
from ..utils import get_openml_classification, preprocess_impute

openml_list = OPENML_LIST
timeout = 20

classifier_dict = {
    "lr":
    LogisticRegression(),
    "rf":
    RandomForestClassifier(),
    "svm":
    SVC(),
    "mlp":
    MLPClassifier(),
    "knn":
    KNeighborsClassifier(),
    "xgb":
    xgb.XGBClassifier(),
    "lgb":
    lgb.LGBMClassifier(),
    "cat":
    CatBoostClassifier(verbose=0),
    "tabpfn_mod":
    TabPFNClassifier(seed=42,
                     model_string="kc",
                     base_path=os.path.join(BASE_DIR, "tabpfn/")),
    "tabpfn":
    TabPFNClassifier(seed=42)
}

# Hyperparameters space for the classifiers
hyperparameters = {
    "lr": {
        "max_iter": [50, 100, 200, 500, 1000],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "fit-intercept": [True, False],
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "C": [0.1, 1.0, 10.0, 100.0],
    },
    "rf": {
        "n_estimators": [10, 50, 100, 200, 500],
        "criterion": ["gini", "entropy"],
        "probability": [True],
        "max_depth": [None, 10, 50, 100, 200],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "probability": [True],
        "degree": [2, 3, 4, 5],
        "gamma": ["scale", "auto"],
    },
    "mlp": {
        "max_iter": [50, 100, 200, 500, 1000],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "learning_rate_init": [0.001, 0.01, 0.1],
    },
    "knn": {
        "n_neighbors": [3, 5, 11, 19],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [30, 50, 100],
        "p": [1, 2],
    },
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
            int(entry.id), max_samples=4000, multiclass=True, shuffled=True)
    except:
        continue

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

                # try if the classifier can be trained on the data
                try:

                    # Timeout after 20 seconds
                    def long_function():
                        return classifier_dict[key].fit(X_train, y_train)

                    try:
                        result = func_timeout(timeout, long_function)
                    except Exception as e:
                        print(e)
                        result = None

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

                values.append(
                    (roc_auc, cross_entropy, accuracy, pred_time, train_time))

            roc_auc, cross_entropy, accuracy, pred_time, train_time = max(
                values, key=lambda x: x[0])

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

        dict_key = f"{entry['Name']}_{key}"
        scores[dict_key] = {
            "Classifier": key,
            "Dataset": entry["Name"],
            "roc": roc_auc,
            "cross_entropy": cross_entropy,
            "accuracy": accuracy,
            "pred_time": pred_time,
            "train_time": train_time,
        }

#Save json just in case
pd.DataFrame(scores).to_json('data/openml_baseline_scores.json')

scores_df = pd.read_json('data/openml_baseline_scores.json')
# Join scores and openml_list on name
scores_df = scores_df.T
scores_df = scores_df.reset_index()
# scores_df['Classifier'] = scores_df['Classifier'].str.split('_').str[-1]
scores_df.columns = [
    "index", "classifier", "Name", "roc", "cross_entropy", "accuracy",
    "pred_time", "train_time"
]

openml_list = openml_list.reset_index()
result = pd.merge(openml_list, scores_df, on="Name")
result.to_csv("data/openml_baseline_scores.csv", index=False)

# Calculate mean scores for each classifier
print(
    scores_df.groupby("classifier")[[
        "roc", "cross_entropy", "accuracy", "pred_time", "train_time"
    ]].mean())

# print to latex
# print(scores_df.groupby("classifier")[["roc", "cross_entropy", "accuracy", "pred_time", "train_time"]].mean().to_latex())
