import time
from utils import get_openml_classification, preprocess_impute
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

# ignore warnings
import warnings

from tqdm import tqdm


def warn(*args, **kwargs):
    pass


warnings.warn = warn

openml_list = pd.read_csv("data/openml_list.csv")

# filter with instances less than 2000 and features less than 100
openml_list = openml_list[openml_list["# Instances"] <= 2000]
openml_list = openml_list[openml_list["# Features"] <= 100]
openml_list = openml_list[openml_list["# Classes"] <= 10]
openml_list = openml_list[openml_list["# Instances"] >= 100]


# Skip certain datasets because they are not working
openml_list = openml_list[openml_list["id"] != 278]
openml_list = openml_list[openml_list["id"] != 480]
openml_list = openml_list[openml_list["id"] != 462]
openml_list = openml_list[openml_list["id"] != 285]

classifier_dict = {
    "lr": LogisticRegression(),
    "rf": RandomForestClassifier(),
    "svm": SVC(),
    "mlp": MLPClassifier(),
    "knn": KNeighborsClassifier(),
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
result.to_csv("data/openml_baseline.csv", index=False)

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
