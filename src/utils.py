import pandas as pd
import torch
import numpy as np
import openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer



def get_openml_classification(did, max_samples, multiclass=True, shuffled=True):
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    if not multiclass:
        X = X[y < 2]
        y = y[y < 2]

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        print("Not a NP Array, skipping")
        return None, None, None, None

    if not shuffled:
        sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
        pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
        X, y = X[sort][-pos * 2 :], y[sort][-pos * 2 :]
        y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
        X = (
            torch.tensor(X)
            .reshape(2, -1, X.shape[1])
            .transpose(0, 1)
            .reshape(-1, X.shape[1])
            .flip([0])
            .float()
        )
    else:
        order = np.arange(y.shape[0])
        np.random.seed(13)
        np.random.shuffle(order)
        X, y = torch.tensor(X[order]), torch.tensor(y[order])
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]

    return X, y, list(np.where(categorical_indicator)[0]), attribute_names


def preprocess_impute(
    x, y, test_x, test_y, impute, one_hot, standardize, cat_features=[]
):

    x, y, test_x, test_y = (
        x.cpu().numpy(),
        y.cpu().long().numpy(),
        test_x.cpu().numpy(),
        test_y.cpu().long().numpy(),
    )

    if impute:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
        imp_mean.fit(x)
        x, test_x = imp_mean.transform(x), imp_mean.transform(test_x)

    if one_hot:

        def make_pd_from_np(x):
            data = pd.DataFrame(x)
            for c in cat_features:
                data.iloc[:, c] = data.iloc[:, c].astype("int")
            return data

        x, test_x = make_pd_from_np(x), make_pd_from_np(test_x)
        transformer = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                    cat_features,
                )
            ],
            remainder="passthrough",
        )
        transformer.fit(x)
        x, test_x = transformer.transform(x), transformer.transform(test_x)

    if standardize:
        scaler = MinMaxScaler()
        scaler.fit(x)
        x, test_x = scaler.transform(x), scaler.transform(test_x)

    return x, y, test_x, test_y
