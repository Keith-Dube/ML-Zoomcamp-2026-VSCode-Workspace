# %%
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "raw" / "car_fuel_efficiency_2026.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)

features = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
]

target = "fuel_efficiency_mpg"


# %%
def download_data():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, DATA_PATH)


def split_data(df, seed):
    n = len(df)

    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)

    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train : n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val :]].reset_index(drop=True)

    return df_train, df_val, df_test


def prepare_X(df, fill_value):
    return df[features].fillna(fill_value).values


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)

    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def train_linear_regression_reg(X, y, r):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)

    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def predict(X, w0, w):
    return w0 + X.dot(w)


def rmse(y, y_pred):
    error = y - y_pred
    mse = (error**2).mean()

    return np.sqrt(mse)


# %%
download_data()

df = pd.read_csv(
    DATA_PATH,
    usecols=features + [target],
)

print(df.shape)
print(df.head())


# %%
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

horsepower_median = df.horsepower.median()
print("Horsepower median:", horsepower_median)


# %%
df_train, df_val, df_test = split_data(df, seed=42)

y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

horsepower_mean = df_train.horsepower.mean()


# %%
scores = {}

for name, fill_value in [
    ("zero", 0),
    ("mean", horsepower_mean),
]:
    X_train = prepare_X(df_train, fill_value)
    X_val = prepare_X(df_val, fill_value)

    w0, w = train_linear_regression(X_train, y_train)
    y_pred = predict(X_val, w0, w)

    score = rmse(y_val, y_pred)
    scores[name] = score

    print(name, round(score, 3))

print("Best:", min(scores, key=scores.get))


# %%
X_train = prepare_X(df_train, 0)
X_val = prepare_X(df_val, 0)

regularization_scores = []

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression_reg(X_train, y_train, r)

    y_pred = predict(X_val, w0, w)
    score = round(rmse(y_val, y_pred), 4)

    regularization_scores.append((r, score))
    print(r, score)

best_r, _ = min(
    regularization_scores,
    key=lambda result: (result[1], result[0]),
)

print("Best r:", best_r)


# %%
seed_scores = []

for seed in range(10):
    df_train, df_val, _ = split_data(df, seed)

    X_train = prepare_X(df_train, 0)
    X_val = prepare_X(df_val, 0)

    y_train = df_train[target].values
    y_val = df_val[target].values

    w0, w = train_linear_regression(X_train, y_train)

    y_pred = predict(X_val, w0, w)
    score = rmse(y_val, y_pred)

    seed_scores.append(score)
    print(seed, score)

std = np.std(seed_scores)
print("Standard deviation:", round(std, 3))


# %%
df_train, df_val, df_test = split_data(df, seed=9)

df_train_full = pd.concat(
    [df_train, df_val],
).reset_index(drop=True)

X_train_full = prepare_X(df_train_full, 0)
X_test = prepare_X(df_test, 0)

y_train_full = df_train_full[target].values
y_test = df_test[target].values

w0, w = train_linear_regression_reg(
    X_train_full,
    y_train_full,
    r=0.001,
)

y_pred = predict(X_test, w0, w)
score = rmse(y_test, y_pred)

print("Test RMSE:", round(score, 3))
