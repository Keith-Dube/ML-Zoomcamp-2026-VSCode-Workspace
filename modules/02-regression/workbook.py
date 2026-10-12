<<<<<<< HEAD
# %% [markdown]
# # Module 2 — Car Price Regression
#
# This workbook follows the Module 2 lecture example.
# The goal is to explore a car-price dataset and prepare it for
# training a linear regression model.
#
# This is separate from Homework 2, which uses the
# `car_fuel_efficiency_2026.csv` dataset.


# %%
# Import the libraries used throughout the workbook.
#
# NumPy:
#   Numerical arrays, shuffling, and mathematical operations.
#
# Pandas:
#   Loading and manipulating tabular data.
#
# Matplotlib and Seaborn:
#   Creating graphs and inspecting distributions.

import matplotlib.pyplot as plt
=======
# %%
from pathlib import Path
from urllib.request import urlretrieve

>>>>>>> dfda2a65d0785c2569737ce2a367d0f73b085997
import numpy as np
import pandas as pd
import seaborn as sns

from mlzoomcamp.paths import RAW_DATA_DIR


<<<<<<< HEAD
# %% [markdown]
# ## 1. Load the dataset
#
# `RAW_DATA_DIR` points to the shared `data/raw` directory in the
# root of the Zoomcamp workspace.
#
# Using the shared path prevents the code from depending on the
# directory from which the script is executed.


# %%
# Construct the full path to the lecture's car-price CSV file.

DATA_PATH = RAW_DATA_DIR / "car_price_chp_2.csv"

print("Attempting to read file from:")
print(DATA_PATH.resolve())

# Stop immediately with a clear message if the dataset does not exist.

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

# Load the CSV file into a Pandas DataFrame.

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape (rows, columns):", df.shape)


# %% [markdown]
# ## 2. Standardize the column names
#
# Real datasets often contain inconsistent column names.
#
# The operations below:
#
# 1. Convert every column name to lowercase.
# 2. Replace spaces with underscores.
#
# For example, `Engine HP` becomes `engine_hp`.


# %%
df.columns = df.columns.str.lower().str.replace(" ", "_")

print("Cleaned column names:")
print(df.columns.tolist())


# %% [markdown]
# ## 3. Preview the dataset
#
# `head()` returns the first five rows by default.
#
# This gives us an initial view of the records and helps confirm
# that the file was loaded correctly.


# %%
print("First five records:")
print(df.head())


# %% [markdown]
# ## 4. Inspect the data types
#
# Pandas assigns a data type to every column:
#
# - `int64` represents whole numbers.
# - `float64` represents decimal values.
# - `object` or `string` usually represents text.
#
# Understanding the types helps us separate numerical and
# categorical features.


# %%
print("Column data types:")
print(df.dtypes)


# %% [markdown]
# ## 5. Find the categorical columns
#
# This cell selects columns containing text.
#
# These columns may need to be encoded into numbers before they
# can be used by a machine-learning model.


# %%
string_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

print("Categorical/string columns:")
print(string_columns)


# %% [markdown]
# ## 6. Inspect the values in every column
#
# For every column, this cell prints:
#
# - five example values;
# - the number of unique values;
# - the number of missing values.
#
# This helps identify:
#
# - categorical columns;
# - high-cardinality columns;
# - unusual values;
# - missing data.


# %%
for column in df.columns:
    print(f"Column: {column}")
    print("Sample values:", df[column].unique()[:5])
    print("Number of unique values:", df[column].nunique())
    print("Number of missing values:", df[column].isna().sum())
    print()


# %% [markdown]
# ## 7. Inspect the target distribution
#
# `msrp` is the value that the lecture model will predict.
#
# We first inspect the complete distribution. Car prices commonly
# have a long right tail because a small number of cars are much
# more expensive than most vehicles.


# %%
sns.histplot(df["msrp"], bins=50)

plt.title("Distribution of Car Prices")
plt.xlabel("Manufacturer Suggested Retail Price")
plt.ylabel("Number of Cars")
plt.show()


# %% [markdown]
# ## 8. Inspect the common price range
#
# Extremely expensive cars can make the main section of the graph
# difficult to see.
#
# This graph only displays cars below $100,000 so that the common
# part of the distribution is easier to examine.
#
# This filtering is only for visualization. It does not remove
# records from the original DataFrame.


# %%
prices_below_100k = df.loc[df["msrp"] < 100000, "msrp"]

sns.histplot(prices_below_100k, bins=50)

plt.title("Distribution of Car Prices Below $100,000")
plt.xlabel("Manufacturer Suggested Retail Price")
plt.ylabel("Number of Cars")
plt.show()


# %% [markdown]
# ## 9. Divide the dataset into training, validation, and test sets
#
# The split is:
#
# - 60% training data
# - 20% validation data
# - 20% test data
#
# Training data fits the model.
#
# Validation data compares model versions and feature choices.
#
# Test data measures the performance of the final selected model.
#
# We shuffle the row indexes before splitting so that each subset
# receives a random selection of cars.


# %%
# Count the total number of records.

n = len(df)

# Calculate the number of records for each subset.

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

print("Expected training records:", n_train)
print("Expected validation records:", n_val)
print("Expected test records:", n_test)


# %%
# Set a random seed so that the shuffle produces the same result
# every time the workbook is run.

np.random.seed(2)

# Create an array containing every row index.

idx = np.arange(n)

# Shuffle the indexes rather than modifying the original DataFrame.

np.random.shuffle(idx)

print("First ten shuffled indexes:")
print(idx[:10])


# %%
# Use the shuffled indexes to create the three datasets.

df_train = df.iloc[idx[:n_train]].copy()

df_val = df.iloc[idx[n_train : n_train + n_val]].copy()

df_test = df.iloc[idx[n_train + n_val :]].copy()

# Reset the indexes so that every subset starts from index 0.

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("Actual training records:", len(df_train))
print("Actual validation records:", len(df_val))
print("Actual test records:", len(df_test))
print(
    "Combined total:",
    len(df_train) + len(df_val) + len(df_test),
)

# Confirm that no records were lost during the split.

assert len(df_train) + len(df_val) + len(df_test) == n


# %% [markdown]
# ## 10. Extract and transform the target
#
# `msrp` is the target, so it must be separated from the feature
# columns before model training.
#
# Car prices have a long-tailed distribution. `np.log1p()` applies:
#
#     log(price + 1)
#
# This compresses very large prices and usually makes the target
# distribution easier for a linear model to learn.
#
# `pop()` performs two operations:
#
# 1. Returns the selected column.
# 2. Removes that column from the DataFrame.
#
# This prevents the target from accidentally being included as an
# input feature, which would cause data leakage.


# %%
y_train = np.log1p(df_train.pop("msrp").to_numpy())

y_val = np.log1p(df_val.pop("msrp").to_numpy())

y_test = np.log1p(df_test.pop("msrp").to_numpy())

print("Training feature shape:", df_train.shape)
print("Training target shape:", y_train.shape)

print("Validation feature shape:", df_val.shape)
print("Validation target shape:", y_val.shape)

print("Test feature shape:", df_test.shape)
print("Test target shape:", y_test.shape)

# Confirm that the target is no longer present among the features.

assert "msrp" not in df_train.columns
assert "msrp" not in df_val.columns
assert "msrp" not in df_test.columns


# %% [markdown]
# ## 11. Inspect the transformed target
#
# This graph shows the price distribution after applying `log1p`.
#
# It should be less heavily skewed than the original MSRP
# distribution.


# %%
sns.histplot(y_train, bins=50)

plt.title("Distribution of Log-Transformed Training Prices")
plt.xlabel("log(MSRP + 1)")
plt.ylabel("Number of Cars")
plt.show()
=======

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
>>>>>>> dfda2a65d0785c2569737ce2a367d0f73b085997
