# %%
# ML Zoomcamp 2026 - Homework 6: Decision Trees and Ensemble Learning

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
)
DATA_PATH = Path(__file__).resolve().parent / "car_fuel_efficiency_2026.csv"
TARGET = "fuel_efficiency_mpg"


# %%
# Load the dataset and split it into training, validation and test sets

def load_data():
    if not DATA_PATH.exists():
        print("Downloading the 2026 fuel-efficiency dataset...")
        urlretrieve(DATA_URL, DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    # Fill missing values with zero, as requested in the homework.
    df = df.fillna(0)

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )

   
    train_dicts = df_train.drop(columns=[TARGET]).to_dict(orient="records")
    val_dicts = df_val.drop(columns=[TARGET]).to_dict(orient="records")

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    # Some scikit-learn versions require 32-bit sparse index arrays.
    for X in (X_train, X_val):
        if X.indices.dtype != np.int32:
            X.indices = X.indices.astype(np.int32)
        if X.indptr.dtype != np.int32:
            X.indptr = X.indptr.astype(np.int32)

    y_train = df_train[TARGET].to_numpy()
    y_val = df_val[TARGET].to_numpy()

    print("Dataset shape:", df.shape)
    print("Train / validation / test:", len(df_train), len(df_val), len(df_test))
    return X_train, X_val, y_train, y_val, dv


def rmse(y_true, predictions):
    return np.sqrt(mean_squared_error(y_true, predictions))


# %%
# Questions 1-6

def main():
    X_train, X_val, y_train, y_val, dv = load_data()
    feature_names = dv.get_feature_names_out()

    # %%
    # Q1 - Which original feature is used by a depth-1 decision tree?
    tree = DecisionTreeRegressor(max_depth=1, random_state=1)
    tree.fit(X_train, y_train)
    split_index = tree.tree_.feature[0]
    if split_index < 0:
        raise ValueError("The decision tree did not produce a root split")
    split_feature = feature_names[split_index]
    original_feature = split_feature.split("=", 1)[0]

    print("\nQ1 - Decision tree split")
    print("Encoded feature:", split_feature)
    print("Answer:", original_feature)

    # %%
    # Q2 - RMSE for a random forest with 10 trees
    forest = RandomForestRegressor(
        n_estimators=10, random_state=1, n_jobs=-1
    )
    forest.fit(X_train, y_train)
    score_q2 = rmse(y_val, forest.predict(X_val))

    print("\nQ2 - Random forest validation RMSE")
    print("RMSE:", round(score_q2, 6))
   

    # %%
    # Q3 - Find the best number of trees
    n_estimators_values = [10, 50, 100, 150]
    forest_results = {}

    print("\nQ3 - Number of trees")
    for n_estimators in n_estimators_values:
        forest = RandomForestRegressor(
            n_estimators=n_estimators, random_state=1, n_jobs=-1
        )
        forest.fit(X_train, y_train)
        score = rmse(y_val, forest.predict(X_val))
        forest_results[n_estimators] = round(float(score), 3)
        print(f"n_estimators={n_estimators}: RMSE={score:.6f}")

    best_n = min(forest_results, key=forest_results.get)
    print("Answer (RMSE rounded to 3 decimals):", best_n)

    # %%
    # Q4 - Compare mean RMSE at different maximum tree depths
    depth_values = [10, 15, 20, 25]
    depth_results = {}

    print("\nQ4 - Maximum depth")
    for depth in depth_values:
        scores = []
        for n_estimators in n_estimators_values:
            forest = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=depth,
                random_state=1,
                n_jobs=-1,
            )
            forest.fit(X_train, y_train)
            scores.append(rmse(y_val, forest.predict(X_val)))

        depth_results[depth] = float(np.mean(scores))
        print(f"max_depth={depth}: mean RMSE={depth_results[depth]:.6f}")

    best_depth = min(depth_results, key=depth_results.get)
    print("Answer:", best_depth)

    # %%
    # Q5 - Find the feature with the greatest importance
    forest = RandomForestRegressor(
        n_estimators=10, max_depth=20, random_state=1, n_jobs=-1
    )
    forest.fit(X_train, y_train)

    importances = pd.Series(
        forest.feature_importances_, index=feature_names
    )
    options = [
        "vehicle_weight", "horsepower", "acceleration", "engine_displacement"
    ]

    print("\nQ5 - Feature importance")
    for feature in options:
        print(f"{feature}: {importances.get(feature, 0.0):.6f}")
    answer_q5 = max(options, key=lambda name: importances.get(name, 0.0))
    print("Answer:", answer_q5)

    # %%
    # Q6 - Compare XGBoost learning rates over 100 rounds
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dtrain, "train"), (dval, "val")]

    xgb_params = {
        "eta": 0.3,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "reg:squarederror",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
        "eval_metric": "rmse",
    }

    xgb_results = {}
    print("\nQ6 - XGBoost learning rate")
    for eta in [0.3, 0.1]:
        params = {**xgb_params, "eta": eta}
        history = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=watchlist,
            evals_result=history,
            verbose_eval=False,
        )
        score = rmse(y_val, model.predict(dval))
        xgb_results[eta] = float(score)
        print(f"eta={eta}: validation RMSE={score:.6f}")
        print(f"         last recorded RMSE={history['val']['rmse'][-1]:.6f}")

    if np.isclose(xgb_results[0.3], xgb_results[0.1], atol=1e-12, rtol=0):
        print("Answer: Both give equal value")
    else:
        print("Answer:", min(xgb_results, key=xgb_results.get))


if __name__ == "__main__":
    main()
