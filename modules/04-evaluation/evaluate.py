# %%
# Homework 4 - Evaluation Metrics
# ML Zoomcamp 2026

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

# %%
# Load and prepare the dataset

data_path = Path(__file__).resolve().parent / "course_lead_scoring_2026.csv"

if not data_path.exists():
    url = (
        "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
        "main/cohorts/2026/data/course_lead_scoring_2026.csv"
    )
    urlretrieve(url, data_path)

df = pd.read_csv(data_path)

# Fill missing values
categorical = df.select_dtypes(include=["object", "string", "category"]).columns
numerical = df.select_dtypes(include=["number"]).columns.drop("converted")

df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())


# %%
# Split the data into training, validation and test sets

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

print("Training:", len(df_train))
print("Validation:", len(df_val))
print("Test:", len(df_test))


# %%
# Question 1
# Find the numerical feature with the highest AUC

numerical_features = [
    "lead_score",
    "number_of_courses_viewed",
    "interaction_count",
    "annual_income",
]

y_train = df_train["converted"]

auc_scores = {}

for feature in numerical_features:
    auc = roc_auc_score(y_train, df_train[feature])

    # Reverse the direction if the AUC is below 0.5
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[feature])

    auc_scores[feature] = auc
    print(f"{feature}: {auc:.4f}")

best_feature = max(auc_scores, key=auc_scores.get)

print("\nQ1:", best_feature)


# %%
# Prepare the features for logistic regression


def prepare_data(train, validation):
    train_dicts = train.drop(columns=["converted"]).to_dict(orient="records")
    val_dicts = validation.drop(columns=["converted"]).to_dict(orient="records")

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    y_train = train["converted"].values
    y_val = validation["converted"].values

    return X_train, X_val, y_train, y_val


def train_model(train, validation, C=1.0):
    X_train, X_val, y_train, y_val = prepare_data(train, validation)

    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, predictions)

    return auc, predictions


# %%
# Question 2
# Train logistic regression and calculate validation AUC

auc, y_pred = train_model(df_train, df_val)

print("\nQ2 - Validation AUC:", round(auc, 3))


# %%
# Question 3
# Calculate precision and recall at different thresholds

thresholds = np.arange(0.0, 1.01, 0.01)

y_val = df_val["converted"].values

precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:
    predicted_positive = y_pred >= threshold
    actual_positive = y_val == 1

    tp = np.sum(predicted_positive & actual_positive)
    fp = np.sum(predicted_positive & ~actual_positive)
    fn = np.sum(~predicted_positive & actual_positive)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)


# Find where precision and recall are closest
differences = np.abs(np.array(precisions) - np.array(recalls))

# Ignore thresholds where both metrics are zero
valid = (np.array(precisions) > 0) | (np.array(recalls) > 0)
differences[~valid] = np.inf

intersection_index = np.argmin(differences)
intersection_threshold = thresholds[intersection_index]

print("\nQ3 - Precision/Recall intersection:", round(intersection_threshold, 2))


# %%
# Question 4
# Find the threshold with the highest F1 score

best_f1_index = np.argmax(f1_scores)

best_f1_threshold = thresholds[best_f1_index]
best_f1 = f1_scores[best_f1_index]

print("\nQ4 - Best F1 threshold:", round(best_f1_threshold, 2))
print("F1 score:", round(best_f1, 4))


# %%
# Plot precision, recall and F1 score

plt.figure(figsize=(9, 5))

plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, f1_scores, label="F1 Score")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall and F1 Score")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# Question 5
# Use 5-fold cross-validation and calculate AUC standard deviation


def cross_validation(data, C=1.0):

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(data):
        train = data.iloc[train_idx]
        validation = data.iloc[val_idx]

        auc, _ = train_model(train, validation, C=C)

        scores.append(auc)

    return np.array(scores)


cv_scores = cross_validation(df_full_train)

print("\nQ5 - AUC scores:", cv_scores)
print("Standard deviation:", round(cv_scores.std(), 3))


# %%
# Question 6
# Find the best value of C using cross-validation

C_values = [0.000001, 0.001, 1]

results = []

for C in C_values:
    scores = cross_validation(df_full_train, C=C)

    mean_auc = round(scores.mean(), 3)
    std_auc = round(scores.std(), 3)

    results.append((C, mean_auc, std_auc))

    print(f"C = {C}")
    print(f"Mean AUC = {mean_auc}")
    print(f"Std = {std_auc}")
    print("-----------------------")


# Select highest mean AUC, then lowest std and smallest C
best_result = sorted(results, key=lambda result: (-result[1], result[2], result[0]))[0]

print("\nQ6 - Best C:", best_result[0])
