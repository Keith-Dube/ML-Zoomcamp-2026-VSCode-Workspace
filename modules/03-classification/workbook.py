# %%
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "raw" / "course_lead_scoring_2026.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/"
    "main/cohorts/2026/data/course_lead_scoring_2026.csv"
)

categorical = [
    "lead_source",
    "industry",
    "employment_status",
    "location",
]

numerical = [
    "annual_income",
    "number_of_courses_viewed",
    "interaction_count",
    "lead_score",
]

features = categorical + numerical
target = "converted"


# %%
if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATA_URL, DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Missing values before filling:")
print(df.isna().sum())


# %%
df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)

industry_mode = df.industry.mode().iloc[0]

print("\nQ1:", industry_mode)


# %%
correlations = df[numerical].corr()

pairs = [
    ("interaction_count", "lead_score"),
    ("number_of_courses_viewed", "lead_score"),
    ("number_of_courses_viewed", "interaction_count"),
    ("annual_income", "interaction_count"),
]

pair_scores = {pair: correlations.loc[pair[0], pair[1]] for pair in pairs}

for pair, score in pair_scores.items():
    print(pair, score)

best_pair = max(pair_scores, key=pair_scores.get)
print("Q2:", best_pair)


# %%
df_full_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
)

df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=42,
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.pop(target).values
y_val = df_val.pop(target).values
y_test = df_test.pop(target).values

print("\nSplit:", len(df_train), len(df_val), len(df_test))


# %%
mi_scores = {
    column: round(mutual_info_score(y_train, df_train[column]), 2) for column in categorical
}

for column, score in mi_scores.items():
    print(column, score)

best_mi_feature = max(mi_scores, key=mi_scores.get)
print("Q3:", best_mi_feature)


# %%
def train_model(df_train, y_train, selected_features, C=1.0):
    dv = DictVectorizer(sparse=False)

    train_dicts = df_train[selected_features].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    model = LogisticRegression(
        solver="liblinear",
        C=C,
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return dv, model


def calculate_accuracy(dv, model, df, y, selected_features):
    records = df[selected_features].to_dict(orient="records")
    X = dv.transform(records)
    predictions = model.predict(X)

    return accuracy_score(y, predictions)


dv, model = train_model(df_train, y_train, features)

baseline_accuracy = calculate_accuracy(
    dv,
    model,
    df_val,
    y_val,
    features,
)

print("\nQ4:", round(baseline_accuracy, 2))


# %%
removal_results = {}

for removed_feature in [
    "lead_source",
    "number_of_courses_viewed",
    "interaction_count",
]:
    selected_features = [feature for feature in features if feature != removed_feature]

    dv_small, model_small = train_model(
        df_train,
        y_train,
        selected_features,
    )

    accuracy = calculate_accuracy(
        dv_small,
        model_small,
        df_val,
        y_val,
        selected_features,
    )

    difference = baseline_accuracy - accuracy
    removal_results[removed_feature] = difference

    print(removed_feature, accuracy, difference)

least_useful_feature = min(removal_results, key=removal_results.get)
print("Q5:", least_useful_feature)


# %%
c_scores = []

for C in [0.000001, 0.00001, 0.0001, 0.001]:
    dv_c, model_c = train_model(
        df_train,
        y_train,
        features,
        C=C,
    )

    accuracy = calculate_accuracy(
        dv_c,
        model_c,
        df_val,
        y_val,
        features,
    )

    rounded_accuracy = round(accuracy, 3)
    c_scores.append((C, rounded_accuracy))

    print(C, rounded_accuracy)

best_c, best_accuracy = max(
    c_scores,
    key=lambda result: (result[1], -result[0]),
)

print("Q6:", best_c, best_accuracy)
