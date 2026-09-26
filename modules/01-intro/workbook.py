# %% [markdown]
# # Module 1 Workbook — Fuel Efficiency Data

# %%
import sys

import numpy as np
import pandas as pd

from mlzoomcamp.paths import RAW_DATA_DIR

# %% [Q0: Read Data]
DATA_PATH = RAW_DATA_DIR / "car_fuel_efficiency_2026.csv"
df = pd.read_csv(DATA_PATH)


# %%
# Q1: Installed Pandas version
print(pd.__version__)


# %%
# Q2: Number of records
print(len(df))


# %%
# Q3: Number of different fuel types
print(df["fuel_type"].nunique())


# %%
# Q4: Number of columns containing missing values
missing_per_column = df.isna().sum()
print(missing_per_column)
print("===============================================================")
print((missing_per_column > 0).sum())


# %%
# Q5: Maximum fuel efficiency for Asian cars
asian_cars = df[df["origin"] == "Asia"]
print(asian_cars["fuel_efficiency_mpg"].max())


# %%
# Q6: Horsepower median before and after filling missing values
median_before = df["horsepower"].median()
horsepower_mode = df["horsepower"].mode().iloc[0]

df_filled = df.copy()
df_filled["horsepower"] = df_filled["horsepower"].fillna(horsepower_mode)

median_after = df_filled["horsepower"].median()

print("Before:", median_before)
print("Mode:", horsepower_mode)
print("After:", median_after)

# %%
# Q7: Sum of weights

X = df[df["origin"] == "Asia"][["vehicle_weight", "model_year"]].head(7).to_numpy()

print("X shape:", X.shape)

# X transpose multiplied by X
XTX = X.T @ X
print("XTX:", XTX)

# Invert XTX
XTX_inv = np.linalg.inv(XTX)

# Target values provided by the assignment
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# Calculate the weights
w = XTX_inv @ X.T @ y

print("w:", w)
print("Sum of w:", w.sum())
