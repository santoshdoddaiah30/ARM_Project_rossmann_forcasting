import pandas as pd
import numpy as np

# Load datasets (with low_memory=False to avoid dtype warnings)
train = pd.read_csv("train.csv", low_memory=False)
test = pd.read_csv("test.csv", low_memory=False)
store = pd.read_csv("store.csv", low_memory=False)

# Merge store data
train = train.merge(store, on="Store", how="left")
test = test.merge(store, on="Store", how="left")

print("  Before cleaning:")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ----------------------------
# 1) Convert Date to datetime
# ----------------------------
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# ----------------------------
# 2) Fix StateHoliday (mixed types)
# Convert to string and clean
# ----------------------------
train["StateHoliday"] = train["StateHoliday"].astype(str)
test["StateHoliday"] = test["StateHoliday"].astype(str)

# Replace '0' with 'None'
train["StateHoliday"] = train["StateHoliday"].replace("0", "None")
test["StateHoliday"] = test["StateHoliday"].replace("0", "None")

# ----------------------------
# 3) Handle missing values
# ----------------------------

# CompetitionDistance missing → fill with median
train["CompetitionDistance"].fillna(train["CompetitionDistance"].median(), inplace=True)
test["CompetitionDistance"].fillna(test["CompetitionDistance"].median(), inplace=True)

# Promo2SinceWeek / Year missing → fill with 0
for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
    train[col].fillna(0, inplace=True)
    test[col].fillna(0, inplace=True)

# PromoInterval missing → fill with "None"
train["PromoInterval"].fillna("None", inplace=True)
test["PromoInterval"].fillna("None", inplace=True)

# Competition open since month/year missing → fill with 0
for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
    train[col].fillna(0, inplace=True)
    test[col].fillna(0, inplace=True)

# Test has 11 missing Open values → fill with 1 (assume store open)
test["Open"].fillna(1, inplace=True)

# ----------------------------
# 4) Remove closed stores rows from train
# ----------------------------
# Rows where Open==0 usually have Sales==0 and don't help demand prediction
train = train[train["Open"] == 1]

print("\n  After cleaning:")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\n  Missing values after cleaning (Train):")
print(train.isnull().sum())

print("\n  Missing values after cleaning (Test):")
print(test.isnull().sum())

# ----------------------------
# 5) Save cleaned datasets
# ----------------------------
train.to_csv("train_clean.csv", index=False)
test.to_csv("test_clean.csv", index=False)

print("\n  Saved files: train_clean.csv and test_clean.csv")
