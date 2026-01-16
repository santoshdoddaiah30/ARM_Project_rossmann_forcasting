import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

import matplotlib.pyplot as plt


# -----------------------------
# PATHS (pipeline is inside scripts/)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")


# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# -----------------------------
# 1) LOAD + MERGE
# -----------------------------
def load_and_merge():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), low_memory=False)
    store = pd.read_csv(os.path.join(DATA_DIR, "store.csv"), low_memory=False)

    train = train.merge(store, on="Store", how="left")
    test = test.merge(store, on="Store", how="left")

    return train, test


# -----------------------------
# 2) CLEAN
# -----------------------------
def clean_data(train, test):
    # Fill missing CompetitionDistance
    for df in [train, test]:
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

    # Fill store-related missing values
    cols_zero = [
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear"
    ]
    for col in cols_zero:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    train["PromoInterval"] = train["PromoInterval"].fillna("None")
    test["PromoInterval"] = test["PromoInterval"].fillna("None")

    # Test Open missing
    if "Open" in test.columns:
        test["Open"] = test["Open"].fillna(1)

    # Remove closed rows from train
    if "Open" in train.columns:
        train = train[train["Open"] == 1].copy()

    return train, test


# -----------------------------
# 3) FEATURE ENGINEERING
# -----------------------------
def add_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    return df


def feature_engineering(train, test):
    train = add_date_features(train)
    test = add_date_features(test)
    return train, test


# -----------------------------
# 4) ENCODING
# -----------------------------
def encode_features(train, test, target_col="Sales"):
    y = train[target_col]
    X = train.drop(columns=[target_col, "Date"])
    test = test.drop(columns=["Date"])


    combined = pd.concat([X, test], axis=0)

    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(combined[cat_cols])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    combined_final = pd.concat(
        [combined[num_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)],
        axis=1
    )

    X_final = combined_final.iloc[:len(X), :]
    test_final = combined_final.iloc[len(X):, :]

    return X_final, y, test_final


# -----------------------------
# 5) TRAIN XGBOOST
# -----------------------------
def train_xgboost(X_train, y_train, X_valid, y_valid):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    metrics = {
        "RMSE": float(rmse(y_valid, preds)),
        "MAE": float(mean_absolute_error(y_valid, preds)),
        "R2": float(r2_score(y_valid, preds))
    }

    return model, preds, metrics


# -----------------------------
# 6) FIGURES (basic set)
# -----------------------------
def generate_figures(train_df):
    ensure_dir(FIG_DIR)

    # Figure 1: Sales trend
    daily_sales = train_df.groupby("Date")["Sales"].sum().sort_index()
    plt.figure(figsize=(10, 4))
    plt.plot(daily_sales.index, daily_sales.values)
    plt.title("Daily Total Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure1_Sales_Trend.png"))
    plt.close()

    # Figure 2: Promo effect
    promo_sales = train_df.groupby("Promo")["Sales"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(["No Promo", "Promo"], [promo_sales.get(0, 0), promo_sales.get(1, 0)])
    plt.title("Average Sales: Promo vs No Promo")
    plt.ylabel("Avg Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure2_Promo_vs_NoPromo.png"))
    plt.close()

    # Figure 3: Holiday effect
    holiday_sales = train_df.groupby("StateHoliday")["Sales"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(holiday_sales.index.astype(str), holiday_sales.values)
    plt.title("Average Sales by StateHoliday")
    plt.xlabel("StateHoliday")
    plt.ylabel("Avg Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure3_Holiday_Effect.png"))
    plt.close()

    # Figure 4: Monthly seasonality
    monthly_sales = train_df.groupby("Month")["Sales"].mean().sort_index()
    plt.figure(figsize=(6, 4))
    plt.plot(monthly_sales.index, monthly_sales.values)
    plt.title("Monthly Seasonality (Avg Sales)")
    plt.xlabel("Month")
    plt.ylabel("Avg Sales")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure4_Monthly_Seasonality.png"))
    plt.close()

    print(f"✅ Figures saved to: {FIG_DIR}")


# -----------------------------
# 7) SUBMISSION
# -----------------------------
def create_submission(model, test_final, test_raw):
    preds = model.predict(test_final)

    submission = pd.DataFrame({
        "Id": test_raw["Id"],
        "Sales": preds
    })

    out_path = os.path.join(DATA_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)

    print(f"✅ submission.csv created at: {out_path}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("Loading & merging data...")
    train_raw, test_raw = load_and_merge()

    print("Cleaning data...")
    train_raw, test_raw = clean_data(train_raw, test_raw)

    print("Feature engineering...")
    train_raw, test_raw = feature_engineering(train_raw, test_raw)

    # Save engineered datasets (optional)
    train_raw.to_csv(os.path.join(DATA_DIR, "train_features.csv"), index=False)
    test_raw.to_csv(os.path.join(DATA_DIR, "test_features.csv"), index=False)

    print("Encoding features...")
    X, y, test_final = encode_features(train_raw, test_raw, target_col="Sales")

    print("Train/validation split...")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=42)

    print("Training XGBoost model...")
    model, valid_preds, metrics = train_xgboost(X_train, y_train, X_valid, y_valid)

    print("\n📌 XGBoost Validation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nGenerating figures...")
    generate_figures(train_raw)

    print("\nGenerating submission file...")
    create_submission(model, test_final, test_raw)

    print("\n✅ Pipeline finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    main()
