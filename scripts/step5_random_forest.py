import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_csv("train_features.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])

# Sort by date
train = train.sort_values("Date")

# Time-based validation split (last 6 weeks)
cutoff_date = train["Date"].max() - pd.Timedelta(days=42)
train_part = train[train["Date"] <= cutoff_date].copy()
valid_part = train[train["Date"] > cutoff_date].copy()

# Target
y_train = train_part["Sales"]
y_valid = valid_part["Sales"]

# Drop columns not usable
drop_cols = ["Sales", "Customers", "Date"]
X_train = train_part.drop(columns=drop_cols)
X_valid = valid_part.drop(columns=drop_cols)

# One-hot encoding categorical columns
cat_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_valid = pd.get_dummies(X_valid, columns=cat_cols, drop_first=True)

# Align columns
X_train, X_valid = X_train.align(X_valid, join="left", axis=1, fill_value=0)

print("✅ Features:", X_train.shape[1])

# -----------------------------
# Random Forest Model
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_valid)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mae = mean_absolute_error(y_valid, y_pred)

print("\n📌 MODEL RESULTS (Random Forest)")
print("✅ RMSE:", rmse)
print("✅ MAE:", mae)
