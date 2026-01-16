import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Load cleaned feature data
train = pd.read_csv("train_features.csv", low_memory=False)
test = pd.read_csv("test_features.csv", low_memory=False)

# Convert Date
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# Sort by date (important for time split)
train = train.sort_values("Date")

# -----------------------------
# 1) Time-based split (last 6 weeks for validation)
# -----------------------------
cutoff_date = train["Date"].max() - pd.Timedelta(days=42)

train_part = train[train["Date"] <= cutoff_date].copy()
valid_part = train[train["Date"] > cutoff_date].copy()

print(" Train part:", train_part.shape)
print(" Valid part:", valid_part.shape)

# -----------------------------
# 2) Target variable
# -----------------------------
y_train = train_part["Sales"]
y_valid = valid_part["Sales"]

# -----------------------------
# 3) Drop columns NOT available in test or should not be used
# -----------------------------
drop_cols = ["Sales", "Customers", "Date"]

X_train = train_part.drop(columns=drop_cols)
X_valid = valid_part.drop(columns=drop_cols)

# -----------------------------
# 4) One-hot encoding (categorical columns)
# -----------------------------
cat_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]

X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_valid = pd.get_dummies(X_valid, columns=cat_cols, drop_first=True)

# Align validation to train columns
X_train, X_valid = X_train.align(X_valid, join="left", axis=1, fill_value=0)

print(" Features after encoding:", X_train.shape[1])

# -----------------------------
# 5) Train baseline model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Predict validation
y_pred = model.predict(X_valid)

# -----------------------------
# 6) Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mae = mean_absolute_error(y_valid, y_pred)

print("\n BASELINE MODEL RESULTS (Linear Regression)")
print(" RMSE:", rmse)
print(" MAE:", mae)
