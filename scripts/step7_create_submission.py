import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Load datasets
train = pd.read_csv("train_features.csv", low_memory=False)
test = pd.read_csv("test_features.csv", low_memory=False)

# Convert Date
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# -----------------------------
# 1) Prepare training data
# -----------------------------
y = train["Sales"]

drop_cols = ["Sales", "Customers", "Date"]
X = train.drop(columns=drop_cols)

# test doesn't have Sales/Customers
X_test = test.drop(columns=["Date"])

# One-hot encoding categorical variables
cat_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Align columns
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

print(" Train features shape:", X.shape)
print(" Test features shape:", X_test.shape)

# -----------------------------
# 2) Train final model on full training set
# -----------------------------
final_model = XGBRegressor(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

# -----------------------------
# 3) Predict on test data
# -----------------------------
test_predictions = final_model.predict(X_test)

# Sales can't be negative
test_predictions = np.clip(test_predictions, 0, None)

# -----------------------------
# 4) Create submission.csv
# -----------------------------
submission = pd.DataFrame({
    "Id": test["Id"],
    "Sales": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("\n submission.csv created successfully!")
print(submission.head())
