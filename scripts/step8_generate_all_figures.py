import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
train = pd.read_csv("train_features.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])
train = train.sort_values("Date")

# --------------------------
# FIGURE 1: Sales Trend Over Time
# --------------------------
daily_sales = train.groupby("Date")["Sales"].sum()
plt.figure(figsize=(12, 5))
plt.plot(daily_sales.index, daily_sales.values)
plt.title("Figure 1: Daily Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("Figure1_Sales_Trend.png", dpi=300)
plt.close()

# --------------------------
# FIGURE 2: Promo vs No Promo
# --------------------------
promo_sales = train.groupby("Promo")["Sales"].mean()
plt.figure(figsize=(6, 4))
plt.bar(["No Promo", "Promo"], promo_sales.values)
plt.title("Figure 2: Average Sales - Promo vs No Promo")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("Figure2_Promo_vs_NoPromo.png", dpi=300)
plt.close()

# --------------------------
# FIGURE 3: Holiday effect
# --------------------------
holiday_sales = train.groupby("StateHoliday")["Sales"].mean().sort_values(ascending=False)
plt.figure(figsize=(7, 4))
plt.bar(holiday_sales.index.astype(str), holiday_sales.values)
plt.title("Figure 3: Average Sales by StateHoliday")
plt.xlabel("StateHoliday")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("Figure3_Holiday_Effect.png", dpi=300)
plt.close()

# --------------------------
# FIGURE 4: Monthly seasonality
# --------------------------
monthly_sales = train.groupby("Month")["Sales"].mean()
plt.figure(figsize=(8, 4))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
plt.title("Figure 4: Monthly Seasonality (Avg Sales by Month)")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig("Figure4_Monthly_Seasonality.png", dpi=300)
plt.close()

# --------------------------
# FIGURE 5: Correlation heatmap (numeric features)
# --------------------------
numeric_cols = train.select_dtypes(include=[np.number]).copy()
corr = numeric_cols.corr(numeric_only=True)

plt.figure(figsize=(10, 7))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.title("Figure 5: Correlation Heatmap (Numeric Features)")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
plt.tight_layout()
plt.savefig("Figure5_Correlation_Heatmap.png", dpi=300)
plt.close()

# --------------------------
# BONUS FIGURE 9: Avg Sales by DayOfWeek
# --------------------------
dow_sales = train.groupby("DayOfWeek")["Sales"].mean()
plt.figure(figsize=(8, 4))
plt.bar(dow_sales.index.astype(str), dow_sales.values)
plt.title("Figure 9: Average Sales by DayOfWeek")
plt.xlabel("DayOfWeek (1=Mon ... 7=Sun)")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("Figure9_AvgSales_by_DayOfWeek.png", dpi=300)
plt.close()

# --------------------------
# BONUS FIGURE 10: Avg Sales by StoreType
# --------------------------
storetype_sales = train.groupby("StoreType")["Sales"].mean().sort_values(ascending=False)
plt.figure(figsize=(7, 4))
plt.bar(storetype_sales.index.astype(str), storetype_sales.values)
plt.title("Figure 10: Average Sales by StoreType")
plt.xlabel("StoreType")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("Figure10_AvgSales_by_StoreType.png", dpi=300)
plt.close()

# --------------------------
# MODEL TRAINING for validation visuals (last 42 days)
# --------------------------
cutoff_date = train["Date"].max() - pd.Timedelta(days=42)
train_part = train[train["Date"] <= cutoff_date].copy()
valid_part = train[train["Date"] > cutoff_date].copy()

y_train = train_part["Sales"]
y_valid = valid_part["Sales"]

drop_cols = ["Sales", "Customers", "Date"]
X_train = train_part.drop(columns=drop_cols)
X_valid = valid_part.drop(columns=drop_cols)

cat_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_valid = pd.get_dummies(X_valid, columns=cat_cols, drop_first=True)
X_train, X_valid = X_train.align(X_valid, join="left", axis=1, fill_value=0)

# XGBoost model
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_valid)

# --------------------------
# FIGURE 6: Model RMSE Comparison
# --------------------------
models = ["Linear Regression", "Random Forest", "XGBoost"]
rmse_values = [2636.46, 1167.59, 925.72]  # your actual results

plt.figure(figsize=(8, 4))
plt.bar(models, rmse_values)
plt.title("Figure 6: RMSE Comparison Across Models")
plt.ylabel("RMSE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("Figure6_Model_RMSE_Comparison.png", dpi=300)
plt.close()

# --------------------------
# FIGURE 7: Actual vs Predicted
# --------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_valid, y_pred, alpha=0.25)
plt.title("Figure 7: Actual vs Predicted Sales (XGBoost)")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.tight_layout()
plt.savefig("Figure7_Actual_vs_Predicted.png", dpi=300)
plt.close()

# --------------------------
# FIGURE 8: Feature Importance (Top 15)
# --------------------------
importances = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 5))
plt.barh(importances.index[::-1], importances.values[::-1])
plt.title("Figure 8: XGBoost Feature Importance (Top 15)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("Figure8_Feature_Importance.png", dpi=300)
plt.close()

# --------------------------
# Print evaluation metrics (for report)
# --------------------------
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(" Figures created successfully (10 PNG files).")
print(" Validation metrics (XGBoost): RMSE =", rmse, "| MAE =", mae, "| R2 =", r2)
