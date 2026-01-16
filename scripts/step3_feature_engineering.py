import pandas as pd

# Load cleaned datasets
train = pd.read_csv("train_clean.csv")
test = pd.read_csv("test_clean.csv")

# Convert Date to datetime again (because csv loses datetime format)
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

def create_date_features(df):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    return df

# Create date-based features
train = create_date_features(train)
test = create_date_features(test)

print(" Feature Engineering Done!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\n New columns added:")
print(["Year", "Month", "Day", "WeekOfYear", "DayOfYear", "IsWeekend"])

print("\n Train preview:")
print(train.head())

# Save new feature datasets
train.to_csv("train_features.csv", index=False)
test.to_csv("test_features.csv", index=False)

print("\n Saved files: train_features.csv and test_features.csv")
