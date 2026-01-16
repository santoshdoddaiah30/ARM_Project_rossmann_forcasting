import pandas as pd

# 1) Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
store = pd.read_csv("store.csv")
sample = pd.read_csv("sample_submission.csv")

# 2) Print dataset shapes
print("✅ Train shape:", train.shape)
print("✅ Test shape:", test.shape)
print("✅ Store shape:", store.shape)
print("✅ Sample submission shape:", sample.shape)

# 3) Print columns
print("\n📌 Train columns:\n", train.columns)
print("\n📌 Test columns:\n", test.columns)
print("\n📌 Store columns:\n", store.columns)

# 4) Missing values
print("\n❗ Missing values in TRAIN:\n", train.isnull().sum())
print("\n❗ Missing values in TEST:\n", test.isnull().sum())
print("\n❗ Missing values in STORE:\n", store.isnull().sum())

# 5) Merge store info into train and test
train_merged = train.merge(store, on="Store", how="left")
test_merged = test.merge(store, on="Store", how="left")

print("\n✅ Train merged shape:", train_merged.shape)
print("✅ Test merged shape:", test_merged.shape)

# 6) Preview first rows
print("\n🔍 Train merged preview:")
print(train_merged.head())

print("\n🔍 Test merged preview:")
print(test_merged.head())
