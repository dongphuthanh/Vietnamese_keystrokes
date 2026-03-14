import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

with open("full.pkl", "rb") as f:
    all_features = pickle.load(f)

df = pd.DataFrame(all_features)
df.fillna(0, inplace=True)

# Encode labels to numbers
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

print(df["label"].value_counts())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train: {len(train_df)} rows")
print(f"Test:  {len(test_df)} rows")