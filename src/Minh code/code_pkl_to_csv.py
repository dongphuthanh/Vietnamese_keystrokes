import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

with open("/home/mtcd001/PycharmProjects/Vietnamese_keystrokes/src/full.pkl", "rb") as f:
    all_features = pickle.load(f)

df = pd.DataFrame(all_features)
df.fillna(0, inplace=True)

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

print(df["label"].value_counts())

# Chia theo user
# Chia theo user
all_users = sorted(df["user_id"].unique())
train_users, test_users = train_test_split(all_users, test_size=0.2, random_state=42)

train_df = df[df["user_id"].isin(train_users)]
test_df = df[df["user_id"].isin(test_users)]

# Drop các cột không phải feature
drop_cols = [c for c in ["user_id", "file", "session"] if c in train_df.columns]
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

train_df.to_csv("/home/mtcd001/PycharmProjects/Vietnamese_keystrokes/src/train.csv", index=False)
test_df.to_csv("/home/mtcd001/PycharmProjects/Vietnamese_keystrokes/src/test.csv", index=False)

print(f"Train: {len(train_users)} users, {len(train_df)} rows")
print(f"Test: {len(test_users)} users, {len(test_df)} rows")
print(f"Test users: {sorted(test_users)}")