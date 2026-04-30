import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

with open("/home/mtcd001/PycharmProjects/Vietnamese_keystrokes/src/attack.pkl", "rb") as f:
    attack_features = pickle.load(f)

df = pd.DataFrame(attack_features)
df.fillna(0, inplace=True)

le = LabelEncoder()
le.fit(["bonafide", "paraphrase", "transcribe"])
df["label"] = le.transform(df["label"])

df.to_csv("/home/mtcd001/PycharmProjects/Vietnamese_keystrokes/src/attack.csv", index=False)

print(f"Saved {len(df)} rows to attack.csv")
print(df["label"].value_counts())