import pandas as pd
raw_data = pd.read_pickle("full.pkl")
df = pd.DataFrame(raw_data) if isinstance(raw_data, list) else raw_data
print(df['file'].unique()[:20])