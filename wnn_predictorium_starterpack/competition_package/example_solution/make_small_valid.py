import pandas as pd
import os

base = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base, "..", "datasets", "valid.parquet")
df = pd.read_parquet(path)
print('loaded', len(df), 'rows')
df2 = df.head(200000)
out = os.path.join(base, 'small_valid.parquet')
df2.to_parquet(out)
print('wrote small file', len(df2), 'rows to', out)
