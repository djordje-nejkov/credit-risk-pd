
import pandas as pd

# run exact-duplicate screening across the 91 columns in the parquet
# only does screening, what actually would get dropped requires a manual check
# says nothing about collinearity, decided at encoding
# returns Series([], dtype: uint64), found no duplicates

c = pd.read_parquet('data/cohort.parquet')
h = c.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
print(h[h.duplicated(keep=False)])