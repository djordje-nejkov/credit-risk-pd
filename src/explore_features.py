import pandas as pd

# run exact-duplicate screening across the 91 columns in the parquet
# only does screening, what actually would get dropped requires a manual check
# says nothing about collinearity, decided at encoding
# returns Series([], dtype: uint64), found no duplicates

c = pd.read_parquet('data/cohort.parquet')
h = c.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
print(h[h.duplicated(keep=False)])

# check for number of unique values for each column, some features like tot_hi_cred_lim or tot_cur_bal are excluded in judging since
# they are continuous metrics that the models interpret based on thresholds.

# candidates are emp_title and zip_code, though columns like title not being there also warrants a check, since it is expected
# to have high cardinality being that its values are free text.
# emp_title returns 86091 levels, will be excluded for cardinality.

print(c.nunique().sort_values(ascending=False).head(30))

# check for null values.
# desc comes back with 282,895 null of 282,934, 39 populated rows, excluded.
# title comes back with 84 null, and has fewer than 161 levels. it is populated and low cardinality, kept.
# zip_code has no null values, requires further checking.
# emp_title returns 18,978 null, but is excluded for cardinality anyway.

print(c[['desc', 'title', 'zip_code', 'emp_title']].isna().sum())

# out of 901 unique zipcodes, there is at least one with only one loan, the bottom 25 percent have 66 or fewer loans, the bottom 50 have 153 or fewer loans.
# since the bottom 25 percent at 14.9 percent pd would have ~10, the whole zip_code feature is excluded since
# they do not hold enough rows to estimate a rate from.

print(c['zip_code'].value_counts().describe())

# create features.csv - see decisions.md #4.
# runs every time, so the .csv should not be changed by hand, since every run discards it.

verdicts = {
    'desc':        ('excluded', 'has 39 populated rows, excluded since the column is empty'),
    'emp_title':   ('excluded', 'has 86,091 levels, excluded by Test 2'),
    'zip_code':    ('excluded', 'at least 1/4 of the levels hold 66 or fewer loans, which is unmodellable, excluded by Test 2'),
    'loan_status': ('target',   'the charge-off the model predicts'),
    'int_rate':    ('set1',     'assigned based on grade/sub_grade, included only in Set 1 by decisions.md #5'),
    'grade':       ('set1',     'derived from the LC model, included only in Set 1 by decisions.md #5'),
    'sub_grade':   ('set1',     'derived from the LC model, included only in Set 1 by decisions.md #5'),
    'installment': ('set1',     'can derive int_rate in combination with loan_amnt, considering every loan is 36 months, included only in Set 1 by Test 3'),
}

f = pd.DataFrame({'column': c.columns})
f['feature_set'] = f['column'].map(lambda x: verdicts.get(x, ('both', ''))[0])
f['reason'] = f['column'].map(lambda x: verdicts.get(x, ('both', ''))[1])
f.to_csv('docs/features.csv', index=False)
