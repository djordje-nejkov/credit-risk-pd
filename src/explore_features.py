import pandas as pd

# run exact-duplicate screening across the 91 columns in the parquet
# only does screening, what actually would get dropped requires a manual check
# says nothing about collinearity, decided at encoding
# returns Series([], dtype: uint64), found no duplicates

c = pd.read_parquet('data/cohort.parquet')
h = c.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
print(h[h.duplicated(keep=False)])

print(c['earliest_cr_line'].dtype)

# the check for Test 1 for all columns, checking number of distinct values and number of empty values per column.
# many candidates excluded - see 'verdicts' dictionary.

print(c.isna().sum().sort_values(ascending=False).head(30))
print(c.nunique().sort_values().to_string())

# check for why such columns have no content.
# bimodal shows either a row has these columns empty or fully filled.

block = ['open_acc_6m','open_act_il','open_il_12m','open_il_24m','total_bal_il','open_rv_12m',
         'open_rv_24m','max_bal_bc','all_util','inq_fi','total_cu_tl','inq_last_12m']
print(c[block].isna().sum(axis=1).value_counts())

# check for what is the cause of the finding above, found that coverage for block is confined to December 2015.

# il_util and mths_since_rcnt_il are excluded from block since some borrowers might not have installment accounts, so the 0 or 14 claim if
# they were included in 'block' wouldn't hold, unlike the 0 or 12 one.

print(c.groupby(c['issue_d'].dt.month)[['open_acc_6m','il_util','mths_since_rcnt_il']].apply(lambda d: d.notna().mean()))

# check for redundancy, every meaningful value for title exists in purpose, and purpose is more machine readable, title dropped.

print(pd.crosstab(c['title'], c['purpose']))

# check for consistent number of inclusion of all zipcodes.
# at least one value is found in a single row, 25% have 66 or fewer and 50% have 153 or fewer.
# excluded since at least 1/4 of the levels hold 66 or fewer loans, roughly 10 bad events, too few to estimate a rate from.

print(c['zip_code'].value_counts().describe())

# create features.csv - see decisions.md #4.
# runs every time, so the .csv should not be changed by hand, since every run discards it.

verdicts = {
    'desc':        ('excluded', 'has 39 populated rows, excluded by Test 1'),
    'emp_title':   ('excluded', 'has 86,091 levels, excluded by Test 3'),
    'zip_code':    ('excluded', 'at least 1/4 of the levels hold 66 or fewer loans, which is unmodellable, excluded by Test 3'),
    'policy_code': ('excluded', 'has 1 value across all rows, excluded by Test 1'),
    'disbursement_method': ('excluded', 'has 1 value across all rows, excluded by Test 1'),
    'term': ('excluded', 'has 1 value across all rows because it was restricted by the cohort, excluded by Test 1'),
    'issue_d': ('excluded', 'not a constant, but a model trained on 2015 loans scored on a new applicant has no use for this field. it is the funding month, which is after approval, excluded by Test 2'),
    'title': ('excluded', 'title and purpose are one variable in two encodings, purpose is the representation kept'),

    'open_acc_6m':        ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'open_act_il':        ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'open_il_12m':        ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'open_il_24m':        ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'mths_since_rcnt_il': ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'total_bal_il':       ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'il_util':            ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'open_rv_12m':        ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'open_rv_24m':        ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'max_bal_bc':         ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'all_util':           ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'inq_fi':             ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'total_cu_tl':        ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),
    'inq_last_12m':       ('excluded', 'coverage is confined to December, so the populated rows carry one month of the vintage rather than a sample of it, excluded by Test 1'),

    'loan_status': ('target',   'the charge-off the model predicts'),
    'int_rate':    ('set1',     'assigned based on grade/sub_grade, included only in Set 1 by decisions.md #5'),
    'grade':       ('set1',     'derived from the LC model, included only in Set 1 by decisions.md #5'),        
    'sub_grade':   ('set1',     'derived from the LC model, included only in Set 1 by decisions.md #5'),
    'installment': ('set1',     'can derive int_rate in combination with loan_amnt, considering every loan is 36 months, included only in Set 1 by Test 4'),
}

f = pd.DataFrame({'column': c.columns})
f['feature_set'] = f['column'].map(lambda x: verdicts.get(x, ('both', ''))[0])
f['reason'] = f['column'].map(lambda x: verdicts.get(x, ('both', ''))[1])
f.to_csv('docs/features.csv', index=False)
print(f['feature_set'].value_counts())


