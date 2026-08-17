import pandas as pd

# run exact-duplicate screening across the 91 columns in the parquet
# only does screening, what actually would get dropped requires a manual check
# says nothing about collinearity, decided at encoding
# returns Series([], dtype: uint64), found no duplicates

c = pd.read_parquet('data/cohort.parquet')
h = c.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
print(h[h.duplicated(keep=False)])

bad = c['loan_status'] == 'Charged Off'

# check for exclusions under decisions.md #4, the verdicts stated in the 'verdicts' dictionary with the exclusion reason.

def check_exclusions():

    # the check for Test 1 for all columns, checking number of distinct values and number of empty values per column.
    # many candidates excluded - see 'verdicts' dictionary.

    print(c.isna().sum().sort_values(ascending=False).head(30))
    print(c.nunique().sort_values().to_string())

    # check for whether length and title of employment are application-time metrics.
    # 99% overlap between the two, meaning LC did not need to independently verify both post-application,
    # which would be indicated by a larger percentage of only one of the two missing.

    print(pd.crosstab(c['emp_length'].notna(), c['emp_title'].notna()))

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
    # decisions.md Test 3 - check for categorical columns.

    print(c['zip_code'].value_counts().describe())

    # confirmation for date claim.

    print(c['earliest_cr_line'].dtype)

    # check for any features that may have a single overrepresented value.
    # found that num_tl_120dpd_2m, with 5 unique values seen from a check above, has 189 rows with values that are not the mode value (the most common one).
    # out of the 189, 33 went bad, therefore 4/5 values have 189 cases between them. dropped.
    # decisions.md Test 3 - check for numerical columns.

    rows = []
    for col in c.select_dtypes('number').columns:
        s = c[col]
        m = s.mode(dropna=True)
        if m.empty:
            continue
        off = s.notna() & (s != m.iloc[0])
        rows.append({
            'column': col,
            'mode': m.iloc[0],
            'off_mode': int(off.sum()),
            'off_mode_bad': int((off & bad).sum()),
            'missing': int(s.isna().sum()),
        })
    print(pd.DataFrame(rows).sort_values('off_mode').to_string(index=False))


check_exclusions()

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
    'num_tl_120dpd_2m': ('excluded', 'non-modal values are thinly represented, 189 rows across 4 distinct non-modal values with 33 bad events, excluded by Test 3'),

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

    # notes on admitted columns, not exclusions.
    
    'earliest_cr_line': ('both', 'a date, not a duration. converted to months of credit history at modelling, see the encoding entry'),
    'emp_length':       ('both', 'an ordinal held as text with a capped top level. mapped to numbers at modelling, see the encoding entry'),
    'mo_sin_old_il_acct':       ('both', 'the column has no published description in either dictionary sheet, and its Test 2 timing is inferred from mo_sin_old_rev_tl_op'),
    'mths_since_recent_bc_dlq': ('both', 'the column has no published description in either dictionary sheet, and its Test 2 timing is inferred from mths_since_recent_bc'),
    'mths_since_recent_inq':    ('both', 'the column has no published description in either dictionary sheet, and its Test 2 timing is inferred from inq_last_6mths'),
}

f = pd.DataFrame({'column': c.columns})
f['feature_set'] = f['column'].map(lambda x: verdicts.get(x, ('both', ''))[0])
f['reason'] = f['column'].map(lambda x: verdicts.get(x, ('both', ''))[1])
f.to_csv('docs/features.csv', index=False)
print(f['feature_set'].value_counts())

# check for columns excluded from standardization under decisions.md #7.

def check_missingness():

    # asks whether when limit = 0, the bc_util (bankcard utilization) is left empty or = 0.
    # 2831 cases of the prior were found, while there were 253 cases where bc_util has a value (0).
    # we cannot determine from the data why this is, and why either are all not = 0 or left empty.

    miss = c['bc_util'].isna()
    print(pd.crosstab(miss, c['total_bc_limit'] == 0))

    # check whether missing indicators mean the borrower does not have a bankcard.
    # all three crosstabs agree that 631 borrowers do not have a bankcard, but around 80% of each column's missing
    # rows belong to borrowers who do.
    # from this we can see that a value missing doesn't mean no bankcard, but rather the value not being reported.

    for col in ['bc_open_to_buy', 'percent_bc_gt_75', 'mths_since_recent_bc']:
        print(col)
        print(pd.crosstab(c[col].isna(), c['num_bc_tl'] == 0), '\n')

    # characterise the odd 253 rows. found that num_bc_tl mean is 3.5, meaning there are borrowers who have bankcards but still have bc_util = 0.
    # this means that these 253 borrowers cannot be separated from the population that has a limit but util = 0, meaning
    # genuine 0% utilization of the limit.
    # the 2831 missing rows will get filled with 0, as well as be assigned a missing indicator.

    odd = (c['bc_util'].notna()) & (c['total_bc_limit'] == 0)
    print(c.loc[odd, ['bc_util', 'num_bc_tl', 'bc_open_to_buy']].describe())
    
    # check for whether the same assignment holds for bc_open_to_buy.
    # every missing row has limit = 0, so the same assignment of 0 applies to bc_open_to_buy's empty rows as well.

    print(pd.crosstab(c['bc_open_to_buy'].isna(), c['total_bc_limit'] == 0))

    # check how many of the bankcard columns are missing per row. out of the 4, all of them are missing in
    # 2,652/3,279 rows, while the remaining 627 are missing one, two or three.

    bc = ['bc_util','bc_open_to_buy','percent_bc_gt_75','mths_since_recent_bc']
    print(c[bc].isna().sum(axis=1).value_counts().sort_index())

    # check if borrowers missing one of the remaining bankcard parameters have a different bad rate than the borrowers that have the parameter.
    # missing bad rate was 0.17, 0.169 and 0.171 for the missing groups respectively, compared to ~0.1485 for the present group.
    # median will be used to fill percent_bc_gt_75 and mths_since_recent_bc, since a deviation of 2 points from the average default rate is acceptable,
    # and bc_util and bc_open_to_buy are already decided to be filled with 0.

    for col in ['percent_bc_gt_75', 'bc_open_to_buy', 'mths_since_recent_bc']:
        m = c[col].isna()
        print(col)
        print('  missing:', m.sum(), 'bad rate:', round(bad[m].mean(), 4))
        print('  present bad rate:', round(bad[~m].mean(), 4))
        print('  present median:', c.loc[~m, col].median())

    # check for whether the no bankcard borrowers have a higher default rate than the ones who have the value missing.
    # 0.1965 bad rate for the no bankcard group, 0.1639 for the unreported group.
    # a shared indicator will be used, collapsing the rates into 0.17, still higher than 0.1485 present rate.

    m = c['percent_bc_gt_75'].isna()
    nocard = c['num_bc_tl'] == 0
    print('no bankcard: ', (m & nocard).sum(), round(bad[m & nocard].mean(), 4))
    print('unreported:  ', (m & ~nocard).sum(), round(bad[m & ~nocard].mean(), 4))

    # check for whether the missing rows have a different default rate than the filled rows.
    # the missing rows have a default rate of 0.2157, while the filled ones range from
    # 0.1333 - 0.1574.
    # since the blank carries more discrimination, it will be flagged with a missingness indicator.

    lvl = c['emp_length'].fillna('MISSING')
    t = lvl.value_counts().rename('n').to_frame()
    t['bad'] = bad.groupby(lvl).sum()
    t['bad_rate'] = (t['bad'] / t['n']).round(4)
    print(t.to_string())


check_missingness()


