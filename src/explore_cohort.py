# setup - import library and explore the dataset fully first.

import pandas as pd

# check the number of columns and rows in the dataset. columns = 151, rows = ~2.2 mil.

df = pd.read_csv("data/accepted_2007_to_2018Q4.csv.gz")

print(df.shape[0], df.shape[1])

# convert the format such that the format for issue periods is correct

df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y')
df['term'] = df['term'].str.strip()

# check for whether running through the feature dictionary (found in /docs) is enough to decide which features to keep for modelling.
# features were found that were not documented at all, so the actual drop list needed to be inferred from the csv column names.
# the reverse direction was also run, finding kept features do not have the same names in the excel file and the .csv

loanstats = pd.read_excel('docs/LCDataDictionary.xlsx', sheet_name='LoanStats')
browsenotes = pd.read_excel('docs/LCDataDictionary.xlsx', sheet_name='browseNotes')

dictnames = set(loanstats.iloc[:, 0].dropna().str.strip()) | set(browsenotes.iloc[:, 0].dropna().str.strip())
columns = set(df.columns)

print("in CSV, not in dictionary:", sorted(columns - dictnames))
print("in dictionary, not in CSV:", sorted(dictnames - columns))

# check the timeframe that the dataset covers: from June 2007, last loan issued December 2018.

print(df['issue_d'].min())
print(df['issue_d'].max())

# using only 36month terms.

df36 = df[df['term'] == '36 months'].copy()

# cost check for the joint exclusion, 239 loans.

print(pd.crosstab(df36['application_type'], df36['issue_d'].dt.year))

# decided on only individual loans, redefine.

df36 = df36[df36['application_type'] == 'Individual'].copy()

# the 2015 vintage: most recent fully-matured 36-month loans - see decisions.md #2.

cohort = df36[df36['issue_d'].dt.year == 2015].copy()

# check for funded_amnt, excluded for 'The total amount committed to that loan at that point in time' meaning post-application 
# redundant as well, but needs to be application-time first, then it is checked for redundancy, since the tests are ordered.

print(loanstats[loanstats.iloc[:, 0].str.strip() == 'funded_amnt'].iloc[:, 1].values)

# check for exclusion of funded_amnt_inv for redundancy. prints 0, since there are no overfunded loans, which means the actual exclusion reason is it not being application-time,
# but rather a marketplace indicator.

print((cohort['loan_amnt'] < cohort['funded_amnt_inv']).sum())

# check the number of good/bad loans, Charged Off = 42,089, Fully Paid = 240,698, Indeterminate = 147

print(pd.crosstab(cohort['loan_status'], cohort['issue_d'].dt.to_period('Q')))

# individual charged-off loans track the on-schedule amortisation curve at p25/p50/p75, so the label
# marks the lender's write-off rather than a defect visible at origination - see decisions.md #3.

chargedoff = cohort[cohort['loan_status'] == 'Charged Off']

recovered = chargedoff['total_rec_prncp'] / chargedoff['funded_amnt']
days_to_last_payment = chargedoff['last_pymnt_d'] - chargedoff['issue_d']

print(recovered.describe())
print(days_to_last_payment.describe())

# 200 charged-off loans have no last payment date. all have total_rec_prncp of exactly 0.0:
# they never made a payment.

nopayment = chargedoff['last_pymnt_d'].isna()
print(chargedoff[nopayment]['total_rec_prncp'].describe())

# int_rate is pinned to sub_grade (generally around std 0.1-0.4 across rates of 5-29%, holds for the populated grades, the tail is thin), so the three grade
# columns can't be split across feature sets - see decisions.md #5.

print(cohort.groupby('sub_grade')['int_rate'].describe())

# bad rate by vintage. 2012-2014 are fully matured too, so maturity is not why they are
# excluded; 2015 is chosen on recency and population stability - see decisions.md #2.

ct = pd.crosstab(df36['issue_d'].dt.year, df36['loan_status'])
print(ct['Charged Off'] / (ct['Charged Off'] + ct['Fully Paid']))

# cost check for decisions.md #2: 306,462 loans in 2012-2014.

print(df36[df36['issue_d'].dt.year.isin([2012, 2013, 2014])].shape[0])

# write the parquet of the cohort, explore_features.py is dependent on this.

identifiers = ['id', 'member_id', 'url']

joint = [c for c in cohort.columns if c.startswith('sec_app_')] + [
    'annual_inc_joint', 'dti_joint', 'verification_status_joint',
    'revol_bal_joint', 'application_type']

post_application = [c for c in cohort.columns
                    if c.startswith(('hardship_', 'settlement_', 'debt_settlement'))] + [
    'collection_recovery_fee', 'deferral_term', 'last_credit_pull_d',
    'last_fico_range_high', 'last_fico_range_low', 'last_pymnt_amnt',
    'last_pymnt_d', 'next_pymnt_d', 'orig_projected_additional_accrued_interest',
    'out_prncp', 'out_prncp_inv', 'payment_plan_start_date', 'pymnt_plan',
    'recoveries', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int',
    'total_rec_late_fee', 'total_rec_prncp', 'funded_amnt_inv', 'funded_amnt']

drop = identifiers + joint + post_application
assert len(drop) == len(set(drop))          
cohort = cohort.drop(columns=drop)          
print(cohort.shape)                         
cohort.to_parquet('data/cohort.parquet')