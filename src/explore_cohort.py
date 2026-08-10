# setup - import library and explore the dataset fully first.

import pandas as pd
path = "data/accepted_2007_to_2018Q4.csv.gz"

# checks the number of columns = 151, nrows = 5 since we do not need all ~2.2 mil rows loaded at this time.
# done here since df is restricted to 8 columns, schema has no restriction.

schema = pd.read_csv(path, nrows=5)
print(schema.shape[1])

# set columns to be used to determine cohort.

df = pd.read_csv(path, usecols=['loan_status', 
                                'issue_d', 
                                'term', 
                                'total_rec_prncp', 
                                'funded_amnt', 
                                'last_pymnt_d', 
                                'sub_grade', 
                                'int_rate',])

# check the number of rows, done here since df already loads 8 columns instead of 151, the count is free here since df is 
# already loaded considering we get 2,260,701 rows.

print(df.shape[0])

# convert the format such that the format for issue periods is correct

df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y')

# check the timeframe that the dataset covers: from June 2007, last loan issued December 2018.

print(df['issue_d'].min())
print(df['issue_d'].max())

# using only 36month terms.

df['term'] = df['term'].str.strip()
df36 = df[df['term'] == '36 months'].copy()

# the 2015 vintage: most recent fully-matured 36-month loans - see decisions.md #2.

cohort = df36[(df36['issue_d'].dt.year == 2015)]
print(cohort.shape)
print(pd.crosstab(cohort['loan_status'], cohort['issue_d'].dt.to_period('Q')))

# charged-off loans track the on-schedule amortisation curve at p25/p50/p75, so the label
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

# int_rate is pinned to sub_grade (std 0.1-0.4 across rates of 5-29%), so the three grade
# columns can't be split across feature sets - see decisions.md #5.

print(cohort.groupby('sub_grade')['int_rate'].describe())

# bad rate by vintage. 2012-2014 are fully matured too, so maturity is not why they are
# excluded; 2015 is chosen on recency and population stability - see decisions.md #2.

ct = pd.crosstab(df36['issue_d'].dt.year, df36['loan_status'])
print(ct['Charged Off'] / (ct['Charged Off'] + ct['Fully Paid']))

# cost check for decisions.md #2: 306,462 loans in 2012-2014.

print(df36[df36['issue_d'].dt.year.isin([2012, 2013, 2014])].shape[0])