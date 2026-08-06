
# setup - import library and set columns to be used to determine cohort.

import pandas as pd
path = "data/accepted_2007_to_2018Q4.csv.gz"
df = pd.read_csv(path, usecols=['loan_status', 'issue_d', 'term'])

# using only 36month terms.

df['term'] = df['term'].str.strip()
df36 = df[df['term'] == '36 months'].copy()

# convert the format such that the the format for issue periods is correct

df36['issue_d'] = pd.to_datetime(df36['issue_d'], format='%b-%Y')

# establish the cohort: 36month loans issued in 2015, since all of them are finished by the time the data is finished being reported, list by quarter
# was also 2016, but excluded: 36-month loans issued in 2016 mature in 2019, past the snapshot. the crosstab showed 5.5k–32k Current per quarter

cohort = df36[df36['issue_d'].dt.year.isin([2015])]
print(cohort.shape)
print(pd.crosstab(cohort['loan_status'], cohort['issue_d'].dt.to_period('Q')))