import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import CategoricalDtype

# load the files

rpar = pd.read_parquet('data/cohort.parquet')
rcsv = pd.read_csv('docs/features.csv')

# set eligible rows and set target as 1 and 0, 1 is Charged Off

rpar = rpar[rpar['loan_status'].isin(['Charged Off', 'Fully Paid'])].copy()
trgt = (rpar['loan_status'] == 'Charged Off').astype(int)

# convert earliest_cr_line

rpar['credit_history_months'] = ((rpar['issue_d'].dt.year - rpar['earliest_cr_line'].dt.year) * 12 + rpar['issue_d'].dt.month - rpar['earliest_cr_line'].dt.month)
rpar = rpar.drop(columns=['earliest_cr_line'])

# sanity check, 317 rows exceed 50 years of credit history

print(rpar['credit_history_months'].describe())
print((rpar['credit_history_months'] > 600).sum())

# emp_length is an ordinal held as text with a capped top level - see decisions.md #7.
# '< 1 year' maps to 0 alongside the LR arm's MISSING fill; the indicator separates them.

emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
           '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
           '10+ years': 10}

rpar['emp_length'] = rpar['emp_length'].map(emp_map)

# sanity check

print(rpar['emp_length'].value_counts(dropna=False).sort_index())

# cohort split into train/validate/test

quarter = rpar['issue_d'].dt.quarter

train = rpar[quarter <= 2]
val = rpar[quarter == 3]
test = rpar[quarter == 4]

y_train = trgt[quarter <= 2]
y_val = trgt[quarter == 3]
y_test = trgt[quarter == 4]

print(train.shape, val.shape, test.shape)
print(round(y_train.mean(), 4), round(y_val.mean(), 4), round(y_test.mean(), 4))

# column selection for both sets

set1 = rcsv.loc[rcsv['feature_set'].isin(['both', 'set1']), 'column'].tolist()
set2 = rcsv.loc[rcsv['feature_set'] == 'both', 'column'].tolist()

# sanity check

print(len(set1), len(set2))

# change to credit_history_months in both sets

set1 = ['credit_history_months' if x == 'earliest_cr_line' else x for x in set1]
set2 = ['credit_history_months' if x == 'earliest_cr_line' else x for x in set2]

# check whether there are any other columns worth examining for fill rules.

named = ['bc_util', 'bc_open_to_buy', 'percent_bc_gt_75', 'mths_since_recent_bc',
         'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog',
         'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq', 'mths_since_recent_inq',
         'emp_length']

n = train[set2].isna().sum()
print(n[(n > 0) & (~n.index.isin(named))])

# build the indicators

n = rpar[set1].isna().sum()

fillcols = n[n > 0].index.tolist()
fillcols = [x for x in fillcols if x not in ['bc_util', 'bc_open_to_buy', 'emp_length']]

medians = {x: train[x].median() for x in fillcols}

print(len(fillcols), medians)

# build the indicators

def encode_lr(df, medians):
    df = df.copy()

    bc = ['bc_util', 'bc_open_to_buy', 'percent_bc_gt_75', 'mths_since_recent_bc']
    since = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog',
             'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq', 'mths_since_recent_inq']

    df['bc_missing'] = df[bc].isna().any(axis=1).astype(int)

    for col in since + ['mo_sin_old_il_acct', 'emp_length']:
        df[col + '_missing'] = df[col].isna().astype(int)

    for col in ['bc_util', 'bc_open_to_buy', 'emp_length']:
        df[col] = df[col].fillna(0)

    for col, val in medians.items():
        df[col] = df[col].fillna(val)

    return df


# LR arm: fills and indicators per decisions.md #7, medians fitted on the training slice per #6

lr1_train = encode_lr(train[set1], medians)
lr1_val   = encode_lr(val[set1], medians)
lr1_test  = encode_lr(test[set1], medians)

lr2_train = encode_lr(train[set2], medians)
lr2_val   = encode_lr(val[set2], medians)
lr2_test  = encode_lr(test[set2], medians)

# checks: no NaN left, column count, indicator counts against the measured figures

print(lr1_train.isna().sum().sum(), lr1_val.isna().sum().sum(), lr1_test.isna().sum().sum())
print(lr1_train.shape, lr2_train.shape)
print(lr1_train['bc_missing'].sum(), lr1_train['mths_since_last_record_missing'].sum())

# make one-hot for categoricals

cats1 = lr1_train.select_dtypes(include=['object', 'string']).columns.tolist()
cont1 = [c for c in lr1_train.columns if c not in cats1 and not c.endswith('_missing')]

enc1 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(lr1_train[cats1])
scaler1 = StandardScaler().fit(lr1_train[cont1])

cats2 = lr2_train.select_dtypes(include=['object', 'string']).columns.tolist()
cont2 = [c for c in lr2_train.columns if c not in cats2 and not c.endswith('_missing')]

enc2 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(lr2_train[cats2])
scaler2 = StandardScaler().fit(lr2_train[cont2])


def onehot_scale(df, enc, scaler, cats, contcols):
    arr = enc.transform(df[cats])
    enc_df = pd.DataFrame(arr, columns=enc.get_feature_names_out(cats), index=df.index)

    cont = pd.DataFrame(scaler.transform(df[contcols]), columns=contcols, index=df.index)
    ind = df[[c for c in df.columns if c.endswith('_missing')]]

    return pd.concat([cont, ind, enc_df], axis=1)

X1_train = onehot_scale(lr1_train, enc1, scaler1, cats1, cont1)
X1_val   = onehot_scale(lr1_val, enc1, scaler1, cats1, cont1)
X1_test  = onehot_scale(lr1_test, enc1, scaler1, cats1, cont1)

X2_train = onehot_scale(lr2_train, enc2, scaler2, cats2, cont2)
X2_val   = onehot_scale(lr2_val, enc2, scaler2, cats2, cont2)
X2_test  = onehot_scale(lr2_test, enc2, scaler2, cats2, cont2)

print(X1_train.shape, X1_val.shape, X1_test.shape)
print(X2_train.shape, X2_val.shape, X2_test.shape)

# categories fixed on the training slice so all three slices share one encoding, since
# addr_state has 46 levels in Q1-Q2 against 49 in Q3 and Q4. levels absent from training
# are set to NaN, 405 rows in Q3 and 984 in Q4, which HGB routes natively - see decisions.md #7.
# the LR arm handles the same rows differently, with handle_unknown='ignore' giving an
# all-zero row across the state blo

dtypes1 = {col: CategoricalDtype(categories=sorted(train[col].dropna().unique()))
           for col in cats1}
dtypes2 = {col: CategoricalDtype(categories=sorted(train[col].dropna().unique()))
           for col in cats2}

def encode_gbm(df, dtypes):
    df = df.copy()
    for col, dt in dtypes.items():
        df[col] = df[col].where(df[col].isin(dt.categories))
        df[col] = df[col].astype(dt)
    return df

# GBM arm: native categoricals, no fills, no indicators, no scaling - see decisions.md #7

gbm1_train = encode_gbm(train[set1], dtypes1)
gbm1_val   = encode_gbm(val[set1], dtypes1)
gbm1_test  = encode_gbm(test[set1], dtypes1)

gbm2_train = encode_gbm(train[set2], dtypes2)
gbm2_val   = encode_gbm(val[set2], dtypes2)
gbm2_test  = encode_gbm(test[set2], dtypes2)

print(gbm1_train.shape, gbm2_train.shape)
print(gbm1_train.isna().sum().sum(), gbm2_train.isna().sum().sum())

print((train[cats1].isna().sum().sum()), (gbm1_train[cats1].isna().sum().sum()))
print(gbm1_val[cats1].isna().sum().sum(), gbm1_test[cats1].isna().sum().sum())