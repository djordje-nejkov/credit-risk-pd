import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier

from preprocess import (X1_train, X1_val, X1_test, X2_train, X2_val, X2_test,
                        gbm1_train, gbm1_val, gbm1_test, gbm2_train, gbm2_val, gbm2_test,
                        y_train, y_val, y_test)

# GBM configuration budget per decisions.md #9: 20 sampled configurations, seeded for reproducibility.
# max_iter is sampled rather than fixed with early stopping, so Q3 selects it like every other
# parameter and no random split is opened inside the temporal design - see #6.

rng = np.random.default_rng(42)

configs = []
for _ in range(20):
    configs.append({
        'learning_rate': float(10 ** rng.uniform(-2, np.log10(0.3))),
        'max_iter': int(rng.integers(100, 601)),
        'max_leaf_nodes': int(rng.integers(15, 64)),
        'min_samples_leaf': int(rng.integers(20, 201)),
        'l2_regularization': float(rng.uniform(0, 10)),
    })

# LR budget per #9: 7 values of the penalty coefficient, 0.001 to 100 plus no penalty.

Cs = [0.001, 0.01, 0.1, 1, 10, 100, np.inf]


def run_lr(Xtr, ytr, Xva, yva):
    out = []
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=1000)
        m.fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        out.append((C, round(auc, 4)))
        print(C, round(auc, 4))
    return out

# check for the best LR configuration for each set

# print('LR Set 1')
# lr1 = run_lr(X1_train, y_train, X1_val, y_val)

# print('LR Set 2')
# lr2 = run_lr(X2_train, y_train, X2_val, y_val)

def run_gbm(Xtr, ytr, Xva, yva):
    out = []
    for i, cfg in enumerate(configs):
        m = HistGradientBoostingClassifier(**cfg, random_state=42)
        m.fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        out.append((cfg, round(auc, 4)))
        print(i, round(auc, 4), cfg)
    return out

# check for the best GBM configuration for each set

# print('GBM Set 1')
# gbm1 = run_gbm(gbm1_train, y_train, gbm1_val, y_val)

# print('GBM Set 2')
# gbm2 = run_gbm(gbm2_train, y_train, gbm2_val, y_val)

# winners selected on Q3 per #9
best_lr1 = LogisticRegression(C=0.1, max_iter=1000).fit(X1_train, y_train)
best_lr2 = LogisticRegression(C=0.1, max_iter=1000).fit(X2_train, y_train)
best_gbm1 = HistGradientBoostingClassifier(**configs[14], random_state=42).fit(gbm1_train, y_train)
best_gbm2 = HistGradientBoostingClassifier(**configs[8], random_state=42).fit(gbm2_train, y_train)

arms = {
    'LR Set 1':  (best_lr1, X1_val, X1_test),
    'LR Set 2':  (best_lr2, X2_val, X2_test),
    'GBM Set 1': (best_gbm1, gbm1_val, gbm1_test),
    'GBM Set 2': (best_gbm2, gbm2_val, gbm2_test),
}

for name, (m, Xva, Xte) in arms.items():
    sva = m.predict_proba(Xva)[:, 1]
    ste = m.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(y_test, ste)
    gini = 2 * auc - 1

    cut = np.quantile(sva, 0.85)                      # 85th percentile of Q3 per #8
    declined = ste > cut
    tn, fp, fn, tp = confusion_matrix(y_test, declined).ravel()

    print(name, 'AUC', round(auc, 4), 'Gini', round(gini, 4),
          'declined', round(declined.mean(), 4))
    print('   approved good', tn, 'declined good', fp,
          'approved bad', fn, 'declined bad', tp)