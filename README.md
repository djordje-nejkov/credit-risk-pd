# Credit Default Risk Model

Predicting the probability that an approved Lending Club loan ends in charge-off,
using only information available at the time of application.

## 1. The question

The model predicts the probability of an approved loan ending up charged off rather
than fully paid, using information available at the time of application. Based on that
probability, applicants are accepted or declined.

Since the decision is a cut-off, what matters is the ordering: the model needs to put
riskier applicants above safer ones. The two errors it can make are labelling a good
applicant bad and labelling a bad applicant good. Declining a good applicant forfeits
the interest the lender would have earned; approving a bad one loses principal that
does not come back.

The project tests whether a gradient boosted model ranks better than logistic
regression, trained both with and without Lending Club's own risk grades.

## 2. The data

The source is Lending Club's data on accepted loans issued between 2007 and 2018. The file contains 151 columns and 2,260,701 rows, covering application-time variables, which the models use, as well as borrower behaviour and post-termination fields, which are excluded because they encode the outcome the model is trying to predict.

The cohort is restricted to individual 36-month loans issued in Q1–Q4 of 2015, which leaves 282,934 loans. The 2016 and 2012–2014 vintages were considered and excluded: the 2016 loans have no terminal outcome, and 2012–2014 are a materially different population. See #2 in decisions.md.

Every loan in the file was approved and funded, so the models estimate default risk conditional on acceptance. The population is itself the output of Lending Club's own filter, which means the models say nothing about anyone Lending Club turned away. The declined population is available and deliberately not used, because it has no outcome to model.

## 3. Target definition

Since the model predicts default, a bad/Charged Off loan is labelled 1 and a good/Fully Paid loan 0. These are the target's two possible states; the model outputs a probability of default between 0 and 1. Of the resolved loans, 14.9% defaulted - 240,698 good against 42,089 bad. A further 147 loans from 2015 are dropped because their outcome is indeterminate, so not yet concluded.

These two loan states are used because they are the only ones that record settled outcomes.

The definition is limited in that a loan reaching 60 or 90 days past due (DPD), the industry standard for default, is still counted as Fully Paid if the borrower recovers. Lending Club records only terminal status, so Charged Off is the closest available proxy. See #3 in decisions.md.

## 4. What "done" means

The difference between LR and GBM either sits inside noise or outside it. Both findings are results.

Since both arms get identical rows, identical slices and a stated configuration budget, there is no prior reason to expect one to win. Furthermore, chasing a win for one of the models has no stopping point, as any model improves with more tuning.

GBM gets 20 configurations, LR gets 7. This might seem like an asymmetry, but keep in mind that GBM has several knobs to adjust. LR has one, the penalty coefficient, with its configuration span running from 0.001 to 100, as well as inf. Equating the counts would be arbitrary.

The score comes from Q4 of the cohort, since it is a slice used for neither fitting nor choosing the winning configuration. This avoids reporting a score from the slice that chose the configuration, which would credit a number for being high on that slice rather than the configuration for being best.

## 5. Features

As the model predicts whether an applicant will default, the only features it can use in the predictions are the ones available at the time of application, therefore everything that can be inferred to be post-application is automatically excluded. The easily observable exclusions can be seen in explore_cohort.py, whereas the ones that would require further examining to confirm whether they are pre-application are examined in explore_features.py.

There are 4 tests that determine whether the feature is kept or excluded from modelling, they are: 1. whether the feature has enough content to model, 2. whether it's knowable at application, 3. whether its levels have enough rows to estimate a rate from, 4. whether it reconstructs an excluded feature. These tests run in order, and a column's stated reason for exclusion is the one it failed first. Test 2's timing for mo_sin_old_il_acct, mths_since_recent_bc_dlq and mths_since_recent_inq is inferred from related columns rather than from the data dictionary, which has no published description for them. Further on this in decisions.md #4.

The parquet created in explore_cohort.py lists 91/151 features that survived the initial check. Of those 91, in the end: 63 features are included in both sets, with 23 excluded from both, 4 only being in set 1 and 1 (loan_status) being the target. The exclusion reasons for the excluded features can be found in features.csv.

Regarding the 4 features only being in set 1, they are features derived from Lending Club's own model output, and set 2 is defined as being without these features. Further on this in decisions.md #5.

## 6. Results

*TODO*

