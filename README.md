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

The cohort is restricted to 36-month loans issued in Q1–Q4 of 2015, which leaves 283,173 loans. The 2016 and 2012–2014 vintages were considered and excluded: the 2016 loans have no terminal outcome, and 2012–2014 are a materially different population. See #2 in decisions.md.

Every loan in the file was approved and funded, so the models estimate default risk conditional on acceptance. The population is itself the output of Lending Club's own filter, which means the models say nothing about anyone Lending Club turned away.


## 3. Target definition

*TODO*

## 4. What "done" means

*TODO*

## 5. Features

*TODO*

## 6. Results

*TODO*

