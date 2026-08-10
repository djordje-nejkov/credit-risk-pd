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

*TODO*

## 3. Target definition

*TODO*

## 4. What "done" means

*TODO*

## 5. Features

*TODO*

## 6. Results

*TODO*

