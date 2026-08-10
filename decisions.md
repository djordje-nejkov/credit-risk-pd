# Decisions

# 1

Decision: Lending Club dataset over German Credit dataset.

Why: 

1. Lending club has a larger dataset, therefore a larger number of defaulted loans, this contributes to:

    1.a The model having a better chance to produce similar results during testing. This is because PD-determining factors would be more stable, since the estimates would be noisier for a dataset that has less defaults.
    
    1.b The thesis of LR vs GBM could be more confidently observed, since german credit findings could more plausibly be attriuted to noise.

2. Since german credit has ~20 precurated features which are all application-time, and lending club has ~150 and a good amount are post-origination, choosing my own feature list from lending club's list allows leakage-hunting, therefore demonstrating leakage-hunting skills.

Cost: Since lending club is a peer to peer marketplace, the general applicant would be of a different nature: it's unsecured, and more often debt-consolidation heavy, therefore the findings could not really transfer for a bank applicant, though the methodology of building a model stands.

# 2

Decision: The cohort is 36-month loans issued in Q1–Q4 of 2015.

Why: Originally, loans issued in both 2015 and 2016 were considered. There are many 2016 loans with the Current status, while almost none in 2015. This shows that many of the 2016 loans did not have the full 36 months to mature and therefore have no terminal outcome. They were excluded because they cannot reliably be placed in either the bad or the good group. The matured 2016 loans are also excluded: whether a loan has resolved by the time data collection ends depends on its outcome, since charged-off loans resolve early while on-schedule borrowers do not resolve until month 36. The resolved subset is therefore not a random sample of the vintage, so keeping it would compare a filtered 2016 against a complete 2015.

The 2012–2014 vintages were also considered but left out. 2015 is the most recent fully-matured vintage, so its underwriting regime is closest to the one a deployed model would score. Lending Club's 36-month volume also grew from ~43k in 2012 to ~283k in 2015, so pooling would combine materially different applicant populations. Bad rates across 2012–2015 run 12.3–14.9%, but a stable marginal bad rate does not imply a stable relationship between features and default, so it is not the evidence for this decision.

Cost: ~270,000 loans from the 2016 vintage, and ~306,000 loans (~40,000 bad events) from 2012–2014.

# 3

Decision: Charged Off = bad, Fully Paid = good, the 147 indeterminate 2015 loans excluded.

Why: Only those two statuses are settled outcomes. Forcing a label on 147 rows out of 283,173 to avoid saying "excluded" is worse than excluding them. Charge-off marks the lender's write-off. There were 200 cases of borrowers returning nothing, and a quarter of Charged Off borrowers returned at least 59% of the principal, meaning that the label spans borrower outcomes from nothing repaid to most repaid, so the label marks the write-off, not a borrower condition.

Cost: Since Lending Club gives terminal status only, no payment history, a borrower who never missed a payment doesn't get distinguished from one who hit the conventional 60 or 90 DPD, which by global bank terms is considered bad, and then recovered. That contamination sits in a class of 238,894, where it is hardest to detect.

# 5

Decision: Restrict int_rate, grade and sub_grade to Set 1 of the 2x2; Set 2 excludes all three.

Why: Firstly, these three parameters run hand in hand: Lending Club assigns the interest rate (int_rate) based on how its own model grades applicants. Secondly, even though int_rate is obviously a very useful parameter for measuring the chance of a loan ending up charged off, it cannot be used in both Set 1 and Set 2. Set 2 is defined as being without Lending Club's model output, and int_rate is that output in another unit. Thirdly, keeping only int_rate leaves the logistic regression worse served: without sub_grade it cannot build the step function a GBM builds from int_rate alone. Grade is subsumed by sub_grade and adds nothing beyond it. This matters for reading the result - if the two models converge in Set 1, that is partly the feature set serving the logistic regression, not only the algorithms performing alike.

Cost: Set 2 cannot include int_rate at all. The rate sets the monthly payment, so it is a causal driver of default, not only a proxy for Lending Club's opinion. Set 2 loses a real variable, and the column cannot be split into its two roles.

# 9

Decision: A finding is defined as either GBM's and Logistic Regression's difference being inside noise, or outside it. V1 is done when all six pieces exist and the budget of 20 configurations is spent, whatever the comparison shows. Scores are reported on a slice used neither for training nor for choosing the winning configuration.

Why: Since both GBM and Logistic Regression get the same treatment regarding training and scoring, either finding is considered sufficient since we have no grounds in advance to reason one being better than the other. Moreover, trying to get one model to be better than the other has no real stopping point, since these kinds of models can always be improved by providing more thorough training or using more configurations.

Cost: The 20 configurations being a relatively low number poses risk that the findings may not represent the true potential of the GBM model, since there is a fair chance the model gets a worse best-case result than a model with 60 configurations would.








