# Decisions

#1

Decision: Lending Club dataset over German Credit dataset.

Why: 

1. Lending club has a larger dataset, therefore a larger number of defaulted loans, this contributes to:

    1.a The model having a better chance to produce similar results during testing. This is because PD-determining factors would be more stable, since the estimates would be noisier for a dataset that has less defaults.
    
    1.b The thesis of LR vs GBM could be more confidently observed, since german credit findings could more plausibly be attriuted to noise.

2. Since german credit has ~20 precurated features which are all application-time, and lending club has ~150 and a good amount are post-origination, choosing my own feature list from lending club's list allows leakage-hunting, therefore demonstrating leakage-hunting skills.

Cost: Since lending club is a peer to peer marketplace, the general applicant would be of a different nature: it's unsecured, and more often debt-consolidation heavy, therefore the findings could not really transfer for a bank applicant, though the methodology of building a model stands.

#2

Decision: The cohort are 36-month loans issued in Q1-Q4 of 2015.

Why: Originally, both loans issued in 2015 and 2016 were considered. Since the data ends at ~Q1 2019, many of the 2016 loans did not have the full 36 months to mature, therefore do not have a terminal outcome. A decision was made to exclude them since we cannot reliably put them in either bad/good group. The matured 2016 loans are also excluded: whether a loan has resolved by the snapshot depends on its outcome, since charged-off loans resolve early while on-schedule borrowers do not resolve until month 36. The resolved subset is therefore not a random sample of the vintage, so keeping it would compare a filtered 2016 against a complete 2015.

Cost: The whole 2016 vintage, which is around 270,000 loans.

#9

Decision: A finding is defined as either GBM's and Logistic Regression's difference being inside noise, or outside it. V1 is done when all six pieces exist and the budget of 20 configurations is spent, whatever the comparison shows. Scores are reported on a slice used neither for training nor for choosing the winning configuration.

Why: Since both GBM and Logistic Regression get the same treatment regarding training and scoring, either finding is considered sufficient since we have no grounds in advance to reason one being better than the other. Moreover, trying to get one model to be better than the other has no real stopping point, since these kinds of models can always be improved by providing more thorough training or using more configurations.

Cost: The 20 configurations being a relatively low number poses risk that the findings may not represent the true potential of the GBM model, since there is a fair chance the model gets a worse best-case result than a model with 60 configurations would.








