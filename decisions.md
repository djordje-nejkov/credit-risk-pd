# Decisions

# 1 Dataset

Decision: Lending Club dataset over German Credit dataset.

Why: 

1. Lending club has a larger dataset, therefore a larger number of defaulted loans, this contributes to:

    1.a The model having a better chance to produce similar results during testing. This is because PD-determining factors would be more stable, since the estimates would be noisier for a dataset that has less defaults.
    
    1.b The thesis of LR vs GBM could be more confidently observed, since german credit findings could more plausibly be attriuted to noise.

2. Since german credit has ~20 precurated features which are all application-time, and lending club has ~150 and a good amount are post-origination, choosing my own feature list from lending club's list allows leakage-hunting, therefore demonstrating leakage-hunting skills.

Cost: Since lending club is a peer to peer marketplace, the general applicant would be of a different nature: it's unsecured, and more often debt-consolidation heavy, therefore the findings could not really transfer for a bank applicant, though the methodology of building a model stands.

# 2 Cohort

Decision: The cohort is individual, 36-month loans issued in Q1–Q4 of 2015.

Why: Originally, loans issued in both 2015 and 2016 were considered. There are many 2016 loans with the Current status, while almost none in 2015. This shows that many of the 2016 loans did not have the full 36 months to mature and therefore have no terminal outcome. They were excluded because they cannot reliably be placed in either the bad or the good group. The matured 2016 loans are also excluded: whether a loan has resolved by the time data collection ends depends on its outcome, since charged-off loans resolve early while on-schedule borrowers do not resolve until month 36. The resolved subset is therefore not a random sample of the vintage, so keeping it would compare a filtered 2016 against a complete 2015. Individual loans are only considered since judging joint loans based on features of individual loans is incoherent, since the features describe only one of the 2 borrowers. Joint features are not used since they are null for every individual loan.

The 2012–2014 vintages were also considered but left out. 2015 is the most recent fully-matured vintage, so its underwriting regime is closest to the one a deployed model would score. Lending Club's 36-month volume also grew from ~43k in 2012 to ~283k in 2015, so pooling would combine materially different applicant populations. Bad rates across 2012–2015 run 12.3–14.9%, but a stable marginal bad rate does not imply a stable relationship between features and default, so it is not the evidence for this decision.

Cost: 318,277 loans from the 2016 vintage, ~306,000 loans (~40,000 bad events) from 2012–2014, and 239 joint loans in 2015.

# 3 Target definition

Decision: Charged Off = bad, Fully Paid = good, the 147 indeterminate 2015 loans excluded.

Why: Only those two statuses are settled outcomes. Forcing a label on 147 rows out of 282,934 to avoid saying "excluded" is worse than excluding them. Charge-off marks the lender's write-off. There were 200 cases of borrowers returning nothing, and a quarter of Charged Off borrowers returned at least 59% of the principal, meaning that the label spans borrower outcomes from nothing repaid to most repaid, so the label marks the write-off, not a borrower condition.

Cost: Since Lending Club gives terminal status only, no payment history, a borrower who never missed a payment doesn't get distinguished from one who hit the conventional 60 or 90 DPD, which by global bank terms is considered bad, and then recovered. That contamination sits in a class of 240,698, where it is hardest to detect.

# 4 Feature eligibility

Decision: Modeling features are judged based on four tests, run sequentially, and admitted only if all four hold: 1. the column has content to model, and that content is not confined to a slice the cohort created, 2. the column is knowable at application, 3. the column's levels hold enough rows to estimate a rate from, and the non-modal values (rows that carry any value other than the most common one) are well represented. This is applied to categorical and numerical columns with different checks (see explore_features.py for the different checks), and 4. the column does not reconstruct a feature deliberately excluded from one of the sets, in combination with others. Test 4 is set relative, so it only applies to Set 2, where features derived from Lending Club's own model are excluded. Installment is a feature that will be included in Set 1, but not in Set 2, since in combination with loan_amnt it can recover int_rate. Further on this in #5. The order decides the stated reason for dropping a column in the event it violates more than one condition. Joint columns are excluded under #2. The column for the relationship between dti and home_ownership will be provided to a separate Logistic Regression run, separately from the 2x2 findings. A .csv file will be produced with every column having a 'set1' or 'both' flag for included features, and an 'excluded' flag marking columns excluded after further evaluation, as well as a reason for exclusion within Set 2. A 'target' flag is also present, used only for loan_status. The exclusion decisions for features decidable from the documentation can be seen in explore_cohort.py, with columns grouped in lists named by exclusion reason, while features requiring measurement on the cohort are excluded in explore_features.py, with the reason recorded in the .csv.

Why: Since the model predicts default, the only features it should use for it to be able to be applied to a new dataset are the ones that would be available at the time of application for the loan. Moreover, columns that have values that are near-unique per row let a GBM split on values that identify individual rows, so the training score rises while nothing transfers to a new applicant. Levels holding too few rows give rates estimated on a handful of events, so the model fits noise rather than a pattern, and the fitted value doesn't hold for the next applicant in that level. This is also true for columns that have their non-modal levels underrepresented, meaning that only the modal level has enough data to estimate a rate from. Furthermore, a feature that, in combination with others, can reconstruct an excluded feature from one of the sets needs to be excluded in order to completely isolate the two cases. Lastly, the inclusion of the relationship column for dti and home_ownership is done since GBM can handle this relationship by itself, where LR cannot, therefore the columns dti x (rent, own, mortgage) will be included in order to help LR separate the dti slope per housing status. The run will happen in Set 2, reported whichever way it goes. 

Regarding Test 1's criterion: it is what determines the missingness, and not necessarily how much there is. For example, mths_since_last_delinq is 48% missing, since missing means the borrower has no delinquency, therefore the feature is included. On the contrary, the 14 features that are excluded under 'coverage is confined to December', which can be seen in explore_features.py have around 95% of their entries missing since the data is confined to December 2015, therefore they are excluded under their content being confined to a slice the cohort created.

Cost: The exclusions can potentially drop columns that carry a signal, but in order to make the model applicable to other datasets, and the actual concept of the 2x2 viable, this is a tradeoff worth having. Furthermore, the columns are carefully excluded by hand, but subjectively justified exclusion remains possible. Test 3's criterion is rows per level, but the threshold is not specified, so where the line falls between zip_code's median of 153 and addr_state's several thousand rests on personal judgement. This extends to the non-modal row count threshold, where num_tl_120dpd_2m was excluded with 189 non-modal rows, while num_tl_30dpd was kept with 1,051. Furthermore, the 14 columns excluded under having their content confined to a slice would presumably be populated in a newer cohort, since December 2015 being populated while the rest of the year is empty leads to an assumption that December 2015 is the period where data for these features started to be collected.

# 5 Feature set split

Decision: Restrict int_rate, grade, sub_grade, and every feature that reconstructs them to Set 1 of the 2x2; Set 2 excludes all of them. Bureau attributes are kept in both sets.

Why: Firstly, these three parameters run hand in hand: Lending Club assigns the interest rate (int_rate) based on how its own model grades applicants. Secondly, even though int_rate is obviously a very useful parameter for measuring the chance of a loan ending up charged off, it cannot be used in both Set 1 and Set 2. Set 2 is defined as being without Lending Club's model output, and int_rate is that output in another unit. Thirdly, keeping only int_rate leaves the logistic regression worse served: without sub_grade it cannot build the step function a GBM builds from int_rate alone. Grade is subsumed by sub_grade and adds nothing beyond it. This matters for reading the result - if the two models converge in Set 1, that is partly the feature set serving the logistic regression, not only the algorithms performing alike.

The 91 columns in the parquet were split into bureau attributes and loan terms. Bureau attributes describe the borrower's credit file, which exists independently of LC pricing anything, so no arithmetic combination of them produces a rate LC set - they can correlate with it, since LC's model reads them, but correlation isn't reconstruction. Only the loan-terms block stands in an arithmetic relation to int_rate, and that block is four columns. Term is constant, installment is Set 1 along with int_rate, so Set 2 holds loan_amnt alone and the equation can't be closed, since it would need installment alongside it. An option to solve this was a correlation screening, but it would flag bureau attributes because they correlate with int_rate by construction, since LC's own grading model uses these metrics to set the rate.

Cost: Set 2 cannot include int_rate at all, nor anything that reconstructs it (installment, etc.). The rate sets the monthly payment, so it is a causal driver of default, not only a proxy for Lending Club's opinion. Set 2 loses a real variable, and the column cannot be split into its two roles. Furthermore, the bureau attributes vs loan terms argument doesn't rest on something reproducible, but rather the hand classification of the columns.

# 6 Data split

Decision: The cohort will be split into three parts: Q1-Q2 will be used for training the models. Q3 will be used for validation, meaning choosing between the 20 model configurations. Q4 will be used for testing, and the reported scores will be measured on this slice of the cohort. The encoding patterns for both models are learned on the training slices and applied to the other two. Moreover, both models will see the identical slices for training, validation and testing. 

Why: The ordering is respected since a deployed model is fitted on the past and used to score the future. Even though the cohort only consists of loans issued in 2015, the stability of it is unverified. A random split would assume exchangeability that there is no evidence for.

Three slices are used rather than two, in order to report the score from a slice used neither for training nor for selecting the winning GBM configuration. Two slices would mean reporting the findings from the same data that picked the configuration, meaning that it would be a score reported by being high on that specific slice, not being the best configuration overall.

Cost: The testing slice is Q4 alone, which carries the possible December changes seen from the 14 excluded features beginning to be kept track of at this time, see #4 - Why. Furthermore, stratification is not possible, so the bad rate for the test slice is whatever the bad rate in Q4 was, though in our example Q4's default rate is actually 14.81%, stable. Lastly, the training split is smaller than a random split would warrant, 120,790 loans in Q1+Q2 to 282,787 in 2015 all together.

# 7 Encoding

Decision: Continuous columns are standardised; dummy columns are left as they are.

The LR model is supplied with additional one-hot columns, filled NaN rows, with each column's rows filled with its median value, and a shared bankcard indicator - see explore_features.py for reasoning about the bankcard indicators.

The GBM model, represented by HistGradientBoostingClassifier, get native categoricals, natively split categorical columns, NaN rows left unfilled, and furthermore no indicator for the NaN columns.

Both models receive identical rows, but the columns are different since the models require different encoding in order to function properly, for example LR needing one-hot columns to differentiate between levels, while GBM does without.

As per 'verdicts' of explore_features.py: 

    1. earliest_cr_line gets converted into months since it was first opened, measured from issue_d, issue date of the loan. This is used instead of the considered option of using an anchor date.

    2. emp_length gets converted into numbers, with the 'MISSING' rows being assigned 0 and having an indicator column alongside it.

Why: Dummy columns are left as they are since standardising the feature stops it from being the actual difference between two levels, which I feel is more important for faithful representation than balancing the dummy's penalties with standardisation. On the contrary, continuous columns are standardised since standardisation corrects for an arbitrary choice of units, and a dummy has no unit to correct.

The two models are supplied differently since LR's requirements are conditions of using it correctly, rather than additions. With GBM, all of these are additions, and the model would function fine without. It would hand it a representation built around LR's constraints.

Regarding LR's filled rows, the rejected alternatives were: 1. to drop the NaN rows, which resulted in 3.07% of the cohort being kept, see explore_features.py and 2. using the median without an indicator. The fill values sit inside the observed range of the columns values. The median fills are fitted on the training slice as per decisions.md #6, and some features like bc_open_to_buy and bc_util are kept constant, rather than fitted, see explore_features.py as for why.

As for the shared bankcard indicators, 2,652/3,279 rows are missing all four indicators, which is 81%. Having four separate indicators would be near-duplicates splitting one signal when we apply L2 penalisation. The rest that have 1-3/4 parameters missing get treated as missing all of them.

For the GBM arm, NaN rows are kept as is since filling would damage the tree when it lands on an existing group at a different risk level, bc_util being the perfect example: filling the rows with 0 like it is done for the LR arm puts 2,831 missing rows at 0.1706 pd inside the genuine zero-utilisation group, and no split can separate them.

Cost: The observable differences in the 2x2 cannot be fully accredited to model performance, but rather the difference in representation of the two models needs to be taken into account. Any gap would have a second possible source, and the design cannot separate representation from the algorithm.

The shared bankcard indicators average the 631 no-bankcard rows at 0.1965 pd and the 2,453 unreported rows at 0.1639 into 0.1706. Two indicators were rejected since the 631 rows carry only 124 bad events.

The 253 degenerate rows stay pooled inside bc_util = 0 in both models, and neither has the ability to separate them from the actual larger pool of borrowers who have bc_util = 0 because they actually used 0% of their bank card limit. LR adds 2,831 rows into that same bucket, but with an indicator, so the bucket has a different composition in the two models.

The fill values sit inside the observed range but are not derived. This was done knowingly, since the model would then include a fitted component inside the preprocessing - its own error and drift, which would be another thing that would need justification. This was deemed not worthy for 3,279/282,787 rows.

# 8 Threshold

Decision:

Why:

Cost:

# 9 Stopping rule

Decision: A finding is defined as either GBM's and Logistic Regression's difference being inside noise, or outside it. V1 is done when all six pieces exist and the budget of LR's 7 and GBM's 20 configurations is spent, whatever the comparison shows. Scores are reported on a slice used neither for training nor for choosing the winning configuration.

Why: Since both GBM and Logistic Regression get the same treatment regarding training and scoring, meaning both get trained, validated and tested on the same slices, either finding is considered sufficient since we have no grounds in advance to reason one being better than the other. Moreover, trying to get one model to be better than the other has no real stopping point, since these kinds of models can always be improved by providing more thorough training or using more configurations. Regarding GBM getting 20 configurations while LR gets 7, GBM adjusts several knobs at once, while LR has one parameter with an already sufficient span, from 0.001 to 100, and also inf. Trying to equate the counts would be arbitrary rather than fair.

Cost: The 20 configurations being a relatively low number poses risk that the findings may not represent the true potential of the GBM model, since there is a fair chance the model gets a worse best-case result than a model with 60 configurations would. LR's 7 configurations is much better coverage for its single changing parameter, though the 0.001 - 100 range is the conventional range, and not necessarily the one that covers the useful range of the penalty coefficient.

# 10 Calibration

Decision:

Why:

Cost: