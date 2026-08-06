# Decisions

#1

Decision: Lending Club dataset over German Credit dataset.

Why: 
1. Lending club has a larger dataset, therefore a larger number of defaulted loans, this contributes to:

    1.a The model having a better chance to produce similar results during testing. This is because PD-determining factors would be more stable, since the estimates would be noisier for a dataset that has less defaults.
    
    1.b The thesis of LR vs GBM could be more confidently observed, since german credit findings could more plausibly be attriuted to noise.

2. Since german credit has ~20 precurated features which are all application-time, and lending club has ~150 and a good amount are post-origination, choosing my own feature list from lending club's list allows leakage-hunting, therefore demonstrating leakage-hunting skills.

Cost: Since lending club is a peer to peer marketplace, the general applicant would be of a different nature: it's unsecured, and more often debt-consolidation heavy, therefore the findings could not really transfer for a bank applicant, though the methodology of building a model stands.






