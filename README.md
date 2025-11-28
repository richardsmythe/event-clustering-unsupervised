# Unsupervised Clustering Analysis for Earthquake Risk Assessment

## Introduction

Unsupervised clustering analysis was conducted on earthquake event data from Puerto Rico and the Virgin Islands. The goal was to identify risk patterns based on loss severity, county-level characteristics, and line of business (LOB) to support reinsurance and pricing decisions.

The clusters (0-4) that are created are based on patterns in the risk features and the risk tiers (High/Medium/Low) are created by scoring the clusters based on loss magnitude + volatility.
<img width="987" height="588" alt="image" src="https://github.com/user-attachments/assets/f31098b9-a93e-48b9-ad5f-59a22d3806d9" />

## Methodology

### Data Preparation and Feature Engineering

The analysis started with 4,740 insurance event records representing earthquake losses across different counties, regions, and lines of business. Feature engineering focused on creating standardized metrics that could capture relative risk across different scales. For each event, we calculated z-scores for regional and LOB losses, allowing fair comparisons regardless of absolute size. County-level volatility measures were constructed by analyzing loss variance within each county. Residual features captured deviations from expected patterns, including ratios of actual losses to regional averages and percentile rankings within regions and LOBs.

### Outlier Detection

Extreme loss events were identified using the 99th percentile of log-transformed losses. This flagged 48 outliers representing 1% of events but accounting for over $2 trillion in catastrophic losses. These were set aside for separate analysis to prevent distortion of the clustering process.

### Dimensionality Reduction and Clustering

Principal Component Analysis reduced six engineered features to three components, explaining 98% of the variance. KMeans clustering was evaluated from 2 to 12 clusters using silhouette scores to measure quality. Five clusters emerged as optimal with a silhouette score of 0.334. Gaussian Mixture Models were tested but produced 11 overlapping components with a lower silhouette score of 0.109, so KMeans was selected.

### Feature Importance

A Random Forest classifier revealed that county volatility was the dominant factor at 31% of predictive power, followed by LOB-specific losses and ratio deviations at 22% each. Regional loss patterns and percentile rankings played smaller but meaningful roles.

<img width="1065" height="647" alt="image" src="https://github.com/user-attachments/assets/343dcc47-3d57-44d4-b09a-daeb1b8aa38e" />


## Results and Risk Tier Classification

### Cluster Profiles

**Cluster 0 (579 events, 12%)** had the highest risk with average losses of $118 million. County volatility was extremely high (z-score 1.23) with strong Virgin Islands concentration. Auto insurance was enriched at 1.63 times baseline. Risk score: 100.

**Cluster 1 (1,290 events, 28%)** showed moderate losses of $7.7 million but elevated volatility (z-score 0.39). Geographically diverse with notable auto insurance underrepresentation at 23% of expected levels. Risk score: 72.

**Cluster 2 (491 events, 11%)** consisted of high-value but stable events averaging $60 million. The defining characteristic was extreme negative deviation in loss ratios (z-score -2.07), suggesting favorable loss-to-premium relationships. Risk score: 81.

**Cluster 3 (662 events, 14%)** was the only medium-risk cluster with tiny average losses of $480,000. County volatility was strongly negative (z-score -1.66), indicating stable areas. Commercial insurance dominated while auto was absent. Risk score: 50.

**Cluster 4 (1,678 events, 36%)** formed the largest segment with $57 million average losses. Puerto Rico was strongly represented at 1.11 times baseline, with auto insurance enriched at 1.51 times. Risk score: 83.

### Risk Tier Distribution

<img width="1186" height="684" alt="image" src="https://github.com/user-attachments/assets/7b41a609-4d46-47fa-b739-895fc168aadf" />

Four clusters qualified as high risk, containing 87% of events and totaling $216 billion in losses. Puerto Rico dominated with 2,635 events, followed by VI20 with 561. The single medium-risk cluster contained 13% of events with just $290 million in total losses. The 48 outliers averaged nearly $6 billion per event, with the largest single event reaching $21.7 billion.

## Key Insights

**Geographic Risk Concentration:** Virgin Islands regions showed disproportionate representation in the highest-risk cluster with enrichment factors approaching 1.8, indicating both severe exposure and lack of geographic diversification. Puerto Rico showed more balanced distribution except in Cluster 4.

**Line of Business Patterns:** Auto insurance consistently marked high-risk clusters with enrichment above 1.5, likely reflecting concentrated population centers rather than inherent coverage risk. Commercial insurance showed enrichment in the medium-risk cluster, suggesting deliberate underwriting toward stable counties.

**Volatility as Primary Driver:** County-level volatility contributed 31% of predictive power, indicating that location matters more than raw loss amount. Two events with similar losses can have dramatically different risk profiles based on county stability.

**Reinsurance Implications:** The medium-risk cluster could be self-insured with retentions around $10 million. High-risk clusters require traditional excess-of-loss coverage with attachment points around $25 to $50 million. Outliers demand catastrophe bonds or retrocessional coverage with attachments above $1 billion.

**Confidence Considerations:** Cluster assignment confidence averaged 0.46, indicating many events fell near boundaries. The 10 lowest-confidence assignments had scores below 0.004, essentially random. These borderline cases should receive manual review rather than automated tier assignment.

## Conclusion

The clustering analysis successfully identified five distinct risk patterns driven primarily by county volatility rather than absolute loss magnitude. The four high-risk clusters accounting for 87% of events share characteristics of geographic concentration, elevated volatility, and auto insurance enrichment. The medium-risk cluster provides opportunity for improved capital efficiency through higher retentions. The extreme tail of 48 outliers requires catastrophe-specific reinsurance structures. These patterns translate directly into pricing and underwriting decisions, with county-level risk factors deserving greater weight in rating algorithms, particularly for Virgin Islands exposures.


### Features Used for Clustering

1. `Region_Loss_Z` - Regional loss z-scores
2. `LOB_Loss_Z` - Line of business loss z-scores
3. `County_Vol_Z` - County volatility z-scores
4. `Log_Ratio_Dev` - Log ratio deviation from regional average
5. `Loss_region_pct` - Loss percentile within region
6. `Vol_region_pct` - Volatility percentile within region


