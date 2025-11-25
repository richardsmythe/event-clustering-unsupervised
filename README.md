# Clustered Catastrophe Events

Unsupervised clustering analysis was conducted on earthquake event data from Puerto Rico and the Virgin Islands. The goal was to identify risk patterns based on loss severity, county-level characteristics, and line of business (LOB) to support reinsurance and pricing decisions.

---

## KMeans Clustering Process

- Analysed approximately **4,740 event records**, removing 48 extreme outliers (~1% of the data).
- Used **Principal Component Analysis (PCA)** to reduce dimensionality, with the first three components explaining **97% of the variance**.
- Applied KMeans clustering, selecting **5 clusters** based on silhouette scores and inertia.
- Merged clusters back into the full dataset and analysed cluster characteristics using feature importance and PCA visualisation.

---

<img width="1184" height="884" alt="cluster_pca_kmeans" src="https://github.com/user-attachments/assets/6df05928-ac5b-45d8-8bd3-5fde96d63b92" />


## Key Results

- **Outliers**: 48 records removed (1% of data).
- **PCA Variance Explained**: 97% (PC1: 44%, PC2: 29%, PC3: 24%).
- **Optimal Clusters**: 5 clusters identified with moderate separation (silhouette score ≈ 0.334).
- **Key Features Driving Clusters**:
  - **County Volatility (County_Vol_Z)**: Most significant factor.
  - **Pricing Deviation (Log_Ratio_Dev)**: Second most important.
  - **LOB Loss Z-Score (LOB_Loss_Z)**: Moderate influence.

---

## Cluster Insights

### General Observations
- The PCA scatter plot shows some **distinct subgroups** but also a **central overlap** where clusters mix. This aligns with the moderate silhouette score and confidence levels.
- Clusters are primarily driven by **county-level volatility** and **pricing deviation**, with residual loss relative to region and LOB playing a secondary role.

### Cluster Profiles
1. **High-Risk Cluster**: 
   - Higher-than-average losses.
   - Very high county-level volatility.
   - Likely represents risky, high-variance counties.

2. **Stable Cluster**: 
   - Lower-than-average losses.
   - Low volatility.
   - Represents stable, low-loss counties.

3. **Distinct Ratio Behaviour Cluster**: 
   - Losses near the average.
   - Very low pricing ratio deviation.
   - Indicates unique pricing behaviour.

4. & 5. **Mixed Clusters**: 
   - Moderate losses and volatility.
   - Differing levels of LOB and regional enrichment.

---

## Key Takeaways
- **Volatility and pricing deviation** are the primary drivers of segmentation.
- The clusters provide actionable insights into risk patterns, with clear distinctions between high-risk, stable, and mixed groups.
- While some clusters are compact and well-defined, others show overlap, suggesting soft boundaries and potential for further refinement.
