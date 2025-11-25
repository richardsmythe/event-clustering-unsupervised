# Clustered Catastrophe Events

K-means clustering analysis of earthquake event data from Puerto Rico and the Virgin Islands, identifying risk patterns based on loss severity, county-level characteristics, and line of business to support reinsurance and pricing decisions.

## Pipeline Steps

1. **Exploratory Data Analysis** - Describe raw data, distributions, and correlations
2. **Data Cleaning** - Handle missing values, drop constant columns, cap outliers, encode categoricals
3. **Feature Engineering** - Create county-level aggregates, LOB×Region interactions, and loss distribution features
4. **Feature Selection** - Select clustering features, check for NaN values and high correlations
5. **Optimal K Selection** - Run elbow method to identify optimal number of clusters
6. **Clustering** - Fit K-means and assign cluster labels

## Current Configuration

**Features:**
- `Loss_log` - Log-transformed capped loss
- `Ratio` - Loss ratio metric
- `LOB_Pure_code` - Line of business (Auto, Commercial, Residential)
- `County_AvgLoss_log` - Log of average loss per county
- `County_LossVolatility` - Loss volatility within county

**Clusters:** 4 (elbow method suggests 2, but 4 provides better business segmentation)

## Results

### Four-Cluster Breakdown (k=4)

| Cluster | Events | Avg Loss | Total Loss | Dominant LOBs | Profile |
|---------|--------|----------|------------|---------------|---------|
| **0** | 636 (13.4%) | $2.1M | $1.3B | Auto (94%) | Low-severity Auto |
| **1** | 1,218 (25.7%) | $121.3M | $147.7B | Commercial (56%), Residential (44%) | High-severity Property |
| **2** | 1,564 (33.0%) | $0.3M | $0.4B | Commercial (56%), Residential (45%) | Routine Property |
| **3** | 1,322 (27.9%) | $254.0M | $335.7B | Commercial (54%), Residential (46%) | Catastrophic Property |

<img width="1390" height="482" alt="Cluster visualization" src="https://github.com/user-attachments/assets/7b39bc3c-4514-4daf-8a99-f0ab1a293b29" />

### Observations:

**Cluster 0 - Auto Events**
- Distinct from property events with consistently lower losses
- 94% of events are Auto LOB
- Average loss of $2.1M vs $100M+ for property clusters

**Property Event Tiers (Clusters 1, 2, 3)**
- Three clear severity levels emerge in the data
- Cluster 2: Routine events ($0.3M avg) - highest frequency, lowest impact
- Cluster 1: Significant events ($121M avg) - moderate frequency
- Cluster 3: Catastrophic events ($254M avg) - accounts for 69% of total losses

**Geographic Patterns**
- PR represents 57-71% across all clusters
- VI regions distributed relatively evenly (8-16% each)
- No strong geographic clustering - all regions appear in all severity levels
- County-level characteristics better predict cluster membership than region

**Loss Concentration**
- Clusters 1 and 3 (54% of events) account for 99.7% of total losses
- Cluster 2 has the most events (33%) but represents <1% of total loss dollars
- Long tail of low-severity events with occasional catastrophic losses

### Business Implications:

1. **Pricing Strategy:** Auto should be priced separately from property. Property events need tiered pricing based on severity rather than simple geographic splits.

2. **Underwriting:** County-level risk assessment is more informative than regional grouping. High-risk counties exist within all geographic regions.

3. **Concentration Risk:** 28% of events (Cluster 3) drive 69% of losses. Portfolio management should focus on limiting exposure to catastrophic property events.

4. **Model Validation:** The lack of geographic clustering suggests that region alone is insufficient for risk segmentation. Event characteristics and county factors are stronger predictors.

## Enhanced Features

The pipeline automatically engineers ~15 additional features from existing data:

### County-Level Features
- `County_AvgLoss_log` - Average loss per county (log-transformed)
- `County_LossVolatility` - Standard deviation / mean loss per county
- `County_EventCount` - Number of events per county
- `County_LossShare_InRegion` - County's percentage of regional losses

### Interaction Features
- `LOB_Region_AvgLoss_log` - Average loss for each LOB×Region combination
- `Ratio_vs_RegionAvg` - Event ratio relative to regional average

### Distribution Features
- `Loss_Percentile_InRegion` - Event loss percentile within region
- `Loss_Percentile_InLOB` - Event loss percentile within LOB
- `IsExtremeLoss_95` / `IsExtremeLoss_99` - Extreme loss indicators (95th/99th percentile)

### Ratio Features
- `Ratio_Category_code` - Binned ratio levels (Very Low to Very High)
- `Ratio_vs_RegionAvg` - Event ratio compared to regional average
