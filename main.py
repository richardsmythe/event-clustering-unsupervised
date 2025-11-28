import os
os.environ['OMP_NUM_THREADS'] = '5'

import numpy as np
import pandas as pd

from data import clean_data, load_data, add_residual_features, tag_outliers
from model_selection import (
    build_pca,
    cluster_enrichment,
    create_risk_tiers,
    plot_risk_tier_distribution,
    report_feature_deltas,
    feature_importance,
    plot_risk_scatter, 
    fit_kmeans_with_metrics,
    compute_kmeans_confidence,
    choose_kmeans_k,
    plot_cluster_feature_heatmap,
    plot_cluster_geography,
    plot_cluster_lob_distribution,
)

FILEPATH = r'\ LOB.csv'
CLUSTER_FEATURES = [
    'Region_Loss_Z', 'LOB_Loss_Z', 'County_Vol_Z', 'Log_Ratio_Dev', 'Loss_region_pct', 'Vol_region_pct'
]
LATENT_COMPONENTS = 3

def _safe_merge_clusters(original_df, base_df, cluster_cols):
    print("\nValidating RecordID uniqueness before merge...")
    orig_unique = original_df['RecordID'].is_unique
    base_unique = base_df['RecordID'].is_unique
    print(f" - original RecordID unique: {orig_unique}")
    print(f" - base RecordID unique: {base_unique}")

    if not base_unique:
        print(" - Warning: base_df has duplicate RecordID values. Aggregating cluster assignments by RecordID.")
        aggs = {}
        for col in cluster_cols:
            if col.endswith('_conf'):
                aggs[col] = 'mean'
            else:
                aggs[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        base_agg = base_df.groupby('RecordID').agg(aggs).reset_index()
    else:
        base_agg = base_df[['RecordID'] + cluster_cols].copy()

    try:
        if base_agg['RecordID'].is_unique and original_df['RecordID'].is_unique:
            merged = original_df.merge(base_agg, on='RecordID', how='left', validate='one_to_one')
        elif base_agg['RecordID'].is_unique and not original_df['RecordID'].is_unique:
            merged = original_df.merge(base_agg, on='RecordID', how='left', validate='one_to_many')
        else:
            merged = original_df.merge(base_agg, on='RecordID', how='left')
    except Exception as e:
        print(f"Merge validation failed: {e}")
        print("Falling back to a left-merge without validation.")
        merged = original_df.merge(base_agg, on='RecordID', how='left')

    return merged


def _diagnose_duplicates(original_df, base_df, top_n=20):
    print("\nDiagnosing RecordID duplicates...")
    orig_counts = original_df['RecordID'].value_counts()
    dup_orig = orig_counts[orig_counts > 1]
    print(f"Original df rows: {len(original_df)}; unique RecordID: {original_df['RecordID'].nunique()}")
    print(f"Original duplicated IDs: {len(dup_orig)} (IDs with count>1). Top {top_n} shown as {{...}}")
    if not dup_orig.empty:
        print(dup_orig.head(top_n).to_dict())
        sample_id = dup_orig.index[0]
        print(f"\nSample duplicate RecordID in original: {sample_id}, count={dup_orig.iloc[0]}")
        print(original_df[original_df['RecordID'] == sample_id].head().to_string(index=False))

    base_counts = base_df['RecordID'].value_counts()
    dup_base = base_counts[base_counts > 1]
    print(f"\nBase df rows: {len(base_df)}; unique RecordID: {base_df['RecordID'].nunique()}")
    print(f"Base duplicated IDs: {len(dup_base)} (IDs with count>1). Top {top_n} shown as {{...}}")
    if not dup_base.empty:
        print(dup_base.head(top_n).to_dict())
        sample_id = dup_base.index[0]
        print(f"\nSample duplicate RecordID in base: {sample_id}, count={dup_base.iloc[0]}")
        print(base_df[base_df['RecordID'] == sample_id].head().to_string(index=False))


def main():
    print("=" * 60)
    print("Event Clustering Analysis using KMeans")
    print("=" * 60)
    
    print("\nLoading data...")
    df = load_data(FILEPATH)
    print(f"Loaded {len(df)} records")
    
    print("\nCleaning data...")
    df = clean_data(df)
    
    print("\nEngineering residual features...")
    df = add_residual_features(df)
    
    print("\nIdentifying outliers...")
    df, outlier_thresh = tag_outliers(df)
    print(f"-Outlier threshold (99th Loss_log): {outlier_thresh:.2f}")
    print(f"-Outliers: {df['IsOutlier'].sum()} ({df['IsOutlier'].mean()*100:.2f}%)")

    print("\nBuilding PCA...")
    base_df = df[df['IsOutlier'] == 0].copy()
    print(f"-Training on {len(base_df)} non-outlier records")
    pca_data, pca, X_scaled = build_pca(base_df, CLUSTER_FEATURES, n_components=LATENT_COMPONENTS)

    print("\nTuning KMeans k (silhouette + inertia)...")
    best_k = choose_kmeans_k(pca_data, k_min=2, k_max=12)
    km, km_labels, km_sil = fit_kmeans_with_metrics(pca_data, best_k)
    base_df['cluster_km'] = km_labels
    base_df['cluster_km_conf'] = compute_kmeans_confidence(pca_data, km)

    print("\n  KMeans cluster sizes:")
    for c in sorted(base_df['cluster_km'].unique()):
        count = (base_df['cluster_km'] == c).sum()
        pct = count / len(base_df) * 100
        print(f"-Cluster {c}: {count:4d} ({pct:.1f}%)")

    _diagnose_duplicates(df, base_df, top_n=20)

    print("\nMerging cluster assignments back to full dataframe...")
    df = _safe_merge_clusters(df, base_df, ['cluster_km', 'cluster_km_conf'])
    
    df['cluster_km'] = df['cluster_km'].fillna(-1).astype(int)
    df['cluster_km_conf'] = df['cluster_km_conf'].fillna(0.0)
    df.loc[df['IsOutlier'] == 1, 'cluster_km'] = -1
    df.loc[df['IsOutlier'] == 1, 'cluster_km_conf'] = 0.0
    
    print(f"KMeans assignments: {df['cluster_km'].value_counts().sort_index().to_dict()}")
    print(f"Outliers (cluster=-1): {(df['cluster_km'] == -1).sum()}")

    print("\nKMeans Analysis...")
    report_feature_deltas(df[df['cluster_km'] != -1], CLUSTER_FEATURES, 'cluster_km')
    cluster_enrichment(df[df['cluster_km'] != -1], 'cluster_km', 'LOB_Pure')
    cluster_enrichment(df[df['cluster_km'] != -1], 'cluster_km', 'Region')
    feature_importance(df, CLUSTER_FEATURES, 'cluster_km')
    
    print("\nGenerating visualizations...")
    plot_risk_scatter(df, 'cluster_km')
    plot_cluster_geography(df, 'cluster_km')
    plot_cluster_lob_distribution(df, 'cluster_km')
    plot_cluster_feature_heatmap(df, CLUSTER_FEATURES, 'cluster_km')

    print("\n" + "=" * 60)
    print("CREATING RISK TIERS")
    print("=" * 60)
    
    df_for_tiers = df[df['cluster_km'] != -1].copy()
    print(f"Creating tiers for {len(df_for_tiers)} non-outlier events")
    
    df_for_tiers, tier_summary = create_risk_tiers(df_for_tiers, cluster_col='cluster_km')

    df = df.merge(
        df_for_tiers[['RecordID', 'RiskTier', 'RiskScore']], 
        on='RecordID', 
        how='left'
    )
    
    df.loc[df['cluster_km'] == -1, 'RiskTier'] = 'Outlier'
    df.loc[df['cluster_km'] == -1, 'RiskScore'] = 100.0

    plot_risk_tier_distribution(df[df['RiskTier'] != 'Outlier'], 'risk_tier_distribution.png')

    print("\n" + "=" * 60)
    print("TIER DISTRIBUTION (excluding outliers)")
    print("=" * 60)
    tier_counts = df[df['RiskTier'] != 'Outlier']['RiskTier'].value_counts()
    print(tier_counts)
    
    print("\n" + "=" * 60)
    print("OUTLIER SUMMARY")
    print("=" * 60)
    outliers = df[df['cluster_km'] == -1]
    print(f"Outlier events: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"Total outlier loss: ${outliers['Loss'].sum()/1e9:.2f}B")
    print(f"Avg outlier loss: ${outliers['Loss'].mean()/1e6:.2f}M")
    if len(outliers) > 0:
        print("Top 5 outliers by loss:")
        print(outliers.nlargest(5, 'Loss')[['RecordID', 'Loss', 'Region', 'LOB_Pure', 'County']].to_string(index=False))

    print("\n" + "=" * 60)
    print("HIGH-RISK EVENTS (non-outliers)")
    print("=" * 60)
    high_risk = df[df['RiskTier'] == 'High']
    if len(high_risk) > 0:
        print(f"High-risk events: {len(high_risk)} ({len(high_risk)/len(df)*100:.2f}%)")
        print(f"Total high-risk loss: ${high_risk['Loss'].sum()/1e9:.2f}B")
        print(f"Avg high-risk loss: ${high_risk['Loss'].mean()/1e6:.2f}M")
        print("Top 5 high-risk events by loss:")
        print(high_risk.nlargest(5, 'Loss')[['RecordID', 'Loss', 'RiskScore', 'Region', 'LOB_Pure', 'County']].to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Selected KMeans k={best_k}; silhouette={km_sil:.3f}; mean confidence={base_df['cluster_km_conf'].mean():.3f}")
    print("\nKMeans confidence distribution:")
    print(base_df['cluster_km_conf'].describe())
    print("\nLowest confidence KMeans assignments:")
    print(base_df.nsmallest(10, 'cluster_km_conf')[['RecordID', 'cluster_km', 'cluster_km_conf']].to_string(index=False))

if __name__ == '__main__':
    main()