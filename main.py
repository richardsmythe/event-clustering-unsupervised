import os
os.environ['OMP_NUM_THREADS'] = '5'

import numpy as np
import pandas as pd

from data import clean_data, load_data, add_residual_features, tag_outliers
from model_selection import (
    build_latent,
    # fit_gmm,  # GMM disabled
    cluster_enrichment,
    report_feature_deltas,
    feature_importance,
    plot_latent,
    fit_kmeans_with_metrics,
    compute_kmeans_confidence,
    choose_kmeans_k,
)

FILEPATH = r'C:\Users\rsmythe\source\repos\event-clustering-unsupervised\TSv11 - YLT - CB-PRVI EQ - By County & LOB.csv'
CLUSTER_FEATURES = [
    'Region_Loss_Z', 'LOB_Loss_Z', 'County_Vol_Z', 'Log_Ratio_Dev', 'Loss_region_pct', 'Vol_region_pct'
]
LATENT_COMPONENTS = 3

def main():
    print("=" * 60)
    print("Event Clustering Analysis using KMeans")
    print("=" * 60)
    
    print("\nLoading data...")
    df = load_data(FILEPATH)
    print(f"Loaded{len(df)} records")
    
    print("\nCleaning data...")
    df = clean_data(df)
    
    print("\nEngineering residual features...")
    df = add_residual_features(df)
    
    print("\nIdentifying outliers...")
    df, outlier_thresh = tag_outliers(df)
    print(f"-Outlier threshold (99th Loss_log): {outlier_thresh:.2f}")
    print(f"-Outliers: {df['IsOutlier'].sum()} ({df['IsOutlier'].mean()*100:.2f}%)")

    print("\nBuilding latent space (PCA on non-outliers)...")
    base_df = df[df['IsOutlier'] == 0].copy()
    print(f"-Training on {len(base_df)} non-outlier records")
    latent, pca, X_scaled = build_latent(base_df, CLUSTER_FEATURES, n_components=LATENT_COMPONENTS)

    print("\nTuning KMeans k (silhouette + inertia)...")
    best_k = choose_kmeans_k(latent, k_min=2, k_max=12)
    km, km_labels, km_sil = fit_kmeans_with_metrics(latent, best_k)
    base_df['cluster_km'] = km_labels
    base_df['cluster_km_conf'] = compute_kmeans_confidence(latent, km)

    print("\n  KMeans cluster sizes:")
    for c in sorted(base_df['cluster_km'].unique()):
        count = (base_df['cluster_km'] == c).sum()
        pct = count / len(base_df) * 100
        print(f"-Cluster {c}: {count:4d} ({pct:.1f}%)")

    print("\nMerging cluster assignments back to full dataframe...")
    df = df.merge(base_df[['RecordID', 'cluster_km', 'cluster_km_conf']], on='RecordID', how='left')
    df.loc[df['IsOutlier'] == 1, 'cluster_km'] = -1

    print("\nKMeans Info...")
    report_feature_deltas(df[df['cluster_km'] != -1], CLUSTER_FEATURES, 'cluster_km')
    cluster_enrichment(df[df['cluster_km'] != -1], 'cluster_km', 'LOB_Pure')
    cluster_enrichment(df[df['cluster_km'] != -1], 'cluster_km', 'Region')
    feature_importance(df, CLUSTER_FEATURES, 'cluster_km')
    plot_latent(latent, base_df, 'cluster_km', filename='cluster_pca_kmeans.png')

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