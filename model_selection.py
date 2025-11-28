import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import seaborn as sns

def find_optimal_k_elbow(X, max_k=10, random_state=42):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, 'o-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(alpha=0.3)
    plt.show()
    
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    k = np.argmax(second_diffs) + 2
    
    print(f"\nSuggested k: {k}")
    return k

def fit_kmeans(X, n_clusters, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
    kmeans.fit(X)
    print(f"Fitted {n_clusters} clusters, inertia: {kmeans.inertia_:.0f}")
    return kmeans

def print_cluster_summary(df):
    print("\nCluster Summary:")
    print("-" * 60)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        n = len(cluster_data)
        
        total_loss = cluster_data['Loss'].sum()
        avg_loss = cluster_data['Loss'].mean()
        
        print(f"\nCluster {cluster_id}: {n} events ({n/len(df)*100:.1f}% of data)")
        print(f"  Total loss: ${total_loss/1e9:.1f}B")
        print(f"  Average loss: ${avg_loss/1e6:.1f}M")        

        regions = cluster_data['Region'].value_counts()
        region_pcts = (regions / n * 100).round(1)
        print(f"  Regions: {dict(regions)}")
        print(f"  Region %: {dict(region_pcts)}")

        lobs = cluster_data['LOB_Pure'].value_counts()
        lob_pcts = (lobs / n * 100).round(1)
        print(f"  LOBs: {dict(lobs)}")
        print(f"  LOB %: {dict(lob_pcts)}")

    print("\n" + "-" * 60)    

    avg_losses = df.groupby('cluster')['Loss'].mean()
    high_cluster = avg_losses.idxmax()
    low_cluster = avg_losses.idxmin()
    
    high_regions = df[df['cluster'] == high_cluster]['Region'].value_counts().head(2).index.tolist()
    low_regions = df[df['cluster'] == low_cluster]['Region'].value_counts().head(2).index.tolist()
    
    print(f"Highest avg loss cluster ({high_cluster}): {', '.join(high_regions)}")
    print(f"Lowest avg loss cluster ({low_cluster}): {', '.join(low_regions)}")

    print("\nGeographic Distribution:")
    all_mixed = True
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        top_region = cluster_data['Region'].value_counts().iloc[0]
        top_region_name = cluster_data['Region'].value_counts().index[0]
        purity = (top_region / len(cluster_data) * 100)
        print(f"  Cluster {cluster_id}: {purity:.1f}% from {top_region_name}")
        
        if purity >= 80:
            all_mixed = False
    
    if all_mixed:
        print("\nNote: All clusters contain mixed regions - patterns cross geographic boundaries")


def build_pca(df, features, n_components=3, random_state=42):
    X = df[features].values.astype(float)
    X_scaled = RobustScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_data = pca.fit_transform(X_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return pca_data, pca, X_scaled


def cluster_enrichment(df, cluster_col, field):
    base = df[field].value_counts(normalize=True)
    print(f"\nEnrichment for {field}:")
    for c in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == c][field].value_counts(normalize=True)
        lift = (sub / base).fillna(0)
        top_pos = lift.sort_values(ascending=False).head(3)
        print(f"  Cluster {c}: {top_pos.to_dict()}")


def report_feature_deltas(df, features, cluster_col):
    means_global = df[features].mean()
    print("\nResidual feature deltas (cluster mean - global mean):")
    for c in sorted(df[cluster_col].unique()):
        m = df[df[cluster_col] == c][features].mean() - means_global
        print("  ", c, {f: round(m[f], 3) for f in features})


def feature_importance(df, features, cluster_col):
    mask = df['IsOutlier'] == 0
    X = df.loc[mask, features].values
    y = df.loc[mask, cluster_col].values
    if len(np.unique(y)) < 2:
        print("Skipping importance (only one cluster).")
        return
    print("\nTraining Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print("Computing permutation importance...")
    imp = permutation_importance(rf, X, y, n_repeats=3, random_state=42, n_jobs=-1)
    order = np.argsort(-imp.importances_mean)
    print("\nPermutation importance (mean decrease):")
    for idx in order:
        print(f"  {features[idx]}: {imp.importances_mean[idx]:.4f} ± {imp.importances_std[idx]:.4f}")


def plot_latent(latent, df, cluster_col, filename='cluster_pca.png'):
    pass


def plot_risk_scatter(df, cluster_col='cluster_km', filename='risk_scatter.png'):
    if 'IsOutlier' in df.columns:
        df_plot = df[df['IsOutlier'] == 0].copy()
    else:
        df_plot = df.copy()
    
    df_plot = df_plot[df_plot[cluster_col] != -1]
    if df_plot.empty:
        print("No valid cluster rows to plot risk scatter.")
        return
    
    if 'Loss' not in df_plot.columns or 'County_Vol_Z' not in df_plot.columns:
        print("Missing required columns (Loss, County_Vol_Z) for risk scatter plot.")
        return
    
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.Set1(np.linspace(0, 1, df_plot[cluster_col].nunique()))
    
    for i, c in enumerate(sorted(df_plot[cluster_col].unique())):
        cluster_data = df_plot[df_plot[cluster_col] == c]
        plt.scatter(
            cluster_data['County_Vol_Z'], 
            cluster_data['Loss'] / 1e6,
            c=[colors[i]], 
            label=f'Cluster {c}',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.xlabel('County Volatility (Z-Score)', fontsize=12)
    plt.ylabel('Loss Amount ($M)', fontsize=12)
    plt.title('Risk Profile: Loss Amount vs County Volatility by Cluster', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title='Cluster', fontsize=10, loc='best')
    
    plt.axhline(y=df_plot['Loss'].median() / 1e6, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def fit_kmeans_with_metrics(latent, k, random_state=42):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
    labels = km.fit_predict(latent)
    sil = silhouette_score(latent, labels) if k > 1 else np.nan
    print(f"KMeans k={k} inertia={km.inertia_:.0f} silhouette={sil:.3f}")
    return km, labels, sil


def compute_kmeans_confidence(latent, km):
    centers = km.cluster_centers_
    dists = np.linalg.norm(latent[:, None, :] - centers[None, :, :], axis=2)
    assigned = np.argmin(dists, axis=1)
    min_dist = dists[np.arange(len(latent)), assigned]
    sorted_dists = np.sort(dists, axis=1)
    second_dist = sorted_dists[:, 1]
    ratio = (min_dist / (second_dist + 1e-6))
    conf = (1 - ratio).clip(0, 1)
    return conf

def choose_kmeans_k(latent, k_min=2, k_max=12, random_state=42):
    results = []
    best_k = None
    best_sil = -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
        labels = km.fit_predict(latent)
        sil = silhouette_score(latent, labels)
        inertia = km.inertia_
        results.append((k, sil, inertia))
        print(f"KMeans tuning k={k}: silhouette={sil:.3f} inertia={inertia:.0f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
    close = [r for r in results if best_sil - r[1] <= 0.01]
    if len(close) > 1:
        best_k = min(close, key=lambda r: r[2])[0]
    print(f"Selected KMeans k={best_k} (silhouette={best_sil:.3f})")
    return best_k

def create_risk_tiers(df, cluster_col='cluster_km', features=None):
    if features is None:
        features = ['Region_Loss_Z', 'LOB_Loss_Z', 'County_Vol_Z', 
                    'Log_Ratio_Dev', 'Loss_region_pct', 'Vol_region_pct']
    
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        raise ValueError(f"None of the specified features found in dataframe. Available: {df.columns.tolist()}")
    
    print(f"\nCreating risk tiers using features: {available_features}")
    
    cluster_stats = df.groupby(cluster_col).agg({
        'Loss': ['mean', 'std', 'count'],
        'County_Vol_Z': 'mean' if 'County_Vol_Z' in df.columns else lambda x: 0,
        'Region_Loss_Z': 'mean' if 'Region_Loss_Z' in df.columns else lambda x: 0,
        'LOB_Loss_Z': 'mean' if 'LOB_Loss_Z' in df.columns else lambda x: 0,
    }).reset_index()
    
    cluster_stats.columns = ['_'.join(col).strip('_') for col in cluster_stats.columns.values]
    cluster_stats.rename(columns={cluster_col: 'cluster'}, inplace=True)
    
    cluster_stats['risk_score'] = 0
    
    if 'County_Vol_Z_mean' in cluster_stats.columns:
        cluster_stats['risk_score'] += cluster_stats['County_Vol_Z_mean'] * 0.4
    
    if 'Region_Loss_Z_mean' in cluster_stats.columns:
        cluster_stats['risk_score'] += cluster_stats['Region_Loss_Z_mean'] * 0.3
    
    if 'LOB_Loss_Z_mean' in cluster_stats.columns:
        cluster_stats['risk_score'] += cluster_stats['LOB_Loss_Z_mean'] * 0.3
    
    min_score = cluster_stats['risk_score'].min()
    max_score = cluster_stats['risk_score'].max()
    cluster_stats['risk_score_normalized'] = 50 + 50 * (cluster_stats['risk_score'] - min_score) / (max_score - min_score + 1e-6)
    
    def assign_tier(score):
        if score >= 70:
            return 'High'
        elif score >= 40:
            return 'Medium'
        else:
            return 'Low'
    
    cluster_stats['RiskTier'] = cluster_stats['risk_score_normalized'].apply(assign_tier)
    
    tier_map = dict(zip(cluster_stats['cluster'], cluster_stats['RiskTier']))
    score_map = dict(zip(cluster_stats['cluster'], cluster_stats['risk_score_normalized']))
    
    df['RiskTier'] = df[cluster_col].map(tier_map)
    df['RiskScore'] = df[cluster_col].map(score_map)
    
    print("\n" + "="*60)
    print("RISK TIER CLASSIFICATION")
    print("="*60)
    
    for cluster in sorted(cluster_stats['cluster'].unique()):
        c_stats = cluster_stats[cluster_stats['cluster'] == cluster].iloc[0]
        print(f"\nCluster {cluster}:")
        print(f"  Risk Tier: {c_stats['RiskTier']}")
        print(f"  Risk Score: {c_stats['risk_score_normalized']:.1f}/100")
        print(f"  Avg Loss: ${c_stats['Loss_mean']/1e6:.2f}M")
        if 'County_Vol_Z_mean' in c_stats:
            print(f"  Volatility Z-score: {c_stats['County_Vol_Z_mean']:.2f}")
        if 'Region_Loss_Z_mean' in c_stats:
            print(f"  Region Loss Z-score: {c_stats['Region_Loss_Z_mean']:.2f}")
        print(f"  Event Count: {int(c_stats['Loss_count'])}")
    
    tier_summary = df.groupby('RiskTier').agg({
        'Loss': ['count', 'sum', 'mean', 'std'],
        'RiskScore': 'mean',
        cluster_col: 'nunique'
    }).round(2)
    
    print("\n" + "="*60)
    print("RISK TIER SUMMARY")
    print("="*60)
    
    for tier in ['High', 'Medium', 'Low']:
        if tier in tier_summary.index:
            print(f"\n{tier} Risk:")
            tier_data = df[df['RiskTier'] == tier]
            print(f"  Events: {len(tier_data)} ({len(tier_data)/len(df)*100:.1f}%)")
            print(f"  Total Loss: ${tier_data['Loss'].sum()/1e9:.2f}B")
            print(f"  Avg Loss: ${tier_data['Loss'].mean()/1e6:.2f}M")
            print(f"  Avg Risk Score: {tier_data['RiskScore'].mean():.1f}")
            print(f"  Clusters in tier: {tier_data[cluster_col].nunique()}")
            
            if 'Region' in tier_data.columns:
                top_regions = tier_data['Region'].value_counts().head(3)
                print(f"  Top Regions: {dict(top_regions)}")
            
            if 'LOB_Pure' in tier_data.columns:
                top_lobs = tier_data['LOB_Pure'].value_counts().head(3)
                print(f"  Top LOBs: {dict(top_lobs)}")
    
    return df, tier_summary


def export_risk_tier_report(df, filename='risk_tier_report.csv', cluster_col='cluster_km'):
    if 'RiskTier' not in df.columns:
        print("Error: RiskTier column not found. Run create_risk_tiers() first.")
        return
    
    export_cols = ['RecordID', 'RiskTier', 'RiskScore', cluster_col, 'Loss', 
                   'Region', 'LOB_Pure', 'County']
    
    optional_cols = ['County_Vol_Z', 'Region_Loss_Z', 'LOB_Loss_Z', 
                     'Log_Ratio_Dev', 'cluster_km_conf', 'IsOutlier', 'Loss_log']
    
    for col in optional_cols:
        if col in df.columns:
            export_cols.append(col)
    
    export_cols = [c for c in export_cols if c in df.columns]
    
    df_export = df[export_cols].copy()
    
    tier_order = {'Outlier': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    df_export['_tier_order'] = df_export['RiskTier'].map(tier_order).fillna(999)
    df_export = df_export.sort_values(['_tier_order', 'RiskScore'], ascending=[True, False])
    df_export = df_export.drop('_tier_order', axis=1)
    
    df_export.to_csv(filename, index=False)
    print(f"\nExported risk tier report to {filename}")
    print(f"  Total records: {len(df_export)}")
    print(f"  Columns: {len(export_cols)}")
    
    tier_summary = df_export.groupby('RiskTier').agg({
        'RecordID': 'count',
        'Loss': ['sum', 'mean']
    }).round(2)
    tier_summary.columns = ['Count', 'TotalLoss', 'AvgLoss']
    print(f"\n  Tier breakdown:")
    for tier in tier_summary.index:
        row = tier_summary.loc[tier]
        print(f"    {tier}: {int(row['Count'])} events, ${row['TotalLoss']/1e9:.2f}B total")


def plot_risk_tier_distribution(df, filename='risk_tier_distribution.png'):
    if 'RiskTier' not in df.columns:
        print("Error: RiskTier column not found. Run create_risk_tiers() first.")
        return
    
    tier_order = ['High', 'Medium', 'Low']
    tier_colors = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
    
    tier_counts = df['RiskTier'].value_counts().reindex(tier_order, fill_value=0)
    tier_losses = df.groupby('RiskTier')['Loss'].sum().reindex(tier_order, fill_value=0) / 1e9
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    bars1 = ax1.bar(tier_order, tier_counts, color=[tier_colors[t] for t in tier_order], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Risk Tier', fontsize=12)
    ax1.set_ylabel('Number of Events', fontsize=12)
    ax1.set_title('Event Count by Risk Tier', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    bars2 = ax2.bar(tier_order, tier_losses, color=[tier_colors[t] for t in tier_order], alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Risk Tier', fontsize=12)
    ax2.set_ylabel('Total Loss ($B)', fontsize=12)
    ax2.set_title('Total Loss by Risk Tier', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}B',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_cluster_feature_heatmap(df, features, cluster_col='cluster_km', filename='cluster_feature_heatmap.png'):
    if 'IsOutlier' in df.columns:
        df_plot = df[df['IsOutlier'] == 0].copy()
    else:
        df_plot = df.copy()
    
    df_plot = df_plot[df_plot[cluster_col] != -1]
    if df_plot.empty:
        print("No valid cluster rows to plot heatmap.")
        return
    feats = [f for f in features if f in df_plot.columns]
    if not feats:
        print("No requested features found for heatmap.")
        return
    
    mat = df_plot.groupby(cluster_col)[feats].mean().round(3)
    standardized = (mat - mat.mean()) / (mat.std(ddof=0) + 1e-6)

    plt.figure(figsize=(1.2 * len(feats) + 4, 0.5 * len(mat) + 4))
    sns.heatmap(standardized, annot=mat, fmt='', cmap='coolwarm', center=0,
                cbar_kws={'label': 'Std (feature relative to mean)'})
    plt.title('Cluster Feature Profiles (raw values annotated; color = standardized)')
    plt.xlabel('Feature')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.show()

def plot_cluster_geography(df, cluster_col='cluster_km', filename='cluster_geography.png'):
    if 'IsOutlier' in df.columns:
        df_plot = df[df['IsOutlier'] == 0].copy()
    else:
        df_plot = df.copy()
    
    df_plot = df_plot[df_plot[cluster_col] != -1]
    if df_plot.empty:
        print("No valid cluster rows to plot geography.")
        return
    
    if 'Region' not in df_plot.columns:
        print("Missing 'Region' column for geography plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    region_cluster = pd.crosstab(df_plot[cluster_col], df_plot['Region'], normalize='index') * 100
    region_cluster.plot(kind='bar', stacked=True, ax=ax1, colormap='Set3', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Region Composition by Cluster', fontsize=14, fontweight='bold')
    ax1.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.set_xticklabels([f'Cluster {i}' for i in region_cluster.index], rotation=0)
    ax1.grid(axis='y', alpha=0.3)
    
    cluster_region = pd.crosstab(df_plot['Region'], df_plot[cluster_col], normalize='index') * 100
    cluster_region.plot(kind='bar', stacked=True, ax=ax2, cmap='tab10', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Region', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Cluster Distribution by Region', fontsize=14, fontweight='bold')
    ax2.legend(title='Cluster', labels=[f'Cluster {i}' for i in sorted(df_plot[cluster_col].unique())], 
               bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.set_xticklabels(cluster_region.index, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("GEOGRAPHIC DISTRIBUTION BY CLUSTER")
    print("="*60)
    
    for cluster_id in sorted(df_plot[cluster_col].unique()):
        cluster_data = df_plot[df_plot[cluster_col] == cluster_id]
        region_counts = cluster_data['Region'].value_counts()
        total = len(cluster_data)
        
        print(f"\nCluster {cluster_id} ({total} events):")
        for region, count in region_counts.items():
            pct = (count / total) * 100
            print(f"  {region}: {count:4d} events ({pct:5.1f}%)")
    
    if 'County' in df_plot.columns:
        print("\n" + "="*60)
        print("TOP COUNTIES BY CLUSTER")
        print("="*60)
        
        for cluster_id in sorted(df_plot[cluster_col].unique()):
            cluster_data = df_plot[df_plot[cluster_col] == cluster_id]
            county_counts = cluster_data['County'].value_counts().head(5)
            
            print(f"\nCluster {cluster_id} - Top 5 Counties:")
            for county, count in county_counts.items():
                pct = (count / len(cluster_data)) * 100
                county_region = cluster_data[cluster_data['County'] == county]['Region'].mode()[0]
                print(f"  County {county} ({county_region}): {count} events ({pct:.1f}%)")


def plot_cluster_lob_distribution(df, cluster_col='cluster_km', filename='cluster_lob.png'):
    if 'IsOutlier' in df.columns:
        df_plot = df[df['IsOutlier'] == 0].copy()
    else:
        df_plot = df.copy()
    
    df_plot = df_plot[df_plot[cluster_col] != -1]
    if df_plot.empty or 'LOB_Pure' not in df_plot.columns:
        print("Cannot plot LOB distribution.")
        return

    lob_cluster = pd.crosstab(df_plot[cluster_col], df_plot['LOB_Pure'], normalize='index') * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    lob_cluster.plot(kind='bar', stacked=True, ax=ax, colormap='Pastel1', edgecolor='black', linewidth=0.7)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Line of Business Composition by Cluster', fontsize=14, fontweight='bold')
    ax.legend(title='Line of Business', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels([f'Cluster {i}' for i in lob_cluster.index], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("LINE OF BUSINESS BY CLUSTER")
    print("="*60)
    
    for cluster_id in sorted(df_plot[cluster_col].unique()):
        cluster_data = df_plot[df_plot[cluster_col] == cluster_id]
        lob_counts = cluster_data['LOB_Pure'].value_counts()
        total = len(cluster_data)
        
        print(f"\nCluster {cluster_id} ({total} events):")
        for lob, count in lob_counts.items():
            pct = (count / total) * 100
            print(f"  {lob}: {count:4d} events ({pct:5.1f}%)")
