import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
import numpy as np

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
    """Fit kmeans model"""
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


def build_latent(df, features, n_components=3, random_state=42):
    X = df[features].values.astype(float)
    X_scaled = RobustScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    latent = pca.fit_transform(X_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return latent, pca, X_scaled


def fit_gmm(latent, k_min=2, k_max=10, random_state=42):
    best_k, best_bic, best_gmm = None, np.inf, None
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
        gmm.fit(latent)
        bic = gmm.bic(latent)
        print(f"GMM k={k} BIC={bic:.0f}")
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm
    print(f"Selected GMM k={best_k} BIC={best_bic:.0f}")
    return best_gmm


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
    plt.figure(figsize=(8, 6))
    for c in sorted(df[cluster_col].unique()):
        pts = latent[df[cluster_col] == c]
        plt.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, label=f'C{c}')
    plt.title('Clusters in PCA latent (PC1 vs PC2)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved cluster plot to {filename}")


def fit_kmeans_with_metrics(latent, k, random_state=42):
    """Fit KMeans and return model plus inertia and silhouette score."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
    labels = km.fit_predict(latent)
    sil = silhouette_score(latent, labels) if k > 1 else np.nan
    print(f"KMeans k={k} inertia={km.inertia_:.0f} silhouette={sil:.3f}")
    return km, labels, sil


def compute_kmeans_confidence(latent, km):
    """Compute a confidence measure based on relative distance to assigned center vs other centers."""
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
    """Evaluate KMeans for k range and select best k by max silhouette; fallback to lowest inertia if silhouettes similar."""
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
