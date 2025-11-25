import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
