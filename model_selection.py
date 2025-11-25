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
    print("\nWhat's in each cluster:")
    print("-" * 60)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        n = len(cluster_data)
        
        total_loss = cluster_data['Loss'].sum()
        avg_loss = cluster_data['Loss'].mean()
        
        print(f"\nCluster {cluster_id}: {n} events")
        print(f"  Total loss: ${total_loss/1e9:.1f}B")
        print(f"  Average loss: ${avg_loss/1e6:.1f}M")        

        regions = cluster_data['Region'].value_counts()
        print(f"  Regions: {dict(regions)}")        

        lobs = cluster_data['LOB_Pure'].value_counts()
        print(f"  LOBs: {dict(lobs)}")

    print("\n" + "-" * 60)    

    avg_losses = df.groupby('cluster')['Loss'].mean()
    high_cluster = avg_losses.idxmax()
    low_cluster = avg_losses.idxmin()
    
    high_regions = df[df['cluster'] == high_cluster]['Region'].value_counts().head(2).index.tolist()
    low_regions = df[df['cluster'] == low_cluster]['Region'].value_counts().head(2).index.tolist()
    
    print(f"High-exposure cluster ({high_cluster}): {', '.join(high_regions)}")
    print(f"Lower-exposure cluster ({low_cluster}): {', '.join(low_regions)}")

    if 'VI10' in high_regions and 'PR' in high_regions:
        print("Note: VI10 clusters with PR despite being geographically VI")
