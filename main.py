import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from data import clean_data, load_data, prepare_features_for_clustering
from model_selection import find_optimal_k_elbow, fit_kmeans, print_cluster_summary

FILEPATH = r'\EQ_LOB.csv'

def plot_clusters(df, x_scaled):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(x_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for cluster_id in sorted(df['cluster'].unique()):
        mask = df['cluster'] == cluster_id
        ax1.scatter(coords[mask, 0], coords[mask, 1], 
                   label=f'Cluster {cluster_id}', alpha=0.6, s=30)
    ax1.set_title('Clusters')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    for region in ['PR', 'VI10', 'VI20', 'VI30']:
        mask = df['Region'] == region
        ax2.scatter(coords[mask, 0], coords[mask, 1], 
                   label=region, alpha=0.6, s=30)
    ax2.set_title('By Region')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    df = load_data(FILEPATH)
    df = clean_data(df)    

    features = ['Loss_log', 'Ratio', 'LOB_Pure_code', 'Region_code']
    x = prepare_features_for_clustering(df, features=features)
    
    k = find_optimal_k_elbow(x, max_k=10)
    model = fit_kmeans(x, n_clusters=k)    
    df['cluster'] = model.labels_
    
    print_cluster_summary(df)
    plot_clusters(df, x)

if __name__ == '__main__':
    main()