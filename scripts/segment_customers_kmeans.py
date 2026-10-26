import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

DATA_DIR = "data"

def run_customer_kmeans():
    print("[1/5] Loading customer features...")
    df = pd.read_csv(os.path.join(DATA_DIR, "customer_features.csv"))
    
    # Selected behavioral features for clustering
    feature_cols = [
        'total_orders',
        'total_spending',
        'avg_order_value',
        'weekend_order_ratio',
        'late_night_order_ratio',
        'ordering_frequency_per_month',
        'avg_rating_given'
    ]
    X = df[feature_cols].copy()

    # 1. Feature Scaling
    print("[2/5] Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Optimal K Analysis: Elbow Method & Silhouette Scores
    print("[3/5] Evaluating K from 2 to 8...")
    k_range = range(2, 9)
    inertias = []
    sil_scores = []

    print(f"\n{'K':<4} | {'Inertia (WCSS)':<16} | {'Silhouette Score':<16}")
    print("-" * 42)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        score = silhouette_score(X_scaled, km.labels_)
        sil_scores.append(score)
        print(f"{k:<4} | {km.inertia_:<16.2f} | {score:<16.4f}")

    # Select K=4 for operational business segments
    selected_k = 4
    print(f"\nSelected K = {selected_k} for operational customer segmentation.")

    # 3. Fit Final K-Means Model
    print(f"[4/5] Fitting final K-Means with K={selected_k}...")
    kmeans_final = KMeans(n_clusters=selected_k, random_state=42, n_init=20)
    df['kmeans_cluster'] = kmeans_final.fit_predict(X_scaled)

    # 4. Dimensionality Reduction with PCA for 2D Visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df['pca_1'] = pca_coords[:, 0].round(3)
    df['pca_2'] = pca_coords[:, 1].round(3)
    print(f"PCA Explained Variance: Component 1 = {pca.explained_variance_ratio_[0]:.2%}, Component 2 = {pca.explained_variance_ratio_[1]:.2%}")

    # 5. Cluster Profiling and Labeling
    print("[5/5] Profiling cluster behavioral centers...")
    cluster_profiles = df.groupby('kmeans_cluster')[feature_cols].mean().round(2)
    cluster_sizes = df['kmeans_cluster'].value_counts().rename('customer_count')
    profile_summary = cluster_profiles.join(cluster_sizes)
    
    # Assign business personas based on behavioral centers
    segment_names = {}
    for cid in range(selected_k):
        row = cluster_profiles.loc[cid]
        if row['avg_order_value'] > cluster_profiles['avg_order_value'].mean() and row['total_spending'] > cluster_profiles['total_spending'].mean():
            segment_names[cid] = "High-Value Frequent VIPs"
        elif row['late_night_order_ratio'] > cluster_profiles['late_night_order_ratio'].mean():
            segment_names[cid] = "Late-Night Diners"
        elif row['weekend_order_ratio'] > cluster_profiles['weekend_order_ratio'].mean():
            segment_names[cid] = "Weekend Family / Social Diners"
        else:
            segment_names[cid] = "Value / Occasional Diners"

    df['segment_persona'] = df['kmeans_cluster'].map(segment_names)
    profile_summary['persona'] = [segment_names[i] for i in profile_summary.index]

    print("\n--- Cluster Profile Summary ---")
    print(profile_summary[['customer_count', 'persona', 'total_spending', 'avg_order_value', 'total_orders', 'weekend_order_ratio', 'late_night_order_ratio']])

    output_path = os.path.join(DATA_DIR, "customer_segments_kmeans.csv")
    df.to_csv(output_path, index=False)
    print(f"\n[Success] Segmented customer data saved to '{output_path}'.")

if __name__ == "__main__":
    run_customer_kmeans()