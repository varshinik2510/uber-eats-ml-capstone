import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

DATA_DIR = "data"

def run_customer_dbscan():
    print("[1/4] Loading customer features...")
    df = pd.read_csv(os.path.join(DATA_DIR, "customer_features.csv"))
    
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

    # 1. Standardize features
    print("[2/4] Normalizing feature space...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Fit DBSCAN
    # eps: maximum distance between two samples for one to be considered as in the neighborhood of the other
    # min_samples: number of samples in a neighborhood for a point to be considered a core point
    print("[3/4] Fitting DBSCAN (eps=1.35, min_samples=6)...")
    dbscan = DBSCAN(eps=1.35, min_samples=6)
    labels = dbscan.fit_predict(X_scaled)
    df['dbscan_cluster'] = labels

    # Count clusters and noise points (-1 indicates noise/outliers)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\nDiscovered Clusters: {n_clusters}")
    print(f"Identified Outliers / Noise Points: {n_noise} ({n_noise / len(df):.1%} of customer base)")

    if n_clusters > 1:
        # Calculate silhouette score only on non-noise points
        non_noise_mask = labels != -1
        if sum(non_noise_mask) > n_clusters:
            sil_score = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
            print(f"DBSCAN Silhouette Score (excluding noise): {sil_score:.4f}")

    # 3. Analyze Outliers vs. Core Clusters
    print("\n[4/4] Comparing Core Segments vs. Behavioral Outliers...")
    comparison = df.groupby('dbscan_cluster')[feature_cols].mean().round(2)
    counts = df['dbscan_cluster'].value_counts().rename('customer_count')
    summary = comparison.join(counts)

    print("\n--- DBSCAN Cluster Summary (Cluster -1 = Anomalies) ---")
    print(summary[['customer_count', 'total_spending', 'avg_order_value', 'total_orders', 'weekend_order_ratio', 'late_night_order_ratio']])

    output_path = os.path.join(DATA_DIR, "customer_segments_dbscan.csv")
    df.to_csv(output_path, index=False)
    print(f"\n[Success] DBSCAN results saved to '{output_path}'.")

if __name__ == "__main__":
    run_customer_dbscan()