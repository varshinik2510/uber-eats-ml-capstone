import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_DIR = "data"

def segment_restaurants():
    print("[1/5] Loading raw tables for restaurant intelligence...")
    df_restaurants = pd.read_csv(os.path.join(DATA_DIR, "restaurants.csv"))
    df_orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    df_deliveries = pd.read_csv(os.path.join(DATA_DIR, "deliveries.csv"))
    df_reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))

    # -------------------------------------------------------------
    # 1. Feature Engineering: Order Metrics per Restaurant
    # -------------------------------------------------------------
    print("[2/5] Calculating kitchen operational metrics...")
    df_orders['is_cancelled'] = (df_orders['order_status'] == 'Cancelled').astype(int)

    order_metrics = df_orders.groupby('restaurant_id').agg(
        total_orders_received=('order_id', 'count'),
        total_revenue=('order_amount', 'sum'),
        avg_order_value=('order_amount', 'mean'),
        cancellation_rate=('is_cancelled', 'mean')
    ).reset_index()

    order_metrics['total_revenue'] = order_metrics['total_revenue'].round(2)
    order_metrics['avg_order_value'] = order_metrics['avg_order_value'].round(2)
    order_metrics['cancellation_rate'] = order_metrics['cancellation_rate'].round(4)

    # -------------------------------------------------------------
    # 2. Delivery & Prep Performance
    # -------------------------------------------------------------
    # Merge orders with deliveries to link restaurants to prep times
    orders_deliv = df_orders[['order_id', 'restaurant_id']].merge(df_deliveries, on='order_id', how='inner')
    delivery_metrics = orders_deliv.groupby('restaurant_id').agg(
        avg_prep_time_min=('prep_time_minutes', 'mean'),
        avg_delivery_time_min=('delivery_time_minutes', 'mean')
    ).reset_index().round(2)

    # -------------------------------------------------------------
    # 3. Customer Satisfaction Feedback per Restaurant
    # -------------------------------------------------------------
    orders_rev = df_orders[['order_id', 'restaurant_id']].merge(df_reviews, on='order_id', how='inner')
    review_metrics = orders_rev.groupby('restaurant_id').agg(
        avg_customer_rating=('rating', 'mean'),
        review_count=('review_id', 'count')
    ).reset_index().round(2)

    # -------------------------------------------------------------
    # 4. Assemble Restaurant Feature Table
    # -------------------------------------------------------------
    print("[3/5] Merging analytical restaurant feature set...")
    rest_features = df_restaurants[['restaurant_id', 'name', 'category', 'price_range']].copy()
    rest_features = rest_features.merge(order_metrics, on='restaurant_id', how='left')
    rest_features = rest_features.merge(delivery_metrics, on='restaurant_id', how='left')
    rest_features = rest_features.merge(review_metrics, on='restaurant_id', how='left')

    # Fill any null values for inactive restaurants
    rest_features['avg_prep_time_min'] = rest_features['avg_prep_time_min'].fillna(rest_features['avg_prep_time_min'].mean()).round(2)
    rest_features['avg_delivery_time_min'] = rest_features['avg_delivery_time_min'].fillna(rest_features['avg_delivery_time_min'].mean()).round(2)
    rest_features['avg_customer_rating'] = rest_features['avg_customer_rating'].fillna(df_restaurants['average_rating']).round(2)
    rest_features['review_count'] = rest_features['review_count'].fillna(0).astype(int)

    # -------------------------------------------------------------
    # 5. Feature Scaling & Clustering
    # -------------------------------------------------------------
    print("[4/5] Scaling operational metrics and running K-Means...")
    cluster_cols = [
        'total_orders_received',
        'total_revenue',
        'avg_order_value',
        'cancellation_rate',
        'avg_prep_time_min',
        'avg_customer_rating'
    ]
    X = rest_features[cluster_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate K=3
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    rest_features['performance_cluster'] = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, rest_features['performance_cluster'])

    # Profile operational tiers
    profiles = rest_features.groupby('performance_cluster')[cluster_cols].mean().round(2)
    sizes = rest_features['performance_cluster'].value_counts().rename('restaurant_count')
    summary = profiles.join(sizes)

    # Assign operational tier names
    tier_names = {}
    for cid in range(k):
        row = profiles.loc[cid]
        if row['total_revenue'] > profiles['total_revenue'].mean() and row['avg_customer_rating'] >= 4.0:
            tier_names[cid] = "Tier 1: High-Volume Flagship Partners"
        elif row['avg_prep_time_min'] > profiles['avg_prep_time_min'].mean() or row['avg_customer_rating'] < 3.8:
            tier_names[cid] = "Tier 3: Kitchen Bottleneck / High Prep Risk"
        else:
            tier_names[cid] = "Tier 2: Steady Everyday Workhorses"

    rest_features['operational_tier'] = rest_features['performance_cluster'].map(tier_names)
    summary['operational_tier'] = [tier_names[i] for i in summary.index]

    print(f"\nSilhouette Score (K=3): {sil:.4f}")
    print("\n--- Restaurant Operational Tier Profiles ---")
    print(summary[['restaurant_count', 'operational_tier', 'total_revenue', 'avg_prep_time_min', 'avg_customer_rating', 'cancellation_rate']])

    output_path = os.path.join(DATA_DIR, "restaurant_segments.csv")
    rest_features.to_csv(output_path, index=False)
    print(f"\n[5/5] Success! Saved restaurant segments to '{output_path}'.")

if __name__ == "__main__":
    segment_restaurants()