import os
import pandas as pd
import numpy as np

DATA_DIR = "data"

def engineer_customer_features():
    print("[1/4] Loading raw CSV datasets...")
    df_customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    df_orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    df_reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))

    # Convert order timestamps to datetime objects
    df_orders['order_timestamp'] = pd.to_datetime(df_orders['order_timestamp'])

    # 1. Behavioral Flags
    df_orders['is_weekend'] = df_orders['order_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    order_hour = df_orders['order_timestamp'].dt.hour
    df_orders['is_late_night'] = ((order_hour >= 22) | (order_hour < 4)).astype(int)

    # 2. Aggregations from Orders
    print("[2/4] Aggregating customer order metrics...")
    order_agg = df_orders.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        total_spending=('order_amount', 'sum'),
        avg_order_value=('order_amount', 'mean'),
        weekend_orders=('is_weekend', 'sum'),
        late_night_orders=('is_late_night', 'sum'),
        first_order=('order_timestamp', 'min'),
        last_order=('order_timestamp', 'max')
    ).reset_index()

    order_agg['weekend_order_ratio'] = (order_agg['weekend_orders'] / order_agg['total_orders']).round(3)
    order_agg['late_night_order_ratio'] = (order_agg['late_night_orders'] / order_agg['total_orders']).round(3)
    order_agg['total_spending'] = order_agg['total_spending'].round(2)
    order_agg['avg_order_value'] = order_agg['avg_order_value'].round(2)

    active_days = (order_agg['last_order'] - order_agg['first_order']).dt.days + 1
    order_agg['ordering_frequency_per_month'] = ((order_agg['total_orders'] / active_days) * 30.0).round(2)

    # 3. Aggregations from Reviews
    print("[3/4] Aggregating customer review feedback...")
    review_agg = df_reviews.groupby('customer_id').agg(
        avg_rating_given=('rating', 'mean'),
        total_reviews_written=('review_id', 'count')
    ).reset_index()
    review_agg['avg_rating_given'] = review_agg['avg_rating_given'].round(2)

    # 4. Merge Analytical Profile Table
    print("[4/4] Building final customer analytical table...")
    customer_features = df_customers[['customer_id', 'age', 'customer_tier', 'city_zone']].copy()
    customer_features = customer_features.merge(order_agg, on='customer_id', how='left')
    customer_features = customer_features.merge(review_agg, on='customer_id', how='left')

    customer_features['total_orders'] = customer_features['total_orders'].fillna(0).astype(int)
    customer_features['total_spending'] = customer_features['total_spending'].fillna(0.0)
    customer_features['avg_order_value'] = customer_features['avg_order_value'].fillna(0.0)
    customer_features['weekend_order_ratio'] = customer_features['weekend_order_ratio'].fillna(0.0)
    customer_features['late_night_order_ratio'] = customer_features['late_night_order_ratio'].fillna(0.0)
    customer_features['ordering_frequency_per_month'] = customer_features['ordering_frequency_per_month'].fillna(0.0)
    customer_features['total_reviews_written'] = customer_features['total_reviews_written'].fillna(0).astype(int)

    overall_avg_rating = df_reviews['rating'].mean()
    customer_features['avg_rating_given'] = customer_features['avg_rating_given'].fillna(overall_avg_rating).round(2)

    customer_features.drop(columns=['first_order', 'last_order', 'weekend_orders', 'late_night_orders'], inplace=True)

    output_file = os.path.join(DATA_DIR, "customer_features.csv")
    customer_features.to_csv(output_file, index=False)

    print(f"\n[Success] Generated {len(customer_features)} customer profiles saved to '{output_file}'.")
    print("\nFeature Preview:")
    print(customer_features[['customer_id', 'total_orders', 'total_spending', 'avg_order_value', 'weekend_order_ratio', 'late_night_order_ratio']].head(5))

if __name__ == "__main__":
    engineer_customer_features()
