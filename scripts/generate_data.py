import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# 1. Reproducibility seed: ensures identical data every run
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. CUSTOMERS
# -------------------------------------------------------------
def generate_customers(num_records=1000):
    customers = []
    tiers = ['Bronze', 'Silver', 'Gold', 'UberOne']
    tier_weights = [0.45, 0.30, 0.15, 0.10]
    
    for i in range(1, num_records + 1):
        c_id = f"CUST_{i:04d}"
        signup_date = fake.date_between(start_date="-2y", end_date="-60d")
        age = int(np.clip(np.random.normal(35, 10), 18, 70))
        
        customers.append({
            "customer_id": c_id,
            "name": fake.name(),
            "email": fake.email(),
            "age": age,
            "customer_tier": np.random.choice(tiers, p=tier_weights),
            "city_zone": np.random.choice(['Downtown', 'Midtown', 'Uptown', 'Suburbs', 'University Zone']),
            "signup_date": signup_date
        })
    return pd.DataFrame(customers)

# -------------------------------------------------------------
# 2. RESTAURANTS
# -------------------------------------------------------------
def generate_restaurants(num_records=100):
    categories = ['Fast Food', 'Italian', 'Indian', 'Japanese', 'Mexican', 'American', 'Healthy / Vegan', 'Desserts']
    restaurants = []
    
    for i in range(1, num_records + 1):
        r_id = f"REST_{i:03d}"
        category = np.random.choice(categories)
        avg_rating = round(np.random.uniform(3.2, 4.9), 2)
        base_prep = {
            'Fast Food': 12, 'Desserts': 10, 'American': 20,
            'Mexican': 18, 'Japanese': 22, 'Italian': 25,
            'Indian': 28, 'Healthy / Vegan': 16
        }[category]
        
        restaurants.append({
            "restaurant_id": r_id,
            "name": f"{fake.company()} Kitchen",
            "category": category,
            "city_zone": np.random.choice(['Downtown', 'Midtown', 'Uptown', 'Suburbs', 'University Zone']),
            "average_rating": avg_rating,
            "price_range": np.random.choice(['$', '$$', '$$$'], p=[0.4, 0.45, 0.15]),
            "base_prep_time_min": base_prep
        })
    return pd.DataFrame(restaurants)

# -------------------------------------------------------------
# 3. DRIVERS
# -------------------------------------------------------------
def generate_drivers(num_records=250):
    drivers = []
    vehicles = ['Bicycle', 'Scooter', 'Motorcycle', 'Car']
    vehicle_weights = [0.15, 0.35, 0.35, 0.15]
    
    for i in range(1, num_records + 1):
        d_id = f"DRIV_{i:04d}"
        drivers.append({
            "driver_id": d_id,
            "name": fake.name(),
            "vehicle_type": np.random.choice(vehicles, p=vehicle_weights),
            "rating": round(np.random.uniform(4.2, 5.0), 2),
            "total_completed_trips": int(np.random.exponential(scale=300) + 10),
            "active_status": np.random.choice(['Active', 'Inactive'], p=[0.9, 0.1])
        })
    return pd.DataFrame(drivers)

# -------------------------------------------------------------
# 4. ORDERS, DELIVERIES, PAYMENTS, REVIEWS
# -------------------------------------------------------------
def generate_orders_and_related(customers_df, restaurants_df, drivers_df, num_orders=10000):
    orders, deliveries, payments, reviews = [], [], [], []
    
    start_date = datetime(2025, 1, 1)
    customer_ids = customers_df['customer_id'].values
    restaurant_dict = restaurants_df.set_index('restaurant_id').to_dict('index')
    driver_ids = drivers_df[drivers_df['active_status'] == 'Active']['driver_id'].values
    
    positive_reviews = [
        "Food was hot and arrived super fast! Excellent service.",
        "Delicious meal, driver followed instructions perfectly.",
        "Packaged neatly and arrived earlier than expected.",
        "Best burger in town, driver was very polite.",
        "Great portion sizes and fresh ingredients. Will order again!"
    ]
    negative_reviews = [
        "Delivery took over an hour! Food was completely cold.",
        "Missing drinks and the bag was crushed. Unacceptable.",
        "Driver was rude and dropped food at the wrong building.",
        "Order arrived late and soup spilled everywhere.",
        "Terrible experience. Cold pizza and incorrect order."
    ]
    neutral_reviews = [
        "Average meal, delivery was on time.",
        "Food was okay, nothing special. Driver was fine.",
        "Standard delivery experience. Reasonable prep time."
    ]

    # Hourly distribution normalized to sum to 1.0
    hour_probs = np.array([
        0.01, 0.01, 0.005, 0.005, 0.005, 0.01, 0.02, 0.03, 0.04, 0.04, 0.05, 0.07,
        0.09, 0.08, 0.05, 0.04, 0.04, 0.06, 0.10, 0.11, 0.08, 0.05, 0.03, 0.015
    ])
    hour_probs = hour_probs / hour_probs.sum()

    for i in range(1, num_orders + 1):
        order_id = f"ORD_{i:06d}"
        cust_id = np.random.choice(customer_ids)
        rest_id = np.random.choice(list(restaurant_dict.keys()))
        rest_info = restaurant_dict[rest_id]
        
        # Real-world peak order curve (lunch 12-14, dinner 18-21)
        hour = int(np.random.choice(range(24), p=hour_probs))
        day_offset = int(np.random.randint(0, 365))
        minute = int(np.random.randint(0, 60))
        second = int(np.random.randint(0, 60))
        order_time = start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
        
        num_items = int(np.random.choice([1, 2, 3, 4, 5, 6], p=[0.25, 0.35, 0.20, 0.10, 0.06, 0.04]))
        base_item_price = {'$': 10, '$$': 18, '$$$': 32}[rest_info['price_range']]
        order_amount = round(num_items * base_item_price * np.random.uniform(0.85, 1.25), 2)
        
        status = np.random.choice(['Delivered', 'Cancelled', 'Refunded'], p=[0.94, 0.04, 0.02])
        
        orders.append({
            "order_id": order_id,
            "customer_id": cust_id,
            "restaurant_id": rest_id,
            "order_timestamp": order_time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_items": num_items,
            "order_amount": order_amount,
            "order_status": status
        })
        
        payments.append({
            "payment_id": f"PAY_{i:06d}",
            "order_id": order_id,
            "payment_method": np.random.choice(['Credit Card', 'Debit Card', 'Uber Cash', 'Apple Pay', 'PayPal']),
            "payment_status": "Successful" if status != 'Cancelled' else "Refunded",
            "transaction_timestamp": order_time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        if status == 'Delivered':
            delivery_id = f"DEL_{i:06d}"
            driver_id = np.random.choice(driver_ids)
            distance_km = round(np.random.exponential(scale=3.5) + 0.8, 2)
            weather = np.random.choice(['Clear', 'Rainy', 'Cloudy', 'Snowy', 'Windy'], p=[0.60, 0.20, 0.12, 0.05, 0.03])
            traffic = np.random.choice(['Low', 'Medium', 'High', 'Congested'], p=[0.25, 0.40, 0.25, 0.10])
            
            prep_time = max(8, int(np.random.normal(rest_info['base_prep_time_min'], 4)))
            traffic_multiplier = {'Low': 1.0, 'Medium': 1.25, 'High': 1.6, 'Congested': 2.0}[traffic]
            weather_delay = {'Clear': 0, 'Cloudy': 1, 'Windy': 3, 'Rainy': 6, 'Snowy': 12}[weather]
            transit_speed_kmh = 24.0 / traffic_multiplier
            transit_time = (distance_km / transit_speed_kmh) * 60.0 + weather_delay
            delivery_time_min = round(prep_time + transit_time, 1)
            
            tip_percent = np.random.choice([0.0, 0.10, 0.15, 0.18, 0.20, 0.25], p=[0.20, 0.20, 0.30, 0.15, 0.10, 0.05])
            tip_amount = round(order_amount * tip_percent, 2) if delivery_time_min < 50 else round(order_amount * (tip_percent * 0.5), 2)
            
            deliveries.append({
                "delivery_id": delivery_id,
                "order_id": order_id,
                "driver_id": driver_id,
                "delivery_distance_km": distance_km,
                "prep_time_minutes": prep_time,
                "delivery_time_minutes": delivery_time_min,
                "weather": weather,
                "traffic_condition": traffic,
                "tip_amount": tip_amount
            })
            
            if np.random.rand() < 0.40:
                rev_id = f"REV_{len(reviews)+1:06d}"
                if delivery_time_min > 45 or prep_time > 35:
                    rev_text = np.random.choice(negative_reviews)
                    rev_rating = np.random.choice([1, 2], p=[0.6, 0.4])
                    sentiment = "Negative"
                elif delivery_time_min < 30:
                    rev_text = np.random.choice(positive_reviews)
                    rev_rating = np.random.choice([4, 5], p=[0.3, 0.7])
                    sentiment = "Positive"
                else:
                    rev_text = np.random.choice(neutral_reviews)
                    rev_rating = 3
                    sentiment = "Neutral"
                    
                reviews.append({
                    "review_id": rev_id,
                    "order_id": order_id,
                    "customer_id": cust_id,
                    "rating": rev_rating,
                    "review_text": rev_text,
                    "sentiment_label": sentiment
                })

    return pd.DataFrame(orders), pd.DataFrame(deliveries), pd.DataFrame(payments), pd.DataFrame(reviews)

def main():
    print("[1/5] Generating Customers...")
    df_cust = generate_customers(1000)
    df_cust.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
    
    print("[2/5] Generating Restaurants...")
    df_rest = generate_restaurants(100)
    df_rest.to_csv(os.path.join(OUTPUT_DIR, "restaurants.csv"), index=False)
    
    print("[3/5] Generating Drivers...")
    df_driv = generate_drivers(250)
    df_driv.to_csv(os.path.join(OUTPUT_DIR, "drivers.csv"), index=False)
    
    print("[4/5] Generating Orders, Deliveries, Payments, and Reviews...")
    df_ord, df_del, df_pay, df_rev = generate_orders_and_related(df_cust, df_rest, df_driv, num_orders=10000)
    
    df_ord.to_csv(os.path.join(OUTPUT_DIR, "orders.csv"), index=False)
    df_del.to_csv(os.path.join(OUTPUT_DIR, "deliveries.csv"), index=False)
    df_pay.to_csv(os.path.join(OUTPUT_DIR, "payments.csv"), index=False)
    df_rev.to_csv(os.path.join(OUTPUT_DIR, "reviews.csv"), index=False)
    
    print("[5/5] Success! All 7 CSV files generated in data/ folder:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.csv'):
            print(f"  - {f}: {len(pd.read_csv(os.path.join(OUTPUT_DIR, f)))} rows")

if __name__ == "__main__":
    main()