import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "data"

def train_eta_models():
    print("[1/4] Loading ETA feature dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, "eta_features.csv"))

    target_col = 'delivery_time_minutes'
    y = df[target_col]
    
    # Feature groups
    numeric_features = ['prep_time_minutes', 'distance_km', 'item_count', 'order_hour', 'is_peak_hour', 'is_weekend']
    categorical_features = ['traffic_density', 'weather_condition', 'vehicle_type']

    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    X = df[numeric_features + categorical_features].copy()

    # -------------------------------------------------------------
    # 1. Train-Test Split (80% Train, 20% Test)
    # -------------------------------------------------------------
    print("[2/4] Splitting train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------------------------------------------------------------
    # 2. Preprocessing Pipeline (Scaling & One-Hot Encoding)
    # -------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    # -------------------------------------------------------------
    # 3. Model Definition and Benchmarking
    # -------------------------------------------------------------
    models = {
        'Linear Regression (Baseline)': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []

    print("\n[3/4] Training and evaluating regression architectures...\n")
    print(f"{'Model Architecture':<30} | {'MAE (min)':<10} | {'RMSE (min)':<10} | {'R2 Score':<10}")
    print("-" * 70)

    best_model_name = None
    best_r2 = -float('inf')

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            'model': name,
            'mae': round(mae, 3),
            'rmse': round(rmse, 3),
            'r2_score': round(r2, 4)
        })

        print(f"{name:<30} | {mae:<10.3f} | {rmse:<10.3f} | {r2:<10.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    print(f"\nOptimal Model: {best_model_name} (R2 = {best_r2:.4f})")

    # -------------------------------------------------------------
    # 4. Save Performance Summary
    # -------------------------------------------------------------
    results_df = pd.DataFrame(results)
    output_path = os.path.join(DATA_DIR, "eta_model_performance.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n[4/4] Model performance summary saved to '{output_path}'.")

if __name__ == "__main__":
    train_eta_models()