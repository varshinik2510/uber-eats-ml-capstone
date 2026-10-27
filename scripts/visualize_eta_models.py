import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

DATA_DIR = "data"
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def visualize_eta_performance():
    print("[1/3] Loading ETA dataset and refitting optimal model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "eta_features.csv"))

    target_col = 'delivery_time_minutes'
    y = df[target_col]

    numeric_features = ['prep_time_minutes', 'distance_km', 'item_count', 'order_hour', 'is_peak_hour', 'is_weekend']
    categorical_features = ['traffic_density', 'weather_condition', 'vehicle_type']

    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    X = df[numeric_features + categorical_features].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    residuals = y_test - y_pred

    # -------------------------------------------------------------
    # 1. Actual vs. Predicted Plot
    # -------------------------------------------------------------
    print("[2/3] Generating Actual vs. Predicted Delivery Time plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, color='#2E86AB', edgecolors='k', linewidth=0.3)
    
    # Ideal prediction line y = x
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (y = x)')

    plt.title("ETA Prediction: Actual vs. Predicted Delivery Time (Test Set)", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Actual Delivery Time (minutes)", fontsize=11)
    plt.ylabel("Predicted Delivery Time (minutes)", fontsize=11)
    plt.legend(loc='upper left')
    plt.tight_layout()

    actual_vs_pred_path = os.path.join(FIGURES_DIR, "eta_actual_vs_predicted.png")
    plt.savefig(actual_vs_pred_path, dpi=300)
    plt.close()
    print(f" -> Saved: {actual_vs_pred_path}")

    # -------------------------------------------------------------
    # 2. Residual Distribution Plot
    # -------------------------------------------------------------
    print("[3/3] Generating Residuals Distribution plot...")
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='#A23B72', bins=30)
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')

    plt.title("ETA Model Residuals Distribution (Errors)", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Residual Error = Actual - Predicted (minutes)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()

    residuals_path = os.path.join(FIGURES_DIR, "eta_residuals_distribution.png")
    plt.savefig(residuals_path, dpi=300)
    plt.close()
    print(f" -> Saved: {residuals_path}")

    print("\n[Done] All Day 3 ETA diagnostic charts successfully generated.")

if __name__ == "__main__":
    visualize_eta_performance()