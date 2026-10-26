import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_visualizations():
    print("[1/3] Loading segmented analytical datasets...")
    df_kmeans = pd.read_csv(os.path.join(DATA_DIR, "customer_segments_kmeans.csv"))
    df_dbscan = pd.read_csv(os.path.join(DATA_DIR, "customer_segments_dbscan.csv"))
    df_rest = pd.read_csv(os.path.join(DATA_DIR, "restaurant_segments.csv"))

    # -------------------------------------------------------------
    # 1. K-Means Customer Personas (PCA 2D Space)
    # -------------------------------------------------------------
    print("[2/3] Generating K-Means 2D PCA cluster map...")
    plt.figure(figsize=(10, 6))
    palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    sns.scatterplot(
        data=df_kmeans,
        x='pca_1',
        y='pca_2',
        hue='segment_persona',
        palette=palette,
        alpha=0.8,
        s=60,
        edgecolor='black',
        linewidth=0.5
    )
    plt.title("Uber Eats Customer Segmentation: 2D PCA Space (K=4)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Principal Component 1 (Spending & Frequency Variance)", fontsize=11)
    plt.ylabel("Principal Component 2 (Time-of-Day & Weekend Ratio)", fontsize=11)
    plt.legend(title="Customer Personas", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    kmeans_plot_path = os.path.join(FIGURES_DIR, "kmeans_customer_segments_pca.png")
    plt.savefig(kmeans_plot_path, dpi=300)
    plt.close()
    print(f" -> Saved: {kmeans_plot_path}")

    # -------------------------------------------------------------
    # 2. DBSCAN Behavioral Anomalies Map
    # -------------------------------------------------------------
    # Merge PCA coordinates into DBSCAN data for consistent plotting
    df_dbscan['pca_1'] = df_kmeans['pca_1']
    df_dbscan['pca_2'] = df_kmeans['pca_2']
    df_dbscan['cluster_label'] = df_dbscan['dbscan_cluster'].map(
        lambda x: "Behavioral Outlier / Anomaly (-1)" if x == -1 else "Core Customer Density (0)"
    )

    plt.figure(figsize=(10, 6))
    dbscan_palette = {'Core Customer Density (0)': '#3A86FF', 'Behavioral Outlier / Anomaly (-1)': '#E63946'}
    
    sns.scatterplot(
        data=df_dbscan,
        x='pca_1',
        y='pca_2',
        hue='cluster_label',
        palette=dbscan_palette,
        style='cluster_label',
        markers={'Core Customer Density (0)': 'o', 'Behavioral Outlier / Anomaly (-1)': 'X'},
        s=70,
        alpha=0.85
    )
    plt.title("DBSCAN Anomaly Detection: Core Patterns vs. Outliers", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.legend(title="DBSCAN Detection", loc='best')
    plt.tight_layout()

    dbscan_plot_path = os.path.join(FIGURES_DIR, "dbscan_customer_anomalies.png")
    plt.savefig(dbscan_plot_path, dpi=300)
    plt.close()
    print(f" -> Saved: {dbscan_plot_path}")

    # -------------------------------------------------------------
    # 3. Restaurant Operational Tiers Plot
    # -------------------------------------------------------------
    print("[3/3] Generating Restaurant Operational Tiers chart...")
    plt.figure(figsize=(10, 6))
    rest_palette = ['#06D6A0', '#118AB2', '#EF476F']

    sns.scatterplot(
        data=df_rest,
        x='avg_prep_time_min',
        y='total_revenue',
        hue='operational_tier',
        size='total_orders_received',
        sizes=(40, 250),
        palette=rest_palette,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5
    )
    plt.title("Restaurant Operational Tiers: Prep Time vs. Revenue", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Average Kitchen Prep Time (Minutes)", fontsize=11)
    plt.ylabel("Total Lifetime Revenue ($)", fontsize=11)
    plt.legend(title="Operational Tiers & Volume", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    rest_plot_path = os.path.join(FIGURES_DIR, "restaurant_operational_tiers.png")
    plt.savefig(rest_plot_path, dpi=300)
    plt.close()
    print(f" -> Saved: {rest_plot_path}")

    print("\n[Done] All Day 2 diagnostic figures generated successfully.")

if __name__ == "__main__":
    generate_visualizations()