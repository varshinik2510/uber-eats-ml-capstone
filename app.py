import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Page layout & styling
st.set_page_config(
    page_title="Uber Eats ML & Operations Portal",
    page_icon="🍔",
    layout="wide"
)

DATA_DIR = "data"
FIGURES_DIR = os.path.join("reports", "figures")

# -------------------------------------------------------------
# Caching Data & Lightweight Models
# -------------------------------------------------------------
@st.cache_data
def load_data():
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    deliveries = pd.read_csv(os.path.join(DATA_DIR, "deliveries.csv"))
    reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
    return orders, deliveries, reviews

@st.cache_resource
def train_live_eta_model():
    df = pd.read_csv(os.path.join(DATA_DIR, "eta_features.csv"))
    target = 'delivery_time_minutes'
    y = df[target]

    numeric_features = ['prep_time_minutes', 'distance_km', 'item_count', 'order_hour', 'is_peak_hour', 'is_weekend']
    categorical_features = ['traffic_density', 'weather_condition', 'vehicle_type']

    numeric_cols = [c for c in numeric_features if c in df.columns]
    cat_cols = [c for c in categorical_features if c in df.columns]

    X = df[numeric_cols + cat_cols].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    pipeline.fit(X, y)
    return pipeline, numeric_cols, cat_cols

@st.cache_resource
def train_live_sentiment_model():
    df = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    df['cleaned'] = df['review_text'].apply(clean_text)
    df = df[df['cleaned'] != ""].copy()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ])
    pipeline.fit(df['cleaned'], df['sentiment_label'])
    return pipeline

# Load assets
df_orders, df_deliveries, df_reviews = load_data()

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.title("🍔 Uber Eats Portal")
st.sidebar.markdown("**Machine Learning Operations Hub**")
view = st.sidebar.radio(
    "Select Module:",
    [
        "1. Executive Overview",
        "2. Customer & Kitchen Clusters",
        "3. Live ETA Predictor",
        "4. Customer Sentiment NLP"
    ]
)

# -------------------------------------------------------------
# View 1: Executive Overview
# -------------------------------------------------------------
if view == "1. Executive Overview":
    st.title("📊 Operations & Fleet Performance")
    st.markdown("Central analytics platform tracking order volume, delivery duration, and customer feedback across operations.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{len(df_orders):,}")
    c2.metric("Completed Deliveries", f"{len(df_deliveries):,}")
    c3.metric("Customer Reviews", f"{len(df_reviews):,}")
    
    deliv_time_col = [c for c in ['delivery_time_minutes', 'delivery_time'] if c in df_deliveries.columns][0]
    c4.metric("Avg Delivery Duration", f"{df_deliveries[deliv_time_col].mean():.1f} min")

    st.markdown("---")
    st.subheader("Recent Deliveries Log")
    st.dataframe(df_deliveries.head(15), use_container_width=True)

# -------------------------------------------------------------
# View 2: Customer & Kitchen Clusters
# -------------------------------------------------------------
elif view == "2. Customer & Kitchen Clusters":
    st.title("🎯 Segmentation & Anomaly Diagnostics (Day 2)")
    st.markdown("Unsupervised behavioral clustering via K-Means, DBSCAN anomaly detection, and operational tiering.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("K-Means Personas (PCA Projection)")
        kmeans_img = os.path.join(FIGURES_DIR, "kmeans_customer_segments_pca.png")
        if os.path.exists(kmeans_img):
            st.image(kmeans_img, use_container_width=True)
        else:
            st.warning("Figure `kmeans_customer_segments_pca.png` not found.")

    with col2:
        st.subheader("DBSCAN Anomaly Detection")
        dbscan_img = os.path.join(FIGURES_DIR, "dbscan_customer_anomalies.png")
        if os.path.exists(dbscan_img):
            st.image(dbscan_img, use_container_width=True)
        else:
            st.warning("Figure `dbscan_customer_anomalies.png` not found.")

    st.markdown("---")
    st.subheader("Restaurant Operational Tiers")
    rest_img = os.path.join(FIGURES_DIR, "restaurant_operational_tiers.png")
    if os.path.exists(rest_img):
        st.image(rest_img, use_container_width=True)

# -------------------------------------------------------------
# View 3: Live ETA Predictor
# -------------------------------------------------------------
elif view == "3. Live ETA Predictor":
    st.title("⏱️ Real-Time Delivery ETA Regressor (Day 3)")
    st.markdown("Predict door-to-door delivery duration based on trip variables, dispatch metrics, and live traffic.")

    eta_pipeline, num_cols, cat_cols = train_live_eta_model()

    with st.form("eta_prediction_form"):
        st.subheader("Trip & Dispatch Parameters")
        c1, c2 = st.columns(2)
        with c1:
            prep_time = st.slider("Kitchen Prep Duration (min)", 5, 50, 18)
            distance = st.slider("Transit Distance (km)", 0.5, 25.0, 4.5, step=0.1)
            item_count = st.number_input("Basket Size (Items)", min_value=1, max_value=20, value=3)
        with c2:
            traffic = st.selectbox("Current Traffic Level", ["Low", "Medium", "High", "Jam"])
            weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Fog", "Storm"])
            vehicle = st.selectbox("Courier Transport Mode", ["Scooter", "Bicycle", "Car"])
            order_hour = st.slider("Order Hour (0 - 23)", 0, 23, 19)

        is_peak = 1 if order_hour in [11, 12, 13, 14, 18, 19, 20, 21] else 0
        is_weekend = st.checkbox("Weekend Order")

        predict_btn = st.form_submit_button("Compute Estimated Delivery ETA")

    if predict_btn:
        sample_dict = {
            'prep_time_minutes': prep_time,
            'distance_km': distance,
            'item_count': item_count,
            'order_hour': order_hour,
            'is_peak_hour': is_peak,
            'is_weekend': int(is_weekend),
            'traffic_density': traffic,
            'weather_condition': weather,
            'vehicle_type': vehicle
        }
        input_row = pd.DataFrame([sample_dict])
        predicted_eta = max(10.0, eta_pipeline.predict(input_row)[0])
        
        st.success(f"### Predicted Total Delivery Time: **{predicted_eta:.1f} minutes**")

    st.markdown("---")
    st.subheader("Model Diagnostic Residuals")
    c_diag1, c_diag2 = st.columns(2)
    with c_diag1:
        res_plot = os.path.join(FIGURES_DIR, "eta_actual_vs_predicted.png")
        if os.path.exists(res_plot):
            st.image(res_plot, caption="Actual vs. Predicted Delivery Time", use_container_width=True)
    with c_diag2:
        dist_plot = os.path.join(FIGURES_DIR, "eta_residuals_distribution.png")
        if os.path.exists(dist_plot):
            st.image(dist_plot, caption="Residual Error Distribution", use_container_width=True)

# -------------------------------------------------------------
# View 4: Customer Sentiment NLP
# -------------------------------------------------------------
elif view == "4. Customer Sentiment NLP":
    st.title("💬 Review Sentiment Classification (Day 4)")
    st.markdown("Real-time sentiment analyzer identifying satisfaction or operational grievances using TF-IDF and Naive Bayes.")

    nlp_pipeline = train_live_sentiment_model()

    user_review = st.text_area(
        "Enter Customer Feedback:",
        value="The food arrived steaming hot and ten minutes earlier than expected! Awesome service.",
        height=100
    )

    if st.button("Classify Sentiment"):
        if user_review.strip():
            clean_str = re.sub(r'[^a-zA-Z\s]', '', user_review.lower()).strip()
            pred = nlp_pipeline.predict([clean_str])[0]

            if pred == "Positive":
                st.success(f"Sentiment: **{pred}** 🌟")
            elif pred == "Neutral":
                st.info(f"Sentiment: **{pred}** ⚖️")
            else:
                st.error(f"Sentiment: **{pred}** ⚠️")
        else:
            st.warning("Please provide feedback text to analyze.")

    st.markdown("---")
    st.subheader("Evaluation Diagnostics")
    c_nlp1, c_nlp2 = st.columns(2)
    with c_nlp1:
        cm_path = os.path.join(FIGURES_DIR, "sentiment_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="NLP Confusion Matrix", use_container_width=True)
    with c_nlp2:
        class_path = os.path.join(FIGURES_DIR, "sentiment_class_distribution.png")
        if os.path.exists(class_path):
            st.image(class_path, caption="Sentiment Class Balance", use_container_width=True)