import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

DATA_DIR = "data"
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def visualize_sentiment_performance():
    print("[1/3] Loading reviews dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
    df['clean_review'] = df['review_text'].apply(clean_text)
    df = df[df['clean_review'] != ""].copy()

    X = df['clean_review']
    y = df['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    # 1. Confusion Matrix
    print("[2/3] Generating Confusion Matrix visualization...")
    labels = ['Negative', 'Neutral', 'Positive']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("NLP Sentiment Model: Confusion Matrix (Multinomial NB)", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Predicted Sentiment Label", fontsize=11)
    plt.ylabel("Actual Ground Truth Label", fontsize=11)
    plt.tight_layout()

    cm_path = os.path.join(FIGURES_DIR, "sentiment_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f" -> Saved: {cm_path}")

    # 2. Class Distribution Plot
    print("[3/3] Generating Sentiment Class Distribution plot...")
    plt.figure(figsize=(8, 5))
    palette = {'Positive': '#2E86AB', 'Neutral': '#F18F01', 'Negative': '#C73E1D'}
    sns.countplot(data=df, x='sentiment_label', palette=palette, order=labels)
    plt.title("Customer Feedback Sentiment Class Distribution", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Sentiment Category", fontsize=11)
    plt.ylabel("Review Count", fontsize=11)
    plt.tight_layout()

    dist_path = os.path.join(FIGURES_DIR, "sentiment_class_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f" -> Saved: {dist_path}")

    print("\n[Done] All Day 4 NLP sentiment figures generated successfully.")

if __name__ == "__main__":
    visualize_sentiment_performance()