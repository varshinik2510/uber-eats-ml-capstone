import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.pipeline import Pipeline

DATA_DIR = "data"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                                    # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)               # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()               # Remove extra whitespace
    return text

def train_sentiment_models():
    print("[1/4] Loading customer review dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))

    # Clean text
    df['clean_review'] = df['review_text'].apply(clean_text)
    
    # Filter empty reviews
    df = df[df['clean_review'] != ""].copy()

    X = df['clean_review']
    y = df['sentiment_label']

    print(f"Loaded {len(df)} valid customer reviews.")
    print("Class Breakdown:")
    print(y.value_counts(normalize=True).round(3))

    # -------------------------------------------------------------
    # 1. Train-Test Split (80% Train, 20% Test)
    # -------------------------------------------------------------
    print("\n[2/4] Splitting dataset into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------------------------------------
    # 2. Define Model Pipelines with TF-IDF
    # -------------------------------------------------------------
    pipelines = {
        'Multinomial Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))),
            ('clf', MultinomialNB())
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest Classifier': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

    results = []

    print("\n[3/4] Benchmarking NLP Classifiers...\n")
    print(f"{'Classifier Model':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)

    best_model_name = None
    best_f1 = -1.0

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        results.append({
            'model': name,
            'accuracy': round(acc, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        })

        print(f"{name:<28} | {acc:<10.4f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\nOptimal Classifier: {best_model_name} (Weighted F1 = {best_f1:.4f})")

    # -------------------------------------------------------------
    # 3. Save Performance Log
    # -------------------------------------------------------------
    results_df = pd.DataFrame(results)
    output_path = os.path.join(DATA_DIR, "nlp_sentiment_performance.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n[4/4] Sentiment classification benchmark saved to '{output_path}'.")

if __name__ == "__main__":
    train_sentiment_models()