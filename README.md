# Uber Eats Operational & Customer Intelligence System

An end-to-end Machine Learning and analytics platform built for food delivery marketplaces. This project integrates unsupervised clustering, natural language processing (NLP), supervised regression, time-series forecasting, and an interactive Streamlit operations dashboard to optimize fleet management, partner kitchen performance, and customer retention.

---

## System Architecture

```text
                    +---------------------------+
                    |    Synthetic Raw Data     |
                    | (Orders, Deliveries, etc) |
                    +-------------+-------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Data Preprocessing & Scaling                  |
+----------+----------------------+--------------------+------------+
           |                      |                    |
           v                      v                    v
  [Customer / Kitchen]     [Customer Feedback]    [Delivery ETA]
     Segmentation                NLP                 Regression
  (K-Means, DBSCAN, PCA)   (TF-IDF + Naive Bayes)  (Linear Reg, RF, GBR)
           |                      |                    |
           +----------------------+--------------------+
                                  |
                                  v
                 +--------------------------------+
                 |    Hourly Demand Forecasting   |
                 |     (ADF Test, 24h ARIMA)      |
                 +----------------+---------------+
                                  |
                                  v
                 +--------------------------------+
                 |   Interactive Streamlit App    |
                 |      & Business Strategy       |
                 +--------------------------------+