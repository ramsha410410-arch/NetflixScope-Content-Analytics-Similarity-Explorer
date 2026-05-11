# 🎬 NetflixScope — Content Analytics & Similarity Explorer

> An end-to-end data science project performing advanced EDA, text vectorization,
> and content similarity on 8,800+ Netflix titles — with an interactive Streamlit dashboard.

---

## 📌 Project Overview

NetflixScope is a full-pipeline data science project built on the Netflix Movies and TV Shows
dataset. It goes beyond basic EDA to demonstrate production-ready skills including advanced
Pandas transformations, NLP-based content similarity, Sklearn preprocessing pipelines,
and an interactive Streamlit web application.

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Data Manipulation | Pandas (explode, melt, str.split, value_counts) |
| Visualization | Matplotlib, Seaborn |
| NLP / Text Analysis | TfidfVectorizer, Cosine Similarity |
| ML Pipeline | ColumnTransformer, Pipeline, SimpleImputer, StandardScaler, OneHotEncoder |
| Outlier Detection | IQR Statistical Method |
| Web App | Streamlit |
| Environment | Python 3.11, virtualenv |

---

## 🚀 Key Features

- Advanced Pandas: explode() multi-value columns (genres, cast, country), str.split(), melt()
- TF-IDF Similarity Engine: vectorized 8,800+ descriptions, cosine similarity for top-N results
- Sklearn Pipeline: ColumnTransformer with numeric and categorical preprocessing
- IQR Outlier Detection: statistical bounds on movie durations and TV seasons
- Streamlit Dashboard: live filters, similarity search, dynamic genre charts

---

## 📈 Key Metrics

| Metric | Value |
|---|---|
| Total Titles | 8,807 |
| Movies | ~69% |
| TV Shows | ~31% |
| Top Country | United States |
| Top Genre | International Movies |
| Peak Year | 2019 |
| TF-IDF Features | 5,000 |

---

## ⚙️ Setup

git clone https://github.com/yourusername/netflix-scope.git
cd netflix-scope
python -m venv venv
venv\Scripts\activate
pip install pandas matplotlib seaborn scikit-learn streamlit
streamlit run src/app.py

---

## 📦 Dataset

Netflix Movies and TV Shows — Kaggle
8,807 titles · 12 columns
kaggle.com/datasets/shivamb/netflix-shows
