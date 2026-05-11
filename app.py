import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="NetflixScope",
    page_icon="🎬",
    layout="wide"
)

# ─── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "netflix_titles.csv")
    df = pd.read_csv(csv_path)

    df['director'].fillna('Unknown', inplace=True)
    df['cast'].fillna('Unknown', inplace=True)
    df['country'].fillna('Unknown', inplace=True)
    df['rating'].fillna('Unknown', inplace=True)
    df['duration'].fillna('Unknown', inplace=True)

    df = df.dropna(subset=['description'])
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df = df.dropna(subset=['date_added'])

    df['year_added'] = df['date_added'].dt.year.astype(int)
    df['month_added'] = df['date_added'].dt.month.astype(int)
    df = df.reset_index(drop=True)
    return df

df = load_data()

# ─── TF-IDF Setup ──────────────────────────────────────────
@st.cache_resource
def build_tfidf(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = tfidf.fit_transform(df['description'].fillna(''))
    return tfidf, matrix

tfidf, tfidf_matrix = build_tfidf(df)
title_to_idx = pd.Series(df.index, index=df['title'].str.lower())

# ─── Header ────────────────────────────────────────────────
st.markdown("""
    <h1 style='color:#E50914;'>🎬 NetflixScope</h1>
    <p style='color:gray;'>Content Analytics & Similarity Explorer</p>
    <hr>
""", unsafe_allow_html=True)

# ─── Sidebar Filters ───────────────────────────────────────
st.sidebar.header("🔎 Filter Content")

content_type_options = df['type'].unique().tolist()
content_type = st.sidebar.multiselect(
    "Content Type",
    options=content_type_options,
    default=content_type_options
)

rating_options = sorted([str(r) for r in df['rating'].unique() if pd.notna(r) and str(r) != 'Unknown'])
ratings = st.sidebar.multiselect(
    "Rating",
    options=rating_options,
    default=[]
)

year_min = int(df['year_added'].min())
year_max = int(df['year_added'].max())
year_range = st.sidebar.slider(
    "Year Added",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# ─── Apply Filters ─────────────────────────────────────────
filtered = df[df['type'].isin(content_type)].copy()

if ratings:
    filtered = filtered[filtered['rating'].isin(ratings)]

filtered = filtered[
    (filtered['year_added'] >= year_range[0]) &
    (filtered['year_added'] <= year_range[1])
]

# ─── Metrics Row ───────────────────────────────────────────
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Titles", len(filtered))
with col2:
    st.metric("Movies", len(filtered[filtered['type'] == 'Movie']))
with col3:
    st.metric("TV Shows", len(filtered[filtered['type'] == 'TV Show']))
with col4:
    st.metric("Countries", filtered['country'].nunique())

st.markdown("---")

# ─── Filtered Data Table ───────────────────────────────────
st.subheader("📋 Filtered Titles")
display_cols = ['title', 'type', 'listed_in', 'rating', 'year_added', 'country']
filtered_display = filtered[display_cols].reset_index(drop=True)

if len(filtered_display) > 0:
    st.dataframe(filtered_display, use_container_width=True, height=300)
else:
    st.warning("No titles match the selected filters.")

st.markdown("---")

# ─── Similarity Finder ─────────────────────────────────────
st.subheader("🤖 Content Similarity Finder")
search_title = st.text_input(
    "Enter a Netflix title to find similar content:",
    placeholder="e.g., Inception, Breaking Bad"
)

if search_title:
    search_lower = search_title.lower()
    if search_lower in title_to_idx.index:
        idx = title_to_idx[search_lower]
        similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][1:6]

        st.success(f"✅ Top 5 titles similar to **{df.iloc[idx]['title']}**")

        for rank, sim_idx in enumerate(top_indices, 1):
            with st.expander(f"{rank}. {df.iloc[sim_idx]['title']} — Score: {similarities[sim_idx]:.4f}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Type:** {df.iloc[sim_idx]['type']}")
                    st.write(f"**Rating:** {df.iloc[sim_idx]['rating']}")
                    st.write(f"**Year Added:** {df.iloc[sim_idx]['year_added']}")
                with col2:
                    st.write(f"**Genre:** {df.iloc[sim_idx]['listed_in']}")
                    st.write(f"**Country:** {df.iloc[sim_idx]['country']}")
                st.write(f"**Description:** {df.iloc[sim_idx]['description']}")
    else:
        st.error(f"❌ Title '{search_title}' not found. Try another title!")

st.markdown("---")

# ─── Genre Distribution ────────────────────────────────────
st.subheader("🎭 Top Genres in Filtered Data")

if len(filtered) > 0:
    df_genres = filtered[['title', 'listed_in']].copy()
    df_genres['genre'] = df_genres['listed_in'].str.split(',')
    df_genres = df_genres.explode('genre')
    df_genres['genre'] = df_genres['genre'].str.strip()

    top_genres = df_genres['genre'].value_counts().head(10)

    if len(top_genres) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_genres.values, y=top_genres.index, palette='Reds_r', ax=ax)
        ax.set_title('Top 10 Genres in Filtered Data', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Genre', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No genres found in the filtered data.")
else:
    st.warning("No data to display. Adjust filters to see genres.")