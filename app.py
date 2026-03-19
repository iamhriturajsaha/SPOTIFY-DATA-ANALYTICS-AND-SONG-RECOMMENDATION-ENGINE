import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
st.set_page_config(page_title="Spotify Recommender", layout="wide")
st.title("🎵 Spotify Intelligence Dashboard")

# LOAD DATA
@st.cache_data
def load_data():
    songs = pd.read_csv("Spotify Wrapped Top 50 Songs 2025.csv")
    artists = pd.read_csv("Spotify Wrapped Top 50 Artists 2025.csv")
    alltime = pd.read_csv("Spotify Alltime Top 100 Songs.csv")
    # Add dataset labels
    songs["dataset"] = "songs"
    artists["dataset"] = "artists"
    alltime["dataset"] = "alltime"
    df = pd.concat([songs, artists, alltime], ignore_index=True)
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    return df
df = load_data()

# HANDLE COLUMN NAME VARIANTS
rename_map = {
    'track_name': 'song_title',
    'track': 'song_title',
    'artist_name': 'artist'
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
df = df.loc[:, ~df.columns.duplicated()]

# USE ONLY SONG DATA FOR ML
if 'dataset' in df.columns:
    df_model = df[df['dataset'] == 'songs'].copy()
else:
    df_model = df.copy()

# FEATURE ENGINEERING
FEATURE_COLS = ['danceability', 'energy', 'valence', 'acousticness']
FEATURE_COLS = [col for col in FEATURE_COLS if col in df_model.columns]
if len(FEATURE_COLS) == 0:
    st.error("No valid feature columns found")
    st.stop()
# Convert to numeric
for col in FEATURE_COLS:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
# Fill missing values
df_model[FEATURE_COLS] = df_model[FEATURE_COLS].fillna(df_model[FEATURE_COLS].median())
# Final safety check
if df_model[FEATURE_COLS].isna().sum().sum() > 0:
    st.error("NaN values still present!")
    st.stop()
# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_model[FEATURE_COLS])

# SIMILARITY
similarity_matrix = cosine_similarity(X_scaled)
def recommend_songs(song_title, n=5):
    if 'song_title' not in df_model.columns:
        return None
    matches = df_model[df_model['song_title'].astype(str).str.lower().str.contains(song_title.lower())]
    if matches.empty:
        return None
    idx = matches.index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    recs = []
    for i, score in scores:
        recs.append({
            "Song": df_model.iloc[i].get('song_title', 'Unknown'),
            "Artist": df_model.iloc[i].get('artist', 'Unknown'),
            "Similarity": round(score, 4)
        })
    return pd.DataFrame(recs)

# SIDEBAR
st.sidebar.header("🔍 Controls")
song_input = st.sidebar.text_input("Enter Song Name", "Blinding Lights")
num_recs = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

# TABS
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Clustering", "🎵 Recommender"])

# TAB 1 — OVERVIEW
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    if 'energy' in df.columns:
        st.subheader("Feature Distribution")
        fig = px.histogram(df, x="energy")
        st.plotly_chart(fig, use_container_width=True)

# TAB 2 — CLUSTERING
with tab2:
    st.subheader("KMeans Clustering")
    k = st.slider("Select K", 2, 10, 5)
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_model['cluster'] = kmeans.fit_predict(X_scaled)
        if 'energy' in df_model.columns and 'valence' in df_model.columns:
            fig = px.scatter(
                df_model,
                x="energy",
                y="valence",
                color=df_model['cluster'].astype(str),
                hover_name=df_model.get("song_title", None)
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# TAB 3 — RECOMMENDER
with tab3:
    st.subheader("🎵 Song Recommendation System")
    if st.button("Recommend"):
        recs = recommend_songs(song_input, num_recs)
        if recs is None or recs.empty:
            st.error("Song not found")
        else:
            st.success(f"Top {num_recs} recommendations:")
            st.dataframe(recs)
            fig = px.bar(recs, x="Similarity", y="Song", orientation='h')
            st.plotly_chart(fig, use_container_width=True)
