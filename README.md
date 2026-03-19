# Spotify Data Analytics & Song Recommendation Engine

A full-stack data science application built with Streamlit that combines exploratory data analysis, unsupervised machine learning and content-based recommendations — all in a single interactive dashboard.

## Features

**Data Exploration -** Visualize and explore Spotify audio features (energy, valence, danceability, acousticness) through interactive Plotly charts.

**KMeans Clustering -** Dynamically group songs into clusters based on musical characteristics, with a user-adjustable value of K.

**Song Recommender -** Enter any song name and receive the top-N most similar songs using cosine similarity on normalized audio features.

## Datasets

Three public Spotify datasets are combined -

| Dataset | Description |
|---|---|
| Spotify Wrapped Top 50 Songs 2025 | Top streamed songs of the year |
| Spotify Wrapped Top 50 Artists 2025 | Top artists of the year |
| Spotify All-Time Top 100 Songs | All-time most popular tracks |

**Audio features used -** `danceability`, `energy`, `valence`, `acousticness`

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| UI Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| ML | Scikit-learn (KMeans, Cosine Similarity) |

## Machine Learning Pipeline

### 1. Preprocessing
- Multiple datasets merged with `pandas.concat()`.
- Column names standardized; duplicates removed.
- Missing values filled using column medians.

### 2. Feature Scaling (MinMaxScaler)
All features normalized to [0, 1] to ensure equal contribution in distance-based algorithms.

### 3. Clustering (KMeans)
Songs are grouped by audio similarity. The user selects K at runtime. The algorithm iteratively assigns songs to the nearest centroid and refines cluster centers until convergence.

### 4. Recommendation (Cosine Similarity)
Cosine similarity measures directional alignment between feature vectors, making it well-suited for normalized data. Given a song, the engine ranks all others by similarity and returns the top N.

## Getting Started

**Install dependencies -**
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

**Run the app -**
```bash
streamlit run app.py
```

**Running in Google Colab (via ngrok) -**
```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
!streamlit run app.py &
public_url = ngrok.connect(8501)
print(public_url)
```
## Known Issues & Fixes

| Issue | Fix |
|---|---|
| Duplicate columns after merge | `df = df.loc[:, ~df.columns.duplicated()]` |
| NaN values in feature columns | `df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())` |
| Inconsistent column names across datasets | Standardized during preprocessing |

## Roadmap

- Mood-based filtering (happy, sad, energetic).
- Live Spotify API integration.
- PCA and heatmap visualizations.
- Spotify-style dark theme.
- Deployment to Streamlit Cloud.
