import pandas as pd
import numpy as np
import pickle
import ast

print("📥 Loading datasets...")

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print("✅ Datasets loaded")

# Merge datasets
movies = movies.merge(credits, on='title')
print("🔗 Datasets merged")

# Select important columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print("📊 Selected features")

# Remove null values
movies.dropna(inplace=True)
print("🧹 Null values removed")

# Convert JSON strings → list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Take top 3 cast
def convert_cast(text):
    L = []
    for i in ast.literal_eval(text)[:3]:
        L.append(i['name'])
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# Fetch director
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

print("🎭 Extracted genres, cast, crew")

# Remove spaces
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

print("🧠 Tags created")

# Final dataset
new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

print("✅ Final dataset ready")

# ✅ SAVE PROCESSED CSV (IMPORTANT FOR YOUR MAM)
new_df.to_csv('processed_movies.csv', index=False)
print("💾 processed_movies.csv saved")

# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()
print("🔢 Text vectorized")

# Similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

print("📐 Similarity calculated")

# Save PKL files
pickle.dump(new_df, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))

print("🎉 All files saved successfully!")