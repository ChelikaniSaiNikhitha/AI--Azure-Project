import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('movies.csv')
df['description'] = df['description'].fillna('')

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend function
def recommend(movie):
    idx = df[df['title'] == movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Choose a movie:", df['title'].values)

if st.button("Get Recommendations"):
    recommendations = recommend(selected_movie)
    st.write("**Top 5 Similar Movies:**")
    for movie in recommendations:
        st.write("👉", movie)
