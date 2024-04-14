import streamlit as st
import pandas as pd
import requests
import pickle
from io import BytesIO

# Define MovieRecommender class
class MovieRecommender:
    def __init__(self, content_df, cosine_similarity_matrix):
        self.content_df = content_df
        self.cosine_similarity_matrix = cosine_similarity_matrix

    def predict(self, movie_id):
        # Add your recommendation logic here
        pass

# Define function to load data from GitHub
def load_data_from_github(file_url):
    response = requests.get(file_url)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to load data from {file_url}")
        return None

# Load content DataFrame from GitHub
content_df_url = 'https://raw.githubusercontent.com/ramansidhuu7/ui/main/content_df.csv'
content_df_content = load_data_from_github(content_df_url)
if content_df_content:
    content_df = pd.read_csv(BytesIO(content_df_content))
else:
    content_df = pd.DataFrame()  # Placeholder DataFrame

# Load cosine similarity matrix from GitHub
cosine_similarity_url = 'https://raw.githubusercontent.com/ramansidhuu7/ui/main/cosine_similarity.pkl'
cosine_similarity_content = load_data_from_github(cosine_similarity_url)
if cosine_similarity_content:
    with BytesIO(cosine_similarity_content) as bio:
        cosine_similarity_matrix = pickle.load(bio)
else:
    cosine_similarity_matrix = None  # Placeholder value

# Load predict function from GitHub
predict_function_url = 'https://raw.githubusercontent.com/ramansidhuu7/ui/main/predict_function.pkl'
predict_function_content = load_data_from_github(predict_function_url)
if predict_function_content:
    with BytesIO(predict_function_content) as bio:
        # Load MovieRecommender instance
        movie_recommender = pickle.load(bio)
else:
    movie_recommender = None  # Placeholder value

# Streamlit app code
st.title("Movie Recommendation App")

# Display loaded data (for demonstration purposes)
st.subheader("Content DataFrame:")
st.write(content_df.head())

st.subheader("Cosine Similarity Matrix:")
st.write(cosine_similarity_matrix)

# Check if MovieRecommender is loaded
if movie_recommender:
    st.subheader("Movie Recommender Predict Function:")
    st.write("Movie Recommender loaded successfully.")
else:
    st.error("Failed to load Movie Recommender.")

