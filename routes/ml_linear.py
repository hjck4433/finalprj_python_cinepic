import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to load vectorized data
def load_vectorized_data():
    # Construct the path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
    data_dir = os.path.join(script_dir, '..', 'data')  # Move up one directory level and then into the 'data' directory
    csv_file_path = os.path.join(data_dir, 'word2vec_results.csv')  # The full path to the CSV file

    # Load the data from the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file_path, encoding='utf-8')

    return data


# 벡터화된 데이터 불러오기
data = load_vectorized_data()


# Function to split data into training and testing sets
def split_data(data, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data


# Split the data
train_data, test_data = split_data(data)


# Example function to generate recommendations based on cosine similarity
def generate_recommendations(user_preferences, movie_features, top_n=10):
    # Calculate similarity scores between user preferences and movie features
    similarity_scores = cosine_similarity(user_preferences, movie_features)

    # Get the top N recommendations
    top_indices = similarity_scores.argsort()[0][-top_n:]
    recommended_movies = movie_features.iloc[top_indices]

    return recommended_movies


# Hypothetical user preferences vector (you will need to replace this with actual data)
user_preferences = np.array([[0.1, 0.2, 0.3, ..., 0.6]])  # Placeholder vector

# Generate recommendations
recommended_movies = generate_recommendations(user_preferences, train_data.drop(columns=['movieId']), top_n=10)

# Display recommended movies
print(recommended_movies)
