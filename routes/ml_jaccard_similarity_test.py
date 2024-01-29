import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np

# Load the dataset
input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data_okt_v4.csv')
data = pd.read_csv(input_path, encoding='utf-8')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize the CountVectorizer
count_vectorizer = CountVectorizer(binary=True)  # Binary output for Jaccard similarity

# Fit the vectorizer on combined text data from the training set
all_text_data = pd.concat([train_data[column] for column in ['movieDirector', 'movieActors', 'movieGenre']])
count_vectorizer.fit(all_text_data)


# Function to calculate weighted Jaccard similarity for each category
def calculate_weighted_jaccard_similarity(column_name, user_preference, weight, dataset):
    user_pref_vector = count_vectorizer.transform([user_preference]).toarray()
    movie_feature_vectors = count_vectorizer.transform(dataset[column_name]).toarray()

    # Calculate Jaccard similarity for each movie and apply the weight
    jaccard_similarities = [jaccard_score(user_pref_vector[0], movie_vector) * weight for movie_vector in
                            movie_feature_vectors]

    return np.array(jaccard_similarities)


# User preferences
user_preferences = {
    'preferActors': '마동석, 손석구, 키아누 리브스',
    'preferDirectors': '김성수, 이상용',
    'preferGenres': '드라마, 범죄, 액션'
}

# Define weights for each category
weights = {
    'genre_weight': 0.5,
    'actor_weight': 0.3,
    'director_weight': 0.2
}


# Calculate weighted Jaccard similarities for each category using test_data
genre_similarity = calculate_weighted_jaccard_similarity('movieGenre', user_preferences['preferGenres'],
                                                         weights['genre_weight'], test_data)
actor_similarity = calculate_weighted_jaccard_similarity('movieActors', user_preferences['preferActors'],
                                                         weights['actor_weight'], test_data)
director_similarity = calculate_weighted_jaccard_similarity('movieDirector', user_preferences['preferDirectors'],
                                                            weights['director_weight'], test_data)

# Combine weighted similarity scores for total similarity score
total_similarity = genre_similarity + actor_similarity + director_similarity

# Add total similarity scores to the test dataframe and sort by these scores
test_data['similarity_score'] = total_similarity
recommended_movies = test_data.sort_values(by='similarity_score', ascending=False)

# Display top recommended movies from the test set
print(recommended_movies[['movieTitle', 'similarity_score']].head(10))