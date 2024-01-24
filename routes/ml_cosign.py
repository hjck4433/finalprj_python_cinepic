import pandas as pd
import os
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def vectorize_text(data, text_columns=['moviePlot_tokens', 'movieDirector_tokens', 'movieActors_tokens']):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Combine the text from multiple columns into a single column
    combined_text = data[text_columns].apply(lambda row: ' '.join(row), axis=1)

    # Fit and transform the TF-IDF vectorizer on the combined text
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)

    return tfidf_matrix

def preprocess_korean_text(data, text_columns=['moviePlot', 'movieDirector', 'movieActors']):
    tokenizer = Okt()

    for column in text_columns:
        data[f'{column}_tokens'] = data[column].apply(lambda x: ' '.join(tokenizer.morphs(x)))

    return data

def calculate_cosine_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def generate_user_profile(user_preferences, feature_vectors, movie_titles, ratings, people_voted):
    indices = []

    for category, preferences in user_preferences.items():
        for preference in preferences:
            indices.extend([i for i, title in enumerate(movie_titles) if preference.lower() in title.lower()])

    user_profile = feature_vectors[indices].mean(axis=0)

    user_rating_scalar = float(user_preferences.get('movieScore', 0.0))
    user_people_voted_scalar = float(user_preferences.get('peopleVoted', 0.0))

    user_profile_with_rating = user_profile + 0.2 * user_rating_scalar
    user_profile_with_both = user_profile_with_rating + 0.1 * user_people_voted_scalar  # Fixed reference

    return user_profile, user_rating_scalar, user_people_voted_scalar, user_profile_with_both  # Updated return

def generate_recommendations(user_preferences, feature_vectors, movie_titles, ratings, people_voted, top_n=10):
    user_profile, user_rating_scalar, user_people_voted_scalar, user_profile_with_both = generate_user_profile(
        user_preferences, feature_vectors, movie_titles, ratings, people_voted
    )

    # Convert to dense array using np.asarray
    user_similarity = cosine_similarity(np.asarray(user_profile).reshape(1, -1), feature_vectors.toarray()).flatten()

    movie_ranking = sorted(list(enumerate(user_similarity)), key=lambda x: x[1], reverse=True)

    filtered_indices = [idx for idx, _ in movie_ranking if pd.notna(ratings[idx]) and pd.notna(people_voted[idx])]

    # Ensure that filtered feature_vectors is a dense array
    feature_vectors_filtered = np.asarray(feature_vectors[filtered_indices].toarray())

    filtered_user_similarity = cosine_similarity(np.asarray(user_profile_with_both).reshape(1, -1), feature_vectors_filtered).flatten()
    filtered_movie_ranking = sorted(list(enumerate(filtered_user_similarity)), key=lambda x: x[1], reverse=True)

    top_recommendations = [
        (
            movie_titles[filtered_indices[idx]],
            float(ratings[filtered_indices[idx]]),
            int(people_voted[filtered_indices[idx]].replace(',', ''))
        )
        for idx, _ in filtered_movie_ranking[:top_n]
    ]

    print("Top Recommendations with Scores and People Voted:", top_recommendations)

    return top_recommendations

def main():
    # Load your data
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'recs_data_set.csv')
    data = load_data(file_path)

    if data is not None:
        # Preprocess Korean text for multiple columns
        data = preprocess_korean_text(data, text_columns=['moviePlot', 'movieDirector', 'movieActors'])

        # TF-IDF vectorization for Korean text
        tfidf_matrix = vectorize_text(data, text_columns=['moviePlot_tokens', 'movieDirector_tokens', 'movieActors_tokens'])

        # User preferences from JSON data (example)
        user_preferences = {
            'movieScore': '8.0',
            'peopleVoted': '1000',
            'directors': ['김성수', '이상용'],
            'actors': ['마동석', '손석구', '키아누 리브스'],
            'genres': ['범죄', '액션']
        }

        # Generate recommendations
        top_recommendations = generate_recommendations(
            user_preferences, tfidf_matrix, data['movieTitle'], data['movieScore'], data['peopleVoted']
        )

if __name__ == "__main__":
    main()