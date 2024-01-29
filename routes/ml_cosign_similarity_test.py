import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 데이터셋 불러오기
input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data_okt_v4.csv')
data = pd.read_csv(input_path, encoding='utf-8')

# 데이터를 훈련 및 테스트 세트로 분할
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# TF-IDF 벡터화기 초기화
tfidf_vectorizer = TfidfVectorizer()


# 각 카테고리별 가중치 설정
weights = {
    'genre_weight': 0.5,
    'actor_weight': 0.3,
    'director_weight': 0.2
}

# 각 카테고리별 가중 유사도를 계산하는 함수 정의
def calculate_weighted_similarity(column_name, user_preference, weight, dataset):
    user_pref_vector = tfidf_vectorizer.transform([user_preference])
    movie_feature_vector = tfidf_vectorizer.transform(dataset[column_name])
    similarity_scores = cosine_similarity(user_pref_vector, movie_feature_vector) * weight
    return similarity_scores.flatten()

# 훈련 세트에서 나온 모든 텍스트 데이터로 벡터화기를 학습
all_text_data = pd.concat([train_data[column] for column in ['movieDirector', 'movieActors', 'movieGenre']])
tfidf_vectorizer.fit(all_text_data)

# 사용자의 선호도 설정
user_preferences = {
    'preferActors': '마동석, 손석구, 키아누 리브스',
    'preferDirectors': '김성수, 이상용',
    'preferGenres': '드라마, 범죄, 액션'
}

# 테스트 데이터를 사용하여 각 카테고리별 가중 유사도 계산
genre_similarity = calculate_weighted_similarity('movieGenre', user_preferences['preferGenres'], weights['genre_weight'], test_data)
actor_similarity = calculate_weighted_similarity('movieActors', user_preferences['preferActors'], weights['actor_weight'], test_data)
director_similarity = calculate_weighted_similarity('movieDirector', user_preferences['preferDirectors'], weights['director_weight'], test_data)

# 각 카테고리별 유사도를 종합하여 총 유사도 점수 생성
total_similarity = genre_similarity + actor_similarity + director_similarity

# 테스트 데이터프레임에 총 유사도 점수를 추가하고, 이를 기준으로 영화를 정렬하여 상위 추천 영화 목록을 생성
test_data['similarity_score'] = total_similarity
recommended_movies = test_data.sort_values(by='similarity_score', ascending=False)

# 테스트 세트에서 상위 추천 영화를 출력
print(recommended_movies[['movieTitle', 'similarity_score']].head(10))