import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np
import time

# 작업 시작
start_time = time.time()

# 데이터셋 불러오기
input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data_okt_v4.csv')
data = pd.read_csv(input_path, encoding='utf-8')

# CountVectorizer 초기화
# Jaccard 유사도를 위한 이진 출력 설정
count_vectorizer = CountVectorizer(binary=True)

# 데이터셋에서 텍스트 데이터를 결합하여 CountVectorizer에 맞게 학습
all_text_data = pd.concat([data[column] for column in ['movieDirector', 'movieActors', 'movieGenre']])
count_vectorizer.fit(all_text_data)

# 각 카테고리별 가중 Jaccard 유사도를 계산하는 함수 정의
def calculate_weighted_jaccard_similarity(column_name, user_preference, weight):
    user_pref_vector = count_vectorizer.transform([user_preference]).toarray()
    movie_feature_vectors = count_vectorizer.transform(data[column_name]).toarray()

    # 각 영화에 대한 Jaccard 유사도 계산하고 가중치 적용
    jaccard_similarities = [jaccard_score(user_pref_vector[0], movie_vector) * weight for movie_vector in movie_feature_vectors]

    return np.array(jaccard_similarities)

# 사용자의 선호도 설정
user_preferences = {
    'preferActors': '',
    'preferDirectors': '',
    'preferGenres': '로맨스'
}

# 각 카테고리별 가중치 설정
weights = {
    'genre_weight': 0.5,
    'actor_weight': 0.3,
    'director_weight': 0.2
}

# 각 카테고리별 가중 Jaccard 유사도 계산
genre_similarity = calculate_weighted_jaccard_similarity('movieGenre', user_preferences['preferGenres'], weights['genre_weight'])
actor_similarity = calculate_weighted_jaccard_similarity('movieActors', user_preferences['preferActors'], weights['actor_weight'])
director_similarity = calculate_weighted_jaccard_similarity('movieDirector', user_preferences['preferDirectors'], weights['director_weight'])

# 각 카테고리별 유사도를 종합하여 총 유사도 점수 생성
total_similarity = genre_similarity + actor_similarity + director_similarity

# 데이터프레임에 총 유사도 점수를 추가하고 이를 기준으로 정렬
data['similarity_score'] = total_similarity
recommended_movies = data.sort_values(by='similarity_score', ascending=False)

# 작업 끝
end_time = time.time()

# 소요 시간
elapsed_time = end_time - start_time

# 상위 추천 영화 출력
print(recommended_movies[['movieId', 'movieTitle','movieScore', 'similarity_score']].head(10))
print(f"The code took {elapsed_time} seconds to run.")