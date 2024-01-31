import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json

def calculate_recommendations(user_preferences):
    # 데이터 불러오기
    input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data_okt_v4.csv')
    data = pd.read_csv(input_path, encoding='utf-8')

    # TF-IDF 벡터화기 초기화
    tfidf_vectorizer = TfidfVectorizer()

    # 각 카테고리별 가중치 설정
    weights = {
        'genre_weight': 0.5,
        'actor_weight': 0.3,
        'director_weight': 0.2
    }

    # 영화 점수의 표준화된 점수의 60번째 백분위수를 계산
    percentile_60th = np.percentile(data['movieScore'], 60)

    # 각 카테고리별 가중 유사도를 계산하는 함수 정의
    def calculate_weighted_similarity(column_name, user_preference, weight):
        user_pref_vector = tfidf_vectorizer.transform([user_preference])
        movie_feature_vector = tfidf_vectorizer.transform(data[column_name])
        similarity_scores = cosine_similarity(user_pref_vector, movie_feature_vector) * weight
        return similarity_scores.flatten()

    # 데이터셋에서 모든 텍스트 데이터를 결합하여 벡터화기를 학습
    all_text_data = pd.concat([data[column] for column in ['movieDirector', 'movieActors', 'movieGenre']])
    tfidf_vectorizer.fit(all_text_data)

    # 북마크된 영화 ID를 추출하고 처리
    bookmarked_movie_ids = user_preferences.pop('movieId', '').split(',')
    # 북마크된 영화 ID가 있는지 확인
    if bookmarked_movie_ids and bookmarked_movie_ids[0]:
        # 북마크된 영화 ID를 정수형으로 변환
        bookmarked_movie_ids = list(map(int, bookmarked_movie_ids))
        # 북마크된 영화의 특징을 추출하고 사용자 선호도에 추가
        for feature in ['movieGenre', 'movieActors', 'movieDirector']:
            # 북마크된 영화의 중복되지 않는 특징을 추출하고, 쉼표로 구분하여 결합
            aggregated_feature = ', '.join(data[data['movieId'].isin(bookmarked_movie_ids)][feature].dropna().unique())
            # 특징에 따라 사용자 선호도 키를 조정하여 user_preferences에 추가 (Actors에는 이미 s가 붙어있음)
            if feature.endswith('Actors'):
                user_preference_key = 'prefer' + feature[5:]
            else:
                user_preference_key = 'prefer' + feature[5:] + 's'
            # 특징 데이터가 있는 경우에만 사용자 선호도를 업데이트
            if aggregated_feature:
                if user_preference_key in user_preferences and user_preferences[user_preference_key]:
                    user_preferences[user_preference_key] += ', ' + aggregated_feature
                else:
                    user_preferences[user_preference_key] = aggregated_feature

    # 북마크된 영화를 제외한 데이터셋 생성
    data = data[~data['movieId'].isin(bookmarked_movie_ids)]

    # 각 카테고리별 가중 유사도 계산
    genre_similarity = calculate_weighted_similarity('movieGenre', user_preferences['preferGenres'],
                                                     weights['genre_weight'])
    actor_similarity = calculate_weighted_similarity('movieActors', user_preferences['preferActors'],
                                                     weights['actor_weight'])
    director_similarity = calculate_weighted_similarity('movieDirector', user_preferences['preferDirectors'],
                                                        weights['director_weight'])

    # 각 카테고리별 유사도를 종합하여 총 유사도 점수 생성
    total_similarity = genre_similarity + actor_similarity + director_similarity

    # 데이터프레임에 총 유사도 점수를 추가하고 이를 기준으로 정렬
    data['similarity_score'] = total_similarity

    # 60번째 백분위수 이상의 영화에 더 높은 우선순위를 할당
    data['priority'] = data['movieScore'].apply(lambda x: 1 if x >= percentile_60th else 0)

    # 정렬 조건을 우선순위, 유사도 순으로 설정
    recommended_movies = data.sort_values(by=['priority', 'similarity_score'], ascending=[False, False])

    recs = recommended_movies.head(4)
    output = {f"recs{i+1}": int(recs.iloc[i]['movieId']) for i in range(len(recs))}
    output_json = json.dumps(output, ensure_ascii=False, indent=4)

    return output_json
