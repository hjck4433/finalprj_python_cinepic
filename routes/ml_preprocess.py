import pandas as pd
import os
from konlpy.tag import Okt, Hannanum
import re
from collections import Counter
from nltk import ngrams
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 전처리
# 영화감독 / 영화배우
def preprocess_names(text):
    if text.strip() == '알수없음':
        return []
    return [name.strip() for name in text.split(',')]

# 장르
def preprocess_genre(genre):
    # "," 와 "/" 제외한 특수문자 제거
    genre = re.sub(r'[^\w\s,\/]', '', genre)

    # "," 또는 "/" 를 기준으로 나눔
    tokens = re.split(r',|\/', genre)

    # 공백 제거
    tokens = [token.strip() for token in tokens if token.strip()]

    return tokens

# 영화제목
def preprocess_title(title, stopwords):
    # 하나의 숫자로만 되어 있는 경우 ex) 65, 1917
    if title.isdigit():
        return [title], Counter()

    # 특수문자 제거
    title = re.sub(r'[^\w\s]', '', title)

    # Hannanum을 사용해 토큰화
    tokenizer = Okt()
    tokens = tokenizer.morphs(title)
    # 불용어 제거
    tokens = [token for token in tokens if token not in stopwords]
    # return tokens

    # Word Frequency Calculation
    word_freq = Counter(tokens)

    return tokens, word_freq

# 영화 줄거리
def preprocess_plot(plot, stopwords):
    #특수문자 제거
    plot = re.sub(r'[^\w\s]', '', plot)

    # Okt를 사용해 토큰화
    tokenizer = Okt()
    tokens = tokenizer.morphs(plot)

    # 불용어 제거
    tokens = [token for token in tokens if token not in stopwords]

    # return tokens

    # Word Frequency Calculation
    word_freq = Counter(tokens)

    # N-gram Extraction (for example, using bigrams)
    bigrams = list(ngrams(tokens, 2))

    return tokens, word_freq, bigrams


# 숫자 데이터
def preprocess_numeric_columns(data):
    numeric_columns = ['audience', 'screen', 'screening', 'movieScore', 'peopleVoted']

    for col in numeric_columns:
        # 숫자형이 아니라면 숫자로 변환
        if data[col].dtype == 'O':  # 'O'은 판다스에서 일반적으로 string 의미
            data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
        else:
            data[col] = data[col].astype(float)

        # Scaling
        if col == 'movieScore':
            scaler = StandardScaler()  # Use StandardScaler for 'movieScore'
        else:
            scaler = MinMaxScaler()  # Use MinMaxScaler for other columns

        scaled_col = scaler.fit_transform(data[[col]])

        # 스케일링 전 후 시각화
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # sns.histplot(data[col], ax=axes[0], kde=True, color='blue', bins=20)
        # axes[0].set_title(f'Distribution of {col} Before Scaling')
        # axes[0].set_xlabel(col)
        # axes[0].set_ylabel('Frequency')
        #
        # sns.histplot(pd.Series(scaled_col.flatten()), ax=axes[1], kde=True, color='red', bins=20)
        # axes[1].set_title(f'Distribution of {col} After Scaling')
        # axes[1].set_xlabel(col)
        # axes[1].set_ylabel('Frequency')
        #
        # plt.tight_layout()
        # plt.show()


        # 스케일링 전 후 비교
        print(f"Descriptive Statistics for '{col}':")
        print("Before Scaling:")
        print(data[col].describe())
        print("After Scaling:")
        print(pd.Series(scaled_col.flatten()).describe())
        print("-----------------------------------")

        data[col] = scaled_col  # Fit and transform on each numerical column

        # 각 컬럼이 가우시안 분포인지 확인
        if (data[col].skew() < 0.5) and (data[col].kurtosis() < 3):
            print(f"The distribution of '{col}' is approximately Gaussian.")
        else:
            print(f"The distribution of '{col}' is not Gaussian.")

    return data

# 개봉일 데이터
def preprocess_date_column(data):
    data['movieRelease'] = pd.to_datetime(data['movieRelease'], errors='coerce')
    return data

# 불용어 리스트
def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'recs_data_set.csv')
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data_okt_v3.csv')
data = pd.read_csv(input_path, encoding='utf-8')

stopword_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'StopWords.txt')

stopwords = load_stopwords(stopword_path)

# 숫자 데이터 처리
data = preprocess_numeric_columns(data)
data = preprocess_date_column(data)

# 토큰화
# data['movieTitle_tokens'] = data['movieTitle'].apply(preprocess_title, args=(stopwords,))
# data['movieGenre_tokens'] = data['movieGenre'].apply(preprocess_genre)
# data['movieDirector_tokens'] = data['movieDirector'].apply(preprocess_names)
# data['movieActors_tokens'] = data['movieActors'].apply(preprocess_names)
data['movieTitle_tokens'], data['movieTitle_word_freq'] = zip(*data['movieTitle'].apply(preprocess_title, args=(stopwords,)))
data['movieGenre_tokens'] = data['movieGenre'].apply(preprocess_genre)
data['movieDirector_tokens'] = data['movieDirector'].apply(preprocess_names)
data['movieActors_tokens'] = data['movieActors'].apply(preprocess_names)
data['moviePlot_tokens'], data['moviePlot_word_freq'], data['moviePlot_bigrams'] = zip(*data['moviePlot'].apply(preprocess_plot, args=(stopwords,)))

# 새 컬럼 생성
data['audience_density'] = data['audience'] / data['screening']
# 북마크 수
data['likes'] = 0
# 정보 조회 수
data['views'] = 0
# 조회 대비 북마크 수
data['like_view_ratio'] = data['likes'] / data['views']
data['moviePlot_tokens'] = data['moviePlot'].apply(preprocess_plot, args=(stopwords,))

unique_directors = set()
unique_actors = set()
unique_genres = set()

for directors in data['movieDirector_tokens']:
    unique_directors.update(directors)

for actors in data['movieActors_tokens']:
    unique_actors.update(actors)

for genres in data['movieGenre_tokens']:
    unique_genres.update(genres)

# 이진 컬럼 생성
binary_columns = []

# 각 유형에 따른 접두사
director_prefix = 'director_'
actor_prefix = 'actor_'
genre_prefix = 'genre_'

for director in unique_directors:
    column_name = director_prefix + director
    binary_columns.append(data['movieDirector_tokens'].apply(lambda x: 1 if director in x else 0).rename(column_name))

for actor in unique_actors:
    column_name = actor_prefix + actor
    binary_columns.append(data['movieActors_tokens'].apply(lambda x: 1 if actor in x else 0).rename(column_name))

for genre in unique_genres:
    column_name = genre_prefix + genre
    binary_columns.append(data['movieGenre_tokens'].apply(lambda x: 1 if genre in x else 0).rename(column_name))

binary_df = pd.concat(binary_columns, axis=1)
data = pd.concat([data, binary_df], axis=1)



data.to_csv(output_path, index=False, encoding='utf-8-sig')

