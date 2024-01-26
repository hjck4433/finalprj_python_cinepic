import pandas as pd
import os
from konlpy.tag import Okt, Hannanum
import re


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
        return [title]

    # 특수문자 제거
    title = re.sub(r'[^\w\s]', '', title)

    # Hannanum을 사용해 토큰화
    tokenizer = Okt()
    tokens = tokenizer.morphs(title)
    # 불용어 제거
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

# 영화 줄거리
def preprocess_plot(plot, stopwords):
    #특수문자 제거
    plot = re.sub(r'[^\w\s]', '', plot)

    # Okt를 사용해 토큰화
    tokenizer = Okt()
    tokens = tokenizer.morphs(plot)

    # 불용어 제거
    tokens = [token for token in tokens if token not in stopwords]

    return tokens

# 숫자 데이터
def preprocess_numeric_columns(data):
    numeric_columns = ['audience', 'screen', 'screening', 'movieScore', 'peopleVoted']

    for col in numeric_columns:
        # 숫자형이 아니라면 숫자로 변환
        if data[col].dtype == 'O':  # 'O'은 판다스에서 일반적으로 string 의미
            data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
        else:
            data[col] = data[col].astype(float)

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
data['movieTitle_tokens'] = data['movieTitle'].apply(preprocess_title, args=(stopwords,))
data['movieGenre_tokens'] = data['movieGenre'].apply(preprocess_genre)
data['movieDirector_tokens'] = data['movieDirector'].apply(preprocess_names)
data['movieActors_tokens'] = data['movieActors'].apply(preprocess_names)

# 새 컬럼 생성
data['audience_density'] = data['audience'] / data['screening']
# 북마크 수
data['likes'] = 0
# 정보 조회 수
data['views'] = 0
# 조회 대비 북마크 수
data['like_view_ratio'] = data['likes'] / data['views']
data['moviePlot_tokens'] = data['moviePlot'].apply(preprocess_plot, args=(stopwords,))


data.to_csv(output_path, index=False, encoding='utf-8-sig')

