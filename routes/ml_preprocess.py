import pandas as pd
import os
from konlpy.tag import Okt
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def preprocess_names(text):
    if text.strip() == '알수없음':
        return '알수없음'
    return ' '.join([name.strip() for name in text.split(',')])

def preprocess_genre(genre):
    genre = re.sub(r'[^\w\s,\/]', '', genre)
    tokens = re.split(r',|\/', genre)
    return ' '.join([token.strip() for token in tokens if token.strip()])

def preprocess_title(title, stopwords, min_token_length=1):
    # 원본 제목을 백업하고, 토큰화 결과가 빈 문자열인 경우에 대비
    original_title = title

    # 제목을 토큰화
    tokenizer = Okt()
    tokens = tokenizer.morphs(title)

    # 불용어 및 최소 토큰 길이 미만의 토큰 제거
    tokens = [token for token in tokens if token not in stopwords and len(token) >= min_token_length]

    # 토큰화 결과가 비어있는 경우 원본 제목 반환
    if not tokens:
        return original_title

    return ' '.join(tokens)

def preprocess_plot(plot, stopwords):
    # 플롯 텍스트를 정제하고 토큰화
    plot = re.sub(r'[^\w\s]', '', plot)  # 구두점 제거
    tokenizer = Okt()
    tokens = tokenizer.morphs(plot)   # 토큰화

    # 불용어 및 한 글자 짜리 단어 필터링
    tokens = [token for token in tokens if token not in stopwords and len(token) > 1]

    # 정제된 플롯을 문자열로 반환하여 벡터화에 적합하게 함
    return ' '.join(tokens)

def preprocess_numeric_columns(data):
    numeric_columns = ['audience', 'screen', 'screening', 'movieScore', 'peopleVoted']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')
        scaler = StandardScaler() if col == 'movieScore' else MinMaxScaler()
        data[col] = scaler.fit_transform(data[[col]].fillna(0))
    return data

def preprocess_date_column(data):
    data['movieRelease'] = pd.to_datetime(data['movieRelease'], errors='coerce')
    return data

def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

# 데이터 및 불용어 로드
input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'recs_data_set.csv')
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data_okt_v4.csv')
stopword_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'StopWords.txt')

data = pd.read_csv(input_path, encoding='utf-8')
stopwords = load_stopwords(stopword_path)

# 데이터 전처리
data = preprocess_numeric_columns(data)
data = preprocess_date_column(data)
data['movieGenre'] = data['movieGenre'].apply(preprocess_genre)
data['movieTitleTokenized'] = data['movieTitle'].apply(preprocess_title, args=(stopwords,))
data['moviePlot'] = data['moviePlot'].apply(preprocess_plot, args=(stopwords,))
data['movieDirector'] = data['movieDirector'].apply(preprocess_names)
data['movieActors'] = data['movieActors'].apply(preprocess_names)


# TF-IDF 벡터화기 초기화 (max_features 제한 없음)
tfidf_vectorizer = TfidfVectorizer()

# 벡터화를 위해 모든 텍스트 데이터를 하나의 시리즈로 연결
all_text_data = pd.concat([data['movieDirector'], data['movieActors'], data['movieGenre'], data['moviePlot'], data['movieTitleTokenized']])

# 텍스트 데이터를 하나의 feature space로 만들기 위해 결합된 텍스트 데이터에 대해 벡터화기를 훈련
tfidf_vectorizer.fit(all_text_data)

# 원하는 SVD 컴포넌트의 최대 수를 정의
desired_n_components = 50

# TF-IDF 변환 및 SVD 차원 축소를 수행하는 함수
def process_text_feature(column_name, tfidf_vectorizer, data, n_components):
    # 텍스트 데이터를 TF-IDF 특성 행렬로 변환
    tfidf_matrix = tfidf_vectorizer.transform(data[column_name])

    # TF-IDF 행렬의 형태에 따라 SVD 컴포넌트 수를 조정
    n_components = min(tfidf_matrix.shape[1] - 1, n_components)

    # TF-IDF 행렬에 대한 차원 축소 수행
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(tfidf_matrix)

    # SVD로부터 생성된 행렬로 DataFrame 생성
    reduced_df = pd.DataFrame(reduced_matrix, columns=[f'{column_name}_tfidf_svd_{i + 1}' for i in range(n_components)])

    return reduced_df

# 각 텍스트 열을 처리하고 생성된 데이터프레임을 원본 데이터에 연결
for column in ['movieDirector', 'movieActors', 'movieGenre', 'moviePlot', 'movieTitleTokenized']:
    reduced_df = process_text_feature(column, tfidf_vectorizer, data, desired_n_components)
    data = pd.concat([data, reduced_df], axis=1)

# 전처리된 데이터 저장
data.to_csv(output_path, index=False, encoding='utf-8-sig')

