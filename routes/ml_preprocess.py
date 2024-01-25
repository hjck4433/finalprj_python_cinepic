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


# 영화제목
def preprocess_title(title):
    # 하나의 숫자로만 되어 있는 경우 ex) 65, 1917
    if title.isdigit():
        return [title]

    # 특수문자 제거
    title = re.sub(r'[^\w\s]', '', title)

    # Hannanum을 사용해 토큰화
    tokenizer = Hannanum()
    tokens = tokenizer.morphs(title)

    return tokens

# 영화 줄거리
def preprocess_plot(plot):
    #특수문자 제거
    plot = re.sub(r'[^\w\s]', '', plot)

    # Okt를 사용해 토큰화
    tokenizer = Hannanum()
    tokens = tokenizer.morphs(plot)

    return tokens

input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'recs_data_set.csv')
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data_hannanum.csv')
data = pd.read_csv(input_path, encoding='utf-8')

data['movieDirector_tokens'] = data['movieDirector'].apply(preprocess_names)
data['movieActors_tokens'] = data['movieActors'].apply(preprocess_names)
data['movieTitle_tokens'] = data['movieTitle'].apply(preprocess_title)
data['moviePlot_tokens'] = data['moviePlot'].apply(preprocess_plot)

data.to_csv(output_path, index=False, encoding='utf-8-sig')

# Display the preprocessed data
# print(data.head())