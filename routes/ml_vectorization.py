import pandas as pd
import os
from gensim.models import Word2Vec

# Word2Vec 모델을 학습하기 위한 함수 정의
def train_word2vec_model(tokens_list):
    model = Word2Vec(tokens_list, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Word2Vec 벡터화를 수행하고 원본 데이터프레임에 연결하는 함수 정의
def word2vec_vectorize_and_concat(data, column_name, model):
    vectors = []
    for tokens in data[column_name]:
        # 각 토큰의 벡터 표현을 계산하고 평균을 취함
        token_vectors = [model.wv[token] for token in tokens if token in model.wv]
        if token_vectors:
            avg_vector = sum(token_vectors) / len(token_vectors)
            vectors.append(avg_vector)
        else:
            # Word2Vec 모델 어휘 사전에 토큰이 없는 경우, 영벡터 사용
            vectors.append([0] * model.vector_size)

    # 벡터로부터 데이터프레임 생성
    word2vec_df = pd.DataFrame(vectors, columns=[f'{column_name}_w2v_{i+1}' for i in range(model.vector_size)])

    # Word2Vec 데이터프레임을 원본 데이터프레임과 연결
    final_df = pd.concat([data.reset_index(drop=True), word2vec_df], axis=1)

    return final_df

# 전처리된 데이터 불러오기
input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data_hannanum_v3.csv')
data = pd.read_csv(input_path, encoding='utf-8')

# 영화 제목 토큰을 사용하여 Word2Vec 모델 학습
title_tokens = data['movieTitle_tokens'].tolist()
title_model = train_word2vec_model(title_tokens)

# 학습된 Word2Vec 모델을 사용하여 영화 제목 벡터화
final_df = word2vec_vectorize_and_concat(data, 'movieTitle_tokens', title_model)

# 영화 줄거리 토큰을 사용하여 Word2Vec 모델 학습
plot_tokens = data['moviePlot_tokens'].tolist()
plot_model = train_word2vec_model(plot_tokens)


# 학습된 Word2Vec 모델을 사용하여 영화 줄거리 벡터화
final_df = word2vec_vectorize_and_concat(final_df, 'moviePlot_tokens', plot_model)

# 사용 되지 않을 컬럼 drop
columns_to_drop = ['movieTitle_tokens', 'movieTitle_word_freq', 'movieGenre_tokens',
                   'movieDirector_tokens', 'movieActors_tokens', 'moviePlot_tokens',
                   'moviePlot_word_freq', 'moviePlot_bigrams']
final_df = final_df.drop(columns=columns_to_drop)

# 최종 데이터프레임을 CSV 파일로 저장
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'word2vec_results.csv')
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# 최종 데이터프레임 출력
print(final_df)