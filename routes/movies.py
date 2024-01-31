import pandas as pd
import json
import os

def send_movie_data():
    input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'movie_db.csv')
    df = pd.read_csv(input_path, encoding='utf-8')

    # NaN(누락된 값)을 빈 문자열로 대체
    df.fillna('', inplace=True)

    # 데이터프레임을 딕셔너리 리스트로 변환
    movies_list = df.to_dict(orient='records')

    # 영화 목록을 반복하며 URL 역 이스케이프 문자를 정상 슬래시로 변환
    for movie in movies_list:
        if 'moviePoster' in movie and isinstance(movie['moviePoster'], str):
            movie['moviePoster'] = movie['moviePoster'].replace('\\/', '/')
        if 'movieStills' in movie and isinstance(movie['movieStills'], str):
            movie['movieStills'] = movie['movieStills'].replace('\\/', '/')

    # 딕셔너리 리스트를 JSON 문자열로 변환 (ASCII 이외 문자를 유니코드로 인코딩, 들여쓰기 설정)
    movies_json = json.dumps(movies_list, ensure_ascii=False, indent=4)

    return movies_json