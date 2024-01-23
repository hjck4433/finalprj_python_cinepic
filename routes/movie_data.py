import csv

import pandas as pd
import os
import json
from naver_movie import get_movie_data

def filter_movie_csv() :
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'boxoffice_kr.csv')
    movie_csv = pd.read_csv(file_path, encoding='utf-8')

    # 개봉일이 없거나 2010년 이전 데이터 제외
    movie_csv = movie_csv.dropna(subset=['movieRelease'])
    movie_csv = movie_csv[movie_csv['movieRelease'].str.extract('(\d{4})', expand=False).astype(float) >= 2010]
    # print(movie_csv)

    # 필터링: '성인물'을 포함하지 않는 행만 선택
    movie_csv = movie_csv[~movie_csv['movieGenre'].fillna('').str.contains('성인물')]

    filtered_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'filtered_boxoffice_kr.csv')
    movie_csv.to_csv(filtered_file_path, index=False, encoding='utf-8-sig')
    # filtered_csv = pd.read_csv(filtered_file_path, encoding='utf-8')
    # print(filtered_csv)

naver_data_cnt = 0
def add_naver_data(row):
    global naver_data_cnt
    try:
        movie_data = get_movie_data(row['movieTitle'])

        if movie_data != {}:
            naver_data_cnt += 1

        return pd.Series({
            'movieTitleEng': movie_data.get('movieTitleEng', ''),
            'movieScore': movie_data.get('movieScore', ''),
            'peopleVoted': movie_data.get('peopleVoted', ''),
            'moviePoster': movie_data.get('moviePoster', ''),
            'movieStills': movie_data.get('movieStills', ''),
            'movieRuntime': movie_data.get('movieRuntime', ''),
            'moviePlot': movie_data.get('moviePlot', ''),
        })
    except Exception as e:
        print(f"error while adding : {str(e)}")
        return pd.Series({})

def save_naver_info():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'filtered_boxoffice.csv')
    try:
        filtered_df = pd.read_csv(file_path, encoding='utf-8')

        result_df = pd.concat([filtered_df, filtered_df.apply(add_naver_data, axis=1)], axis=1)

        result_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'info_added_boxoffice.csv')
        result_df.to_csv(result_file_path, index=False, encoding='utf-8-sig')
        print(f"저장 완료 : {naver_data_cnt} 건")

    except Exception as e:
        print(f"error while saving : {str(e)}")


# filter_movie_csv()

# get_movie_data()

save_naver_info()