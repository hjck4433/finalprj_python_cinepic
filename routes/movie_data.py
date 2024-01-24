import pandas as pd
import os
from naver_movie import get_movie_data, get_director

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

# filter_movie_csv()

naver_data_cnt = 0
naver_empty_cnt = 0

# 네이버 크롤링 정보 추가
def add_naver_data(row):
    global naver_data_cnt
    global naver_empty_cnt
    try:
        movie_data = get_movie_data(row['movieTitle'])

        if movie_data != {}:
            naver_data_cnt += 1
        else:
            naver_empty_cnt += 1

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
    # 처음부터
    # file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'filtered_boxoffice_test.csv')
    # 중간부터
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'info_added_boxoffice_2.csv')
    try:
        filtered_df = pd.read_csv(file_path, encoding='utf-8')

        # 처음부터 끝가지 크롤링 시도 후 저장-> 300개 조금 넘어가면 네이버에서 403 forbidden
        #
        # result_df = pd.concat([filtered_df, filtered_df.apply(add_naver_data, axis=1).reset_index(drop=True)], axis=1)
        #
        # result_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'info_added_boxoffice_test.csv')
        # result_df.to_csv(result_file_path, index=False, encoding='utf-8-sig')

        #  - row 지정 -> 200개 정도씩 나눠서 진행
        start_row = 874
        end_row = start_row + 200 - 1
        rows_to_update = (filtered_df.index >= start_row - 1) & (filtered_df.index <= end_row - 1)

        # 기존 네이버 크롤링 정보가 추가 하기 위해 업데이트 필요 컬럼 지정 (하지않으면 다른 컬럼의 데이터가 삭제되어 저장됨)
        naver_data_df = filtered_df.loc[rows_to_update].apply(add_naver_data, axis=1)
        update_columns = ['movieTitleEng', 'movieScore', 'peopleVoted', 'moviePoster', 'movieStills', 'movieRuntime',
                          'moviePlot']

        filtered_df.loc[rows_to_update, update_columns] = naver_data_df[update_columns]

        filtered_df.to_csv(file_path, index=False, encoding='utf-8-sig')

        print(f"저장 완료 : {naver_data_cnt} 건 / 저장 실패 : {naver_empty_cnt} 건")

    except Exception as e:
        print(f"error while saving : {str(e)}")

# save_naver_info()

def final_filter_data():
    input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'info_added_boxoffice_2.csv')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'final_boxoffice.csv')

    try:
        data_df = pd.read_csv(input_path, encoding='utf-8')

        # movieGenre 및 movieScore 열에 데이터가 있는 행만 필터링
        filtered_df = data_df.dropna(subset=['movieGenre', 'movieScore']).copy()

        # movieId 값을 row number로 바꿈(고유값 처리)
        filtered_df.reset_index(drop=True, inplace=True)
        # filtered_df['movieId'] = filtered_df.index + 1
        filtered_df.loc[:, 'movieId'] = filtered_df.index + 1

        # 필터링된 데이터를 새로운 CSV 파일로 저장
        filtered_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("저장 완료")

    except Exception as e:
        print(f"최종 저장 중 오류 발생: {str(e)}")

# final_filter_data()

def add_director():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'final_boxoffice.csv')

    try:
        data_df = pd.read_csv(file_path, encoding='utf-8')
        # 감독 정보가 없으면
        rows_to_update = data_df['movieDirector'].isnull()
        # 감독 정보 추가
        data_df.loc[rows_to_update, 'movieDirector'] = data_df.loc[rows_to_update, 'movieTitle'].apply(get_director)

        data_df.to_csv(file_path, index=False, encoding='utf-8-sig')

        print("저장 완료")
    except Exception as e:
        print(f"감독 정보 추가하는 중 오류: {str(e)}")

# add_director()

# DB 저장용 csv 파일 저장
def csv_for_db():
    input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'final_boxoffice.csv')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'movie_db.csv')

    try:
        data_df = pd.read_csv(input_path, encoding='utf-8')

        columns_to_keep = ['movieId', 'movieTitle', 'movieTitleEng', 'movieRelease', 'movieGenre',
                           'movieNation', 'movieGrade', 'movieRuntime', 'movieScore', 'movieDirector', 'movieActors',
                           'moviePlot', 'moviePoster', 'movieStills']

        db_df = data_df[columns_to_keep]

        db_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print("저장 완료")

        db_df = data_df[columns_to_keep]

    except Exception as e:
        print(f"db용 파일 저장 중 오류 : {str(e)}")

csv_for_db()