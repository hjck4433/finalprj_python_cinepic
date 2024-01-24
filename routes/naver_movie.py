import requests
import json
from bs4 import BeautifulSoup
from flask import jsonify
import time
import random
import re

DEBUG_MODE = False

# 네이버 영화 포토 탭 접근 - 포스터/ 스틸컷 url 가져오기
def get_image_url(url,title, max_retries = 3) :
    retries = 0
    while retries < max_retries:
        if DEBUG_MODE:
            print(f'{title}의 포스터 가져오는 중 / 시도 : {retries + 1}')
        try :
            # 지연
            sleep_interval = random.uniform(0.5, 1.5)
            # sleep_interval = random.uniform(2.0, 4.0)
            time.sleep(sleep_interval)

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36 (compatible; Yeti/1.1; +https://naver.me/spd)'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raises HTTPError for bad responses

            soup = BeautifulSoup(response.text, 'html.parser')
            main_el = soup.find('body', class_='wrap-new api_animation').find('div', class_='sec_movie_photo')

            poster_outer = main_el.find('div', class_='_image_base_poster').find('div', class_='movie_photo_list')
            poster_el = poster_outer.find('li', class_='item').find('img')
            poster = poster_el['data-img-src']

            stills_outer = main_el.find('div', class_='_image_base_stillcut')

            stills_urls_ls = []

            try:
                if stills_outer:
                    stills_list = stills_outer.find('div', class_='movie_photo_list')
                    stills_els = stills_list.find_all('li', class_='item')

                    for index, li in enumerate(stills_els):
                        stills_el = li.find('img')

                        if stills_el:
                            stills_url = stills_el.get('data-img-src')
                            stills_urls_ls.append(stills_url)

                        if index == 7:
                            break

            except Exception as e :
                if DEBUG_MODE:
                    print(f"스틸가져오는 중 에러 : {e}")
                stills_urls_ls = []

            stills_urls = "|".join(stills_urls_ls)

            # if DEBUG_MODE:
                # print(f"{title} poster : {poster}")
                # print(f"{title} stlls : {stlls_urls}")

            result = {"poster": poster, "stills":stills_urls}
            return result

        except requests.exceptions.RequestException as e:
            if DEBUG_MODE:
                print(f'Error: {str(e)} - Retry {retries + 1}/{max_retries}')
        retries += 1

    return {'error': '3번 시도 후에도 실패'}

# 네이버 영화 기본정보 탭 접근 - 러닝타임 / 줄거리
def get_basic_data(url,title, max_retries = 3):
    retries = 0
    while retries < max_retries:
        if DEBUG_MODE:
            print(f'{title}의 기본 정보 가져오는 중 / 시도 : {retries + 1}')
        try :
            # 지연
            sleep_interval = random.uniform(0.5, 1.5)
            # sleep_interval = random.uniform(2.0, 4.0)
            time.sleep(sleep_interval)

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36 (compatible; Yeti/1.1; +https://naver.me/spd)'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raises HTTPError for bad responses

            soup = BeautifulSoup(response.text, 'html.parser')
            main_el = soup.find('body', class_='wrap-new api_animation')

            # 러닝타임
            outer_runtime_el = main_el.find('div', class_='detail_info').find('dl', class_='info')
            info_groups = outer_runtime_el.find_all('div', class_='info_group')

            for info_group in info_groups:
                dt_element = info_group.find('dt')
                if dt_element and dt_element.text.strip() == '러닝타임':
                    runtime_value = info_group.find_next('dd')
                    if runtime_value:
                        runtime_text = runtime_value.text.strip()
                        match = re.search(r'\d+', runtime_text)
                        runtime = int(match.group())
                        break

            # 줄거리
            outer_plot_el = main_el.find('div', class_='_cm_content_area_synopsis')
            plot_el = outer_plot_el.find('p', class_='_content_text')
            plot = plot_el.get_text(strip=True)

            result = {"runtime": runtime, "plot": plot}
            return result

        except requests.exceptions.RequestException as e:
            if DEBUG_MODE:
                print(f'Error: {str(e)} - Retry {retries + 1}/{max_retries}')
        retries += 1

    return {'error': '3번 시도 후에도 실패'}


# 네이버 영화 전체 탭 접근 - 영문 제목 / 평점 / 참여자
def get_movie_data(title, max_retries = 3):
    retries = 0
    while retries < max_retries:
        if DEBUG_MODE:
            print(f'{title}의 추가 정보 가져오는 중 / 시도 : {retries + 1}')
        try:

            url = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=영화+' + title

            #지연
            sleep_interval = random.uniform(0.5, 1.5)
            # sleep_interval = random.uniform(2.0, 4.0)
            time.sleep(sleep_interval)

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36 (compatible; Yeti/1.1; +https://naver.me/spd)'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raises HTTPError for bad responses

            soup = BeautifulSoup(response.text, 'html.parser')
            # main_el = soup.find('body', class_='wrap-new api_animation').find('dl', class_='info')
            # outer_els = main_el.find_all('div', class_='info_group')

            main_el = soup.find('body', class_='wrap-new api_animation')

            # 영문 제목
            outer_title_el = main_el.find('div', class_='title_area type_keep _title_area').find('div', class_='sub_title')
            english_title = outer_title_el.find_all('span', class_='txt')[1].get_text(strip=True)

            # 평점 영역
            outer_score_el = main_el.find('div', class_='lego_rating_slide_outer').find('a', class_='lego_rating_box_see')

            # 종합 평점
            movie_score_el = outer_score_el.find('span', class_='area_star_number')
            movie_score_inner = movie_score_el.find('span', class_='area_star_total_number')
            movie_score_inner.extract()
            movie_score = movie_score_el.get_text(strip=True)

            # 참여수
            people_voted = outer_score_el.find('span', class_='area_people').get_text(strip=True)
            people_voted = re.sub(r'명\s참여', '', people_voted)

            # 추가 정보 url
            outer_url_el = main_el.find('div', class_='_main_tab').find('ul', class_='tab_list')

            # 기본정보 url
            basic_info_el = outer_url_el.find('span', string='기본정보')

            if basic_info_el:
                outer_a_tag = basic_info_el.find_parent('a')
                if outer_a_tag:
                    basic_href_value = outer_a_tag.get('href')

            basic_url = 'https://search.naver.com/search.naver'+basic_href_value
            # print(basic_url)

            # 포토 url
            photo_info_el = outer_url_el.find('span', string='포토')

            if photo_info_el:
                outer_a_tag = photo_info_el.find_parent('a')
                if outer_a_tag:
                    photo_href_value = outer_a_tag.get('href')

            photo_url = 'https://search.naver.com/search.naver' + photo_href_value
            # print(photo_url)


            # 이미지 (포스터/ 스틸컷 가져오기)
            # image_urls = get_image_url(url+'포토', title)
            image_urls = get_image_url(photo_url, title)


            # 상영시간 / 줄거리
            # basic_info = get_basic_data(url+'정보', title)
            basic_info = get_basic_data(basic_url, title)

            result = {"movieTitleEng": english_title, "movieScore": movie_score, "peopleVoted": people_voted, "moviePoster": image_urls.get("poster"), "movieStills": image_urls.get("stills"), "movieRuntime": basic_info.get("runtime"), "moviePlot": basic_info.get("plot")}
            # result = {"movieTitleEng": english_title, "movieScore": movie_score, "peopleVoted": people_voted}
            # print(result)
            return result

        except requests.exceptions.RequestException as e:
            if DEBUG_MODE:
                print(f'Error: {str(e)} - Retry {retries + 1}/{max_retries}')
            retries += 1
        except Exception as e :
            # if DEBUG_MODE:
            print(f'Error {title} : {str(e)}')
            return {}

    return {'error': '3번 시도 후에도 실패'}


get_movie_data("미션 임파서블: 폴아웃")