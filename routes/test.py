import pandas as pd
from konlpy.tag import Okt, Hannanum
import re

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


# 데이터 전처리
def preprocess_data(data):
    # KoNLPy 활용한 토큰화
    tokenizer = Okt()

    # movieTitle, movieDirector, movieActors, movieGenre, moviePlot 컬럼의 한글 텍스트를 전처리 및 토큰화
    text_columns = ['movieTitle', 'movieDirector', 'movieActors', 'movieGenre']


def tokenize_names(text):
    # text = text.replace(',', ' ')
    tokenizer = Okt()

    # 규칙 커스텀
    tokens = text.split(',')

    return tokens


# sentence = '마동석,손석구'
# tokens = tokenize_names(sentence)

def tokenize_title(title):
    # 제목이 숫자로만 이루어진 경우
    if title.isdigit():
        return [title]

    # 특수 문자 제거
    title = re.sub(r'[^a-zA-Z가-힣]', ' ', title)

    tokenizer = Hannanum()
    tokens = tokenizer.morphs(title)

    return tokens


title = '1979년 12월 12일, 수도 서울 군사반란 발생 그날, 대한민국의 운명이 바뀌었다 대한민국을 뒤흔든 10월 26일 이후, 서울에 새로운 바람이 불어온 것도 잠시 12월 12일, 보안사령관 전두광이 반란을 일으키고 군 내 사조직을 총동원하여 최전선의 전방부대까지 서울로 불러들인다. 권력에 눈이 먼 전두광의 반란군과 이에 맞선 수도경비사령관 이태신을 비롯한 진압군 사이, 일촉즉발의 9시간이 흘러가는데… 목숨을 건 두 세력의 팽팽한 대립 오늘 밤, 대한민국 수도에서 가장 치열한 전쟁이 펼쳐진다!'
tokens = tokenize_title(title)

print(tokens)