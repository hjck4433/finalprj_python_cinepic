from elasticsearch import Elasticsearch
import movies
import json

es = Elasticsearch('http://localhost:9200')

# Elasticsearch 인덱스 이름 설정
index_name = "movies"

# 인덱스 매핑 정의
mapping = {
     "mappings": {
        "properties": {
            "movieId": {"type": "integer"},
            "movieTitle": {"type": "text"},
            "movieTitleEng": {"type": "text"},
            "movieRelease": {"type": "date"},
            "movieGenre": {"type": "text"},
            "movieNation": {"type": "text"},
            "movieGrade": {"type": "text"},
            "movieRuntime": {"type": "integer"},
            "movieScore": {"type": "float"},
            "movieDirector": {"type": "text"},
            "movieActors": {"type": "text"},
            "moviePlot": {"type": "text"},
            "moviePoster": {"type": "text"},
            "movieStills": {"type": "text"}
        }
    }
}

# 인덱스가 존재하지 않으면 정의된 매핑과 함께 인덱스 생성
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)

# movies 모듈에서 영화 데이터를 가져옴
movies_json = movies.send_movie_data()
movies_list = json.loads(movies_json)

# 각 영화 데이터를 Elasticsearch에 색인화
for movie in movies_list:
    es.index(index=index_name, body=movie)

# 인덱스 새로고침 (색인을 즉시 사용 가능하게 함)
es.indices.refresh(index=index_name)

# Elasticsearch에서 모든 영화를 검색하여 총 영화 수 출력
response = es.search(index=index_name, body={"query": {"match_all": {}}})
print(f"Total movies indexed: {response['hits']['total']['value']}")
