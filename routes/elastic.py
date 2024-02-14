import logging
from elasticsearch import Elasticsearch, helpers
from . import movies
import json
from env import settings

logging.basicConfig(level=logging.INFO)


def index_movies():
    es_config = settings.ELASTICSEARCH_CONFIG

    es = Elasticsearch(
        [es_config['host']],
        http_auth=(es_config['username'], es_config['password'])
    )

    # Elasticsearch 인덱스 이름 설정
    index_name = "movies"

    # 인덱스 설정 및 매핑 정의
    index_body = {
        "settings": {
            "analysis": {
                "tokenizer": {
                    "nori_user_dict_tokenizer": {
                        "type": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "discard_punctuation": "false"
                    }
                },
                "filter": {
                    "korean_stop": {
                        "type": "stop",
                        "stopwords_path": "analysis/stopwords/korean_stopwords.txt"
                    },
                    "nori_filter": {
                      "type": "nori_part_of_speech",
                      "stoptags": [
                        "E", "IC", "J", "MAG", "MAJ", "MM", "SP", "SSC", "SSO", "SC", "SE", "XPN", "XSA", "XSN", "XSV",
                        "UNA", "NA", "VSV", "NP"
                      ]
                    },
                    "ngram_filter": {
                        "type": "ngram",
                        "min_gram": 2,
                        "max_gram": 3
                    },
                    "english_ngram_filter": {
                        "type": "ngram",
                        "min_gram": 2,
                        "max_gram": 3
                    },
                },
                "analyzer": {
                    "nori_analyzer_with_stopwords": {
                        "type": "custom",
                        "tokenizer": "nori_user_dict_tokenizer",
                        "filter": ["nori_readingform", "korean_stop", "nori_filter", "trim"]
                    },
                    "nori_ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_user_dict_tokenizer",
                        "filter": ["nori_readingform", "ngram_filter", "trim"]
                    },
                    "english_ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "english_ngram_filter", "trim"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "movieId": {"type": "long"},
                "movieTitle": {
                    "type": "text",
                    "analyzer": "nori_analyzer_with_stopwords",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "nori_ngram_analyzer"
                        }
                    }
                },
                "movieTitleEng": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "english_ngram_analyzer"
                        }
                    }
                },
                "movieRelease": {
                    "type": "date",
                    "format": "yyyy-MM-dd"
                },
                "movieGenre": {
                    "type": "text",
                    "analyzer": "nori_analyzer_with_stopwords",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "nori_ngram_analyzer"
                        }
                    }
                },
                "movieNation": {"type": "text"},
                "movieGrade": {"type": "text"},
                "movieRuntime": {"type": "keyword"},
                "movieScore": {"type": "keyword"},
                "movieDirector": {
                    "type": "text",
                    "analyzer": "nori_analyzer_with_stopwords",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "nori_ngram_analyzer"
                        }
                    }
                },
                "movieActors": {
                    "type": "text",
                    "analyzer": "nori_analyzer_with_stopwords",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "nori_ngram_analyzer"
                        }
                    }
                },
                "moviePlot": {"type": "text", "analyzer": "nori_analyzer_with_stopwords"},
                "moviePoster": {"type": "keyword"},
                "movieStills": {"type": "keyword"}
            }
        }
    }

    try:
        # 인덱스가 존재하지 않으면 정의된 매핑과 함께 인덱스 생성
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, body=index_body)

        # movies 모듈에서 영화 데이터를 가져옴
        movies_json = movies.send_movie_data()
        movies_list = json.loads(movies_json)

        # 벌크 인덱싱 준비
        actions = [
            {
                "_index": index_name,
                "_source": movie
            }
            for movie in movies_list
        ]

        #  helpers.bulk() 를 사용해 벌크 인덱싱 수행
        helpers.bulk(es, actions)

        # 인덱스 새로고침 (색인을 즉시 사용 가능하게 함)
        es.indices.refresh(index=index_name)

        # Elasticsearch에서 모든 영화를 검색하여 총 영화 수 출력
        response = es.search(index=index_name, body={"query": {"match_all": {}}})
        total_movies = response['hits']['total']['value']
        success_message = f"Indexing completed successfully. Total movies indexed: {total_movies}"
        return success_message

    except Exception as e:
        logging.error("An error occurred during indexing:", exc_info=True)
        return f"An error occurred during indexing: {str(e)}"