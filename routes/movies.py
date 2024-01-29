import pandas as pd
import json
import os

def send_movie_data():
    input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'movie_db.csv')
    df = pd.read_csv(input_path, encoding='utf-8')

    # Replace NaN with an empty string
    df.fillna('', inplace=True)

    # Convert DataFrame to a list of dictionaries
    movies_list = df.to_dict(orient='records')

    # Iterate over the movies and unescape the URLs
    for movie in movies_list:
        if 'moviePoster' in movie and isinstance(movie['moviePoster'], str):
            movie['moviePoster'] = movie['moviePoster'].replace('\\/', '/')
        if 'movieStills' in movie and isinstance(movie['movieStills'], str):
            movie['movieStills'] = movie['movieStills'].replace('\\/', '/')

    # Convert the list of dictionaries to a JSON string
    movies_json = json.dumps(movies_list, ensure_ascii=False, indent=4)

    return movies_json