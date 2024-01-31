import pandas as pd
import os
import json

def convert_boolean(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return value.lower() == 'true'
    else:
        return False
def get_theater_data() :
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'independent_theater_info.csv')
    theater_csv = pd.read_csv(file_path, encoding='utf-8')

    # boolean
    theater_csv['isSpecialScreen'] = theater_csv['isSpecialScreen'].apply(convert_boolean)

    theater_csv['seats'] = theater_csv['seats'].str.replace(',', '').astype(int)

    # 숫자로 변환
    columns_to_convert = ['theaterId','screens', 'seats', 'screenFilm', 'screen2D', 'screen3D', 'screen4D', 'screenImax',
                          'latitude', 'longitude']
    for column in columns_to_convert:
        theater_csv[column] = pd.to_numeric(theater_csv[column], errors='coerce')

    #JSON으로 변환
    theater_json = json.dumps(json.loads(theater_csv.to_json(orient='records', force_ascii=False)), ensure_ascii=False)

    # print(theater_csv)

    return theater_json

# get_theater_data()