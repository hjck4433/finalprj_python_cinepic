import pandas as pd
import os


file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'recs_data_set.csv')

df = pd.read_csv(file_path, encoding='utf-8')

print(df)