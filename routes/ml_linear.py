import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec


# 벡터화된 데이터 불러오기
input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'word2vec_results.csv')
data = pd.read_csv(input_path, encoding='utf-8')

# Calculate the correlation matrix
correlation_matrix = data.corr(numeric_only=True)

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()