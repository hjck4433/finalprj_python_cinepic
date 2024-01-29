import pandas as pd
import os
from konlpy.tag import Okt, Hannanum
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def preprocess_names(text):
    if text.strip() == '알수없음':
        return '알수없음'
    return ' '.join([name.strip() for name in text.split(',')])

def preprocess_genre(genre):
    genre = re.sub(r'[^\w\s,\/]', '', genre)
    tokens = re.split(r',|\/', genre)
    return ' '.join([token.strip() for token in tokens if token.strip()])

def preprocess_title(title, stopwords, min_token_length=1):
    # Original title backup in case tokenization results in an empty string
    original_title = title

    # Tokenize the title
    tokenizer = Okt()
    tokens = tokenizer.morphs(title)

    # Remove stopwords and tokens below the minimum length threshold
    tokens = [token for token in tokens if token not in stopwords and len(token) >= min_token_length]

    # Check if the tokenization result is empty
    if not tokens:
        return original_title  # Return the original title if no tokens remain

    return ' '.join(tokens)  # Return the processed title otherwise

def preprocess_plot(plot, stopwords):
    # Clean and tokenize the plot text
    plot = re.sub(r'[^\w\s]', '', plot)  # Remove punctuation
    tokenizer = Okt()
    tokens = tokenizer.morphs(plot)  # Tokenize

    # Filter out stopwords and single-character words
    tokens = [token for token in tokens if token not in stopwords and len(token) > 1]

    # Return the cleaned plot as a string, suitable for vectorization
    return ' '.join(tokens)

def preprocess_numeric_columns(data):
    numeric_columns = ['audience', 'screen', 'screening', 'movieScore', 'peopleVoted']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')
        scaler = StandardScaler() if col == 'movieScore' else MinMaxScaler()
        data[col] = scaler.fit_transform(data[[col]].fillna(0))
    return data

def preprocess_date_column(data):
    data['movieRelease'] = pd.to_datetime(data['movieRelease'], errors='coerce')
    return data

def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

# Load data and stopwords
input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'recs_data_set.csv')
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data_okt_v4.csv')
stopword_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'StopWords.txt')

data = pd.read_csv(input_path, encoding='utf-8')
stopwords = load_stopwords(stopword_path)

# Preprocess data
data = preprocess_numeric_columns(data)
data = preprocess_date_column(data)
data['movieGenre'] = data['movieGenre'].apply(preprocess_genre)
data['movieTitleTokenized'] = data['movieTitle'].apply(preprocess_title, args=(stopwords,))
data['moviePlot'] = data['moviePlot'].apply(preprocess_plot, args=(stopwords,))
data['movieDirector'] = data['movieDirector'].apply(preprocess_names)
data['movieActors'] = data['movieActors'].apply(preprocess_names)


# Initialize the TF-IDF vectorizer without a max_features limit
tfidf_vectorizer = TfidfVectorizer()

# Prepare the data for vectorization by concatenating all textual data into one series
# Ensure you have a 'movieTitleTokenized' column with preprocessed titles
all_text_data = pd.concat([data['movieDirector'], data['movieActors'], data['movieGenre'], data['moviePlot'], data['movieTitleTokenized']])

# Fit the vectorizer on the combined text data to create a unified feature space
tfidf_vectorizer.fit(all_text_data)

# Define the desired maximum number of SVD components
desired_n_components = 50

# Function to perform TF-IDF transformation and SVD dimensionality reduction
def process_text_feature(column_name, tfidf_vectorizer, data, n_components):
    # Transform the text data to TF-IDF feature matrix
    tfidf_matrix = tfidf_vectorizer.transform(data[column_name])

    # Adjust the number of SVD components based on the shape of the TF-IDF matrix
    n_components = min(tfidf_matrix.shape[1] - 1, n_components)

    # Perform dimensionality reduction on the TF-IDF matrix
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(tfidf_matrix)

    # Create a DataFrame from the SVD-transformed matrix
    reduced_df = pd.DataFrame(reduced_matrix, columns=[f'{column_name}_tfidf_svd_{i + 1}' for i in range(n_components)])

    return reduced_df

# Process each text column and concatenate the resulting dataframes to the original data
for column in ['movieDirector', 'movieActors', 'movieGenre', 'moviePlot', 'movieTitleTokenized']:
    reduced_df = process_text_feature(column, tfidf_vectorizer, data, desired_n_components)
    data = pd.concat([data, reduced_df], axis=1)

# Save the preprocessed data
data.to_csv(output_path, index=False, encoding='utf-8-sig')

