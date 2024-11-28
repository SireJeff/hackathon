import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from hazm import Normalizer, word_tokenize
import joblib
import os

# Persian Preprocessing
def preprocess_persian_text(text):
    """Normalize and tokenize Persian text."""
    if not isinstance(text, str):
        return ""
    normalizer = Normalizer()
    text = normalizer.normalize(text)  # Normalize text
    tokens = word_tokenize(text)  # Tokenize into words
    return " ".join(tokens)

def preprocess_persian_texts(adverts_df, products_df):
    """Apply Persian text preprocessing to datasets."""
    adverts_df['processed_text'] = adverts_df['full_text'].apply(preprocess_persian_text)
    products_df['processed_text'] = products_df['full_text'].apply(preprocess_persian_text)
    return adverts_df, products_df

# TF-IDF Vectorization
def vectorize_persian_text(adverts_df, products_df, max_features=5000):
    """Vectorize Persian text data using TF-IDF."""
    # Extract processed text
    adverts_text = adverts_df['processed_text'].tolist()
    products_text = products_df['processed_text'].tolist()

    # Combine text for consistent vectorization
    combined_text = adverts_text + products_text

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='farsi')

    # Fit and transform the combined text
    combined_tfidf = vectorizer.fit_transform(combined_text)

    # Split back into individual datasets
    adverts_tfidf = combined_tfidf[:len(adverts_text)]
    products_tfidf = combined_tfidf[len(adverts_text):]

    return adverts_tfidf, products_tfidf, vectorizer

# Saving Results
def save_outputs(adverts_df, products_df, adverts_tfidf, products_tfidf, vectorizer, output_dir='outputs/'):
    """Save processed data and TF-IDF outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrames with processed Persian text
    adverts_df.to_csv(f'{output_dir}/adverts_processed.csv', index=False)
    products_df.to_csv(f'{output_dir}/products_processed.csv', index=False)

    # Save TF-IDF matrices and vectorizer
    joblib.dump(adverts_tfidf, f'{output_dir}/adverts_tfidf.pkl')
    joblib.dump(products_tfidf, f'{output_dir}/products_tfidf.pkl')
    joblib.dump(vectorizer, f'{output_dir}/tfidf_vectorizer.pkl')

if __name__ == '__main__':
    # File paths
    adverts_path = 'data/processed/adverts_preprocessed.csv'
    products_path = 'data/processed/products_preprocessed.csv'

    # Load preprocessed data
    adverts_df = pd.read_csv(adverts_path)
    products_df = pd.read_csv(products_path)

    # Preprocess Persian text
    adverts_df, products_df = preprocess_persian_texts(adverts_df, products_df)

    # Vectorize Persian text
    adverts_tfidf, products_tfidf, vectorizer = vectorize_persian_text(adverts_df, products_df)

    # Save outputs
    save_outputs(adverts_df, products_df, adverts_tfidf, products_tfidf, vectorizer)
