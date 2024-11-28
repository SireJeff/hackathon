import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
import joblib
import os
import time
import re

def load_preprocessed_data(adverts_path, products_path):
    """Load preprocessed datasets."""
    adverts_df = pd.read_csv(adverts_path)
    products_df = pd.read_csv(products_path)
    return adverts_df, products_df

def clean_text(text):
    """Remove unwanted patterns or characters from text."""
    if not isinstance(text, str):
        return ""
    # Remove JSON-like patterns
    text = re.sub(r'\[\[.*?\]\]', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def translate_text(adverts_df, products_df):
    """Translate Persian text to English with retry logic."""
    translator = Translator()

    def safe_translate(text):
        if not isinstance(text, str) or not text.strip():
            return ""
        retries = 3
        while retries > 0:
            try:
                return translator.translate(text, src='fa', dest='en').text
            except Exception as e:
                print(f"Translation error for text: {text[:30]}... -> {e}")
                retries -= 1
                time.sleep(2)  # Wait before retrying
        return ""

    # Clean text before translation
    adverts_df['full_text'] = adverts_df['full_text'].apply(clean_text)
    products_df['full_text'] = products_df['full_text'].apply(clean_text)

    # Translate text
    adverts_df['translated_text'] = adverts_df['full_text'].apply(safe_translate)
    products_df['translated_text'] = products_df['full_text'].apply(safe_translate)

    return adverts_df, products_df

def vectorize_translated_text(adverts_df, products_df, max_features=5000):
    """Vectorize translated text data using TF-IDF."""
    # Extract translated text
    adverts_text = adverts_df['translated_text'].tolist()
    products_text = products_df['translated_text'].tolist()

    # Combine text for consistent vectorization
    combined_text = adverts_text + products_text

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    # Fit and transform the combined text
    combined_tfidf = vectorizer.fit_transform(combined_text)

    # Split back into individual datasets
    adverts_tfidf = combined_tfidf[:len(adverts_text)]
    products_tfidf = combined_tfidf[len(adverts_text):]

    return adverts_tfidf, products_tfidf, vectorizer

def save_outputs(adverts_df, products_df, adverts_tfidf, products_tfidf, vectorizer, output_dir='outputs/'):
    """Save processed data and TF-IDF outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrames with translations
    adverts_df.to_csv(f'{output_dir}/adverts_translated.csv', index=False)
    products_df.to_csv(f'{output_dir}/products_translated.csv', index=False)

    # Save TF-IDF matrices and vectorizer
    joblib.dump(adverts_tfidf, f'{output_dir}/adverts_tfidf.pkl')
    joblib.dump(products_tfidf, f'{output_dir}/products_tfidf.pkl')
    joblib.dump(vectorizer, f'{output_dir}/tfidf_vectorizer.pkl')

if __name__ == '__main__':
    # File paths
    adverts_path = 'data/processed/adverts_preprocessed.csv'
    products_path = 'data/processed/products_preprocessed.csv'

    # Load preprocessed data
    adverts_df, products_df = load_preprocessed_data(adverts_path, products_path)

    # Translate text
    adverts_df, products_df = translate_text(adverts_df, products_df)

    # Vectorize translated text
    adverts_tfidf, products_tfidf, vectorizer = vectorize_translated_text(adverts_df, products_df)

    # Save outputs
    save_outputs(adverts_df, products_df, adverts_tfidf, products_tfidf, vectorizer)
