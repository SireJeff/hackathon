import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
import os
import joblib

# Load spaCy Persian model
nlp = spacy.blank('xx')  # Use multi-language model that includes Persian processing

def preprocess_persian_text(text):
    """Normalize, tokenize, and clean Persian text."""
    if not isinstance(text, str):
        return ""
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Tokenize, lemmatize, and remove stopwords and punctuation
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(tokens)

def preprocess_persian_texts(adverts_df, products_df, batch_size=1000, n_processes=4):
    """Apply Persian text preprocessing to datasets in batches with parallelization."""
    
    # Using Parallel and delayed to process in parallel across multiple cores
    def preprocess_batch(batch_texts):
        return [preprocess_persian_text(text) for text in batch_texts]
    
    # Process the advertisements and products texts in batches
    adverts_batches = [adverts_df['full_text'][i:i+batch_size] for i in range(0, len(adverts_df), batch_size)]
    products_batches = [products_df['full_text'][i:i+batch_size] for i in range(0, len(products_df), batch_size)]
    
    with Parallel(n_jobs=n_processes) as parallel:
        adverts_processed = parallel(delayed(preprocess_batch)(batch) for batch in adverts_batches)
        products_processed = parallel(delayed(preprocess_batch)(batch) for batch in products_batches)

    # Flatten the list of lists
    adverts_df['processed_text'] = [item for sublist in adverts_processed for item in sublist]
    products_df['processed_text'] = [item for sublist in products_processed for item in sublist]
    
    return adverts_df, products_df

# TF-IDF Vectorization with parallelization
def vectorize_persian_text(adverts_df, products_df, max_features=5000, n_processes=4):
    """Vectorize Persian text data using TF-IDF with parallelization."""
    # Extract processed text
    adverts_text = adverts_df['processed_text'].tolist()
    products_text = products_df['processed_text'].tolist()

    # Combine text for consistent vectorization
    combined_text = adverts_text + products_text

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')  # 'farsi' option can be problematic

    # Parallelize TF-IDF fitting and transforming
    def fit_transform_batch(batch_texts):
        return vectorizer.fit_transform(batch_texts)

    # Split the combined text into smaller chunks
    chunks = [combined_text[i:i + 10000] for i in range(0, len(combined_text), 10000)]

    with Parallel(n_jobs=n_processes) as parallel:
        tfidf_batches = parallel(delayed(fit_transform_batch)(chunk) for chunk in chunks)

    # Combine all the TF-IDF results
    from scipy.sparse import vstack
    combined_tfidf = vstack(tfidf_batches)

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
    adverts_df, products_df = preprocess_persian_texts(adverts_df, products_df, batch_size=1000, n_processes=4)

    # Vectorize Persian text
    adverts_tfidf, products_tfidf, vectorizer = vectorize_persian_text(adverts_df, products_df, max_features=5000, n_processes=4)

    # Save outputs
    save_outputs(adverts_df, products_df, adverts_tfidf, products_tfidf, vectorizer)
