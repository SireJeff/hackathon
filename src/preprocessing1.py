import pandas as pd
import spacy
import re
import os
from spacy.lang.fa import Persian
from spacy.util import minibatch
from concurrent.futures import ProcessPoolExecutor

# Load spaCy Persian model
nlp = Persian()

def preprocess_persian_text(text):
    """Full preprocessing pipeline for Persian text using spaCy."""
    if not isinstance(text, str):
        return ""
    
    # Normalize text (similar to hazm Normalizer but done with regex)
    text = re.sub(r'[^آ-ی\s]', '', text)  # Remove non-Persian characters

    # Process text using spaCy NLP pipeline
    doc = nlp(text)
    
    # Tokenize and lemmatize, removing stopwords and punctuation
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(tokens)

def preprocess_data_batch(adverts_df, products_df, batch_size=1000, n_processes=4):
    """Preprocess advertisements and products data in batches for faster processing."""
    # Define a function to preprocess in parallel
    def preprocess_batch(batch_texts):
        return [preprocess_persian_text(text) for text in batch_texts]
    
    # Using minibatch for efficient batch processing
    adverts_batches = minibatch(adverts_df['full_text'], size=batch_size)
    products_batches = minibatch(products_df['full_text'], size=batch_size)

    # Use ProcessPoolExecutor to parallelize processing across multiple CPU cores
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        adverts_processed = list(executor.map(preprocess_batch, adverts_batches))
        products_processed = list(executor.map(preprocess_batch, products_batches))

    # Flatten the list of lists returned from executor.map
    adverts_df['processed_text'] = [item for sublist in adverts_processed for item in sublist]
    products_df['processed_text'] = [item for sublist in products_processed for item in sublist]
    
    return adverts_df, products_df

def preprocess_data(adverts_df, products_df):
    """Preprocess advertisements and products data."""
    # Fill missing values
    adverts_df['advertisement_description'] = adverts_df['advertisement_description'].fillna('')
    products_df['product_description'] = products_df['product_description'].fillna('')
    print("Finished filling missing values")

    # Combine title and description into one field for easier processing
    adverts_df['full_text'] = adverts_df['advertisement_title'] + ' ' + adverts_df['advertisement_description']
    products_df['full_text'] = products_df['product_title'] + ' ' + products_df['product_description']
    print("Finished combining title and description")

    # Preprocess data in batches using spaCy and parallelization
    adverts_df, products_df = preprocess_data_batch(adverts_df, products_df, batch_size=1000, n_processes=4)
    print("Finished text preprocessing")

    return adverts_df, products_df

def save_preprocessed_data(adverts_df, products_df, output_dir='data/processed/'):
    """Save preprocessed datasets."""
    os.makedirs(output_dir, exist_ok=True)
    adverts_df.to_csv(f'{output_dir}/adverts_preprocessed.csv', index=False)
    products_df.to_csv(f'{output_dir}/products_preprocessed.csv', index=False)

if __name__ == '__main__':
    # Define file paths
    adverts_path = 'data/adverts.csv'
    products_path = 'data/productss.csv'

    # Load datasets
    adverts_df = pd.read_csv(adverts_path)
    products_df = pd.read_csv(products_path)

    # Preprocess data
    adverts_df, products_df = preprocess_data(adverts_df, products_df)

    # Save preprocessed data
    save_preprocessed_data(adverts_df, products_df)
