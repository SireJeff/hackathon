import pandas as pd
from hazm import Normalizer, word_tokenize, Lemmatizer, stopwords_list
import re
import os


def preprocess_persian_text(text):
    """Full preprocessing pipeline for Persian text."""
    if not isinstance(text, str):
        return ""
    
    # Initialize tools
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()
    stopwords = set(stopwords_list())

    # Normalize text
    text = normalizer.normalize(text)
    #print("finished normalization")
    # Remove non-Persian characters
    text = re.sub(r'[^آ-ی\s]', '', text)
    #print("finished removing non-persian charachters")
    # Tokenize
    tokens = word_tokenize(text)
    #print("finished tokenization")
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords]
    #print("stepwords removed")
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    #print("finished lemmatization")
    return " ".join(tokens)


def preprocess_data(adverts_df, products_df):
    """Preprocess advertisements and products data."""
    # Fill missing values
    adverts_df['advertisement_description'] = adverts_df['advertisement_description'].fillna('')
    products_df['product_description'] = products_df['product_description'].fillna('')

    print("finished filling missed values")

    # Combine title and description into one field for easier processing
    adverts_df['full_text'] = adverts_df['advertisement_title'] + ' ' + adverts_df['advertisement_description']
    products_df['full_text'] = products_df['product_title'] + ' ' + products_df['product_description']
    print("finished Combine title ")
    # Apply Persian text preprocessing
    adverts_df['processed_text'] = adverts_df['full_text'].apply(preprocess_persian_text)
    products_df['processed_text'] = products_df['full_text'].apply(preprocess_persian_text)
    print("finished Apply Persian text")
    
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
