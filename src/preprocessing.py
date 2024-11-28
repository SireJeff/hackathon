import pandas as pd
import os

def load_data(adverts_path, products_path):
    """Load raw advertisements and products datasets."""
    adverts_df = pd.read_csv(adverts_path)
    products_df = pd.read_csv(products_path)
    return adverts_df, products_df

def preprocess_data(adverts_df, products_df):
    """Preprocess advertisements and products data."""
    # Fill missing values
    adverts_df['advertisement_description'] = adverts_df['advertisement_description'].fillna('')
    products_df['product_description'] = products_df['product_description'].fillna('')

    # Combine title and description into one field for easier processing
    adverts_df['full_text'] = adverts_df['advertisement_title'] + ' ' + adverts_df['advertisement_description']
    products_df['full_text'] = products_df['product_title'] + ' ' + products_df['product_description']

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

    # Load data
    adverts_df, products_df = load_data(adverts_path, products_path)

    # Preprocess data
    adverts_df, products_df = preprocess_data(adverts_df, products_df)

    # Save preprocessed data
    save_preprocessed_data(adverts_df, products_df)
