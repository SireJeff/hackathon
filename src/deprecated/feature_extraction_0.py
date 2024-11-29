import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from transformers import MarianMTModel, MarianTokenizer

# Translation Functions
def local_translate_texts(texts, model_name="Helsinki-NLP/opus-mt-fa-en"):
    """Translate texts using a local machine translation model."""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            translated.append("")
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(**inputs)
        translated.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return translated

def translate_text_local(adverts_df, products_df):
    """Translate text fields using a local translation model."""
    model_name = "Helsinki-NLP/opus-mt-fa-en"

    # Translate advertisements
    adverts_df['translated_text'] = local_translate_texts(adverts_df['full_text'].tolist(), model_name)

    # Translate products
    products_df['translated_text'] = local_translate_texts(products_df['full_text'].tolist(), model_name)

    return adverts_df, products_df

# TF-IDF Vectorization
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

# Saving Intermediate Results
def save_partial_results(adverts_df, products_df, output_dir='outputs/'):
    """Save translated data periodically."""
    os.makedirs(output_dir, exist_ok=True)
    adverts_df.to_csv(f'{output_dir}/adverts_translated_partial.csv', index=False)
    products_df.to_csv(f'{output_dir}/products_translated_partial.csv', index=False)

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
    adverts_df = pd.read_csv(adverts_path)
    products_df = pd.read_csv(products_path)

    # Translate text locally
    adverts_df, products_df = translate_text_local(adverts_df, products_df)

    # Save intermediate results
    save_partial_results(adverts_df, products_df)

    # Vectorize translated text
    adverts_tfidf, products_tfidf, vectorizer = vectorize_translated_text(adverts_df, products_df)

    # Save outputs
    save_outputs(adverts_df, products_df, adverts_tfidf, products_tfidf, vectorizer)
