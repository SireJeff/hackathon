import pandas as pd
import joblib
import faiss  # Use FAISS for Approximate Nearest Neighbors (ANN)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import os

def calculate_similarity(adverts_tfidf, products_tfidf, adverts_df, products_df, top_n=5):
    """Calculate similarity between advertisements and products using cosine similarity."""
    # Convert to dense if necessary, but avoid if using FAISS for memory efficiency
    similarity_matrix = cosine_similarity(adverts_tfidf, products_tfidf)

    results = []
    for ad_idx, similarities in enumerate(similarity_matrix):
        top_indices = similarities.argsort()[-top_n:][::-1]
        for product_idx in top_indices:
            results.append({
                "advertisement_id": adverts_df.iloc[ad_idx]["hash_id"],
                "product_id": products_df.iloc[product_idx]["hash_id"],
                "similarity_score": similarities[product_idx],
            })

    return pd.DataFrame(results)

def find_nearest_neighbors_faiss(adverts_tfidf, products_tfidf, adverts_df, products_df, top_n=5):
    """Find top N nearest neighbors using FAISS for Approximate Nearest Neighbors."""
    
    # FAISS requires dense format, so we need to convert sparse to dense
    adverts_dense = adverts_tfidf.toarray()
    products_dense = products_tfidf.toarray()
    
    # Create FAISS index for Approximate Nearest Neighbor search
    index = faiss.IndexFlatL2(products_dense.shape[1])  # L2 distance metric (cosine is similar to L2)
    index.add(products_dense)  # Add product vectors to the index
    
    # Search for nearest neighbors for each advert
    distances, indices = index.search(adverts_dense, top_n)  # Search for the top_n nearest neighbors

    results = []
    for ad_idx, product_indices in enumerate(indices):
        for rank, product_idx in enumerate(product_indices):
            results.append({
                "advertisement_id": adverts_df.iloc[ad_idx]["hash_id"],
                "product_id": products_df.iloc[product_idx]["hash_id"],
                "rank": rank + 1,
                "similarity_score": 1 - distances[ad_idx][rank],  # Convert L2 distance to cosine similarity
            })

    return pd.DataFrame(results)

def find_nearest_neighbors_sklearn(adverts_tfidf, products_tfidf, adverts_df, products_df, top_n=5):
    """Find top N nearest neighbors using Nearest Neighbors (sklearn)."""
    nn = NearestNeighbors(n_neighbors=top_n, metric='cosine', algorithm='brute')
    nn.fit(products_tfidf)

    distances, indices = nn.kneighbors(adverts_tfidf)

    results = []
    for ad_idx, product_indices in enumerate(indices):
        for rank, product_idx in enumerate(product_indices):
            results.append({
                "advertisement_id": adverts_df.iloc[ad_idx]["hash_id"],
                "product_id": products_df.iloc[product_idx]["hash_id"],
                "rank": rank + 1,
                "similarity_score": 1 - distances[ad_idx][rank],  # Convert to cosine similarity
            })

    return pd.DataFrame(results)

def save_mappings(mappings_df, output_file='outputs/adverts_to_products_mappings.csv'):
    """Save mappings to a CSV file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mappings_df.to_csv(output_file, index=False)
    print(f"Mappings saved to {output_file}")

if __name__ == '__main__':
    # File paths from feature_extraction.py output
    adverts_path = 'data/processed/adverts_preprocessed.csv'  # Path from feature_extraction.py output
    products_path = 'data/processed/products_preprocessed.csv'  # Path from feature_extraction.py output
    tfidf_vectorizer_path = 'outputs/tfidf_vectorizer.pkl'  # Path for saved TF-IDF vectorizer

    # Load preprocessed data and TF-IDF vectorizer
    adverts_df = pd.read_csv(adverts_path)
    products_df = pd.read_csv(products_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

    # Vectorize the processed texts using the saved TF-IDF vectorizer
    adverts_tfidf = tfidf_vectorizer.transform(adverts_df['processed_text'])
    products_tfidf = tfidf_vectorizer.transform(products_df['processed_text'])

    # Calculate similarities using FAISS (or use sklearn for smaller datasets)
    mappings_df = find_nearest_neighbors_faiss(adverts_tfidf, products_tfidf, adverts_df, products_df, top_n=5)

    # Save results
    save_mappings(mappings_df)
