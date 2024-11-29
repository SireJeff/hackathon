import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def calculate_similarity(adverts_tfidf, products_tfidf, adverts_df, products_df, top_n=5):
    """Calculate similarity between advertisements and products."""
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

def find_nearest_neighbors(adverts_tfidf, products_tfidf, adverts_df, products_df, top_n=5):
    """Find top N nearest neighbors using Approximate Nearest Neighbors."""
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
                "similarity_score": 1 - distances[ad_idx][rank],
            })

    return pd.DataFrame(results)

def save_mappings(mappings_df, output_file='outputs/mappings.csv'):
    """Save mappings to a CSV file."""
    mappings_df.to_csv(output_file, index=False)
    print(f"Mappings saved to {output_file}")

if __name__ == '__main__':
    # Load preprocessed data and TF-IDF matrices
    adverts_df = pd.read_csv('outputs/adverts_processed.csv')
    products_df = pd.read_csv('outputs/products_processed.csv')
    adverts_tfidf = joblib.load('outputs/adverts_tfidf.pkl')
    products_tfidf = joblib.load('outputs/products_tfidf.pkl')

    # Calculate similarities
    # Use either exact similarity or ANN for optimization
    mappings_df = find_nearest_neighbors(adverts_tfidf, products_tfidf, adverts_df, products_df, top_n=5)

    # Save results
    save_mappings(mappings_df)
