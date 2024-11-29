Sure! Here's a detailed **README.md** file that explains the entire process, from preprocessing the raw data to feature extraction and similarity calculation, based on the directory structure you've provided.

---

# Hackathon Project: Advertisement-Product Matching

## Overview

This project involves processing advertisement and product data to calculate similarities between them using Natural Language Processing (NLP) and machine learning techniques. The goal is to match advertisements to products based on textual similarity, which can be helpful in recommendation systems, search engines, and other related applications.

### Workflow:
1. **Preprocessing**: Clean and normalize the raw text data.
2. **Feature Extraction**: Convert the cleaned text data into numerical features (TF-IDF).
3. **Similarity Calculation**: Compute the similarity between advertisements and products using various methods, including cosine similarity, FAISS, and Nearest Neighbors.

---

## Directory Structure

```
hackathon
├── .gitignore
├── adverts.csv
├── data
│   ├── adverts.csv
│   ├── processed
│   │   ├── adverts_preprocessed.csv
│   │   └── products_preprocessed.csv
│   └── productss.csv
├── notebooks
├── outputs
│   ├── adverts_to_products_mappings.csv
│   └── tfidf_vectorizer.pkl
├── productss.csv
├── README.md
├── requirements.txt
└── src
    ├── feature_extraction.py
    ├── feature_extraction1.py
    ├── feature_extraction_0.py
    ├── preprocessing.py
    ├── preprocessing1.py
    ├── processingReadMe.md
    ├── similarity.py
    └── similarity1.py
```

---

## File Descriptions

### `data/`
This folder contains the raw and processed data:
- **`adverts.csv`**: The raw advertisements dataset.
- **`productss.csv`**: The raw products dataset.
- **`processed/`**: Contains the cleaned and preprocessed datasets:
  - **`adverts_preprocessed.csv`**: The preprocessed advertisements data.
  - **`products_preprocessed.csv`**: The preprocessed products data.

### `outputs/`
This folder contains the output files generated from the feature extraction and similarity calculations:
- **`adverts_to_products_mappings.csv`**: Contains the results of the similarity calculation. Each row represents a pair of an advertisement and a product with a similarity score.
- **`tfidf_vectorizer.pkl`**: The saved TF-IDF vectorizer used for transforming raw text into numerical features.

### `src/`
This folder contains the Python scripts that perform the different stages of the pipeline:
- **`preprocessing.py`**: Contains functions to preprocess the raw advertisement and product data (tokenization, stopword removal, etc.).
- **`feature_extraction.py`**: Extracts features from the preprocessed text using the TF-IDF method.
- **`similarity.py`**: Calculates the similarity between advertisements and products using cosine similarity, FAISS, and Nearest Neighbors.

---

## Workflow

### 1. Preprocessing (`preprocessing.py`)

The preprocessing script performs the following steps:
- **Normalization**: Converts all text to lowercase, removes extra spaces, and applies other text normalization techniques.
- **Tokenization**: Splits the text into tokens (words).
- **Stopword Removal**: Removes common words (like "the", "is", etc.) that don't carry meaningful information.
- **Punctuation Removal**: Removes punctuation marks from the text.

#### Running the Preprocessing:
To preprocess the raw data, run the following command:
```bash
python src/preprocessing.py
```
This will generate the processed files:
- `adverts_preprocessed.csv`
- `products_preprocessed.csv`

---

### 2. Feature Extraction (`feature_extraction.py`)

In this step, we convert the preprocessed text data into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**. The script performs the following steps:
- **Loading Processed Text**: Reads the preprocessed text data.
- **TF-IDF Vectorization**: Transforms the processed text into numerical features (TF-IDF matrices), where each document (advertisement/product) is represented as a vector.
- **Saving Output**: Saves the vectorized TF-IDF matrices and the vectorizer (`tfidf_vectorizer.pkl`) for later use.

#### Running Feature Extraction:
To perform feature extraction, run the following command:
```bash
python src/feature_extraction.py
```

This will generate:
- **TF-IDF Matrices** (`adverts_tfidf.pkl`, `products_tfidf.pkl`) - Saved as `joblib` files.
- **TF-IDF Vectorizer** (`tfidf_vectorizer.pkl`) - Saved for future use.

---

### 3. Similarity Calculation (`similarity.py`)

In this step, we calculate the similarity between advertisements and products based on their TF-IDF vectors. We use different similarity calculation methods:
- **Cosine Similarity**: Measures the cosine of the angle between two vectors, providing a similarity score.
- **FAISS (Approximate Nearest Neighbors)**: A highly efficient method for nearest neighbor search using **L2 distance** (which is similar to cosine similarity).
- **Nearest Neighbors (sklearn)**: A method using brute-force search to calculate cosine similarity between TF-IDF vectors.

The final output will be saved in the **`adverts_to_products_mappings.csv`** file.

#### Running Similarity Calculation:
To calculate the similarity, run the following command:
```bash
python src/similarity.py
```

This will generate:
- **`adverts_to_products_mappings.csv`**: A CSV file containing advertisement-product pairs with their similarity scores.

---

## Detailed Functionality

### `preprocessing.py`
1. **Tokenization**: Breaks down raw text into individual words.
2. **Stopword Removal**: Removes Persian stopwords using the Hazm library.
3. **Punctuation Removal**: Removes unnecessary punctuation.

### `feature_extraction.py`
1. **Load Processed Data**: Loads the preprocessed advertisement and product data.
2. **TF-IDF Vectorization**: Uses the **TF-IDF vectorizer** to transform the text data into numerical vectors.
3. **Save Vectorizer and TF-IDF Matrices**: Saves the vectorizer and matrices to be used later in similarity calculations.

### `similarity.py`
1. **Cosine Similarity**: Calculates similarity between each advertisement and each product.
2. **FAISS**: Uses the **FAISS** library for faster approximate nearest neighbor search.
3. **Nearest Neighbors**: Uses **sklearn**'s `NearestNeighbors` to find the most similar products to each advertisement based on cosine similarity.

---

## Requirements

The following libraries are required to run the scripts:
- **pandas**: For data manipulation.
- **spacy**: For natural language processing tasks.
- **scikit-learn**: For TF-IDF vectorization and similarity calculation.
- **faiss**: For fast nearest neighbor search.
- **hazm**: For Persian language processing (e.g., stopword removal).

Install the necessary packages by running:
```bash
pip install -r requirements.txt
```

---

## Output Files

1. **`adverts_to_products_mappings.csv`**:
   - Contains the advertisement-product pairs with similarity scores. Each row includes:
     - `advertisement_id`: The ID of the advertisement.
     - `product_id`: The ID of the product.
     - `similarity_score`: The cosine similarity score between the advertisement and the product.

2. **`tfidf_vectorizer.pkl`**:
   - The saved TF-IDF vectorizer that can be used for transforming new text data.

---

## Conclusion

This project automates the process of matching advertisements to products based on text similarity using advanced NLP techniques like TF-IDF and FAISS. The results can be used for recommendations, product searches, and enhancing user experiences on e-commerce platforms.

---

### **Next Steps**:
1. Run `feature_extraction.py` to generate processed data and TF-IDF matrices.
2. Run `similarity.py` to calculate and save the similarity results between advertisements and products.
3. Use the `adverts_to_products_mappings.csv` file for further analysis or integration into a recommendation system.

---

Let me know if you'd like any further clarifications or adjustments to the documentation!