 explaining both `preprocessing.py` and `feature_extraction.py` modules in detail:

---

# **Hackathon Project**

This project processes and analyzes advertisements and product data to establish relationships and mappings for a Persian dataset. The project pipeline consists of two main components:

1. **Preprocessing Module (`preprocessing.py`)**
2. **Feature Extraction Module (`feature_extraction.py`)**

---

## **1. Preprocessing Module (`preprocessing.py`)**

### **Purpose**
The `preprocessing.py` module handles the initial preparation of the raw Persian text data from the `adverts.csv` and `productss.csv` files. It performs advanced text preprocessing to clean, normalize, and tokenize the data.

### **Features**
1. **Handles Missing Data**:
   - Fills missing `description` fields with empty strings.

2. **Combines Fields**:
   - Merges `title` and `description` into a single `full_text` field for unified processing.

3. **Persian Text Preprocessing**:
   - **Normalization**: Converts Arabic characters (e.g., "ي" → "ی", "ك" → "ک") to their Persian equivalents, removes diacritics, and standardizes text.
   - **Tokenization**: Splits text into individual words using Persian-specific rules.
   - **Stopword Removal**: Removes common Persian words (e.g., "و", "در") using `Hazm`’s stopwords list.
   - **Lemmatization**: Reduces words to their base forms (e.g., "می‌خواند" → "خوان").
   - **Non-Persian Character Removal**: Filters out any non-Persian characters.

4. **Saves Outputs**:
   - Produces `adverts_preprocessed.csv` and `products_preprocessed.csv` in the `data/processed/` directory.

### **How to Use**
1. Ensure the input files (`adverts.csv` and `productss.csv`) are in the `data/` directory.
2. Run the script:
   ```bash
   python preprocessing.py
   ```
3. Processed files will be saved in the `data/processed/` directory.

---

## **2. Feature Extraction Module (`feature_extraction.py`)**

### **Purpose**
The `feature_extraction.py` module processes the preprocessed Persian text data to extract numerical features for similarity calculations. It uses **TF-IDF** (Term Frequency-Inverse Document Frequency) to vectorize the text into sparse matrices for further analysis.

### **Features**
1. **Reads Preprocessed Data**:
   - Loads `adverts_preprocessed.csv` and `products_preprocessed.csv` as input.

2. **Persian-Specific Processing**:
   - Works directly with Persian text (`processed_text`) from the preprocessing step.
   - Avoids translation, leveraging Persian tokenization and stopwords.

3. **TF-IDF Vectorization**:
   - Converts Persian text into numerical vectors.
   - Removes Persian stopwords during vectorization for better accuracy.

4. **Saves Outputs**:
   - Transformed datasets:
     - `adverts_processed.csv`
     - `products_processed.csv`
   - TF-IDF matrices:
     - `adverts_tfidf.pkl`
     - `products_tfidf.pkl`
   - TF-IDF vectorizer:
     - `tfidf_vectorizer.pkl`

### **How to Use**
1. Ensure the preprocessed files (`adverts_preprocessed.csv` and `products_preprocessed.csv`) are in the `data/processed/` directory.
2. Run the script:
   ```bash
   python feature_extraction.py
   ```
3. Outputs will be saved in the `outputs/` directory.

---

## **Pipeline Workflow**

1. Run `preprocessing.py`:
   - Cleans and processes raw text into `adverts_preprocessed.csv` and `products_preprocessed.csv`.

2. Run `feature_extraction.py`:
   - Extracts numerical features for similarity calculations and generates TF-IDF vectors.

---

## **Directory Structure**

```
hackathon_project/
├── data/
│   ├── adverts.csv              # Raw advertisements dataset
│   ├── productss.csv            # Raw products dataset
│   ├── processed/               # Directory for preprocessed outputs
│       ├── adverts_preprocessed.csv
│       └── products_preprocessed.csv
├── outputs/                     # Directory for feature extraction outputs
│   ├── adverts_processed.csv
│   ├── products_processed.csv
│   ├── adverts_tfidf.pkl
│   ├── products_tfidf.pkl
│   ├── tfidf_vectorizer.pkl
├── src/
│   ├── preprocessing.py         # Preprocessing module
│   ├── feature_extraction.py    # Feature extraction module
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
```

---

## **Python Dependencies**

The project requires the following Python libraries:
- `pandas`
- `hazm`
- `sklearn`
- `joblib`

Install dependencies:
```bash
pip install pandas hazm scikit-learn joblib
```

---
