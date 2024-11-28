```markdown
# Preprocessing Module

The `preprocessing.py` module handles the initial stages of the data pipeline, including loading, cleaning, and preparing datasets for feature extraction and similarity calculations.

---

## **Module Overview**

Key functionalities of this module include:

1. **Data Loading**: Importing advertisement and product data from CSV files.
2. **Data Cleaning**:
   - Resolving missing values (e.g., in descriptions).
   - Merging title and description fields for feature extraction.
3. **Preprocessed Data Export**: Saving cleaned data for future use.

---

## **Directory Structure**

The module assumes the following project structure:

```
hackathon_project/
├── data/
│   ├── adverts.csv              # Original advertisement dataset
│   ├── productss.csv            # Original products dataset
│   ├── processed/               # Folder for preprocessed outputs
├── src/
│   ├── preprocessing.py         # Preprocessing script
```

---

## **Script Functions**

### `load_data(adverts_path, products_path)`
- **Description**: Loads advertisement and product datasets from the specified file paths.
- **Parameters**:
  - `adverts_path`: Path to the advertisements CSV file.
  - `products_path`: Path to the products CSV file.
- **Returns**: Two DataFrames: `adverts_df` and `products_df`.

---

### `preprocess_data(adverts_df, products_df)`
- **Description**: Cleans and combines data fields for further analysis.
- **Steps**:
  1. Fills missing descriptions with empty strings.
  2. Combines `title` and `description` into a `full_text` field.
- **Parameters**:
  - `adverts_df`: Advertisements DataFrame.
  - `products_df`: Products DataFrame.
- **Returns**: Preprocessed DataFrames for advertisements and products.

---

### `save_preprocessed_data(adverts_df, products_df, output_dir='data/processed/')`
- **Description**: Saves preprocessed DataFrames to the specified directory.
- **Parameters**:
  - `adverts_df`: Preprocessed advertisements DataFrame.
  - `products_df`: Preprocessed products DataFrame.
  - `output_dir`: Path to save the output files (default: `data/processed/`).
- **Outputs**:
  - `adverts_preprocessed.csv`
  - `products_preprocessed.csv`

---

## **Usage**

### **Prerequisites**
Ensure `pandas` is installed:
```bash
pip install pandas
```

### **Running the Script**
1. Place `adverts.csv` and `productss.csv` in the `data/` directory.
2. Run the script:
   ```bash
   python src/preprocessing.py
   ```
3. Preprocessed files will appear in the `data/processed/` directory.

---

## **Example Code Usage**

```python
from src.preprocessing import load_data, preprocess_data, save_preprocessed_data

# File paths
adverts_path = 'data/adverts.csv'
products_path = 'data/productss.csv'

# Load data
adverts_df, products_df = load_data(adverts_path, products_path)

# Preprocess data
adverts_df, products_df = preprocess_data(adverts_df, products_df)

# Save preprocessed data
save_preprocessed_data(adverts_df, products_df, output_dir='data/processed/')
```

---

## **Output Files**

- **`adverts_preprocessed.csv`**: Preprocessed advertisements dataset.
- **`products_preprocessed.csv`**: Preprocessed products dataset.

--- 
```