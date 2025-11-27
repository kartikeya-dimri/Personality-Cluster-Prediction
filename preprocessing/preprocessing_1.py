import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import sys

# --- 1. Define Paths ---
# Get the directory where this script is located (the 'preprocessing' folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory to find 'data'
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')

# CHANGE: Output directory is now the same as the script directory
OUTPUT_DIR = SCRIPT_DIR 

TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

def preprocess_data_1():
    print("Starting Preprocessing 1 (Standard Scaling & Label Encoding)...")

    # --- 2. Load Data ---
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find data files in {DATA_DIR}")
        sys.exit(1)

    print(f"Original Train Shape: {df_train.shape}")
    print(f"Original Test Shape: {df_test.shape}")

    # --- 3. Separate Target and IDs ---
    target_col = 'personality_cluster'
    id_col = 'participant_id'

    # Save IDs for later use if needed
    train_ids = df_train[id_col]
    test_ids = df_test[id_col]

    y_train = df_train[target_col]
    X_train = df_train.drop(columns=[target_col, id_col])
    X_test = df_test.drop(columns=[id_col])

    # --- 4. Encode Target ---
    # Converts class labels (e.g., 'Cluster A', 'Cluster B') to integers (0, 1, ...)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Save the mapping for reference
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\nTarget Encoding Mapping: {label_mapping}")

    # --- 5. Handle Missing Values (Imputation) ---
    # Using Median strategy for all features (robust to outliers)
    imputer = SimpleImputer(strategy='median')
    
    # Fit on train, transform both
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # --- 6. Feature Scaling ---
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)

    # --- 7. Reassemble and Save ---
    # Add the encoded target back to the training set
    train_processed = X_train_scaled.copy()
    train_processed[target_col] = y_train_encoded
    
    # Add IDs back to test set (optional, usually good for tracking)
    test_processed = X_test_scaled.copy()
    test_processed[id_col] = test_ids.values

    # Save to CSV in the current directory (preprocessing/)
    train_out_path = os.path.join(OUTPUT_DIR, 'train_preprocessing_1.csv')
    test_out_path = os.path.join(OUTPUT_DIR, 'test_preprocessing_1.csv')

    train_processed.to_csv(train_out_path, index=False)
    test_processed.to_csv(test_out_path, index=False)

    print("\nPreprocessing complete!")
    print(f"Saved processed train data to: {train_out_path}")
    print(f"Saved processed test data to: {test_out_path}")
    print(f"Features scaled: {len(X_train.columns)}")

if __name__ == "__main__":
    preprocess_data_1()