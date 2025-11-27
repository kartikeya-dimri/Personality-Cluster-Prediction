import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import sys

# --- 1. Define Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = SCRIPT_DIR

TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

def preprocess_data_1_modified():
    print("Starting Preprocessing 1 Modified (Outlier Removal + Standard Scaling)...")

    # --- 2. Load Data ---
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find data files in {DATA_DIR}")
        sys.exit(1)

    # --- 3. Define Features and Target ---
    target_col = 'personality_cluster'
    id_col = 'participant_id'
    
    # Separate target for train
    y_train = df_train[target_col]
    
    # Features to check for outliers (All numeric features except IDs)
    # In this dataset, almost all features are numeric or integer-encoded
    feature_cols = [col for col in df_train.columns if col not in [target_col, id_col]]

    # --- 4. Outlier Removal (IQR Method) on Training Data ---
    print(f"Original Train Shape: {df_train.shape}")
    
    # Calculate IQR only on training features
    Q1 = df_train[feature_cols].quantile(0.25)
    Q3 = df_train[feature_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    # any(axis=1) checks if a row has an outlier in ANY feature
    outlier_mask = ((df_train[feature_cols] < lower_bound) | (df_train[feature_cols] > upper_bound)).any(axis=1)
    
    # Filter training data
    df_train_clean = df_train[~outlier_mask].copy()
    
    print(f"Dropped {outlier_mask.sum()} outlier rows.")
    print(f"New Train Shape: {df_train_clean.shape}")
    
    # --- 5. Prepare X and y after cleaning ---
    X_train = df_train_clean.drop(columns=[target_col, id_col])
    y_train = df_train_clean[target_col]
    
    # Test data (we keep all test rows, but will cap them later to match train distribution)
    X_test = df_test.drop(columns=[id_col])
    test_ids = df_test[id_col]

    # --- 6. Encode Target ---
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # --- 7. Handle Missing Values (Imputation) ---
    imputer = SimpleImputer(strategy='median')
    
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # --- 8. Cap Test Data Outliers (Winsorization) ---
    # We shouldn't drop test rows, but we can cap them to the training bounds
    # so they don't throw off the scaler or model.
    for col in feature_cols:
        X_test_imputed[col] = np.clip(X_test_imputed[col], lower_bound[col], upper_bound[col])

    # --- 9. Feature Scaling (StandardScaler) ---
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)

    # --- 10. Reassemble and Save ---
    train_processed = X_train_scaled.copy()
    train_processed[target_col] = y_train_encoded
    
    test_processed = X_test_scaled.copy()
    test_processed[id_col] = test_ids.values

    train_out_path = os.path.join(OUTPUT_DIR, 'train_preprocessing_1_modified.csv')
    test_out_path = os.path.join(OUTPUT_DIR, 'test_preprocessing_1_modified.csv')

    train_processed.to_csv(train_out_path, index=False)
    test_processed.to_csv(test_out_path, index=False)

    print("\nPreprocessing complete!")
    print(f"Saved processed train data to: {train_out_path}")
    print(f"Saved processed test data to: {test_out_path}")

if __name__ == "__main__":
    preprocess_data_1_modified()