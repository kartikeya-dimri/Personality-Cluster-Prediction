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
OUTPUT_DIR = SCRIPT_DIR # Save processed files in the preprocessing folder

TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

def preprocess_data_3():
    print("Starting Preprocessing 3 (Correlation-based Feature Selection)...")

    # --- 2. Load Data ---
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find data files in {DATA_DIR}")
        sys.exit(1)

    # --- 3. Define Features to Drop (Correlation < 0.05) ---
    # Based on analysis, these features have very little linear impact on the target
    low_correlation_features = [
        'age_group', 
        'altruism_score', 
        'identity_code', 
        'physical_activity_index', 
        'cultural_background', 
        'upbringing_influence'
    ]
    
    target_col = 'personality_cluster'
    id_col = 'participant_id'
    
    # Features to drop from X (IDs + Low Correlation)
    cols_to_drop = [id_col] + low_correlation_features

    print(f"Dropping {len(low_correlation_features)} low-impact features: {low_correlation_features}")

    # --- 4. Separate Target and Features ---
    # Save IDs for tracking
    train_ids = df_train[id_col]
    test_ids = df_test[id_col]

    y_train = df_train[target_col]
    
    # Drop columns
    X_train = df_train.drop(columns=[target_col] + cols_to_drop)
    X_test = df_test.drop(columns=cols_to_drop)

    print(f"Remaining Feature Count: {X_train.shape[1]}")
    print(f"Remaining Features: {list(X_train.columns)}")

    # --- 5. Encode Target ---
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # --- 6. Handle Missing Values (Median Imputation) ---
    imputer = SimpleImputer(strategy='median')
    
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # --- 7. Feature Scaling (StandardScaler) ---
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)

    # --- 8. Reassemble and Save ---
    # Train file: Features + Encoded Target
    train_processed = X_train_scaled.copy()
    train_processed[target_col] = y_train_encoded
    
    # Test file: Features + ID
    test_processed = X_test_scaled.copy()
    test_processed[id_col] = test_ids.values

    # Save to CSV
    train_out_path = os.path.join(OUTPUT_DIR, 'train_preprocessing_3.csv')
    test_out_path = os.path.join(OUTPUT_DIR, 'test_preprocessing_3.csv')

    train_processed.to_csv(train_out_path, index=False)
    test_processed.to_csv(test_out_path, index=False)

    print("\nPreprocessing 3 complete!")
    print(f"Saved processed train data to: {train_out_path}")
    print(f"Saved processed test data to: {test_out_path}")

if __name__ == "__main__":
    preprocess_data_3()