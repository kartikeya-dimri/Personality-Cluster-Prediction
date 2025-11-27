import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import sys

# --- 1. Define Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = SCRIPT_DIR # Save in preprocessing folder

TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

def preprocess_data_2():
    print("Starting Preprocessing 2 (Robust Scaling & Winsorization)...")

    # --- 2. Load Data ---
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find data files in {DATA_DIR}")
        sys.exit(1)

    # --- 3. Feature Selection (Drop IDs and potential noise) ---
    target_col = 'personality_cluster'
    id_col = 'participant_id'
    
    # Columns that might be just IDs or noise (based on common dataset patterns)
    # You can comment these out if you want to keep them
    cols_to_drop = [id_col, 'identity_code', 'cultural_background']
    
    # Save IDs for submission
    train_ids = df_train[id_col]
    test_ids = df_test[id_col]

    print(f"Dropping columns: {cols_to_drop}")
    
    # Separate Target
    y_train = df_train[target_col]
    
    # Drop columns from features
    X_train = df_train.drop(columns=[target_col] + cols_to_drop, errors='ignore')
    X_test = df_test.drop(columns=cols_to_drop, errors='ignore')

    # --- 4. Encode Target ---
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # --- 5. Handle Missing Values ---
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # --- 6. Outlier Handling (Winsorization / Capping) ---
    # Cap values at 1st and 99th percentiles to reduce outlier impact
    print("Capping outliers at 1% and 99% percentiles...")
    
    for col in X_train_imputed.columns:
        # Calculate bounds on training data
        lower_bound = X_train_imputed[col].quantile(0.01)
        upper_bound = X_train_imputed[col].quantile(0.99)
        
        # Apply to train
        X_train_imputed[col] = np.clip(X_train_imputed[col], lower_bound, upper_bound)
        
        # Apply same bounds to test (prevent data leakage)
        X_test_imputed[col] = np.clip(X_test_imputed[col], lower_bound, upper_bound)

    # --- 7. Robust Scaling ---
    # RobustScaler scales data using statistics that are robust to outliers
    scaler = RobustScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test.columns)

    # --- 8. Reassemble and Save ---
    train_processed = X_train_scaled.copy()
    train_processed[target_col] = y_train_encoded
    
    test_processed = X_test_scaled.copy()
    test_processed[id_col] = test_ids.values

    train_out_path = os.path.join(OUTPUT_DIR, 'train_preprocessing_2.csv')
    test_out_path = os.path.join(OUTPUT_DIR, 'test_preprocessing_2.csv')

    train_processed.to_csv(train_out_path, index=False)
    test_processed.to_csv(test_out_path, index=False)

    print("\nPreprocessing 2 complete!")
    print(f"Saved processed train data to: {train_out_path}")
    print(f"Saved processed test data to: {test_out_path}")
    print(f"Features used: {len(X_train.columns)}")

if __name__ == "__main__":
    preprocess_data_2()