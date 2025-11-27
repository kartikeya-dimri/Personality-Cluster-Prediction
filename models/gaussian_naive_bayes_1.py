import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import os
import sys

# --- 1. Define Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

PREPROCESSING_DIR = os.path.join(PARENT_DIR, 'preprocessing')
DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PROCESSED_FILE = os.path.join(PREPROCESSING_DIR, 'train_preprocessing_1.csv')
TEST_PROCESSED_FILE = os.path.join(PREPROCESSING_DIR, 'test_preprocessing_1.csv')
ORIGINAL_TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')

def run_model():
    print("--- Loading Data ---")
    try:
        train_df = pd.read_csv(TRAIN_PROCESSED_FILE)
        test_df = pd.read_csv(TEST_PROCESSED_FILE)
        original_train = pd.read_csv(ORIGINAL_TRAIN_FILE)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure 'preprocessing/preprocessing_1.py' has been run.")
        sys.exit(1)

    # --- 2. Prepare Data ---
    target_col = 'personality_cluster'
    id_col = 'participant_id'

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[id_col])
    test_ids = test_df[id_col]

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # --- 3. Train Gaussian Naive Bayes Model ---
    # GaussianNB assumes features are normally distributed.
    # The StandardScaler used in preprocessing_1 helps with this assumption.
    print("\nTraining Gaussian Naive Bayes model...")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # --- 4. Predict ---
    print("Making predictions...")
    y_pred_encoded = gnb.predict(X_test)

    # --- 5. Inverse Transform Predictions ---
    le = LabelEncoder()
    le.fit(original_train[target_col])
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    # --- 6. Create Submission File ---
    submission_df = pd.DataFrame({
        id_col: test_ids,
        target_col: y_pred_labels
    })

    # Updated filename as requested
    submission_file = os.path.join(OUTPUT_DIR, 'gaussian_naive_bayes_1.csv')
    submission_df.to_csv(submission_file, index=False)

    print(f"\nSubmission file generated: {submission_file}")
    print(submission_df.head())

if __name__ == "__main__":
    run_model()