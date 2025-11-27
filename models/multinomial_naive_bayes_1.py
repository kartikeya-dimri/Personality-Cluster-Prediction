import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import sys

# --- 1. Define Paths ---
# Get the directory where this script is located (models/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

# Path to the preprocessed data (saved in the 'preprocessing' folder)
PREPROCESSING_DIR = os.path.join(PARENT_DIR, 'preprocessing')
# Path to the original data (for LabelEncoder fitting)
DATA_DIR = os.path.join(PARENT_DIR, 'data')
# Path to save the submission
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
        
        # We need the original train data to fit the LabelEncoder correctly
        # so we can transform predictions back to the original class names.
        original_train = pd.read_csv(ORIGINAL_TRAIN_FILE)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print(f"Please ensure 'preprocessing/preprocessing_1.py' has been run.")
        sys.exit(1)

    # --- 2. Prepare Data ---
    target_col = 'personality_cluster'
    id_col = 'participant_id'

    # Separate features and target
    # Note: train_preprocessing_1.csv has the target column encoded as integers
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # The test file from preprocessing_1 has the ID column. We need to separate it.
    X_test = test_df.drop(columns=[id_col])
    test_ids = test_df[id_col]

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # --- 3. Scale Data for MultinomialNB ---
    # MultinomialNB requires non-negative input. 
    # Since preprocessing_1 used StandardScaler (which creates negatives),
    # we must rescale the data to be positive [0, 1].
    print("\nScaling features to [0, 1] for Multinomial Naive Bayes...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. Train Model ---
    print("Training Multinomial Naive Bayes model...")
    mnb = MultinomialNB()
    mnb.fit(X_train_scaled, y_train)

    # --- 5. Predict ---
    print("Making predictions...")
    y_pred_encoded = mnb.predict(X_test_scaled)

    # --- 6. Inverse Transform Predictions ---
    # Fit LabelEncoder on original targets to get the class mapping
    le = LabelEncoder()
    le.fit(original_train[target_col])
    
    # Convert integer predictions (0, 1, 2...) back to strings (Cluster A, B...)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    # --- 7. Create Submission File ---
    submission_df = pd.DataFrame({
        id_col: test_ids,
        target_col: y_pred_labels
    })

    submission_file = os.path.join(OUTPUT_DIR, 'multinomial_naive_bayes_1.csv')
    submission_df.to_csv(submission_file, index=False)

    print(f"\nSubmission file generated: {submission_file}")
    print(submission_df.head())

if __name__ == "__main__":
    run_model()