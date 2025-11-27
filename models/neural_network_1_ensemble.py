import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os
import sys
import torch.nn.functional as F

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

# --- 2. Define Dataset Class ---
class PersonalityDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# --- 3. Define Neural Network Architecture (Exact Baseline) ---
class PersonalityNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PersonalityNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out

def run_model():
    print("--- Loading Data ---")
    try:
        train_df = pd.read_csv(TRAIN_PROCESSED_FILE)
        test_df = pd.read_csv(TEST_PROCESSED_FILE)
        original_train = pd.read_csv(ORIGINAL_TRAIN_FILE)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # --- 4. Prepare Data ---
    target_col = 'personality_cluster'
    id_col = 'participant_id'

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[id_col])
    test_ids = test_df[id_col]

    input_size = X_train.shape[1]
    num_classes = len(y_train.unique())
    
    print(f"Features: {input_size}, Classes: {num_classes}")

    # Create DataLoaders
    train_dataset = PersonalityDataset(X_train, y_train)
    test_dataset = PersonalityDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 5. Ensemble Training Loop ---
    num_seeds = 5
    all_test_probs = None
    
    print(f"\nStarting Ensemble Training ({num_seeds} models)...")

    for seed in range(num_seeds):
        print(f"\n--- Training Model {seed + 1}/{num_seeds} (Seed {42 + seed}) ---")
        
        # Set seed for reproducibility of this specific run
        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)
        
        # Initialize Model (Fresh for each seed)
        model = PersonalityNet(input_size, 64, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train Loop
        model.train()
        for epoch in range(100): # 100 Epochs per model
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Predict Probabilities (Softmax)
        model.eval()
        probs_list = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                # Apply Softmax to get probabilities
                probs = F.softmax(outputs, dim=1)
                probs_list.append(probs.cpu().numpy())
        
        current_probs = np.concatenate(probs_list)
        
        # Accumulate probabilities
        if all_test_probs is None:
            all_test_probs = current_probs
        else:
            all_test_probs += current_probs

    # --- 6. Average and Argmax ---
    avg_probs = all_test_probs / num_seeds
    final_preds_indices = np.argmax(avg_probs, axis=1)

    # --- 7. Inverse Transform ---
    le = LabelEncoder()
    le.fit(original_train[target_col])
    y_pred_labels = le.inverse_transform(final_preds_indices)

    # --- 8. Submission ---
    submission_df = pd.DataFrame({
        id_col: test_ids,
        target_col: y_pred_labels
    })

    submission_file = os.path.join(OUTPUT_DIR, 'neural_network_1_ensemble.csv')
    submission_df.to_csv(submission_file, index=False)

    print(f"\nEnsemble submission file generated: {submission_file}")
    print(submission_df.head())

if __name__ == "__main__":
    run_model()