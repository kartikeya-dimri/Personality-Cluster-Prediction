import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
import sys
import copy

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

# --- 3. Define Neural Network Architecture ---
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

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    X_test_final = test_df.drop(columns=[id_col])
    test_ids = test_df[id_col]

    input_size = X.shape[1]
    num_classes = len(y.unique())
    
    # --- 5. Split for Early Stopping Validation ---
    print("\nSplitting data for Early Stopping validation...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # DataLoaders
    train_dataset = PersonalityDataset(X_train, y_train)
    val_dataset = PersonalityDataset(X_val, y_val)
    test_dataset = PersonalityDataset(X_test_final)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PersonalityNet(input_size, 64, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 6. Training Loop with Early Stopping ---
    max_epochs = 1000
    patience = 10
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    print(f"\nStarting training with Early Stopping (Patience: {patience})...")

    for epoch in range(max_epochs):
        # Train
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Check F1 Score
        current_f1 = f1_score(val_targets, val_preds, average='macro')
        
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            print(f"Epoch {epoch+1}: New Best Validation F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

    # --- 7. Final Prediction using Best Model ---
    print("\nRestoring best model weights and predicting...")
    model.load_state_dict(best_model_state)
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    # --- 8. Submission ---
    le = LabelEncoder()
    le.fit(original_train[target_col])
    y_pred_labels = le.inverse_transform(all_preds)

    submission_df = pd.DataFrame({
        id_col: test_ids,
        target_col: y_pred_labels
    })

    submission_file = os.path.join(OUTPUT_DIR, 'neural_network_1_early_stopping.csv')
    submission_df.to_csv(submission_file, index=False)

    print(f"\nSubmission file generated: {submission_file}")
    print(submission_df.head())

if __name__ == "__main__":
    run_model()