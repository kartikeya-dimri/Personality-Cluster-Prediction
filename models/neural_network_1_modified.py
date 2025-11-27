import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os
import sys
import random

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

# --- 2. Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# --- 3. Dataset Class ---
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

# --- 4. Modified Architecture (Wider & Deeper) ---
class ModifiedPersonalityNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ModifiedPersonalityNet, self).__init__()
        
        # Wider input layer (Input -> 256)
        self.layer1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256) # Batch Norm for stability
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4) # Increased dropout for wider layer 
        
        # Middle layer (256 -> 128)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        # Narrowing layer (128 -> 64)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Output layer (64 -> Classes)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.output(x)
        return x

def run_model():
    print("--- Loading Data ---")
    try:
        train_df = pd.read_csv(TRAIN_PROCESSED_FILE)
        test_df = pd.read_csv(TEST_PROCESSED_FILE)
        original_train = pd.read_csv(ORIGINAL_TRAIN_FILE)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # --- 5. Prepare Data ---
    target_col = 'personality_cluster'
    id_col = 'participant_id'

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[id_col])
    test_ids = test_df[id_col]

    input_size = X_train.shape[1]
    num_classes = len(original_train[target_col].unique())
    
    print(f"Features: {input_size}, Classes: {num_classes}")

    # DataLoaders (Full Data)
    train_dataset = PersonalityDataset(X_train, y_train)
    test_dataset = PersonalityDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- 6. Setup Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ModifiedPersonalityNet(input_size, num_classes).to(device)
    
    # MODIFICATION: Label Smoothing
    # Helps with calibration and minority classes without aggressive weighting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # MODIFICATION: AdamW Optimizer
    # Better regularization handling
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # --- 7. Training Loop ---
    num_epochs = 100
    print(f"\nStarting training for {num_epochs} epochs...")
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # --- 8. Prediction ---
    print("\nMaking predictions on test set...")
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    # --- 9. Submission ---
    le = LabelEncoder()
    le.fit(original_train[target_col])
    y_pred_labels = le.inverse_transform(all_preds)

    submission_df = pd.DataFrame({
        id_col: test_ids,
        target_col: y_pred_labels
    })

    submission_file = os.path.join(OUTPUT_DIR, 'neural_network_1_modified.csv')
    submission_df.to_csv(submission_file, index=False)

    print(f"\nSubmission file generated: {submission_file}")
    print(submission_df.head())

if __name__ == "__main__":
    run_model()