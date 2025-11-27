import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
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

TRAIN_PROCESSED_FILE = os.path.join(PREPROCESSING_DIR, 'train_preprocessing_3.csv')
TEST_PROCESSED_FILE = os.path.join(PREPROCESSING_DIR, 'test_preprocessing_3.csv')
ORIGINAL_TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')

# --- 2. Reproducibility (Fixing the Seed) ---
def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

# Call it immediately
set_seed(42)

# --- 3. Define Dataset Class ---
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

# --- 4. Define Network Architecture ---
class FeatureSelectedNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FeatureSelectedNet, self).__init__()
        
        # Since input features are fewer, we start with a focused hidden layer
        self.layer1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.output(x)
        return x

def run_model():
    print("--- Loading Data (Preprocessing 3) ---")
    try:
        train_df = pd.read_csv(TRAIN_PROCESSED_FILE)
        test_df = pd.read_csv(TEST_PROCESSED_FILE)
        original_train = pd.read_csv(ORIGINAL_TRAIN_FILE)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure 'preprocessing/preprocessing_3.py' has been run.")
        sys.exit(1)

    # --- 5. Prepare Data ---
    target_col = 'personality_cluster'
    id_col = 'participant_id'

    # preprocessing_3 already encoded the target and handled IDs
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[id_col])
    test_ids = test_df[id_col]

    input_size = X_train.shape[1]
    num_classes = len(original_train[target_col].unique())
    
    print(f"Features: {input_size} (Reduced from original), Classes: {num_classes}")

    # --- 6. Compute Class Weights ---
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    print(f"Class Weights: {class_weights}")

    # Create DataLoaders
    train_dataset = PersonalityDataset(X_train, y_train)
    test_dataset = PersonalityDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- 7. Setup Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = FeatureSelectedNet(input_size, num_classes).to(device)
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Cosine Annealing for smooth convergence on full dataset
    num_epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- 8. Training Loop (Full Data) ---
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
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # --- 9. Prediction ---
    print("\nMaking predictions on test set...")
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    # --- 10. Save Submission ---
    le = LabelEncoder()
    le.fit(original_train[target_col])
    y_pred_labels = le.inverse_transform(all_preds)

    submission_df = pd.DataFrame({
        id_col: test_ids,
        target_col: y_pred_labels
    })

    submission_file = os.path.join(OUTPUT_DIR, 'neural_network_3.csv')
    submission_df.to_csv(submission_file, index=False)

    print(f"\nSubmission file generated: {submission_file}")
    print(submission_df.head())

if __name__ == "__main__":
    run_model()