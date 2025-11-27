import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Define Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

PLOTS_DIR = os.path.join(SCRIPT_DIR, 'eda_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Load Data ---
print("Loading data...")
try:
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print(f"Error: Could not find data files in {DATA_DIR}")
    sys.exit(1)

# --- 1. Basic Inspection ---
print("\n--- Train Data Info ---")
df_train.info()

print("\n--- Train Data Missing Values ---")
print(df_train.isnull().sum())

print("\n--- Target Variable Analysis (personality_cluster) ---")
target_col = 'personality_cluster'
print(df_train[target_col].value_counts(normalize=True))

# --- 2. Visualization Settings ---
sns.set_style("whitegrid")

# --- 3. Target Distribution ---
plt.figure(figsize=(10, 6))
sns.countplot(x=target_col, data=df_train, order=df_train[target_col].value_counts().index)
plt.title('Distribution of Personality Clusters')
plt.xticks(rotation=45)
plt.savefig(os.path.join(PLOTS_DIR, '1_target_distribution.png'))
print(f"Saved plot: 1_target_distribution.png")

# --- 4. Categorical/Demographic Features Analysis ---
# Even though these are 'int64', they represent categories or groups
categorical_cols = [
    'age_group', 'identity_code', 'cultural_background', 'upbringing_influence'
]

for i, col in enumerate(categorical_cols):
    plt.figure(figsize=(12, 6))
    # Treat as category for plotting
    sns.countplot(data=df_train, x=col, hue=target_col, palette='viridis')
    plt.title(f'{col} Distribution by Personality Cluster')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'2_cat_dist_{col}.png'))
    print(f"Saved plot: 2_cat_dist_{col}.png")

# --- 5. Numerical/Score Features Analysis ---
numerical_cols = [
    'focus_intensity', 'consistency_score', 'external_guidance_usage',
    'support_environment_score', 'hobby_engagement_level', 
    'physical_activity_index', 'creative_expression_index', 'altruism_score'
]

# Plot Boxplots to see distribution/outliers per cluster
for i, col in enumerate(numerical_cols):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_train, x=target_col, y=col)
    plt.title(f'{col} by Personality Cluster')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, f'3_num_boxplot_{col}.png'))
    print(f"Saved plot: 3_num_boxplot_{col}.png")

# Correlation Heatmap (Numerical Features)
plt.figure(figsize=(12, 10))
corr_matrix = df_train[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap (Score Features)')
plt.savefig(os.path.join(PLOTS_DIR, '4_correlation_heatmap.png'))
print(f"Saved plot: 4_correlation_heatmap.png")

print(f"\nEDA complete. All plots saved to: {PLOTS_DIR}")