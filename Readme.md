# Personality Cluster Prediction

## Team Information

**Team Name:** Predictify

**Team Members:**
1. Kartikeya Dimri – IMT2023126  
2. Ayush Mishra – IMT2023129  
3. Harsh Sinha – IMT2023571  

---

## Overview

This project focuses on classifying individuals into five distinct **personality clusters** using high-dimensional behavioral, lifestyle, and demographic data.  
We explore the full pipeline—from dataset understanding and EDA to multiple preprocessing strategies and a large suite of machine-learning models.

The goal is to predict personality segments accurately in a **multiclass setting with class imbalance**, using **Macro F1 Score** as the evaluation metric.

Our approach involves:
- Multiple preprocessing pipelines  
- Deep EDA insights  
- Neural Networks, SVMs, Linear Models, Tree-based models, and Ensembles  
- Model comparison & interpretation  
- Detailed analysis explaining *why* the best model worked  

---

## Directory Structure

```
Personality-Cluster-Prediction/
│
├── data/                          # Raw datasets
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── preprocessing/                 # Data preprocessing scripts
│   ├── preprocessing_1.py
│   ├── preprocessing_2.py
│   ├── preprocessing_3.py
│   ├── preprocessing_1_modified.py
│   ├── train_preprocessing_*.csv
│   └── test_preprocessing_*.csv
│
├── eda/                           # Exploratory Data Analysis
│   ├── EDA.py
│   └── eda_plots/
│
├── models/                        # Model training scripts
│   ├── neural_network_*.py
│   ├── random_forest_*.py
│   ├── xg_boost_*.py
│   ├── gradient_boosting_classifier_*.py
│   ├── logistic_regression_*.py
│   ├── SVM_*.py
│   ├── knn_classifier_*.py
│   ├── adaboost_*.py
│   ├── decision_tree_*.py
│   └── *_naive_bayes_*.py
│
├── output/                        # Model predictions and saved outputs
│   ├── *.csv
│   └── *.joblib
```

---

## Dataset Summary

The dataset consists of rich behavioral features such as:
- consistency score  
- focus intensity  
- upbringing influence  
- cultural background  
- creative and altruistic indicators  
- physical & hobby engagement  
- support-environment scores  

Target variable: **personality_cluster (A–E)**.

Key characteristics:
- **Class imbalance** (Cluster E dominates; Cluster A very small)
- **Near-zero correlations** between features (independent attributes)
- **Mixed feature types** (categorical + numerical)
- Some **saturated / binary-like** features  
- Sparse or skewed features (e.g., altruism, creative expression)

---

## Exploratory Data Analysis

Major insights include:

### 🔹 Class Imbalance  
Cluster E has the highest frequency (~1000 samples), while Cluster A is rare.  
Macro F1 is necessary to evaluate minority cluster performance properly.

### 🔹 Correlation & Redundancy  
Most features show correlations between **-0.04 to +0.03**, indicating:
- Minimal redundancy  
- Low multicollinearity  
- Limited scope for PCA  

### 🔹 Feature Separability  
Some key differentiators:
- **Consistency score** strongly separates clusters  
- **Focus intensity** differentiates A/B vs D/E  
- Several features (hobby level, activity index) are nearly binary and saturated  
- Sparse features require careful modeling  

### 🔹 Demographic Trends  
Cluster E dominates across all age groups (15–18).  
Cultural & identity categories show mild differences in cluster proportions.

---

## Preprocessing Pipelines

We built **four** major preprocessing scripts, each tailored for different model families.

### 1. **preprocessing_1.py (Baseline Full Pipeline)**
- Median/Mode imputation  
- One-Hot encoding for categorical features  
- Standard Scaling  
- Label Encoding for target  
- Best suited for: Neural Networks, SVM, Logistic Regression  

### 2. **preprocessing_1_modified.py (Outlier Removal)**
- IQR-based outlier removal for key continuous features  
- Same encoding & scaling as baseline  
- Reduced dataset size significantly  
- Helped some models but harmed minority cluster learning  

### 3. **preprocessing_2.py (Tree-Based Optimization)**
- Ordinal Encoding for high-cardinality features  
- MinMax scaling  
- Binary-like features treated as categorical  
- Designed specifically for Random Forest, XGBoost, LightGBM  

### 4. **preprocessing_3.py (Feature Selection)**
- Dropped extremely low-variance & saturated features  
- Robust Scaling to preserve edge cases  
- Lower dimensionality to prevent overfitting  

---

## Models Implemented

We trained almost every major model family to compare performance thoroughly:

### 🔹 Primary Models
- **Neural Networks (MLPs)**  
- **Support Vector Machines (SVM)**  
- **Logistic Regression**

### 🔹 Secondary / Benchmark Models
- **Naive Bayes (Gaussian, Multinomial, Bernoulli)**  
- **KNN Classifiers**  
- **Decision Trees**  
- **Ensemble Models**
  - Random Forest  
  - Gradient Boosting  
  - AdaBoost  
  - XGBoost  
  - LightGBM  

Each model family has multiple variants (1,2,3...), early stopping versions, ensemble variants, and tuning attempts.

---

## Results

| Model Type             | Best Script             | Kaggle Macro F1 |
|------------------------|-------------------------|-----------------|
| Neural Network         | neural_network_1.csv    | **0.643**       |
| SVM                    | SVM_3.csv               | 0.586           |
| XGBoost                | xg_boost_3.csv          | 0.567           |
| Random Forest          | random_forest_2.csv     | 0.564           |
| Logistic Regression    | logistic_regression_2.csv | 0.477        |

**Winner:** Neural Network + Preprocessing_1.py  
This pairing consistently outperformed all others.

---

## Why the Neural Network Won

### 1. Non-Linear Interactions  
Personality traits interact in complex, non-linear ways.  
NN hidden layers captured these interactions far better than:
- logistic regression (too linear)  
- tree models (axis-aligned splits only)

### 2. High-Dimensional “Manifold” Structure  
SVM outperforming RF gave a big hint:  
Clusters behave like curved blobs in higher dimensions.  
NNs model such smooth boundaries perfectly.

### 3. Outlier Preservation  
Removing outliers wiped out rare samples from Cluster A.  
Keeping them helped NN understand the minority classes.

### 4. Independent Features Advantage  
When every feature carries unique information,  
dense NN layers aggregate signals more effectively than trees.

---

## Final Takeaway

The project demonstrates that:
- Complex human behavioral segmentation **benefits heavily from non-linear models**  
- Preprocessing choices dramatically affect model performance  
- Neural Networks, when paired with carefully standardized full-feature data, deliver the most reliable results  



---

