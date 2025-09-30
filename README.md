# Zindi New User Engagement Prediction Challenge

This project implements machine learning models to predict user engagement in the second month based on first-month activity data.

## Project Structure

```
zindi-new-user-engagement-prediction-challenge/
├── code/
│   ├── master_dataset.py          # Data preprocessing and feature engineering
│   ├── decision_tree_model.py     # Decision Tree classifier
│   ├── random_forest_model.py     # Random Forest classifier
│   ├── lightgbm_model.py         # LightGBM classifier (XGBoost alternative)
│   └── notes.txt                  # Project notes
├── data/
│   ├── master_dataset.csv         # Processed dataset
│   ├── decision_tree_results.txt  # Decision Tree results
│   ├── random_forest_results.txt  # Random Forest results
│   └── lightgbm_results.txt      # LightGBM results
└── README.md
```

## Dataset Overview

- **Total Users**: 12,413
- **Features**: 23 features including demographics, activity counts, and engagement patterns
- **Target Variable**: `active_second_month` (binary: 0=inactive, 1=active)
- **Class Distribution**: 90.7% inactive vs 9.3% active (imbalanced dataset)

## Models Implemented

### 1. Decision Tree
- **Test F1 Score**: 0.3907
- **Test AUC-ROC**: 0.8270
- **Key Features**: Signup timing, first-month activity levels

### 2. Random Forest
- **Hyperparameter tuning** with GridSearchCV
- **Class imbalance handling** with balanced weights
- **Feature importance analysis**

### 3. LightGBM
- **Alternative to XGBoost** with similar performance
- **Faster training** and lower memory usage
- **Built-in categorical feature handling**

## Key Features

- **Feature Engineering**: Month offset calculation for temporal features
- **Class Imbalance Handling**: Balanced class weights and appropriate metrics
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Comprehensive Evaluation**: F1, AUC-ROC, precision, recall, confusion matrix
- **Feature Importance**: Analysis of most predictive features
- **Text Output**: Results saved in readable text format

## Usage

1. **Prepare Dataset**:
   ```bash
   python3 code/master_dataset.py
   ```

2. **Train Decision Tree**:
   ```bash
   python3 code/decision_tree_model.py
   ```

3. **Train Random Forest**:
   ```bash
   python3 code/random_forest_model.py
   ```

4. **Train LightGBM**:
   ```bash
   python3 code/lightgbm_model.py
   ```

## Key Findings

- **Signup timing** (`Created At Month`) is the strongest predictor
- **First-month activity levels** are highly predictive of second-month engagement
- **Community engagement** (discussions, learning pages) shows strong correlation
- **Geographic patterns** (`Countries_ID`) provide additional predictive power

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- lightgbm (or xgboost)
- joblib

## Results

All models show good discriminative ability with AUC scores above 0.8. The Decision Tree model provides interpretable results with clear feature importance rankings, while ensemble methods (Random Forest, LightGBM) offer improved performance through hyperparameter optimization.

## Files Generated

- `*.pkl`: Trained model files
- `*_results.txt`: Detailed evaluation metrics and analysis
- `master_dataset.csv`: Processed dataset ready for modeling
