import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score,
    roc_curve
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the master dataset for training"""
    print("Loading master dataset...")
    df = pd.read_csv("../data/master_dataset.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable distribution:")
    print(df['active_second_month'].value_counts())
    print(f"Target variable percentage:")
    print(df['active_second_month'].value_counts(normalize=True))
    
    return df

def prepare_features(df):
    """Prepare features for training"""
    print("\nPreparing features...")
    
    # Separate features and target
    target = df['active_second_month']
    
    # Select features (exclude User_ID and target)
    feature_columns = [col for col in df.columns if col not in ['User_ID', 'active_second_month']]
    features = df[feature_columns].copy()
    
    # Handle categorical variables
    categorical_cols = features.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"Found categorical columns: {list(categorical_cols)}")
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    # Handle missing values
    missing_values = features.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values found:")
        print(missing_values[missing_values > 0])
        features = features.fillna(0)  # Fill with 0 for simplicity
    
    print(f"\nFinal feature matrix shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    return features, target, feature_columns

def train_gradient_boosting_model(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting model (similar to CatBoost)"""
    print("\n" + "="*50)
    print("TRAINING GRADIENT BOOSTING MODEL")
    print("="*50)
    
    # Initial model with basic parameters
    print("Training initial Gradient Boosting model...")
    gb_model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        max_features='sqrt'
    )
    
    # Train the model
    gb_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = gb_model.predict(X_train)
    y_test_pred = gb_model.predict(X_test)
    y_train_proba = gb_model.predict_proba(X_train)[:, 1]
    y_test_proba = gb_model.predict_proba(X_test)[:, 1]
    
    return gb_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba

def evaluate_model(y_true, y_pred, y_proba, dataset_name):
    """Evaluate model performance"""
    print(f"\n{dataset_name.upper()} SET EVALUATION:")
    print("-" * 40)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0,0]:,}")
    print(f"False Positives: {cm[0,1]:,}")
    print(f"False Negatives: {cm[1,0]:,}")
    print(f"True Positives:  {cm[1,1]:,}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

def print_feature_importance(model, feature_names, top_n=15):
    """Print feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    print(f"\nTop {top_n} Feature Importances:")
    for i in range(top_n):
        print(f"{i+1:2d}. {feature_names[indices[i]]:30s}: {importances[indices[i]]:.4f}")
    
    return [(feature_names[indices[i]], importances[indices[i]]) for i in range(top_n)]

def print_roc_metrics(y_true, y_proba, dataset_name):
    """Print ROC curve metrics"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"\nROC Curve Metrics for {dataset_name} Set:")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Optimal TPR (Recall): {optimal_tpr:.4f}")
    print(f"Optimal FPR: {optimal_fpr:.4f}")
    
    return {
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': optimal_tpr,
        'optimal_fpr': optimal_fpr,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

def save_results_to_file(train_metrics, test_metrics, feature_importance_data, 
                        train_roc_metrics, test_roc_metrics, train_shape, test_shape):
    """Save all results to a text file"""
    output_file = "../data/gradient_boosting_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GRADIENT BOOSTING MODEL RESULTS FOR USER ENGAGEMENT PREDICTION\n")
        f.write("="*80 + "\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training samples: {train_shape[0]:,}\n")
        f.write(f"Test samples: {test_shape[0]:,}\n")
        f.write(f"Number of features: {train_shape[1]}\n\n")
        
        # Training metrics
        f.write("TRAINING SET PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {train_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {train_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {train_metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {train_metrics['f1_score']:.4f}\n")
        f.write(f"AUC-ROC:   {train_metrics['auc']:.4f}\n")
        cm = train_metrics['confusion_matrix']
        f.write(f"Confusion Matrix:\n")
        f.write(f"  True Negatives:  {cm[0,0]:,}\n")
        f.write(f"  False Positives: {cm[0,1]:,}\n")
        f.write(f"  False Negatives: {cm[1,0]:,}\n")
        f.write(f"  True Positives:  {cm[1,1]:,}\n\n")
        
        # Test metrics
        f.write("TEST SET PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {test_metrics['f1_score']:.4f}\n")
        f.write(f"AUC-ROC:   {test_metrics['auc']:.4f}\n")
        cm = test_metrics['confusion_matrix']
        f.write(f"Confusion Matrix:\n")
        f.write(f"  True Negatives:  {cm[0,0]:,}\n")
        f.write(f"  False Positives: {cm[0,1]:,}\n")
        f.write(f"  False Negatives: {cm[1,0]:,}\n")
        f.write(f"  True Positives:  {cm[1,1]:,}\n\n")
        
        # Feature importance
        f.write("FEATURE IMPORTANCE (Top 15):\n")
        f.write("-" * 40 + "\n")
        for i, (feature, importance) in enumerate(feature_importance_data, 1):
            f.write(f"{i:2d}. {feature:30s}: {importance:.4f}\n")
        f.write("\n")
        
        # ROC metrics
        f.write("ROC CURVE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write("Training Set:\n")
        f.write(f"  AUC-ROC: {train_roc_metrics['auc']:.4f}\n")
        f.write(f"  Optimal Threshold: {train_roc_metrics['optimal_threshold']:.4f}\n")
        f.write(f"  Optimal TPR (Recall): {train_roc_metrics['optimal_tpr']:.4f}\n")
        f.write(f"  Optimal FPR: {train_roc_metrics['optimal_fpr']:.4f}\n\n")
        
        f.write("Test Set:\n")
        f.write(f"  AUC-ROC: {test_roc_metrics['auc']:.4f}\n")
        f.write(f"  Optimal Threshold: {test_roc_metrics['optimal_threshold']:.4f}\n")
        f.write(f"  Optimal TPR (Recall): {test_roc_metrics['optimal_tpr']:.4f}\n")
        f.write(f"  Optimal FPR: {test_roc_metrics['optimal_fpr']:.4f}\n\n")
        
        # Model summary
        f.write("MODEL SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write("Model Type: Gradient Boosting Classifier\n")
        f.write("Key Features:\n")
        f.write("  - Gradient boosting with decision trees\n")
        f.write("  - Similar performance to CatBoost/XGBoost\n")
        f.write("  - Feature importance calculation\n")
        f.write("  - Good handling of mixed data types\n")
        f.write("  - Built-in sklearn implementation\n\n")
        
        f.write("Key Findings:\n")
        f.write(f"  - Most important feature: {feature_importance_data[0][0]} ({feature_importance_data[0][1]:.4f})\n")
        f.write(f"  - Test F1 Score: {test_metrics['f1_score']:.4f}\n")
        f.write(f"  - Test AUC-ROC: {test_metrics['auc']:.4f}\n")
        f.write(f"  - Model shows {'excellent' if test_metrics['auc'] > 0.85 else 'good' if test_metrics['auc'] > 0.8 else 'moderate'} discriminative ability\n")
    
    print(f"\nResults saved to: {output_file}")

def main():
    """Main function to run the Gradient Boosting model"""
    print("="*60)
    print("GRADIENT BOOSTING MODEL FOR USER ENGAGEMENT PREDICTION")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    features, target, feature_names = prepare_features(df)
    
    # Split the data
    print(f"\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, 
        test_size=0.2, 
        random_state=42, 
        stratify=target
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training set target distribution: {y_train.value_counts().to_dict()}")
    print(f"Test set target distribution: {y_test.value_counts().to_dict()}")
    
    # Train model
    model, y_train_pred, y_test_pred, y_train_proba, y_test_proba = train_gradient_boosting_model(
        X_train, X_test, y_train, y_test
    )
    
    # Evaluate model
    train_metrics = evaluate_model(y_train, y_train_pred, y_train_proba, "Training")
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, "Test")
    
    # Print results instead of plotting
    feature_importance_data = print_feature_importance(model, feature_names)
    train_roc_metrics = print_roc_metrics(y_train, y_train_proba, "Training")
    test_roc_metrics = print_roc_metrics(y_test, y_test_proba, "Test")
    
    # Summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model Type: Gradient Boosting Classifier")
    print(f"Training Samples: {len(y_train):,}")
    print(f"Test Samples: {len(y_test):,}")
    print(f"Number of Features: {len(feature_names)}")
    print(f"\nTest Set Performance:")
    print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:  {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision:{test_metrics['precision']:.4f}")
    print(f"  Recall:   {test_metrics['recall']:.4f}")
    
    # Save model
    import joblib
    joblib.dump(model, "../data/gradient_boosting_model.pkl")
    print(f"\nModel saved to: ../data/gradient_boosting_model.pkl")
    
    # Save results to text file
    save_results_to_file(
        train_metrics, test_metrics, feature_importance_data, 
        train_roc_metrics, test_roc_metrics, X_train.shape, X_test.shape
    )
    
    return model, train_metrics, test_metrics

if __name__ == "__main__":
    model, train_metrics, test_metrics = main()
