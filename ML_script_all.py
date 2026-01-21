# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:43:00 2025

@author: User
"""
# Save this script as: D:\ML_Pore_Typing\Final_run\ML_classification.py

"""
ML Classification Script for Processed Pore Features
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, f1_score, 
                            classification_report, roc_curve, auc, 
                            precision_recall_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize

# Import classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                             GradientBoostingClassifier, AdaBoostClassifier,
                             HistGradientBoostingClassifier, BaggingClassifier)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import (SGDClassifier, PassiveAggressiveClassifier, 
                                 Perceptron, RidgeClassifier)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import json
import os
from pathlib import Path


# Global hyperparameters - simplified for efficiency
LOGISTIC_REGRESSION_PARAMS = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear'], 'max_iter': [100000]}
LDA_PARAMS = {'solver': ['svd', 'lsqr']}
#QDA_PARAMS = {}
KNN_PARAMS = {'n_neighbors': [3, 5, 7]}
DECISION_TREE_PARAMS = {'max_depth': [3, 5, 7]}
RANDOM_FOREST_PARAMS = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7]}
EXTRA_TREES_PARAMS = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7]}
GRADIENT_BOOSTING_PARAMS = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
XGBOOST_PARAMS = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
CATBOOST_PARAMS = {'iterations': [50, 100], 'learning_rate': [0.01, 0.1], 'depth': [3, 5]}
SVM_PARAMS = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
NU_SVC_PARAMS = {'nu': [0.1, 0.5], 'kernel': ['rbf', 'linear']}
LINEAR_SVC_PARAMS = {'C': [0.1, 1, 10]}
GAUSSIAN_NB_PARAMS = {}
MLP_PARAMS = {'hidden_layer_sizes': [(50,), (100,)], 'max_iter': [500, 1000]}
SGD_PARAMS = {'max_iter': [500, 1000], 'tol': [1e-3, 1e-4]}
PASSIVE_AGGRESSIVE_PARAMS = {'C': [0.1, 1, 10], 'max_iter': [500, 1000], 'tol': [1e-3, 1e-4]}
PERCEPTRON_PARAMS = {'max_iter': [500, 1000], 'tol': [1e-3, 1e-4]}
RIDGE_PARAMS = {'alpha': [0.1, 1, 10]}
BAGGING_PARAMS = {'n_estimators': [5, 10]}
ADABOOST_PARAMS = {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]}
HISTOGRAM_GBM_PARAMS = {'max_iter': [50, 100], 'learning_rate': [0.1, 1.0]}

# Enable or disable grid search
ENABLE_GRID_SEARCH = True

# Add these functions for checkpointing

def get_checkpoint_file():
    """Return the path to the checkpoint file."""
    base_dir = r"D:\ML_Pore_Typing\Final_run"
    return os.path.join(base_dir, "ml_progress_checkpoint.json")

def save_checkpoint(dataset_name, feature_folder, fold, model_name=None):
    """Save current progress to a checkpoint file."""
    checkpoint_file = get_checkpoint_file()
    
    # Create or load existing checkpoint data
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
    else:
        checkpoint_data = {
            "completed_folds": {},
            "current_progress": {}
        }
    
    # Update checkpoint data
    fold_key = f"{dataset_name}/{feature_folder}/Fold_{fold}"
    
    if model_name is None:
        # The entire fold is completed
        if "completed_folds" not in checkpoint_data:
            checkpoint_data["completed_folds"] = {}
        
        checkpoint_data["completed_folds"][fold_key] = True
        
        # Clear current progress for this fold since it's complete
        if fold_key in checkpoint_data.get("current_progress", {}):
            del checkpoint_data["current_progress"][fold_key]
    else:
        # Update progress within a fold
        if "current_progress" not in checkpoint_data:
            checkpoint_data["current_progress"] = {}
        
        if fold_key not in checkpoint_data["current_progress"]:
            checkpoint_data["current_progress"][fold_key] = []
        
        if model_name not in checkpoint_data["current_progress"][fold_key]:
            checkpoint_data["current_progress"][fold_key].append(model_name)
    
    # Save the checkpoint data
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"Checkpoint saved: {dataset_name}/{feature_folder}/Fold_{fold}" + 
          (f"/{model_name}" if model_name else " (completed)"))

def load_checkpoint():
    """Load the checkpoint data if it exists."""
    checkpoint_file = get_checkpoint_file()
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    
    return {"completed_folds": {}, "current_progress": {}}

def is_fold_completed(dataset_name, feature_folder, fold):
    """Check if a fold has been fully processed."""
    checkpoint_data = load_checkpoint()
    fold_key = f"{dataset_name}/{feature_folder}/Fold_{fold}"
    
    return fold_key in checkpoint_data.get("completed_folds", {})

def get_completed_models(dataset_name, feature_folder, fold):
    """Get list of models already completed for a fold."""
    checkpoint_data = load_checkpoint()
    fold_key = f"{dataset_name}/{feature_folder}/Fold_{fold}"
    
    return checkpoint_data.get("current_progress", {}).get(fold_key, [])


def get_used_hyperparameters(model, model_name):
    """Retrieve the hyperparameters used by the model after fitting."""
    used_params = model.get_params()
    print(f"Hyperparameters used by {model_name}:")
    for param, value in used_params.items():
        print(f"{param}: {value}")
    return used_params

def check_data_leakage(train_data, test_data, label_col):
    """Check for potential data leakage between train and test sets."""
    print("\n==== Checking for potential data leakage ====")
    
    # Ensure the label column exists
    if label_col not in train_data.columns or label_col not in test_data.columns:
        print(f"⚠️ Warning: '{label_col}' column not found in data")
        return
    
    # Check class distribution
    print(f"\nClass distribution in train set:")
    train_class_dist = train_data[label_col].value_counts(normalize=True) * 100
    print(train_class_dist)
    
    print(f"\nClass distribution in test set:")
    test_class_dist = test_data[label_col].value_counts(normalize=True) * 100
    print(test_class_dist)
    
    # Check for metadata columns that could cause leakage
    metadata_columns = ['Label', 'Centroid', 'source_file']
    leakage_cols = [col for col in metadata_columns if col in train_data.columns]
    if leakage_cols:
        print(f"\n⚠️ WARNING: Potential leakage columns found: {leakage_cols}")
    else:
        print("\n✓ No known metadata leakage columns found")
    
    # Check for overlapping samples
    print("\nChecking for overlapping samples...")
    
    # Function to convert row to tuple for comparison
    def row_to_tuple(row):
        # Convert all values to strings to handle different data types
        return tuple(str(val) for val in row)
    
    # Convert train and test data to sets of tuples
    train_set = set(train_data.apply(row_to_tuple, axis=1))
    test_set = set(test_data.apply(row_to_tuple, axis=1))
    
    # Check for overlap
    overlap = train_set.intersection(test_set)
    if len(overlap) > 0:
        print(f"\n⚠️ WARNING: Found {len(overlap)} overlapping samples between train and test")
        print("Sample of overlapping rows:")
        overlap_list = list(overlap)[:3]  # Show up to 3 examples
        for row in overlap_list:
            print(row)
    else:
        print("✓ No overlapping samples found between train and test")
    
    # Check for feature correlations with class
    print("\nChecking for features with high correlation to class...")
    
    # Separate features and target
    X_train = train_data.drop(columns=[label_col])
    y_train = train_data[label_col]
    
    # Convert class to numeric if it's not already
    if not pd.api.types.is_numeric_dtype(y_train):
        print("Converting class labels to numeric...")
        class_mapping = {label: i for i, label in enumerate(y_train.unique())}
        y_train_numeric = y_train.map(class_mapping)
    else:
        y_train_numeric = y_train
    
    # Calculate correlation for numeric features only
    numeric_features = X_train.select_dtypes(include=['number'])
    if not numeric_features.empty:
        # Add class to calculate correlation
        numeric_with_class = numeric_features.copy()
        numeric_with_class['Class'] = y_train_numeric
        
        # Calculate correlation with class
        correlations = numeric_with_class.corr()['Class'].drop('Class').abs()
        high_corr_features = correlations[correlations > 0.9].sort_values(ascending=False)
        
        if not high_corr_features.empty:
            print(f"\n⚠️ WARNING: Found {len(high_corr_features)} features with >0.9 correlation to class:")
            for feat, corr in high_corr_features.items():
                print(f"  - {feat}: {corr:.4f}")
        else:
            print("✓ No features with suspiciously high correlation (>0.9) to class")
    else:
        print("No numeric features found to check correlations")
    
    print("\n==== Data leakage check completed ====")
    
    # Return True if no leakage is found, False otherwise
    return len(overlap) == 0 and len(leakage_cols) == 0 and (
        high_corr_features.empty if 'high_corr_features' in locals() else True)

def load_processed_data(base_dir, dataset_name, feature_folder, fold):
    """Load processed train and test datasets from the Boruta results."""
    print(f"Loading fold {fold} data from {dataset_name}/{feature_folder}...")
    
    # Construct the path to the train and test csv files
    # Path pattern: BASE_DIR/DATASET_NAME/FEATURE_FOLDER/Preprocessing_Results/boruta_results/csv_files/split_X
    data_path = os.path.join(base_dir, dataset_name, feature_folder, 
                             "Preprocessing_Results", "boruta_results", 
                             "csv_files", f"split_{fold-1}")
    
    train_file = os.path.join(data_path, "train.csv")
    test_file = os.path.join(data_path, "test.csv")
    
    # Verify files exist
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Train or test file not found at {data_path}")
    
    # Load the data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # Check for data leakage
    label_col = 'Class' if 'Class' in train_data.columns else 'label'
    leakage_check_passed = check_data_leakage(train_data, test_data, label_col)
    
    if not leakage_check_passed:
        print("⚠️ Warning: Potential data leakage detected. Consider reviewing your data.")
    
    # Rename 'Class' column to 'label' if necessary
    if 'Class' in train_data.columns and 'label' not in train_data.columns:
        print(f"Renaming 'Class' column to 'label' for consistency")
        train_data = train_data.rename(columns={'Class': 'label'})
        test_data = test_data.rename(columns={'Class': 'label'})
    
    # Extract features and labels
    label_col = 'label'  # Now we know it's called 'label'
    X_train = train_data.drop(columns=[label_col])
    y_train = train_data[label_col]
    
    X_test = test_data.drop(columns=[label_col])
    y_test = test_data[label_col]
    
    print(f"Loaded: Train samples: {X_train.shape[0]}, features: {X_train.shape[1]}, "
          f"Test samples: {X_test.shape[0]}, features: {X_test.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def setup_output_directories(dataset_name, feature_folder, fold):
    """Create all required output directories for the given dataset and fold."""
    # Create output directory within the base directory
    output_base = os.path.join(BASE_DIR, "ML_results", dataset_name, feature_folder)
    
    # Create all needed directories
    directories = [
        output_base,
        os.path.join(output_base, f"Fold_{fold}"),
        os.path.join(output_base, f"Fold_{fold}", "precision_recall_curves"),
        os.path.join(output_base, f"Fold_{fold}", "ROC_curves"),
        os.path.join(output_base, f"Fold_{fold}", "models"),
        os.path.join(output_base, f"Fold_{fold}", "summary")
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    return output_base

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, 
                            params, output_base, fold):
    """Train, evaluate and save model performance metrics and visualizations."""
    print(f"Training and evaluating {model_name}...")
    starting_time = datetime.now()
    
    # Grid search for hyperparameter tuning if enabled
    if ENABLE_GRID_SEARCH and hasattr(model, 'get_params'):
        try:
            grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        except Exception as e:
            print(f"Grid search failed for {model_name}: {str(e)}")
            print("Falling back to default parameters.")
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Predictions and metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    
    # Binarize labels for multi-class ROC/PR curves
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_binarized.shape[1]
    
    # Get prediction scores
    try:
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)
        else:
            y_scores = model.decision_function(X_test)
    except:
        print(f"Warning: {model_name} doesn't support predict_proba or decision_function")
        # Create a dummy y_scores for plotting
        y_scores = np.zeros((len(y_test), n_classes))
        for i, pred in enumerate(y_pred):
            y_scores[i, list(np.unique(y_test)).index(pred)] = 1
    
    # Plot settings
    font_size = 25
    legend_font_size = 20
    
    # Precision-Recall Curve
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_test_binarized[:, i], y_scores[:, i])
            pr_auc = auc(recall_curve, precision_curve)
            plt.plot(recall_curve, precision_curve, lw=2, 
                    label=f'Class {i} (area = {pr_auc:.4f})')
        except:
            print(f"Warning: Could not create PR curve for class {i}")
    
    plt.xlabel('Recall', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.legend(loc='lower left', fontsize=legend_font_size)
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.savefig(f"{output_base}/Fold_{fold}/precision_recall_curves/{model_name}_PR_curve.jpeg")
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.4f})')
        except:
            print(f"Warning: Could not create ROC curve for class {i}")
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.legend(loc='lower right', fontsize=legend_font_size)
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.savefig(f"{output_base}/Fold_{fold}/ROC_curves/{model_name}_ROC_curve.jpeg")
    plt.close()
    
    # Get used hyperparameters
    used_params = get_used_hyperparameters(model, model_name)
    
    print(f"{model_name} trained and evaluated successfully. Accuracy: {accuracy:.4f}")
    end_time = datetime.now()
    total_time = end_time - starting_time
    
    return model, accuracy, precision, f1, cv_scores, report, used_params, total_time

def save_model(model, model_name, output_base, fold):
    """Save trained model to disk."""
    print(f"Saving {model_name}...")
    try:
        model_dir = f'{output_base}/Fold_{fold}/models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Create a filename
        filename = f"{model_name}.pkl"
        model_path = os.path.join(model_dir, filename)
        
        # Save the model
        joblib.dump(model, model_path)
        print(f"{model_name} saved successfully to {model_path}")
    except Exception as e:
        print(f"Error in saving {model_name}: {str(e)}")

def save_model_metadata(model_name, X_train, output_base, fold):
    """Save feature names for the model for later post-processing."""
    try:
        metadata_dir = f'{output_base}/Fold_{fold}/models/metadata'
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)
        
        # Save feature names if available
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
            feature_df = pd.DataFrame({'feature_name': feature_names})
            feature_df.to_csv(f"{metadata_dir}/{model_name}_features.csv", index=False)
            print(f"Feature metadata saved for {model_name}")
    except Exception as e:
        print(f"Error saving metadata for {model_name}: {str(e)}")

def save_model_rankings(results_df, output_base, fold):
    """Save a ranking of models by performance metrics."""
    try:
        # Create a sorted version by accuracy
        top_models = results_df.sort_values('Accuracy', ascending=False)
        
        # Save to summary directory
        summary_dir = f'{output_base}/Fold_{fold}/summary'
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        
        # Save all models
        top_models[['Model', 'Accuracy', 'Precision', 'F1']].to_csv(
            f"{summary_dir}/all_models.csv", index=False)
        
        # Save top 5 models separately
        top5 = top_models.head(5)[['Model', 'Accuracy', 'Precision', 'F1']]
        top5.to_csv(f"{summary_dir}/top_models.csv", index=False)
        print(f"Model rankings saved for Fold {fold}")
    except Exception as e:
        print(f"Error saving model rankings: {str(e)}")

# Modify the process_dataset_fold function to use checkpointing
def process_dataset_fold(base_dir, dataset_name, feature_folder, fold):
    """Process a single fold of a dataset with checkpointing."""
    # Check if this fold is already completed
    if is_fold_completed(dataset_name, feature_folder, fold):
        print(f"\n{'='*50}")
        print(f"Skipping {dataset_name}/{feature_folder} - Fold {fold} (already completed)")
        print(f"{'='*50}\n")
        return True
    
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name}/{feature_folder} - Fold {fold}")
    print(f"{'='*50}\n")
    
    try:
        # Set up output directories
        output_base = setup_output_directories(dataset_name, feature_folder, fold)
        
        # Load data
        X_train, X_test, y_train, y_test = load_processed_data(
            base_dir, dataset_name, feature_folder, fold)
        
        # Check number of classes
        n_classes = len(np.unique(y_train))
        print(f"Number of classes in the dataset: {n_classes}")
        if n_classes < 2:
            raise ValueError("The dataset must contain at least 2 classes for classification tasks.")
        
        # Get the list of completed models
        completed_models = get_completed_models(dataset_name, feature_folder, fold)
        print(f"Resuming from checkpoint: {len(completed_models)} models already processed")
        
        # Define models to train
        models = [
            (LinearDiscriminantAnalysis(), "LDA", LDA_PARAMS),
            #(QuadraticDiscriminantAnalysis(), "QDA", QDA_PARAMS),
            (KNeighborsClassifier(), "KNN", KNN_PARAMS),
            (DecisionTreeClassifier(), "DecisionTree", DECISION_TREE_PARAMS),
            (RandomForestClassifier(), "RandomForest", RANDOM_FOREST_PARAMS),
            (ExtraTreesClassifier(), "ExtraTrees", EXTRA_TREES_PARAMS),
            (GradientBoostingClassifier(), "GradientBoosting", GRADIENT_BOOSTING_PARAMS),
            (XGBClassifier(), "XGBoost", XGBOOST_PARAMS),
            (CatBoostClassifier(verbose=False), "CatBoost", CATBOOST_PARAMS),
            (SVC(probability=True), "SVM", SVM_PARAMS),
            (NuSVC(probability=True), "NuSVC", NU_SVC_PARAMS),
            (LinearSVC(dual="auto"), "LinearSVC", LINEAR_SVC_PARAMS),
            (GaussianNB(), "GaussianNB", GAUSSIAN_NB_PARAMS),
            (MLPClassifier(), "MLP", MLP_PARAMS),
            (SGDClassifier(), "SGD", SGD_PARAMS),
            (PassiveAggressiveClassifier(), "PassiveAggressive", PASSIVE_AGGRESSIVE_PARAMS),
            (Perceptron(), "Perceptron", PERCEPTRON_PARAMS),
            (RidgeClassifier(), "Ridge", RIDGE_PARAMS),
            (BaggingClassifier(), "Bagging", BAGGING_PARAMS),
            (AdaBoostClassifier(), "AdaBoost", ADABOOST_PARAMS),
            (HistGradientBoostingClassifier(), "HistGradientBoosting", HISTOGRAM_GBM_PARAMS),
        ]
        
        # Filter out models that have already been processed
        models_to_process = [(model, name, params) for model, name, params in models 
                            if name not in completed_models]
        
        if not models_to_process:
            print(f"All models for this fold have been processed. Marking fold as complete.")
            save_checkpoint(dataset_name, feature_folder, fold)
            return True
        
        # Load existing results if any
        results = []
        results_csv_path = None
        
        # Look for the most recent results CSV
        result_files = list(Path(f"{output_base}/Fold_{fold}").glob("results_*.csv"))
        if result_files:
            most_recent = max(result_files, key=os.path.getmtime)
            results_csv_path = str(most_recent)
            print(f"Found existing results file: {results_csv_path}")
            results_df = pd.read_csv(results_csv_path)
            results = results_df.to_dict('records')
        
        # If no existing results file, create a new one
        if not results_csv_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_csv_path = f"{output_base}/Fold_{fold}/results_{timestamp}.csv"
        
        for model, model_name, params in models_to_process:
            try:
                print(f"Processing model {model_name} ({len(completed_models) + len(results) + 1}/{len(models)})")
                
                trained_model, accuracy, precision, f1, cv_scores, report, used_params, total_time = (
                    train_and_evaluate_model(
                        model, X_train, X_test, y_train, y_test, 
                        model_name, params, output_base, fold)
                )
                
                if trained_model is not None:
                    save_model(trained_model, model_name, output_base, fold)
                    save_model_metadata(model_name, X_train, output_base, fold)
                    
                    # Create a results entry for this model
                    model_result = {
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'F1': f1,
                        'CV_Scores_Mean': np.mean(cv_scores),
                        'CV_Scores_Std': np.std(cv_scores),
                        'Total_Runtime': total_time,
                        'Hyperparameters': str(used_params),
                        'Report': report
                    }
                    
                    # Add to our running list of results
                    results.append(model_result)
                    
                    # Write/update the results CSV after each model
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(results_csv_path, index=False)
                    print(f"Updated results CSV with {model_name} results")
                    
                    # Update checkpoint for this model
                    save_checkpoint(dataset_name, feature_folder, fold, model_name)
                
            except Exception as e:
                print(f"Error processing model {model_name}: {str(e)}")
        
        # Save model rankings for easier post-processing
        save_model_rankings(pd.DataFrame(results), output_base, fold)
        
        # Mark the fold as completed
        save_checkpoint(dataset_name, feature_folder, fold)
        
        print(f"All models for {dataset_name}/{feature_folder} Fold {fold} completed.")
        return True
    
    except Exception as e:
        print(f"Error processing {dataset_name}/{feature_folder} Fold {fold}: {str(e)}")
        return False

def create_aggregate_results(base_dir, dataset_name, feature_folder):
    """Aggregate results across all folds for a specific dataset and feature folder."""
    print(f"\n{'='*50}")
    print(f"Creating aggregate results for {dataset_name}/{feature_folder}")
    print(f"{'='*50}\n")
    
    output_dir = os.path.join(base_dir, "ML_results", dataset_name, feature_folder, "aggregate")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_fold_results = []
    
    # Collect results from each fold
    for fold in range(1, 6):
        summary_path = os.path.join(base_dir, "ML_results", dataset_name, feature_folder, 
                                   f"Fold_{fold}", "summary", "all_models.csv")
        
        if os.path.exists(summary_path):
            fold_results = pd.read_csv(summary_path)
            fold_results['Fold'] = fold
            all_fold_results.append(fold_results)
    
    if not all_fold_results:
        print(f"No results found for {dataset_name}/{feature_folder}")
        return
    
    # Combine all fold results
    combined_results = pd.concat(all_fold_results, ignore_index=True)
    
    # Group by model and calculate mean and std of metrics
    aggregate_results = combined_results.groupby('Model').agg({
        'Accuracy': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'F1': ['mean', 'std']
    }).reset_index()
    
    # Flatten the MultiIndex columns
    aggregate_results.columns = ['Model', 
                               'Accuracy_Mean', 'Accuracy_Std',
                               'Precision_Mean', 'Precision_Std',
                               'F1_Mean', 'F1_Std']
    
    # Sort by accuracy
    aggregate_results = aggregate_results.sort_values('Accuracy_Mean', ascending=False)
    
    # Save aggregate results
    aggregate_results.to_csv(f"{output_dir}/aggregate_results.csv", index=False)
    
    # Save top 5 models
    top5 = aggregate_results.head(5)
    top5.to_csv(f"{output_dir}/top5_models.csv", index=False)
    
    print(f"Aggregate results saved to {output_dir}/aggregate_results.csv")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy with error bars
    models = aggregate_results['Model'].values
    accuracy_means = aggregate_results['Accuracy_Mean'].values
    accuracy_stds = aggregate_results['Accuracy_Std'].values
    
    # Sort by accuracy for better visualization
    idx = np.argsort(accuracy_means)[::-1]  # descending order
    models = models[idx]
    accuracy_means = accuracy_means[idx]
    accuracy_stds = accuracy_stds[idx]
    
    # Plot top 10 models only for clarity
    plt.barh(models[:10], accuracy_means[:10], xerr=accuracy_stds[:10], 
             alpha=0.7, capsize=5)
    plt.xlabel('Accuracy', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.title(f'Model Accuracy for {dataset_name}/{feature_folder}', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300)
    plt.close()
    
    print(f"Model comparison visualization saved to {output_dir}/model_comparison.png")

# Modify the main function to use checkpointing
def main():
    """Main function to run the ML classification pipeline with checkpointing."""
    global BASE_DIR
    # Base directory where processed data is located
    BASE_DIR = r"D:\ML_Pore_Typing\Final_run"
    
    # Print checkpoint status
    checkpoint_data = load_checkpoint()
    completed_count = len(checkpoint_data.get("completed_folds", {}))
    in_progress_count = len(checkpoint_data.get("current_progress", {}))
    
    print(f"Starting ML pipeline with checkpoint system")
    print(f"Checkpoint status: {completed_count} folds completed, {in_progress_count} in progress")
    
    # Define datasets and their feature folders to process
    DATASETS = [
        {
            "name": "Pore_features",
            "folders": ["All_Features", "DL_Features", "Traditional_Features"]
        },
        {
            "name": "Pore_and_neighbourhood_features",
            "folders": ["All_Features_NI", "DL_Features_NI", "Traditional_Features_NI"]
        }
    ]
    
    # Track total progress
    total_folds = 0
    completed_folds = 0
    
    for dataset in DATASETS:
        dataset_name = dataset["name"]
        for feature_folder in dataset["folders"]:
            total_folds += 5  # 5 folds per dataset/feature combination
    
    print(f"Total tasks: {total_folds} folds across {len(DATASETS) * 3} datasets/feature combinations")
    
    # Process each dataset and its folders
    for dataset in DATASETS:
        dataset_name = dataset["name"]
        
        for feature_folder in dataset["folders"]:
            # Process all 5 folds for this dataset and feature folder
            success_count = 0
            for fold in range(1, 6):
                if process_dataset_fold(BASE_DIR, dataset_name, feature_folder, fold):
                    success_count += 1
                    completed_folds += 1
                    
                print(f"Overall progress: {completed_folds}/{total_folds} folds completed ({completed_folds/total_folds*100:.1f}%)")
            
            # Create aggregate results if at least one fold was successful
            if success_count > 0:
                create_aggregate_results(BASE_DIR, dataset_name, feature_folder)
            else:
                print(f"No successful folds for {dataset_name}/{feature_folder}, "
                      f"skipping aggregate results.")
    
    print("\nAll datasets and folds have been processed.")

if __name__ == "__main__":
    main()