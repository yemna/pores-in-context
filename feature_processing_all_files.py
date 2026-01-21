# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:07:33 2025

@author: User
"""



import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import gaussian_kde
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import joblib
from boruta import BorutaPy
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.model_selection import StratifiedKFold

# Define base directory and datasets to process
BASE_DIR = r"D:\ML_Pore_Typing\Final_run"

# Define dataset paths and their structure
DATASETS = [
    {
        "name": "Pore_features",
        "folders": ["All_Features", "DL_Features", "Traditional_Features"],
        "files": ["all_features.csv", "dl_features.csv", "traditional_features.csv"]
    },
    {
        "name": "Pore_and_neighbourhood_features",
        "folders": ["All_Features_NI", "DL_Features_NI", "Traditional_Features_NI"],
        "files": ["all_features_NI.csv", "dl_features_NI.csv", "traditional_features_NI.csv"]
    }
]

# Helper functions
def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Modified preprocessing pipeline to run for each dataset
def run_preprocessing_pipeline(dataset_path, output_dir, csv_filename):
    print(f"\n==== PROCESSING: {csv_filename} in {dataset_path} ====")
    
    # Create dataset-specific output directory
    create_directory(output_dir)
    os.chdir(output_dir)  # Change to output directory for all outputs
    
    # Load the data
    data_file_path = os.path.join(dataset_path, csv_filename)
    print(f"Loading data from: {data_file_path}")
    
    try:
        data = pd.read_csv(data_file_path)
        print(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 1: Remove features with high missing values
    print("\n-- STEP 1: Removing features with high missing values --")
    cleaned_data, removed_features = remove_high_missing_features(data, threshold=0.05)
    
    # Step 2: Create 5-fold stratified cross-validation splits
    print("\n-- STEP 2: Creating 5-fold stratified cross-validation splits --")
    splits = generate_kfold_splits(cleaned_data, "Class")
    
    # Check fold integrity
    print("\n-- Checking fold integrity --")
    check_fold_integrity(splits)
    
    # Step 3: Remove zero variance and zero IQR features
    print("\n-- STEP 3: Removing zero variance and zero IQR features --")
    splits_filtered, removed_features_variance_iqr = remove_zero_variance_iqr(splits)
    
    # Save and align splits
    print("\n-- Saving and aligning splits --")
    processed_data = save_and_align_splits(splits_filtered)
    
    # Step 4: Apply robust scaling
    print("\n-- STEP 4: Applying robust scaling --")
    robust_scaled_splits = apply_robust_scaling(processed_data)
    
    # Step 5: Enhanced mutual information and correlation feature selection
    print("\n-- STEP 5: Performing feature selection with mutual information --")
    feature_selected_splits = feature_selection_with_mutual_info(robust_scaled_splits, correlation_threshold=0.7)
    
    # Step 6: Boruta feature selection
    print("\n-- STEP 6: Performing Boruta feature selection --")
    boruta_splits = boruta_feature_selection(
        feature_selected_splits,
        n_estimators=250,
        max_iter=50,
        perc=70
    )
    
    # Export Boruta results
    print("\n-- Exporting Boruta results --")
    export_boruta_csv()
    
    # Extract feature importance
    extract_feature_importance(0)
    
    # Create summary and visualization
    print("\n-- Creating feature tracking summary --")
    summary = create_feature_tracking_summary(data)
    
    print(f"\n==== COMPLETED PROCESSING: {csv_filename} ====\n")
    
    # Return to base directory
    os.chdir(BASE_DIR)


# Step 1: Remove features with high missing values
def remove_high_missing_features(data, threshold=0.05, save_results=True):
    """
    Remove features with high percentage of missing values.
    Works with either a DataFrame or a file path.
    Returns cleaned data and list of removed features.
    """
    if save_results:
        create_directory("missing_value_analysis")
    
    # Handle both DataFrame and path input
    if isinstance(data, str):
        # It's a file path
        data = pd.read_csv(data)
        print(f"Loaded data with shape: {data.shape}")
    
    print(f"Original data shape: {data.shape}")
    
    # Calculate missing value percentages
    missing_percentages = data.isnull().mean()
    high_missing_features = missing_percentages[missing_percentages > threshold].index.tolist()
    
    if save_results:
        # Save missing value information
        missing_info = pd.DataFrame({
            'Feature': missing_percentages.index,
            'Missing_Percentage': missing_percentages.values * 100
        })
        missing_info = missing_info.sort_values('Missing_Percentage', ascending=False)
        missing_info.to_csv("missing_value_analysis/missing_percentages.csv", index=False)
    
    # Remove features with high missing percentages
    data_clean = data.drop(columns=high_missing_features)
    
    if save_results:
        # Save cleaned data
        data_clean.to_csv("missing_value_analysis/data_cleaned.csv", index=False)
        
        # Save summary
        summary = pd.DataFrame({
            'Metric': ['Initial Features', 'Features Removed', 'Remaining Features'],
            'Count': [len(data.columns), len(high_missing_features), len(data_clean.columns)]
        })
        summary.to_csv("missing_value_analysis/missing_values_summary.csv", index=False)
    
    print(f"Removed {len(high_missing_features)} features with >{threshold*100}% missing values")
    print(f"Cleaned data shape: {data_clean.shape}")
    
    return data_clean, high_missing_features

# Step 2: Create 5-fold stratified cross-validation splits
def generate_kfold_splits(data, target_col, n_splits=5, random_seed=42):
    """Generate stratified k-fold cross-validation splits."""
    # Create output directory
    create_directory("KFold_Splits")
    
    # Initialize the k-fold cross-validator
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Lists to store all splits
    splits = []
    
    # Generate the k folds
    fold_idx = 1
    for train_index, test_index in skf.split(data.drop(columns=[target_col]), data[target_col]):
        # Split the data for this fold
        train_data = data.iloc[train_index].copy()
        test_data = data.iloc[test_index].copy()
        
        # Store class distribution
        train_classes = pd.DataFrame(train_data[target_col].value_counts()).reset_index()
        train_classes.columns = ['Class', 'Count']
        test_classes = pd.DataFrame(test_data[target_col].value_counts()).reset_index()
        test_classes.columns = ['Class', 'Count']
        
        # Save to CSV with fold number
        fold_dir = os.path.join("KFold_Splits", f"Fold_{fold_idx}")
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
            
        train_data.to_csv(f"{fold_dir}/train.csv", index=False)
        test_data.to_csv(f"{fold_dir}/test.csv", index=False)
        train_classes.to_csv(f"{fold_dir}/train_classes.csv", index=False)
        test_classes.to_csv(f"{fold_dir}/test_classes.csv", index=False)
        
        # Add this split to our list
        splits.append({
            'fold': fold_idx,
            'train': train_data,
            'test': test_data,
            'train_classes': train_classes,
            'test_classes': test_classes
        })
        
        fold_idx += 1
    
    print(f"Created {n_splits}-fold stratified cross-validation splits (random_seed={random_seed})")
    return splits

# Function to check fold integrity
def check_fold_integrity(splits):
    """Check if cross-validation folds are properly created with no data leakage."""
    print("\n=== CHECKING CROSS-VALIDATION FOLD INTEGRITY ===")
    
    # Verify no sample appears in train and test of same fold
    for i, split in enumerate(splits):
        train_rows = set(split['train'].astype(str).apply(tuple, axis=1))
        test_rows = set(split['test'].astype(str).apply(tuple, axis=1))
        
        overlap = train_rows.intersection(test_rows)
        if len(overlap) > 0:
            print(f"WARNING: Fold {i+1} has {len(overlap)} overlapping samples between train and test!")
        else:
            print(f"Fold {i+1}: No overlap between train and test sets ✓")
    
    # Verify each sample appears in test set exactly once
    all_test_indices = []
    for split in splits:
        all_test_indices.extend(split['test'].astype(str).apply(tuple, axis=1).tolist())
    
    unique_indices = set(all_test_indices)
    
    if len(all_test_indices) == len(unique_indices):
        print(f"Each sample appears in exactly one test set ✓")
    else:
        duplicate_count = len(all_test_indices) - len(unique_indices)
        print(f"WARNING: {duplicate_count} samples appear in multiple test sets!")
    
    # Verify class distribution is maintained
    if len(splits) > 0:
        # Get the full dataset by combining all train and test sets
        first_fold = splits[0]
        full_data = pd.concat([first_fold['train'], first_fold['test']])
        for split in splits[1:]:
            full_data = pd.concat([full_data, split['train'], split['test']])
        full_data = full_data.drop_duplicates()
        
        original_dist = full_data['Class'].value_counts(normalize=True)
        
        for i, split in enumerate(splits):
            train_dist = split['train']['Class'].value_counts(normalize=True)
            test_dist = split['test']['Class'].value_counts(normalize=True)
            
            train_diff = sum((train_dist.get(c, 0) - original_dist.get(c, 0))**2 for c in set(original_dist.index) | set(train_dist.index))**0.5
            test_diff = sum((test_dist.get(c, 0) - original_dist.get(c, 0))**2 for c in set(original_dist.index) | set(test_dist.index))**0.5
            
            if train_diff > 0.1 or test_diff > 0.1:
                print(f"WARNING: Fold {i+1} has class distribution differences greater than 10%")
            else:
                print(f"Fold {i+1}: Class distribution properly maintained ✓")
    
    print("=== FOLD INTEGRITY CHECK COMPLETE ===\n")

# Step 3: Remove zero variance and zero IQR features
def remove_zero_variance_iqr(splits, save_summary=True):
    """Remove features with zero variance or zero IQR."""
    create_directory("features_processing_zero_variance_iqr")
    
    # Track removed features
    removed_features = pd.DataFrame(columns=['Feature', 'Reason', 'Split'])
    processed_splits = []
    
    for i, split in enumerate(splits):
        train_data = split['train']
        test_data = split['test']
        
        # Get feature columns (exclude Label and Class if they exist)
        meta_cols = ['Label', 'Class'] if 'Label' in train_data.columns else ['Class']
        features = train_data.drop(columns=meta_cols)
        
        # First identify and handle non-numeric columns
        non_numeric_cols = []
        for col in features.columns:
            try:
                # Check if we can compute variance (this catches non-numeric columns)
                _ = features[col].var()
            except:
                non_numeric_cols.append(col)
                # Add to removed_features
                removed_features = pd.concat([removed_features, 
                                            pd.DataFrame({'Feature': [col], 
                                                        'Reason': ['Non-numeric data'], 
                                                        'Split': [i+1]})], 
                                            ignore_index=True)
        
        if non_numeric_cols:
            print(f"Split {i+1}: Removing {len(non_numeric_cols)} non-numeric columns")
            features = features.drop(columns=non_numeric_cols)
            
        # Now proceed with numeric columns only
        # Identify zero variance features
        variance = features.var()
        zero_var_features = variance[variance <= 1e-8].index.tolist()
        
        # Identify zero IQR features
        try:
            iqr = features.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
            zero_iqr_features = iqr[iqr <= 1e-8].index.tolist()
            zero_iqr_features = [f for f in zero_iqr_features if f not in zero_var_features]
        except:
            print(f"Warning: IQR calculation failed for split {i+1}. Skipping IQR check.")
            zero_iqr_features = []
        
        # Track removed features
        for feature in zero_var_features:
            removed_features = pd.concat([removed_features, 
                                         pd.DataFrame({'Feature': [feature], 
                                                      'Reason': ['Zero variance'], 
                                                      'Split': [i+1]})], 
                                         ignore_index=True)
        
        for feature in zero_iqr_features:
            removed_features = pd.concat([removed_features, 
                                         pd.DataFrame({'Feature': [feature], 
                                                      'Reason': ['Zero IQR'], 
                                                      'Split': [i+1]})], 
                                         ignore_index=True)
        
        # Remove identified features
        features_to_remove = non_numeric_cols + zero_var_features + zero_iqr_features
        train_data_clean = train_data.drop(columns=features_to_remove)
        
        processed_splits.append({
            'train': train_data_clean,
            'test': test_data,  # Keep test data unchanged
            'removed_features': features_to_remove
        })
    
    # Find common features across all splits
    common_features = set(processed_splits[0]['train'].columns)
    for split in processed_splits[1:]:
        common_features = common_features.intersection(set(split['train'].columns))
    
    # Ensure all splits have the same features
    final_splits = []
    for i, split in enumerate(processed_splits):
        uncommon_features = set(split['train'].columns) - common_features
        
        # Track uncommon features
        for feature in uncommon_features:
            removed_features = pd.concat([removed_features, 
                                         pd.DataFrame({'Feature': [feature], 
                                                      'Reason': ['Not common across splits'], 
                                                      'Split': [i+1]})], 
                                         ignore_index=True)
        
        # Keep only common features
        final_splits.append({
            'train': split['train'][list(common_features)],
            'test': split['test'],
            'fold': splits[i].get('fold', i+1)
        })
    
    # Create summary
    if save_summary:
        # Count unique features removed by reason
        summary = pd.DataFrame({
            'Summary': [
                'Initial number of features',
                'Features removed due to non-numeric data',
                'Features removed due to zero variance',
                'Features removed due to zero IQR',
                'Features removed for not being common across splits',
                'Total features removed',
                'Remaining features after processing'
            ],
            'Count': [
                len(splits[0]['train'].columns) - (2 if 'Label' in splits[0]['train'].columns else 1),
                len(set(removed_features[removed_features['Reason'] == 'Non-numeric data']['Feature'])),
                len(set(removed_features[removed_features['Reason'] == 'Zero variance']['Feature'])),
                len(set(removed_features[removed_features['Reason'] == 'Zero IQR']['Feature'])),
                len(set(removed_features[removed_features['Reason'] == 'Not common across splits']['Feature'])),
                len(splits[0]['train'].columns) - len(common_features),
                len(common_features) - (2 if 'Label' in splits[0]['train'].columns else 1)
            ]
        })
        
        summary.to_csv("features_processing_zero_variance_iqr/features_processing_zero_variance_iqr_summary.csv", index=False)
        removed_features.to_csv("features_processing_zero_variance_iqr/removed_features_zero_variance_iqr_details.csv", index=False)
    
    print(f"Removed zero variance and zero IQR features, keeping {len(common_features)} common features")
    return final_splits, removed_features

# Define the function for aligning splits
def save_and_align_splits(splits, directory="Processed_Data_after_zero_variance_iqr"):
    """
    Save processed train and test data and ensure test has the same features as train.
    """
    create_directory(directory)
    
    # Save data and align test features
    for i, split in enumerate(splits):
        train_data = split['train']
        test_data = split['test']
        
        # Save original data
        train_data.to_csv(f"{directory}/train_split{i+1}_processed.csv", index=False)
        test_data.to_csv(f"{directory}/test_split{i+1}_original.csv", index=False)
        
        # Get train features
        meta_cols = ['Label', 'Class'] if 'Label' in train_data.columns else ['Class']
        train_features = [col for col in train_data.columns if col not in meta_cols]
        
        # Process test data to match train features
        test_features_to_keep = meta_cols + [col for col in train_features if col in test_data.columns]
        test_data_processed = test_data[test_features_to_keep]
        
        # Save processed test data
        test_data_processed.to_csv(f"{directory}/test_split{i+1}_processed.csv", index=False)
        
        # Update split with processed test data
        splits[i]['test'] = test_data_processed
        # Ensure fold information is preserved
        if 'fold' not in splits[i] and i < len(splits):
            splits[i]['fold'] = i+1
    
    # Save a summary
    summary = pd.DataFrame({
        'Split': range(1, len(splits) + 1),
        'Train_Samples': [len(split['train']) for split in splits],
        'Test_Samples': [len(split['test']) for split in splits],
        'Feature_Count': [len(split['train'].columns) - len(['Label', 'Class'] 
                                                         if 'Label' in split['train'].columns 
                                                         else ['Class']) 
                       for split in splits]
    })
    summary.to_csv(f"{directory}/summary.csv", index=False)
    
    print(f"Saved all processed train and test data to {directory}")
    return splits

# Step 4: Apply robust scaling
def apply_robust_scaling(splits):
    create_directory("Scaled_Results")
    
    scaled_splits = []
    
    for i, split in enumerate(splits):
        train_data = split['train']
        test_data = split['test']
        
        # Separate metadata columns
        meta_cols = ['Label', 'Class'] if 'Label' in train_data.columns else ['Class']
        train_meta = train_data[meta_cols]
        train_features = train_data.drop(columns=meta_cols)
        
        # Same for test data
        test_meta = test_data[meta_cols]
        test_features = test_data.drop(columns=meta_cols)
        
        # Fit scaler on train data only
        scaler = RobustScaler()
        scaler.fit(train_features)
        
        # Transform both train and test data
        train_scaled = pd.DataFrame(
            scaler.transform(train_features),
            columns=train_features.columns
        )
        
        test_scaled = pd.DataFrame(
            scaler.transform(test_features),
            columns=test_features.columns
        )
        
        # Combine with metadata
        scaled_train = pd.concat([train_meta.reset_index(drop=True), 
                                 train_scaled.reset_index(drop=True)], axis=1)
        
        scaled_test = pd.concat([test_meta.reset_index(drop=True), 
                                test_scaled.reset_index(drop=True)], axis=1)
        
        # Save scaled data
        scaled_train.to_csv(f"Scaled_Results/scaled_train_split{i+1}.csv", index=False)
        scaled_test.to_csv(f"Scaled_Results/scaled_test_split{i+1}.csv", index=False)
        
        # Store the scaler for potential future use
        scaled_splits.append({
            'train': scaled_train,
            'test': scaled_test,
            'scaler': scaler,
            'fold': split.get('fold', i+1)
        })
    
    # Create a summary
    summary = pd.DataFrame({
        'Split': range(1, len(splits) + 1),
        'Train_Samples': [len(split['train']) for split in scaled_splits],
        'Test_Samples': [len(split['test']) for split in scaled_splits],
        'Feature_Count': [len(split['train'].columns) - len(['Label', 'Class'] 
                                                         if 'Label' in split['train'].columns 
                                                         else ['Class']) 
                       for split in scaled_splits]
    })
    summary.to_csv("Scaled_Results/scaling_summary.csv", index=False)
    
    print(f"Applied robust scaling to train and test data")
    return scaled_splits

# Step 5: Enhanced mutual information and correlation feature selection
def feature_selection_with_mutual_info(splits, correlation_threshold=0.7, mi_percentile=70):
    create_directory("feature_selection")
    
    # Track feature selection information
    all_info = pd.DataFrame()
    
    for i, split in enumerate(splits):
        train_data = split['train']
        test_data = split['test']
        
        # Separate metadata columns
        meta_cols = ['Label', 'Class'] if 'Label' in train_data.columns else ['Class']
        y = train_data['Class']
        X = train_data.drop(columns=meta_cols)
        
        print(f"Split {i+1}: Starting with {X.shape[1]} features")
        
        # Calculate mutual information
        mi_values = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({'Feature': X.columns, 'MI': mi_values})
        mi_df = mi_df.sort_values('MI', ascending=False)
        
        # Save mutual information
        mi_df.to_csv(f"feature_selection/mutual_info_split{i+1}.csv", index=False)
        
        # Keep only features above the MI percentile threshold
        mi_threshold = np.percentile(mi_values, 100 - mi_percentile)
        significant_features = mi_df[mi_df['MI'] > mi_threshold]['Feature'].tolist()
        
        print(f"Split {i+1}: After MI thresholding: {len(significant_features)} features")
        
        # Filter X to only include significant features
        X_significant = X[significant_features]
        
        # Calculate correlation matrix for significant features
        corr_matrix = X_significant.corr().abs()
        
        # Create DataFrame of highly correlated features
        high_corr_pairs = []
        for j in range(len(corr_matrix.columns)):
            for k in range(j+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[j, k] >= correlation_threshold:
                    col1, col2 = corr_matrix.columns[j], corr_matrix.columns[k]
                    mi1 = mi_df[mi_df['Feature'] == col1]['MI'].values[0]
                    mi2 = mi_df[mi_df['Feature'] == col2]['MI'].values[0]
                    high_corr_pairs.append({
                        'Feature1': col1,
                        'Feature2': col2,
                        'Correlation': corr_matrix.iloc[j, k],
                        'MI1': mi1,
                        'MI2': mi2
                    })
        
        corr_df = pd.DataFrame(high_corr_pairs)
        corr_df.to_csv(f"feature_selection/correlations_split{i+1}.csv", index=False)
        
        # Enhanced correlation removal with recursive approach
        if len(corr_df) > 0:
            # Sort by correlation strength (highest first)
            corr_df = corr_df.sort_values('Correlation', ascending=False)
            
            # Initialize feature sets
            features_to_keep = set(significant_features)
            
            # Process pairs in order of correlation strength
            for _, row in corr_df.iterrows():
                f1, f2 = row['Feature1'], row['Feature2']
                # Skip if either feature already removed
                if f1 not in features_to_keep or f2 not in features_to_keep:
                    continue
                
                # Remove feature with lower MI
                if row['MI1'] >= row['MI2']:
                    features_to_keep.remove(f2)
                else:
                    features_to_keep.remove(f1)
            
            # Convert back to list
            features_to_keep = list(features_to_keep)
        else:
            # Keep all significant features if no correlations found
            features_to_keep = significant_features
        
        print(f"Split {i+1}: After correlation removal: {len(features_to_keep)} features")
        
        # Add metadata
        selected_features = meta_cols + features_to_keep
        
        # Filter datasets
        filtered_train = train_data[selected_features]
        filtered_test = test_data[selected_features]  # Also filter test data
        
        # Save filtered datasets
        create_directory(f"feature_selection/split_{i+1}")
        filtered_train.to_csv(f"feature_selection/split_{i+1}/train_filtered.csv", index=False)
        filtered_test.to_csv(f"feature_selection/split_{i+1}/test_filtered.csv", index=False)
        
        # Save list of selected features
        pd.DataFrame({'Selected_Features': features_to_keep}).to_csv(
            f"feature_selection/split_{i+1}/selected_features.csv", index=False
        )
        
        # Track information
        split_info = pd.DataFrame({
            'Split': i+1,
            'Initial_Features': len(X.columns),
            'Features_After_MI': len(significant_features),
            'Features_After_Correlation': len(features_to_keep),
            'Total_Features_Removed': len(X.columns) - len(features_to_keep),
            'Percent_Removed': 100 * (len(X.columns) - len(features_to_keep)) / len(X.columns)
        }, index=[0])
        
        all_info = pd.concat([all_info, split_info], ignore_index=True)
        
        # Update splits
        splits[i]['train'] = filtered_train
        splits[i]['test'] = filtered_test
        splits[i]['selected_features'] = selected_features
    
    # Save summary
    all_info.to_csv("feature_selection/feature_selection_summary.csv", index=False)
    
    print(f"Completed enhanced feature selection")
    return splits

# Step 6: Robust Boruta feature selection
def boruta_feature_selection(splits, n_estimators=250, max_iter=50, perc=70):
    """Apply Boruta feature selection to the splits."""
    # Create output directory
    create_directory("boruta_results")
    
    # Process each split sequentially and save progress
    results = []
    
    for i, split_data in enumerate(splits):
        # Extract train and test data
        train_data = split_data['train']
        test_data = split_data['test']
        
        # Identify metadata columns
        meta_cols = ['Label', 'Class'] if 'Label' in train_data.columns else ['Class']
        
        # Separate features and target
        y_train = train_data['Class']
        X_train = train_data.drop(columns=meta_cols)
        
        y_test = test_data['Class']
        X_test = test_data.drop(columns=meta_cols)
        
        # Output filepath for this split
        split_path = f"boruta_results/boruta_split_{i}.joblib"
        
        # Check if split was already processed
        if os.path.exists(split_path):
            print(f"Loading pre-processed split {i}")
            result = joblib.load(split_path)
            results.append(result)
            continue
        
        print(f"Processing split {i}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        # Initialize Boruta
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        boruta = BorutaPy(rf, n_estimators=n_estimators, max_iter=max_iter, 
                          perc=perc, random_state=42, verbose=0)
        
        # Fit Boruta
        boruta.fit(X_train.values, y_train.values)
        
        # Get selected features
        selected_features = X_train.columns[boruta.support_].tolist()
        print(f"Split {i}: Selected {len(selected_features)} features")
        
        # Filter datasets to include only selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Re-create train and test DataFrames with selected features and target
        train_selected = pd.concat([X_train_selected, y_train], axis=1)
        test_selected = pd.concat([X_test_selected, y_test], axis=1)
        
        # Create result dictionary
        result = {
            'split_id': i,
            'selected_features': selected_features,
            'boruta': boruta,
            'train': train_selected,
            'test': test_selected,
            'X_train': X_train_selected,
            'y_train': y_train,
            'X_test': X_test_selected,
            'y_test': y_test,
            'original_X_train_columns': X_train.columns.tolist(),  # Save original columns for later analysis
            'fold': split_data.get('fold', i+1)
        }
        
        # Save the split result
        joblib.dump(result, split_path)
        print(f"Saved split {i}")
        
        # Add to results list
        results.append(result)
        
    # Save all results
    all_results_path = "boruta_results/final_boruta_splits.joblib"  # Changed from all_boruta_results.joblib
    joblib.dump(results, all_results_path)
    print(f"Saved all results to {all_results_path}")
    
    return results

# Function to export Boruta results as CSV files
def export_boruta_csv():
    # Create a directory for CSV exports
    create_directory("boruta_results/csv_files")
    
    # Check if the file exists
    boruta_results_path = "boruta_results/final_boruta_splits.joblib"
    
    if os.path.exists(boruta_results_path):
        # Load all splits
        boruta_splits = joblib.load(boruta_results_path)
        print(f"Loaded {len(boruta_splits)} Boruta feature-selected splits")
        
        # Save each split as CSV
        for i, split in enumerate(boruta_splits):
            # Create a directory for this split
            split_dir = f"boruta_results/csv_files/split_{i}"
            create_directory(split_dir)
            
            # Save the train and test DataFrames
            split['train'].to_csv(f"{split_dir}/train.csv", index=False)
            split['test'].to_csv(f"{split_dir}/test.csv", index=False)
            
            # Save the selected features as a CSV
            pd.DataFrame({'selected_features': split['selected_features']}).to_csv(
                f"{split_dir}/selected_features.csv", index=False
            )
            
            # Save feature importance information if available
            if hasattr(split['boruta'], 'ranking_'):
                try:
                    # Get original feature names
                    original_features = split.get('original_X_train_columns', split['X_train'].columns)
                    
                    # Check if arrays are the same length
                    if len(original_features) == len(split['boruta'].ranking_):
                        # Create a DataFrame with feature names and their rankings
                        feature_importance = pd.DataFrame({
                            'feature': original_features,
                            'ranking': split['boruta'].ranking_,
                            'support': split['boruta'].support_,
                            'support_weak': split['boruta'].support_weak_
                        })
                        feature_importance.to_csv(f"{split_dir}/feature_importance.csv", index=False)
                    else:
                        print(f"Warning: Skipping feature importance for split {i} due to length mismatch")
                except Exception as e:
                    print(f"Error saving feature importance for split {i}: {str(e)}")
            
            print(f"Exported CSV files for split {i}")
        
        # Create a summary file with information about each split
        summary_data = []
        for i, split in enumerate(boruta_splits):
            summary_data.append({
                'split': i,
                'selected_features': len(split['selected_features']),
                'train_samples': split['train'].shape[0],
                'test_samples': split['test'].shape[0]
            })
        
        # Create and save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("boruta_results/csv_files/summary.csv", index=False)
        print("Exported summary information")
    else:
        print(f"Error: Could not find Boruta results at {boruta_results_path}")

# Function to extract feature importance for analysis
def extract_feature_importance(split_index=0):
    boruta_results_path = "boruta_results/final_boruta_splits.joblib"
    
    if os.path.exists(boruta_results_path):
        # Load the splits
        boruta_splits = joblib.load(boruta_results_path)
        
        # Get the requested split
        if split_index < len(boruta_splits):
            split = boruta_splits[split_index]
            
            # Extract the Boruta object
            boruta = split['boruta']
            
            # Get original feature names
            if 'original_X_train_columns' in split:
                original_features = split['original_X_train_columns']
            else:
                original_features = split['X_train'].columns.tolist()
            
            # Create DataFrame with feature importance info
            feature_importance = pd.DataFrame({
                'feature': original_features,
                'ranking': boruta.ranking_,
                'support': boruta.support_,
                'support_weak': boruta.support_weak_
            })
            
            # Sort by importance (lower ranking = more important)
            feature_importance = feature_importance.sort_values('ranking')
            
            # Save to CSV
            output_path = f"boruta_results/all_features_importance_split_{split_index}.csv"
            feature_importance.to_csv(output_path, index=False)
            print(f"Saved complete feature importance to {output_path}")
        else:
            print(f"Error: Split index {split_index} is out of range")
    else:
        print(f"Error: Could not find Boruta results at {boruta_results_path}")

# Function to create comprehensive feature tracking summary
def create_feature_tracking_summary(original_data):
    """Create a summary CSV tracking feature counts through all processing steps."""
    create_directory("summary")
    
    # Collect information from each processing step
    stages = []
    
    # 1. Initial data
    initial_features = len(original_data.columns) - 1  # Subtract 1 for Class
    stages.append({
        'Stage': '1. Initial Data',
        'Features': initial_features,
        'Features Removed': 0,
        'Features Retained (%)': 100.0,
        'Description': 'Raw feature set before any processing'
    })
    
    # 2. Missing value removal
    if os.path.exists("missing_value_analysis/missing_values_summary.csv"):
        missing_summary = pd.read_csv("missing_value_analysis/missing_values_summary.csv")
        features_after_missing = missing_summary[missing_summary['Metric'] == 'Remaining Features']['Count'].values[0] - 1
        removed_in_step = initial_features - features_after_missing
        retention_rate = (features_after_missing / initial_features) * 100
        
        stages.append({
            'Stage': '2. Missing Value Removal',
            'Features': features_after_missing,
            'Features Removed': removed_in_step,
            'Features Retained (%)': retention_rate,
            'Description': f'Removed features with >{0.05*100}% missing values'
        })
    
    # 3. Zero variance/IQR removal
    if os.path.exists("features_processing_zero_variance_iqr/features_processing_zero_variance_iqr_summary.csv"):
        zero_var_summary = pd.read_csv("features_processing_zero_variance_iqr/features_processing_zero_variance_iqr_summary.csv")
        features_after_zero_var = zero_var_summary[zero_var_summary['Summary'] == 'Remaining features after processing']['Count'].values[0]
        prev_features = stages[-1]['Features']
        removed_in_step = prev_features - features_after_zero_var
        retention_rate = (features_after_zero_var / initial_features) * 100
        
        stages.append({
            'Stage': '3. Zero Variance & IQR Removal',
            'Features': features_after_zero_var,
            'Features Removed': removed_in_step,
            'Features Retained (%)': retention_rate,
            'Description': 'Removed features with zero/near-zero variance or IQR'
        })
    
    # 4. Feature selection
    if os.path.exists("feature_selection/feature_selection_summary.csv"):
        selection_summary = pd.read_csv("feature_selection/feature_selection_summary.csv")
        # Calculate average number of features across all splits
        avg_features_after_selection = selection_summary['Features_After_Correlation'].mean()
        prev_features = stages[-1]['Features'] if stages else initial_features
        removed_in_step = prev_features - avg_features_after_selection
        retention_rate = (avg_features_after_selection / initial_features) * 100
        
        stages.append({
            'Stage': '4. Mutual Info & Correlation Selection',
            'Features': avg_features_after_selection,
            'Features Removed': removed_in_step,
            'Features Retained (%)': retention_rate,
            'Description': 'Selected features based on MI and removed highly correlated features'
        })
    
    # 5. Boruta
    if os.path.exists("boruta_results/csv_files/summary.csv"):
        boruta_summary = pd.read_csv("boruta_results/csv_files/summary.csv")
        avg_features_after_boruta = boruta_summary['selected_features'].mean()
        prev_features = stages[-1]['Features'] if stages else initial_features
        removed_in_step = prev_features - avg_features_after_boruta
        retention_rate = (avg_features_after_boruta / initial_features) * 100
        
        stages.append({
            'Stage': '5. Boruta Feature Selection',
            'Features': avg_features_after_boruta,
            'Features Removed': removed_in_step,
            'Features Retained (%)': retention_rate,
            'Description': 'Selected features using Boruta algorithm'
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(stages)
    
    # Save summary
    summary_df.to_csv("summary/feature_tracking_summary.csv", index=False)
    
    # Also create a more detailed per-fold summary if possible
    try:
        fold_summaries = []
        
        # Get number of folds from Boruta results
        if os.path.exists("boruta_results/csv_files/summary.csv"):
            boruta_summary = pd.read_csv("boruta_results/csv_files/summary.csv")
            num_folds = len(boruta_summary)
            
            for fold in range(num_folds):
                # Get feature counts for each processing stage for this fold
                # Feature selection
                if os.path.exists(f"feature_selection/split_{fold+1}/selected_features.csv"):
                    fs_features = len(pd.read_csv(f"feature_selection/split_{fold+1}/selected_features.csv"))
                else:
                    fs_features = None
                
                # Boruta
                if os.path.exists(f"boruta_results/csv_files/split_{fold}/selected_features.csv"):
                    boruta_features = len(pd.read_csv(f"boruta_results/csv_files/split_{fold}/selected_features.csv"))
                else:
                    boruta_features = None
                
                fold_summaries.append({
                    'Fold': fold+1,
                    'Initial Features': initial_features,
                    'After Missing Value Removal': stages[1]['Features'] if len(stages) > 1 else None,
                    'After Zero Var/IQR Removal': stages[2]['Features'] if len(stages) > 2 else None,
                    'After Feature Selection': fs_features,
                    'After Boruta': boruta_features,
                    'Final Feature Retention (%)': (boruta_features / initial_features * 100) if boruta_features else None
                })
        
        # Create and save fold-specific summary
        if fold_summaries:
            fold_df = pd.DataFrame(fold_summaries)
            fold_df.to_csv("summary/per_fold_feature_summary.csv", index=False)
            print("Created per-fold feature tracking summary")
    except Exception as e:
        print(f"Error creating per-fold summary: {e}")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 6))
        
        # Create bar chart of features at each stage
        plt.bar(
            summary_df['Stage'], 
            summary_df['Features'],
            color=sns.color_palette("viridis", len(summary_df))
        )
        
        # Add text labels on top of each bar
        for i, v in enumerate(summary_df['Features']):
            plt.text(
                i, 
                v + max(summary_df['Features'])*0.02,  # Position text slightly above bar
                f"{v:.0f}\n({summary_df['Features Retained (%)'][i]:.1f}%)",
                ha='center',
                fontweight='bold'
            )
        
        # Formatting
        plt.title('Feature Count Reduction Through Processing Pipeline', fontsize=16)
        plt.ylabel('Number of Features', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig("summary/feature_reduction_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Created feature reduction visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    print("Created feature tracking summary")
    return summary_df

def validate_dataset_structure():
    """
    Validates that all specified folders and files exist before processing.
    Returns a tuple (is_valid, missing_items) where missing_items is a list of missing paths.
    """
    print("Validating dataset structure...")
    missing_items = []
    
    for dataset in DATASETS:
        dataset_name = dataset["name"]
        dataset_path = os.path.join(BASE_DIR, dataset_name)
        
        # Check if dataset directory exists
        if not os.path.exists(dataset_path):
            missing_items.append(dataset_path)
            continue
            
        # Check each folder in the dataset
        for folder, file in zip(dataset["folders"], dataset["files"]):
            folder_path = os.path.join(dataset_path, folder)
            
            # Check if folder exists
            if not os.path.exists(folder_path):
                missing_items.append(folder_path)
                continue
                
            # Check if the file exists in the folder
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                missing_items.append(file_path)
    
    is_valid = len(missing_items) == 0
    
    if is_valid:
        print("✓ All folders and files exist!")
    else:
        print(f"✗ Found {len(missing_items)} missing items:")
        for item in missing_items:
            print(f"  - {item}")
    
    return is_valid, missing_items


# Main execution function that processes all datasets
def process_all_datasets():
    """Process all datasets in the specified folders"""
    print(f"Starting preprocessing pipeline for all datasets in {BASE_DIR}")
    
    # Validate that all required folders and files exist
    is_valid, missing_items = validate_dataset_structure()
    
    if not is_valid:
        print("\nERROR: Cannot proceed with processing due to missing folders or files.")
        user_input = input("Do you want to continue anyway? (y/n): ").lower()
        if user_input != 'y':
            print("Aborting processing.")
            return
        print("Continuing despite missing items...")
    
    # Save original working directory
    original_dir = os.getcwd()
    
    # Process each dataset
    for dataset in DATASETS:
        dataset_name = dataset["name"]
        dataset_path = os.path.join(BASE_DIR, dataset_name)
        
        # Skip if dataset directory doesn't exist
        if not os.path.exists(dataset_path):
            print(f"\n\n======== SKIPPING MISSING DATASET: {dataset_name} ========\n")
            continue
            
        print(f"\n\n======== PROCESSING DATASET: {dataset_name} ========\n")
        
        # Process each folder in the dataset
        for folder, file in zip(dataset["folders"], dataset["files"]):
            folder_path = os.path.join(dataset_path, folder)
            
            # Skip if folder doesn't exist
            if not os.path.exists(folder_path):
                print(f"\n---- SKIPPING MISSING FOLDER: {folder} ----")
                continue
                
            file_path = os.path.join(folder_path, file)
            
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                print(f"\n---- SKIPPING MISSING FILE: {file} ----")
                continue
                
            output_dir = os.path.join(folder_path, "Preprocessing_Results")
            
            print(f"\n---- Processing folder: {folder} ----")
            
            # Run the preprocessing pipeline for this folder/file
            try:
                run_preprocessing_pipeline(folder_path, output_dir, file)
                print(f"Successfully processed {folder}/{file}")
            except Exception as e:
                print(f"Error processing {folder}/{file}: {e}")
                import traceback
                traceback.print_exc()
    
    # Return to original directory
    os.chdir(original_dir)
    print("\nAll datasets processed successfully!")

# Execute the main function if the script is run directly
if __name__ == "__main__":
    process_all_datasets()
        