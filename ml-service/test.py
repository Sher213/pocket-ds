#!/usr/bin/env python
"""
Test script for Pocket Data Scientist functionality.
This script demonstrates the complete workflow from data ingestion to model training and evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging
from data_processor import DataProcessor
from auto_modeler import AutoModeler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_script")

def test_data_processing(dataset_path):
    """Test the data processing functionality."""
    logger.info("Testing data processing functionality")
    
    # Initialize DataProcessor
    processor = DataProcessor(dataset_path, target_column="Average gross")
    
    # Display initial dataset info
    logger.info(f"Initial dataset shape: {processor.df.shape}")
    logger.info(f"Initial column types: {processor.detect_column_types()}")
    
    # Test column type detection
    column_types = processor.detect_column_types()
    logger.info(f"Detected column types: {column_types}")
    
    # Test missing value handling
    missing_counts = processor.handle_missing_values(strategy="auto")
    logger.info(f"Missing values handled: {missing_counts}")
    
    # Test outlier detection
    outlier_counts = processor.detect_outliers(method="zscore", threshold=3.0)
    logger.info(f"Detected outliers: {outlier_counts}")
    
    # Test outlier removal
    removed_rows = processor.remove_outliers(method="zscore", threshold=3.0)
    logger.info(f"Removed {removed_rows} rows with outliers")
    
    # Test categorical encoding
    encoded_columns = processor.encode_categorical_variables(categorical_columns=column_types, method="auto", threshold=10)
    logger.info(f"Encoded categorical columns: {encoded_columns}")
    
    # Test datetime feature extraction
    datetime_features = processor.extract_datetime_features()
    logger.info(f"Extracted datetime features: {datetime_features}")
    
    # Test column name standardization
    standardized_columns = processor.standardize_column_names()
    logger.info(f"Standardized column names: {standardized_columns}")
    
    # Test duplicate removal
    removed_duplicates = processor.remove_duplicates()
    logger.info(f"Removed {removed_duplicates} duplicate rows")
    
    # Test summary statistics generation
    summary_stats = processor.generate_summary_statistics()
    logger.info("Generated summary statistics")
    
    # Test correlation matrix generation
    correlation_matrix = processor.generate_correlation_matrix()
    logger.info("Generated correlation matrix")
    
    # Test visualization generation
    visualization_paths = processor.generate_visualizations()
    logger.info(f"Generated visualizations: {visualization_paths}")
    
    # Test EDA report generation
    report_path = processor.generate_eda_report()
    logger.info(f"Generated EDA report: {report_path}")
    
    # Test data preparation for modeling
    X_train, y_train, X_test, y_test = processor.prepare_for_modeling(
        target_column=processor.target_column,
        test_size=0.2,
        random_state=42
    )
    logger.info(f"Prepared data for modeling: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Test saving cleaned dataset
    cleaned_path = processor.save_cleaned_dataset()
    logger.info(f"Saved cleaned dataset: {cleaned_path}")
    
    # Test saving analysis results
    analysis_path = processor.save_analysis_results()
    logger.info(f"Saved analysis results: {analysis_path}")
    
    return processor, X_train, y_train, X_test, y_test

def test_auto_modeling(X_train, y_train, X_test, y_test):
    """Test the automated modeling functionality."""
    logger.info("Testing automated modeling functionality")
    
    # Initialize AutoModeler
    modeler = AutoModeler(X_train, y_train, X_test, y_test)
    
    # Display problem type detection
    logger.info(f"Detected problem type: {modeler.problem_type}")
    
    # Test model initialization
    logger.info(f"Initialized models: {list(modeler.models.keys())}")
    
    # Test data preprocessing
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = modeler.preprocess_data()
    logger.info(f"Preprocessed data: X_train_scaled shape: {X_train_scaled.shape}")
    
    # Test model training and evaluation
    model_results = modeler.train_and_evaluate_models(cv_folds=5)
    logger.info(f"Model evaluation results: {model_results}")
    
    # Test best model selection
    best_model, best_model_name, best_score = modeler.select_best_model()
    logger.info(f"Selected best model: {best_model_name} with score: {best_score}")
    
    # Test hyperparameter tuning
    tuning_results = modeler.tune_hyperparameters(n_iter=10, cv=3)
    logger.info(f"Hyperparameter tuning results: {tuning_results}")
    
    # Test model report generation
    report_path = modeler.generate_model_report()
    logger.info(f"Generated model report: {report_path}")
    
    # Test model saving
    model_path = modeler.save_model()
    logger.info(f"Saved model: {model_path}")
    
    # Test visualization generation
    visualization_paths = modeler.generate_visualizations()
    logger.info(f"Generated model visualizations: {visualization_paths}")
    
    # Test comprehensive auto modeling
    auto_results = modeler.auto_model(tune_hyperparameters=True)
    logger.info(f"Auto modeling completed: {auto_results}")
    
    return modeler, auto_results

def main():
    """Main function to run the test script."""
    logger.info("Starting Pocket Data Scientist test script")
    
    # Create sample dataset
    dataset_path = 'datasets/my_file.csv'
    
    # Test data processing
    processor, X_train, X_test, y_train, y_test = test_data_processing(dataset_path)
    
    # Test auto modeling
    modeler, auto_results = test_auto_modeling(X_train, y_train, X_test, y_test)
    
    logger.info("Test script completed successfully")
    print("\n" + "="*50)
    print("Pocket Data Scientist Test Results")
    print("="*50)
    print(f"Sample dataset created: {dataset_path}")
    print(f"EDA report generated: {processor.generate_eda_report()}")
    print(f"Best model: {modeler.best_model_name} with score: {modeler.best_score}")
    print(f"Model report: {modeler.generate_model_report()}")
    print("="*50)
    print("Check the ml-service/reports directory for generated reports and visualizations")
    print("Check the ml-service/logs directory for detailed logs")
    print("="*50)

if __name__ == "__main__":
    main() 