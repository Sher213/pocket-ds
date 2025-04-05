import pandas as pd
import numpy as np
import re
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

def clean_text(text):
    return text.encode('latin-1', errors='ignore').decode('latin-1')

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_processor")

class DatasetProcessor:
    """
    A comprehensive class for data ingestion, cleaning, and analysis.
    """
    
    def __init__(self, dataset_path: str, target_column: str):
        """
        Initialize the DataProcessor with a dataset path.
        
        Args:
            dataset_path: Path to the dataset file
        """
        self.dataset_path = dataset_path
        self.df = None
        self.original_df = None
        self.cleaning_log = []
        self.analysis_results = {}
        self.backup_path = f"{dataset_path}.backup"
        self.target_column = target_column
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        # Load the dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset based on file extension."""
        try:
            file_ext = os.path.splitext(self.dataset_path)[1].lower()
            
            if file_ext == '.csv':
                self.df = pd.read_csv(self.dataset_path)
            elif file_ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.dataset_path)
            elif file_ext == '.json':
                self.df = pd.read_json(self.dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Create a backup of the original dataset
            self.original_df = self.df.copy()
            self.original_df.to_csv(self.backup_path, index=False)
            
            logger.info(f"Dataset loaded successfully: {self.dataset_path}")
            logger.info(f"Dataset shape: {self.df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def detect_column_types(self) -> Dict[str, str]:
        """
        Automatically detect column types (numeric, categorical, datetime, text).

        Returns:
            Dictionary mapping column names to their detected types
        """
        column_types = {}

        # Improved regex for numeric detection
        numeric_regex = re.compile(
            r"""^[\'"]?           # optional opening quote
                [\$\€\£\¥\₹\₽]?    # optional currency symbol
                -?                 # optional negative sign
                (\d{1,3}(,\d{3})*|\d+)?  # integer part with optional comma separators or just digits
                (\.\d+)?           # optional decimal part (dot separated)
                (%|[\'"]?$)?       # optional percentage sign or optional closing quote
                $""",  # Added end of string anchor
            re.VERBOSE,
        )

        for column in self.df.columns:
            # Handle datetime conversion
            try:
                # Try converting the entire column to datetime (excluding errors)
                pd.to_datetime(self.df[column], errors='raise')
                column_types[column] = "datetime"
            except Exception as e:
                # Handle specific error if necessary
                pass

            # Check if numeric dtype
            if pd.api.types.is_numeric_dtype(self.df[column]):
                column_types[column] = "numeric"
                continue

            # Check if it looks like a numeric string (by sampling)
            try:
                sample = self.df[column].dropna().sample(min(10, len(self.df)), random_state=42).astype(str)
                if all(numeric_regex.fullmatch(x) for x in sample):
                    column_types[column] = "numeric"
            except Exception as e:
                # Handle sample error
                pass

            # Determine if categorical or text based on unique value count
            unique_count = self.df[column].nunique()
            if unique_count < min(50, len(self.df) * 0.5):
                column_types[column] = "categorical"
                continue
            else:
                column_types[column] = "text"

        self.analysis_results["column_types"] = column_types
        print(f"Detected column types: {column_types}")
        return column_types
    
    def handle_missing_values(self, strategy: str = "auto") -> Dict[str, int]:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: Strategy for handling missing values ("auto", "drop", "impute")
            
        Returns:
            Dictionary with counts of missing values handled per column
        """
        missing_counts = self.df.isnull().sum().to_dict()
        total_missing = sum(missing_counts.values())
        
        if total_missing == 0:
            logger.info("No missing values found in the dataset")
            return missing_counts
        
        logger.info(f"Found {total_missing} missing values across {sum(1 for v in missing_counts.values() if v > 0)} columns")
        
        if strategy == "drop":
            # Drop rows with any missing values
            initial_rows = len(self.df)
            self.df.dropna(inplace=True)
            dropped_rows = initial_rows - len(self.df)
            logger.info(f"Dropped {dropped_rows} rows with missing values")
            self.cleaning_log.append(f"Dropped {dropped_rows} rows with missing values")
            
        elif strategy == "impute" or strategy == "auto":
            # Impute missing values based on column type
            column_types = self.detect_column_types()
            
            for column, missing_count in missing_counts.items():
                if missing_count > 0:
                    col_type = column_types.get(column, "text")
                    
                    if col_type == "numeric":
                        # Use median for numeric columns
                        median_val = self.df[column].median()
                        self.df[column].fillna(median_val, inplace=True)
                        logger.info(f"Filled {missing_count} missing values in {column} with median: {median_val}")
                        self.cleaning_log.append(f"Filled {missing_count} missing values in {column} with median: {median_val}")
                    
                    elif col_type == "categorical":
                        # Use mode for categorical columns
                        mode_val = self.df[column].mode()[0]
                        self.df[column].fillna(mode_val, inplace=True)
                        logger.info(f"Filled {missing_count} missing values in {column} with mode: {mode_val}")
                        self.cleaning_log.append(f"Filled {missing_count} missing values in {column} with mode: {mode_val}")
                    
                    else:
                        # For text or datetime, use a placeholder
                        placeholder = "MISSING" if col_type == "text" else pd.NaT
                        self.df[column].fillna(placeholder, inplace=True)
                        logger.info(f"Filled {missing_count} missing values in {column} with placeholder")
                        self.cleaning_log.append(f"Filled {missing_count} missing values in {column} with placeholder")
        
        return missing_counts
    
    def detect_outliers(self, method: str = "zscore", threshold: float = 3.0) -> Dict[str, int]:
        """
        Detect outliers in numeric columns.
        
        Args:
            method: Method for outlier detection ("zscore" or "iqr")
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with counts of outliers detected per column
        """
        outlier_counts = {}
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == "zscore":
                # Z-score method
                z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
                outliers = (z_scores > threshold).sum()
            
            elif method == "iqr":
                # IQR method
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[column] < (Q1 - 1.5 * IQR)) | (self.df[column] > (Q3 + 1.5 * IQR))).sum()
            
            outlier_counts[column] = int(outliers)
            
            if outliers > 0:
                logger.info(f"Detected {outliers} outliers in {column} using {method} method")
                self.cleaning_log.append(f"Detected {outliers} outliers in {column} using {method} method")
        
        self.analysis_results["outliers"] = outlier_counts
        return outlier_counts
    
    def remove_outliers(self, method: str = "zscore", threshold: float = 3.0) -> int:
        """
        Remove outliers from numeric columns.
        
        Args:
            method: Method for outlier detection ("zscore" or "iqr")
            threshold: Threshold for outlier detection
            
        Returns:
            Number of rows removed
        """
        initial_rows = len(self.df)
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == "zscore":
                # Z-score method
                z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
                self.df = self.df[z_scores <= threshold]
            
            elif method == "iqr":
                # IQR method
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                self.df = self.df[(self.df[column] >= (Q1 - 1.5 * IQR)) & (self.df[column] <= (Q3 + 1.5 * IQR))]
        
        removed_rows = initial_rows - len(self.df)
        logger.info(f"Removed {removed_rows} rows with outliers using {method} method")
        self.cleaning_log.append(f"Removed {removed_rows} rows with outliers using {method} method")
        
        return removed_rows
    
    def format_num_vars(self, column_types: dict) -> None:
        for column, col_type in column_types.items():
            if col_type == "numeric":
                # Convert to numeric type
                self.df[column] = self.df[column].replace(r'[^\d\.\-]', '', regex=True)
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')

    def encode_categorical_variables(self, categorical_columns: dict, method: str = "auto", threshold: int = 10) -> Dict[str, str]:
        """
        Encode specified categorical variables based on their type.

        Args:
            categorical_columns: Dictionary where keys are column names and values are their types.
            method: Encoding method ("label", "onehot", or "auto")
            threshold: Threshold for one-hot encoding (number of unique values)

        Returns:
            Dictionary mapping original column names to their encoded versions
        """
        encoded_columns = {}

        # Filter only 'categorical' or 'text' columns
        valid_columns = [col for col, col_type in categorical_columns.items() if col_type in ["categorical", "text"]]

        for column in valid_columns:
            if column not in self.df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame. Skipping.")
                continue

            unique_count = self.df[column].nunique()

            if method == "label" or (method == "auto" and unique_count > threshold):
                # Label encoding
                le = LabelEncoder()
                self.df[f"{column}_encoded"] = le.fit_transform(self.df[column].astype(str))
                encoded_columns[column] = f"{column}_encoded"
                logger.info(f"Applied label encoding to {column}")
                self.cleaning_log.append(f"Applied label encoding to {column}")

            elif method == "onehot" or (method == "auto" and unique_count <= threshold):
                # One-hot encoding
                dummies = pd.get_dummies(self.df[column], prefix=column)
                self.df = pd.concat([self.df, dummies], axis=1)
                encoded_columns[column] = f"{column}_onehot"
                logger.info(f"Applied one-hot encoding to {column}")
                self.cleaning_log.append(f"Applied one-hot encoding to {column}")

            # Drop the original column
            self.df.drop(columns=[column], inplace=True)

        self.analysis_results["encoded_columns"] = encoded_columns
        return encoded_columns
    
    def extract_datetime_features(self, columns: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Extract features from datetime columns.
        
        Args:
            columns: List of datetime columns to process (if None, all datetime columns are processed)
            
        Returns:
            Dictionary mapping original column names to their extracted feature columns
        """
        extracted_features = {}
        
        if columns is None:
            # Try to identify datetime columns
            for column in self.df.columns:
                try:
                    pd.to_datetime(self.df[column])
                    columns = [column]

                    
                except:
                    pass
        
        if not columns:
            logger.info("No datetime columns found for feature extraction")
            return extracted_features
        
        for column in columns:
            try:
                # Convert to datetime
                self.df[column] = pd.to_datetime(self.df[column])
                
                # Extract features
                self.df[f"{column}_year"] = self.df[column].dt.year
                self.df[f"{column}_month"] = self.df[column].dt.month
                self.df[f"{column}_day"] = self.df[column].dt.day
                self.df[f"{column}_dayofweek"] = self.df[column].dt.dayofweek
                self.df[f"{column}_quarter"] = self.df[column].dt.quarter
                
                extracted_features[column] = [
                    f"{column}_year", f"{column}_month", f"{column}_day",
                    f"{column}_dayofweek", f"{column}_quarter"
                ]
                
                logger.info(f"Extracted datetime features from {column}")
                self.cleaning_log.append(f"Extracted datetime features from {column}")
                
            except Exception as e:
                logger.error(f"Error extracting datetime features from {column}: {str(e)}")
        
        self.analysis_results["datetime_features"] = extracted_features
        return extracted_features
    
    def standardize_column_names(self) -> Dict[str, str]:
        """
        Standardize column names (lowercase, replace spaces with underscores).
        
        Returns:
            Dictionary mapping original column names to standardized names
        """
        original_columns = self.df.columns.tolist()
        new_columns = {col: col.lower().replace(' ', '_').replace('-', '_') for col in original_columns}
        
        # Rename columns
        self.df.rename(columns=new_columns, inplace=True)
        
        logger.info("Standardized column names")
        self.cleaning_log.append("Standardized column names")
        
        self.target_column = self.target_column.lower().replace(' ', '_').replace('-', '_')

        return new_columns
    
    def remove_duplicates(self) -> int:
        """
        Remove duplicate rows.
        
        Returns:
            Number of duplicate rows removed
        """
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed_rows = initial_rows - len(self.df)
        
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows")
            self.cleaning_log.append(f"Removed {removed_rows} duplicate rows")
        
        return removed_rows
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics for the dataset.
        
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric summary
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            summary["numeric_summary"][column] = {
                "mean": float(self.df[column].mean()),
                "median": float(self.df[column].median()),
                "std": float(self.df[column].std()),
                "min": float(self.df[column].min()),
                "max": float(self.df[column].max()),
                "skew": float(self.df[column].skew()),
                "kurtosis": float(self.df[column].kurtosis())
            }
        
        # Categorical summary
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            value_counts = self.df[column].value_counts().to_dict()
            summary["categorical_summary"][column] = {
                "unique_values": len(value_counts),
                "top_values": dict(list(value_counts.items())[:5])
            }
        
        self.analysis_results["summary_statistics"] = summary
        return summary
    
    def clean_text_columns(self) -> None:
        """
        Clean text columns by removing non-ASCII characters.
        """
        text_columns = self.df.select_dtypes(include=['object', 'category']).columns
        
        for column in text_columns:
            self.df[column] = self.df[column].apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)
        
        logger.info("Cleaned text columns")
        self.cleaning_log.append("Cleaned text columns")

    def generate_correlation_matrix(self) -> pd.DataFrame:
        """
        Generate correlation matrix for numeric columns.
        
        Returns:
            Correlation matrix as a DataFrame
        """
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            logger.warning("Not enough numeric columns to generate correlation matrix")
            return pd.DataFrame()
        
        correlation_matrix = self.df[numeric_columns].corr()
        self.analysis_results["correlation_matrix"] = correlation_matrix.to_dict()
        
        return correlation_matrix
    
    def generate_visualizations(self, output_dir: str = "reports/visualizations") -> Dict[str, str]:
        """
        Generate visualizations for the dataset.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        visualization_paths = {}
        
        # Numeric distributions
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            # Histograms
            fig = make_subplots(rows=len(numeric_columns), cols=1, subplot_titles=numeric_columns)
            for i, column in enumerate(numeric_columns, 1):
                fig.add_trace(go.Histogram(x=self.df[column], name=column), row=i, col=1)
            fig.update_layout(height=300*len(numeric_columns), width=800, title_text="Numeric Distributions")
            hist_path = f"{output_dir}/numeric_distributions.html"
            fig.write_html(hist_path)
            visualization_paths["numeric_distributions"] = hist_path
            
            # Correlation heatmap
            if len(numeric_columns) > 1:
                correlation_matrix = self.df[numeric_columns].corr()
                fig = go.Figure(data=go.Heatmap(z=correlation_matrix, x=correlation_matrix.columns, y=correlation_matrix.columns))
                fig.update_layout(title="Correlation Heatmap")
                heatmap_path = f"{output_dir}/correlation_heatmap.html"
                fig.write_html(heatmap_path)
                visualization_paths["correlation_heatmap"] = heatmap_path
        
        # Categorical distributions
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            # Bar charts
            for column in categorical_columns:
                value_counts = self.df[column].value_counts().head(10)
                fig = go.Figure(data=go.Bar(x=value_counts.index, y=value_counts.values))
                fig.update_layout(title=f"Top 10 Values in {column}", xaxis_title=column, yaxis_title="Count")
                bar_path = f"{output_dir}/{column}_bar_chart.html"
                fig.write_html(bar_path)
                visualization_paths[f"{column}_bar_chart"] = bar_path
        
        # Scatter plots for numeric columns
        if len(numeric_columns) >= 2:
            for i in range(len(numeric_columns)):
                for j in range(i+1, len(numeric_columns)):
                    col1, col2 = numeric_columns[i], numeric_columns[j]
                    fig = go.Figure(data=go.Scatter(x=self.df[col1], y=self.df[col2], mode='markers'))
                    fig.update_layout(title=f"{col1} vs {col2}", xaxis_title=col1, yaxis_title=col2)
                    scatter_path = f"{output_dir}/{col1}_{col2}_scatter.html"
                    fig.write_html(scatter_path)
                    visualization_paths[f"{col1}_{col2}_scatter"] = scatter_path
        
        self.analysis_results["visualizations"] = visualization_paths
        return visualization_paths
    
    def generate_eda_report(self, output_path: str = "reports/eda_report.pdf") -> str:
        """
        Generate a comprehensive EDA report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # Generate summary statistics if not already done
        if "summary_statistics" not in self.analysis_results:
            self.generate_summary_statistics()
        
        # Generate correlation matrix if not already done
        if "correlation_matrix" not in self.analysis_results:
            self.generate_correlation_matrix()
        
        # Generate visualizations if not already done
        if "visualizations" not in self.analysis_results:
            self.generate_visualizations()
        
        # Create PDF report
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Use built-in fonts instead of custom fonts
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Exploratory Data Analysis Report", ln=True, align="C")
        pdf.ln(10)
        
        # Dataset Overview
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Dataset Overview", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Dataset: {os.path.basename(self.dataset_path)}", ln=True)
        pdf.cell(0, 10, f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}", ln=True)
        pdf.ln(5)
        
        # Column Types
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Column Types", ln=True)
        pdf.set_font("Arial", "", 12)
        column_types = self.detect_column_types()
        for column, col_type in column_types.items():
            pdf.cell(0, 10, clean_text(f"{column}: {col_type}"), ln=True)
        pdf.ln(5)
        
        # Missing Values
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Missing Values", ln=True)
        pdf.set_font("Arial", "", 12)
        missing_values = self.df.isnull().sum()
        for column, count in missing_values.items():
            if count > 0:
                pdf.cell(0, 10, clean_text(f"{column}: {count} missing values"), ln=True)
        pdf.ln(5)
        
        # Numeric Summary
        if self.analysis_results["summary_statistics"]["numeric_summary"]:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Numeric Summary", ln=True)
            pdf.set_font("Arial", "", 12)
            for column, stats in self.analysis_results["summary_statistics"]["numeric_summary"].items():
                pdf.cell(0, 10, clean_text(f"{column}:"), ln=True)
                pdf.cell(0, 10, clean_text(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, Std: {stats['std']:.2f}"), ln=True)
                pdf.cell(0, 10, clean_text(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Skew: {stats['skew']:.2f}"), ln=True)
            pdf.ln(5)
        
        # Categorical Summary
        if self.analysis_results["summary_statistics"]["categorical_summary"]:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Categorical Summary", ln=True)
            pdf.set_font("Arial", "", 12)
            for column, stats in self.analysis_results["summary_statistics"]["categorical_summary"].items():
                pdf.cell(0, 10, clean_text(f"{column}: {stats['unique_values']} unique values"), ln=True)
                pdf.cell(0, 10, "  Top values:", ln=True)
                for value, count in stats['top_values'].items():
                    pdf.cell(0, 10, clean_text(f"    {value}: {count}"), ln=True)
            pdf.ln(5)
        
        # Cleaning Log
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Data Cleaning Steps", ln=True)
        pdf.set_font("Arial", "", 12)
        for step in self.cleaning_log:
            pdf.cell(0, 10, clean_text(f"- {step}"), ln=True)
        pdf.ln(5)
        
        # Save the report
        pdf.output(output_path)
        logger.info(f"EDA report generated: {output_path}")
        
        return output_path
    
    def prepare_for_modeling(self, target_column: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare the dataset for modeling.

        Args:
            target_column: Name of the target column
            test_size: Proportion of the dataset to include in the test split
            random_state: Random state for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Check if target column exists
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset")

        # Separate features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # Handle datetime columns by extracting features
        datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        for col in datetime_cols:
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_day"] = X[col].dt.day
            X[f"{col}_weekday"] = X[col].dt.weekday
            X.drop(columns=[col], inplace=True)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

        return X_train, X_test, y_train, y_test
    
    def save_cleaned_dataset(self, output_path: Optional[str] = None) -> str:
        """
        Save the cleaned dataset.
        
        Args:
            output_path: Path to save the cleaned dataset (if None, a default path is used)
            
        Returns:
            Path to the saved dataset
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
            output_path = f"datasets/{base_name}_cleaned.csv"
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved to: {output_path}")
        
        return output_path
    
    def save_analysis_results(self, output_path: Optional[str] = None) -> str:
        """
        Save the analysis results as JSON.
        
        Args:
            output_path: Path to save the analysis results (if None, a default path is used)
            
        Returns:
            Path to the saved analysis results
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
            output_path = f"reports/{base_name}_analysis.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.analysis_results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, dict):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Analysis results saved to: {output_path}")
        
        return output_path
    
    def clean_dataset(self) -> Dict[str, Any]:
        """
        Clean the dataset using a comprehensive strategy.
        
        Returns:
            Dictionary with cleaning results
        """
        try:
            # Create backup before any changes
            if self.original_df is None:
                self.original_df = self.df.copy()
                self.original_df.to_csv(self.backup_path, index=False)
                logger.info(f"Backup created at {self.backup_path}")
                self.cleaning_log.append(f"Backup created at {self.backup_path}")
            
            # Execute cleaning steps in sequence
            logger.info("Starting dataset cleaning process")
            self.cleaning_log.append("Starting dataset cleaning process")
            
            # Step 1: Detect column types
            column_types = self.detect_column_types()
            
            # Step 2: Standardize column names
            self.standardize_column_names()
            
            # Step 3: Handle missing values
            missing_counts = self.handle_missing_values()
            
            # Step 4: Detect and remove outliers
            outlier_counts = self.detect_outliers()
            if sum(outlier_counts.values()) > 0:
                self.remove_outliers()
            
            # Step 4.5: Clean text columns
            self.clean_text_columns()

            # Step 4.6: Format numeric variables
            self.format_num_vars(column_types)
                
            # Step 5: Remove duplicates
            self.remove_duplicates()
            
            # Step 6: Encode categorical variables
            self.encode_categorical_variables(column_types)
            
            # Step 7: Extract datetime features
            self.extract_datetime_features()

            ##### Step 8: Drop all Old Columns Compare new vs old vals?????
            #print(self.df.columns)
            #self.df.drop(columns=[col for col in self.df.columns if col in self.original_df.columns], axis=1, inplace=True)

            # Save the cleaned dataset
            cleaned_path = self.save_cleaned_dataset()
            
            # Generate summary statistics
            self.generate_summary_statistics()
            
            # Save analysis results
            self.save_analysis_results()
            
            # Generate EDA report
            report_path = self.generate_eda_report()
            
            logger.info("Dataset cleaning completed successfully")
            self.cleaning_log.append("Dataset cleaning completed successfully")
            
            # Print cleaning summary
            print("\nCleaning Summary:")
            for log in self.cleaning_log:
                print(f"- {log}")
            
            return {
                "cleaned_dataset_path": cleaned_path,
                "report_path": report_path,
                "cleaning_log": self.cleaning_log,
                "column_types": column_types,
                "missing_values": missing_counts,
                "outliers": outlier_counts
            }
            
        except Exception as e:
            # Restore from backup if something goes wrong
            if os.path.exists(self.backup_path):
                self.df = pd.read_csv(self.backup_path)
                logger.error(f"Error during cleaning: {str(e)}")
                logger.info("Dataset restored from backup")
                print(f"Error during cleaning: {str(e)}")
                print("Dataset restored from backup")
            else:
                logger.error(f"Critical error: {str(e)}")
                logger.error("No backup available for restoration")
                print(f"Critical error: {str(e)}")
                print("No backup available for restoration")
            
            raise 