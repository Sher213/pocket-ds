import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, roc_curve
)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/auto_modeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("auto_modeler")

class AutoModeler:
    """
    A comprehensive class for automated model selection, training, and evaluation.
    """
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initialize the AutoModeler with training and test data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.model_results = {}
        self.training_log = []
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Detect problem type
        self.problem_type = self._detect_problem_type()
        logger.info(f"Detected problem type: {self.problem_type}")
        self.training_log.append(f"Detected problem type: {self.problem_type}")
        
        # Initialize models based on problem type
        self._initialize_models()
    
    def _detect_problem_type(self) -> str:
        """
        Detect the type of problem (classification or regression).
        
        Returns:
            Problem type ("classification" or "regression")
        """
        # Try to convert y_train to numeric, coercing errors to NaN
        y_train_numeric = pd.to_numeric(self.y_train, errors='coerce')
        
        # If conversion has no NaN values, treat as regression
        if y_train_numeric.notna().all():
            # Check if target has few unique values (classification)
            unique_values = self.y_train.nunique()
            if unique_values < min(50, len(self.y_train) * 0.5):
                return "classification"
            else:
                return "regression"
        else:
            # If conversion failed (contains NaN), treat as classification
            return "classification"
    
    def _initialize_models(self):
        """Initialize models based on the problem type."""
        if self.problem_type == "classification":
            self.models = {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier(),
                "gradient_boosting": GradientBoostingClassifier(),
                "svm": SVC(probability=True),
                "xgboost": xgb.XGBClassifier(),
                "neural_network": MLPClassifier(max_iter=1000)
            }
        else:  # regression
            self.models = {
                "linear_regression": LinearRegression(),
                "ridge_regression": Ridge(),
                "lasso_regression": Lasso(),
                "random_forest": RandomForestRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "svm": SVR(),
                "xgboost": xgb.XGBRegressor(),
                "neural_network": MLPRegressor(max_iter=1000)
            }
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data for modeling.
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded)
        """

        # Ensure 2D input
        if isinstance(self.X_train, pd.Series):
            self.X_train = self.X_train.to_frame()
        if isinstance(self.X_test, pd.Series):
            self.X_test = self.X_test.to_frame()

        # Align test columns to match train (important after encoding)
        self.X_test = self.X_test.reindex(columns=self.X_train.columns, fill_value=0)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Encode target if classification
        if self.problem_type == "classification" and not pd.api.types.is_numeric_dtype(self.y_train):
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(self.y_train)
            y_test_encoded = le.transform(self.y_test)
        else:
            y_train_encoded = self.y_train
            y_test_encoded = self.y_test
        
        logger.info("Data preprocessed for modeling")
        self.training_log.append("Data preprocessed for modeling")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
    
    def train_and_evaluate_models(self, cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all models.
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = self.preprocess_data()
        
        results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name} model")
                self.training_log.append(f"Training {name} model")
                
                # Train the model
                model.fit(X_train_scaled, y_train_encoded)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Evaluate the model
                metrics = self._evaluate_model(y_test_encoded, y_pred, model, X_test_scaled)
                
                # Store results
                results[name] = metrics
                self.model_results[name] = {
                    "model": model,
                    "metrics": metrics
                }
                
                logger.info(f"{name} model trained and evaluated: {metrics}")
                self.training_log.append(f"{name} model trained and evaluated: {metrics}")
                
            except Exception as e:
                logger.error(f"Error training {name} model: {str(e)}")
                self.training_log.append(f"Error training {name} model: {str(e)}")
        
        return results
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model: Any, X_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a model based on the problem type.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model: Trained model
            X_test: Test features
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        if self.problem_type == "classification":
            # Classification metrics
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, average='weighted')
            metrics["recall"] = recall_score(y_true, y_pred, average='weighted')
            metrics["f1"] = f1_score(y_true, y_pred, average='weighted')
            
            # ROC-AUC if model supports predict_proba
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                except:
                    pass
        
        else:  # regression
            # Regression metrics
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)
        
        return metrics
    
    from typing import Any, Tuple

    def select_best_model(self, metric: str = None) -> Tuple[Any, str, float]:
        """
        Select the best model based on a specific metric.
        
        Args:
            metric: Metric to use for model selection (if None, a default is used)
            
        Returns:
            Tuple of (best_model, best_model_name, best_score)
        """
        if not self.model_results:
            raise ValueError("No models have been trained yet")
        
        # Choose default metric
        if metric is None:
            metric = "accuracy" if self.problem_type == "classification" else "r2"
        metric = metric.lower()
        
        # Define which metrics should be minimized vs. maximized
        minimize_metrics = {"rmse", "mae", "mse", "log_loss"}
        is_minimize = metric in minimize_metrics
        
        # Initialize best score
        if is_minimize:
            best_score = float("inf")
        else:
            best_score = -float("inf")
        best_model_name = None
        
        # Search through trained models
        for name, result in self.model_results.items():
            metrics = result.get("metrics", {})
            if metric not in metrics:
                continue
            score = metrics[metric]
            
            # Compare appropriately
            if (is_minimize and score < best_score) or (not is_minimize and score > best_score):
                best_score = score
                best_model_name = name
        
        if best_model_name is None:
            raise ValueError(f"Could not find any model with the metric '{metric}'")
        
        # Store and return
        self.best_model = self.model_results[best_model_name]["model"]
        self.best_model_name = best_model_name
        self.best_score = best_score
        
        logger.info(f"Selected '{best_model_name}' as best with {metric} = {best_score}")
        self.training_log.append(f"Selected {best_model_name} as best with {metric} = {best_score}")
        
        return self.best_model, self.best_model_name, self.best_score

    
    def tune_hyperparameters(self, n_iter: int = 10, cv: int = 3) -> Dict[str, Any]:
        """
        Tune hyperparameters for the best model.
        
        Args:
            n_iter: Number of parameter settings sampled
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with tuned hyperparameters
        """
        if self.best_model is None:
            raise ValueError("No best model has been selected yet")
        
        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = self.preprocess_data()
        
        # Define parameter space based on model type
        param_space = self._get_param_space(self.best_model_name)
        
        # Create RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=self.best_model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring=self._get_scoring_metric(),
            n_jobs=-1,
            random_state=42
        )
        
        # Fit RandomizedSearchCV
        logger.info(f"Tuning hyperparameters for {self.best_model_name}")
        self.training_log.append(f"Tuning hyperparameters for {self.best_model_name}")
        
        random_search.fit(X_train_scaled, y_train_encoded)
        
        # Update best model with tuned hyperparameters
        self.best_model = random_search.best_estimator_
        
        # Evaluate tuned model
        y_pred = self.best_model.predict(X_test_scaled)
        tuned_metrics = self._evaluate_model(y_test_encoded, y_pred, self.best_model, X_test_scaled)
        
        logger.info(f"Tuned hyperparameters for {self.best_model_name}: {random_search.best_params_}")
        self.training_log.append(f"Tuned hyperparameters for {self.best_model_name}: {random_search.best_params_}")
        
        return {
            "best_params": random_search.best_params_,
            "best_score": random_search.best_score_,
            "tuned_metrics": tuned_metrics
        }
    
    def _get_param_space(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameter space for hyperparameter tuning.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with parameter space
        """
        if model_name == "logistic_regression":
            return {
                "C": stats.uniform(0.1, 10.0),
                "penalty": ["l1", "l2", "elasticnet", None],
                "solver": ["liblinear", "saga"]
            }
        
        elif model_name == "random_forest":
            if self.problem_type == "classification":
                return {
                    "n_estimators": stats.randint(10, 200),
                    "max_depth": [None] + list(range(5, 30, 5)),
                    "min_samples_split": stats.randint(2, 20),
                    "min_samples_leaf": stats.randint(1, 10)
                }
            else:  # regression
                return {
                    "n_estimators": stats.randint(10, 200),
                    "max_depth": [None] + list(range(5, 30, 5)),
                    "min_samples_split": stats.randint(2, 20),
                    "min_samples_leaf": stats.randint(1, 10)
                }
        
        elif model_name == "gradient_boosting":
            if self.problem_type == "classification":
                return {
                    "n_estimators": stats.randint(10, 200),
                    "learning_rate": stats.uniform(0.01, 0.3),
                    "max_depth": stats.randint(3, 10),
                    "min_samples_split": stats.randint(2, 20),
                    "min_samples_leaf": stats.randint(1, 10)
                }
            else:  # regression
                return {
                    "n_estimators": stats.randint(10, 200),
                    "learning_rate": stats.uniform(0.01, 0.3),
                    "max_depth": stats.randint(3, 10),
                    "min_samples_split": stats.randint(2, 20),
                    "min_samples_leaf": stats.randint(1, 10)
                }
        
        elif model_name == "svm":
            if self.problem_type == "classification":
                return {
                    "C": stats.uniform(0.1, 10.0),
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"] + list(stats.uniform(0.001, 1.0).rvs(5))
                }
            else:  # regression
                return {
                    "C": stats.uniform(0.1, 10.0),
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"] + list(stats.uniform(0.001, 1.0).rvs(5))
                }
        
        elif model_name == "xgboost":
            if self.problem_type == "classification":
                return {
                    "n_estimators": stats.randint(10, 200),
                    "learning_rate": stats.uniform(0.01, 0.3),
                    "max_depth": stats.randint(3, 10),
                    "min_child_weight": stats.randint(1, 10),
                    "subsample": stats.uniform(0.6, 0.4),
                    "colsample_bytree": stats.uniform(0.6, 0.4)
                }
            else:  # regression
                return {
                    "n_estimators": stats.randint(10, 200),
                    "learning_rate": stats.uniform(0.01, 0.3),
                    "max_depth": stats.randint(3, 10),
                    "min_child_weight": stats.randint(1, 10),
                    "subsample": stats.uniform(0.6, 0.4),
                    "colsample_bytree": stats.uniform(0.6, 0.4)
                }
        
        elif model_name == "neural_network":
            if self.problem_type == "classification":
                return {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "activation": ["relu", "tanh"],
                    "alpha": stats.uniform(0.0001, 0.1),
                    "learning_rate": ["constant", "adaptive"]
                }
            else:  # regression
                return {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "activation": ["relu", "tanh"],
                    "alpha": stats.uniform(0.0001, 0.1),
                    "learning_rate": ["constant", "adaptive"]
                }
        
        else:
            return {}
    
    def _get_scoring_metric(self) -> str:
        """
        Get the appropriate scoring metric based on the problem type.
        
        Returns:
            Scoring metric
        """
        if self.problem_type == "classification":
            return "accuracy"
        else:  # regression
            return "neg_mean_squared_error"
    
    def generate_model_report_txt(self, output_dir: str = "reports") -> str:
        """
        Generate a comprehensive model report in text format with a timestamped filename.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        if self.best_model is None:
            raise ValueError("No best model has been selected yet")

        from datetime import datetime
        import os

        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_report_{timestamp}.txt"
        output_path = os.path.join(output_dir, filename)

        os.makedirs(output_dir, exist_ok=True)

        lines = []

        # Header
        lines.append("=== Model Evaluation Report ===\n")

        # Problem Type
        lines.append(">> Problem Type")
        lines.append(f"Type: {self.problem_type.capitalize()}\n")

        # Best Model
        lines.append(">> Best Model")
        lines.append(f"Model: {self.best_model_name}")
        lines.append(f"Score: {self.best_score:.4f}\n")

        # Model Comparison Table
        lines.append(">> Model Comparison")
        if self.problem_type == "classification":
            header = "Model\tAccuracy\tPrecision\tRecall\tF1 Score"
        else:
            header = "Model\tMSE\tRMSE\tMAE\tR²"
        lines.append(header)

        for name, result in self.model_results.items():
            if self.problem_type == "classification":
                row = (
                    f"{name}\t"
                    f"{result['metrics'].get('accuracy', 0):.4f}\t"
                    f"{result['metrics'].get('precision', 0):.4f}\t"
                    f"{result['metrics'].get('recall', 0):.4f}\t"
                    f"{result['metrics'].get('f1', 0):.4f}"
                )
            else:
                row = (
                    f"{name}\t"
                    f"{result['metrics'].get('mse', 0):.4f}\t"
                    f"{result['metrics'].get('rmse', 0):.4f}\t"
                    f"{result['metrics'].get('mae', 0):.4f}\t"
                    f"{result['metrics'].get('r2', 0):.4f}"
                )
            lines.append(row)
        lines.append("")

        # Feature Importance
        if hasattr(self.best_model, "feature_importances_"):
            lines.append(">> Feature Importance")
            importances = self.best_model.feature_importances_
            feature_names = self.X_train.columns.tolist()
            indices = np.argsort(importances)[::-1]

            lines.append("Feature\tImportance")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                lines.append(f"{feature_names[idx]}\t{importances[idx]:.4f}")
            lines.append("")

        # Training Log
        lines.append(">> Training Log")
        for step in self.training_log:
            lines.append(f"- {step}")

        # Save to file
        with open(output_path, "w", encoding='utf-8') as f:
            f.write("\n".join(lines))

        logger.info(f"Model TXT report generated: {output_path}")
        return output_path

    def generate_model_pdf_report(self, txt_file_path: str, output_dir: str = "reports") -> str:
        """
        Generate a PDF report from a TXT file with the same filename (but .pdf extension).
        
        Args:
            txt_file_path: Path to the input TXT report file.
            output_dir: Directory to save the generated PDF report.
            
        Returns:
            Path to the generated PDF report.
        """
        if not os.path.exists(txt_file_path):
            raise FileNotFoundError(f"The specified TXT file does not exist: {txt_file_path}")
        
        # Extract filename without extension and create output path for the PDF
        filename = os.path.basename(txt_file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{filename_without_ext}.pdf")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Read the content of the TXT file
        with open(txt_file_path, "r", encoding='utf-8') as file:
            report_text = file.read()

        # Create PDF report
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Add a title
        pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", "", 14)
        pdf.cell(0, 10, "Model Evaluation Report", ln=True, align="C")
        pdf.ln(10)

        # Add the content from the TXT file
        pdf.set_font("DejaVu", "", 14)
        pdf.multi_cell(0, 10, report_text)

        # Save the generated PDF
        pdf.output(output_path)
        logger.info(f"PDF report generated from TXT file: {output_path}")

        return output_path
    
    def save_model(self, output_path: Optional[str] = None) -> str:
        """
        Save the best model.
        
        Args:
            output_path: Path to save the model (if None, a default path is used)
            
        Returns:
            Path to the saved model
        """
        if self.best_model is None:
            raise ValueError("No best model has been selected yet")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"models/{self.best_model_name}_{timestamp}.joblib"
        
        # Save the model
        joblib.dump(self.best_model, output_path)
        
        # Save model metadata
        metadata = {
            "model_name": self.best_model_name,
            "problem_type": self.problem_type,
            "feature_columns": self.X_train.columns.tolist(),
            "target_column": self.y_train.name if hasattr(self.y_train, "name") else "target",
            "metrics": self.model_results[self.best_model_name]["metrics"],
            "training_log": self.training_log
        }
        
        metadata_path = f"{output_path}.metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Model saved to: {output_path}")
        logger.info(f"Model metadata saved to: {metadata_path}")
        
        return output_path
    
    def generate_visualizations(self, output_dir: str = "reports/visualizations") -> Dict[str, str]:
        """
        Generate visualizations for model evaluation.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        if self.best_model is None:
            raise ValueError("No best model has been selected yet")
        
        os.makedirs(output_dir, exist_ok=True)
        visualization_paths = {}
        
        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = self.preprocess_data()
        y_pred = self.best_model.predict(X_test_scaled)
        
        if self.problem_type == "classification":
            # Confusion Matrix
            cm = confusion_matrix(y_test_encoded, y_pred)
            fig = go.Figure(data=go.Heatmap(z=cm, x=list(range(cm.shape[1])), y=list(range(cm.shape[0]))))
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            cm_path = f"{output_dir}/confusion_matrix.html"
            fig.write_html(cm_path)
            visualization_paths["confusion_matrix"] = cm_path
            
            # ROC Curve (if model supports predict_proba)
            if hasattr(self.best_model, "predict_proba"):
                try:
                    y_prob = self.best_model.predict_proba(X_test_scaled)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test_encoded, y_prob)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc_score(y_test_encoded, y_prob):.4f})'))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    roc_path = f"{output_dir}/roc_curve.html"
                    fig.write_html(roc_path)
                    visualization_paths["roc_curve"] = roc_path
                except:
                    pass
        
        else:  # regression
            # Actual vs Predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test_encoded, y=y_pred, mode='markers', name='Predictions'))
            fig.add_trace(go.Scatter(x=[min(y_test_encoded), max(y_test_encoded)], y=[min(y_test_encoded), max(y_test_encoded)], mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
            fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
            scatter_path = f"{output_dir}/actual_vs_predicted.html"
            fig.write_html(scatter_path)
            visualization_paths["actual_vs_predicted"] = scatter_path
            
            # Residuals
            y_test_encoded = pd.to_numeric(y_test_encoded, errors='coerce')
            y_pred = pd.to_numeric(y_pred, errors='coerce')

            residuals = y_test_encoded - y_pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
            fig.add_trace(go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0], mode='lines', name='Zero Residual', line=dict(dash='dash')))
            fig.update_layout(title="Residuals", xaxis_title="Predicted", yaxis_title="Residuals")
            residuals_path = f"{output_dir}/residuals.html"
            fig.write_html(residuals_path)
            visualization_paths["residuals"] = residuals_path
        
        # Feature Importance (if available)
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
            feature_names = self.X_train.columns.tolist()
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            top_indices = indices[:10]  # Top 10 features
            
            fig = go.Figure(data=go.Bar(x=[feature_names[i] for i in top_indices], y=[importances[i] for i in top_indices]))
            fig.update_layout(title="Top 10 Feature Importance", xaxis_title="Feature", yaxis_title="Importance")
            importance_path = f"{output_dir}/feature_importance.html"
            fig.write_html(importance_path)
            visualization_paths["feature_importance"] = importance_path
        
        return visualization_paths
    
    def auto_model(self, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Automatically model the data.
        
        Args:
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary with modeling results
        """
        try:
            # Train and evaluate all models
            model_results = self.train_and_evaluate_models()
            
            # Select the best model
            best_model, best_model_name, best_score = self.select_best_model()
            
            # Tune hyperparameters if requested
            tuning_results = None
            if tune_hyperparameters:
                tuning_results = self.tune_hyperparameters()
            

            
            # Save the model
            model_path = self.save_model()
            
            logger.info("Auto modeling completed successfully")
            self.training_log.append("Auto modeling completed successfully")
            
            return {
                "best_model_name": self.best_model_name,
                "best_score": self.best_score,
                "model_results": model_results,
                "tuning_results": tuning_results,
                "model_path": model_path,
                "training_log": self.training_log
            }
            
        except Exception as e:
            logger.error(f"Error during auto modeling: {str(e)}")
            self.training_log.append(f"Error during auto modeling: {str(e)}")
            raise 