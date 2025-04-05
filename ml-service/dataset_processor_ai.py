import re
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from statsmodels.tsa.arima.model import ARIMA
import os
from dataset_processor import DatasetProcessor

load_dotenv()

class LLM_API:
    def __init__(self, system_message):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_message = system_message
        self.messages = [
            {"role": "system", "content": self.system_message}
        ]  # Initialize messages with the system message

    def send_prompt(self, prompt):
        # Send a prompt to the OpenAI API
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Corrected the model name to "gpt-4"
            messages=self.messages + [{"role": "user", "content": prompt}]
        )
        # Return the content of the first choice's message
        return completion.choices[0].message.content

    def prompt_whist(self, prompt):
        # Appending the user prompt to the message list
        self.messages.append({"role": "user", "content": prompt})
        
        # Get the response from the API
        response = self.send_prompt(prompt)
        
        # Return the response from the model
        return response

class AnalysisPredictor(LLM_API):
    def __init__(self, dataset, target_class):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_message = "You are a data analyst expert."

        self.dataset = f"datasets/{dataset}"
        
        self.problem = None
        self.target_class = target_class
        self.model = None

        self.messages = []
        
        dataset = pd.read_csv(self.dataset)
        
        if not self.target_class.lower() in [l.lower() for l in dataset.columns]:
            print("target class not in dataset")
            raise Exception

    def predict_problem(self):
        data = pd.read_csv(self.dataset)

        message = f"""{list(data.columns)}
                    {list(data.iloc[0])}
                    This is a dataset with one row of data. The target class is {self.target_class}. Predict whether it is a:
                    1) Regression Problem
                    2) Classification Problem
                    3) Sentiment Analysis Problem
                    4) Time-Series Problem.

                    Determine and return the corresponding number. Return only the number.
                    """
        
        prediction = super().send_prompt(message)
        
        print("Prediction: ", prediction)

        match = re.match(r"\d+", prediction)  # Match one or more digits
        if match:
            prediction = int(match.group())  # Convert to integer

            if prediction == 1:
                self.problem = "regression"
            elif prediction == 2:
                self.problem = "classification"
            elif prediction == 3:
                self.problem = "sentiment"
            elif prediction == 4:
                self.problem = "time-series"
            else:
                self.problem = "unknown"  # Handle unexpected values
        else:
            self.problem = "invalid input"  # Handle cases where no number is found

        return self.problem

class BestModelIdentifier():
    def __init__(self, problem_type):
        """
        Initialize with the problem type.
        :param problem_type: "regression", "classification", "sentiment", or "time-series"
        """
        self.problem_type = problem_type
        self.models = self._get_models()
        self.best_model = None
        self.best_score = None

    def _get_models(self):
        """Return a dictionary of models based on the problem type."""
        if self.problem_type == "classification":
            return {
                "LogisticRegression": LogisticRegression(),
                "RandomForestClassifier": RandomForestClassifier(),
                "SVM": SVC(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "MLPClassifier": MLPClassifier()
            }
        elif self.problem_type == "regression":
            return {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(),
                "SVR": SVR(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "Ridge": Ridge(),
                "Lasso": Lasso()
            }
        elif self.problem_type == "sentiment":
            return {
                "NaiveBayes": MultinomialNB(),
                "RandomForestClassifier": RandomForestClassifier(),
                "SVM": SVC(),
                "MLPClassifier": MLPClassifier(),
                "LogisticRegression": LogisticRegression()
            }
        elif self.problem_type == "time-series":
            return {"ARIMA": None}  # ARIMA is trained separately
        else:
            raise ValueError("Invalid problem type. Choose 'regression', 'classification', 'sentiment', or 'time-series'.")

    def process_csv(self, csv, target):
        threshold = 50

        def ask_chatgpt_to_identify_irrelevant_columns(columns, sample_data):
            prompt = f"""
            Given the following dataset columns and sample data, identify which columns are irrelevant for predicting the target variable:
            
            Columns: {columns}
            Sample Data:
            {sample_data}
            
            Provide a list of column names that should be removed.
            """
            
            llm = LLM_API("""You are an expert data analyst well trained in the ability to identify irrelevant features for model training."
                      Ensure to follow these rules: 1) Only include the names of the columns - no explanation is necessary, 
                      2) Ignore the target feature, 3) Organize the irrelevant feature names neatly with no other decorators or text.""")
            response = llm.send_prompt(prompt)
            
            irrelevant_columns = response.split("\n")

            return [col.strip() for col in irrelevant_columns if col.strip() in columns]
        
        dataset = pd.read_csv(f"datasets/{csv}")
        
        # Ask ChatGPT to check which columns are irrelevant
        sample_data = dataset.head(5).to_dict()
        irrelevant_columns = ask_chatgpt_to_identify_irrelevant_columns(dataset.columns.tolist(), sample_data)
        dataset.drop(columns=irrelevant_columns, errors='ignore', inplace=True)

        encoded_dataset = dataset.select_dtypes(include=[np.number]).copy()
    
        for col in dataset.select_dtypes(exclude=[np.number]).columns:
            if not col == target:
                unique_count = dataset[col].nunique()

                # Apply One-Hot Encoding if column is categorical (low cardinality)
                if unique_count <= threshold:
                    encoded_cols = pd.get_dummies(dataset[col], prefix=col, drop_first=True)
                    encoded_dataset = pd.concat([encoded_dataset, encoded_cols], axis=1)

                # Apply Label Encoding if column has high cardinality
                else:
                    le = LabelEncoder()
                    encoded_dataset[col] = le.fit_transform(dataset[col].astype(str))
            else:
                 encoded_dataset[col] = dataset[target]

        return encoded_dataset[[c for c in encoded_dataset.columns if not c == target]], encoded_dataset[target]

    def fit_and_evaluate(self, X, y, test_size=0.2, metric=None, cv_folds=5):
        """
        Train multiple models and evaluate them.
        :param X: Features
        :param y: Target
        :param test_size: Test set proportion
        :param metric: Custom metric function (optional)
        :param cv_folds: Number of cross-validation folds
        """
        if self.problem_type == "time-series":
            return self._fit_time_series(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        best_score = -np.inf if self.problem_type in ["classification", "sentiment"] else np.inf
        best_model = None

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Classification & Sentiment
            if self.problem_type in ["classification", "sentiment"]:
                score = accuracy_score(y_test, predictions) if not metric else metric(y_test, predictions)
                cv_score = np.mean(cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy'))
            
            # Regression
            else:
                score = mean_squared_error(y_test, predictions) if not metric else metric(y_test, predictions)
                cv_score = np.mean(cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error'))
            
            print(f"{name} - Test Score: {score:.4f}, CV Score: {cv_score:.4f}")

            # Update best model
            if (self.problem_type in ["classification", "sentiment"] and score > best_score) or \
               (self.problem_type == "regression" and score < best_score):
                best_score = score
                best_model = model

        self.best_model = best_model
        self.best_score = best_score
        print(f"Best Model: {type(self.best_model).__name__} with score: {self.best_score:.4f}")

    def predict(self, X, y):
        """
        Make predictions and evaluate the best model.
        :param X: Features
        :param y: Target
        :return: predictions, evaluation metrics
        """
        if self.best_model is None:
            raise Exception("Best model not trained yet. Call 'fit_and_evaluate' first.")

        # Make predictions using the best model
        predictions = self.best_model.predict(X)

        # Evaluate the model based on the problem type
        if self.problem_type == "classification" or self.problem_type == "sentiment":
            accuracy = accuracy_score(y, predictions)
            print(f"Accuracy: {accuracy:.4f}")
            return predictions, {"accuracy": accuracy}

        elif self.problem_type == "regression":
            mse = mean_squared_error(y, predictions)
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            return predictions, {"mse": mse}

        elif self.problem_type == "time-series":
            # For time-series, you can evaluate with some time-series metrics or AIC for ARIMA
            if hasattr(self.best_model, 'aic'):
                aic = self.best_model.aic
                print(f"AIC: {aic:.4f}")
                return predictions, {"aic": aic}
            else:
                raise Exception("Time-series model did not return valid AIC.")

        return predictions, {}

    def _fit_time_series(self, y, order=(5,1,0)):
        """Train ARIMA model for time series data."""
        model = ARIMA(y, order=order)
        model = model.fit()
        self.best_model = model
        self.best_score = model.aic  # Lower AIC is better
        print(f"Trained ARIMA model with AIC: {self.best_score:.4f}")

class ReportCreator():
    def __init__(self, dataset, model, problem_type):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.problem_type = problem_type
        self.dataset = f"datasets/{dataset}"
        print(model)
        self.model = model

        self.report = None

        self.reports_dir = "datasets/"
    
    def create_visuals(self, X_test, y_test, predictions):
        """Generate and save visualizations for model evaluation.
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', label='ROC Curve (AUC = {:.4f})'.format(auc(fpr, tpr)))
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig('reports/roc_curve.png')
            plt.close()"""

        if self.problem_type in ['classification', 'sentiment']:
            cm = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig('reports/confusion_matrix.png')
            plt.close()

        print("Visuals saved in 'reports/' directory.")
    
    def generate_text_report(self, X_train, X_test, y_train, y_test, predictions, score):
        """Generate a detailed text-based model evaluation report using OpenAI API."""

        # Generate report with OpenAI API
        prompt = f"""
        Generate a comprehensive report analyzing a machine learning model.
        Model Type: {type(self.model).__name__}
        Dataset: {pd.read_csv(self.dataset).head()}
        Problem Type: {self.problem_type}
        Model Score: {list(score.keys())[0]}: {score[list(score.keys())[0]]}

        Include:
        - Overview of the model.
        - Explanation of feature importance (if available).
        - Analysis of evaluation metrics (make them specific to task, i.e regression considers MSE, classification considers accuracy, etc.).
        - Recommendations for model improvement.

        Do Not Include:
        - ROC Curve
        - Confusion Matrix

        Provide a detailed and well-structured report.
        """

        # Call OpenAI API
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        report_text = completion.choices[0].message.content

        # Save text to file
        report_path = f"{self.reports_dir}/model_report.txt"
        with open(report_path, "w") as file:
            file.write(report_text)

        print(f"✅ Text report saved as '{report_path}'")
        return report_text

    def save_report_as_pdataset(self, report_text):
        """Generate and save the full report as a Pdataset."""
        pdataset = FPDF()
        pdataset.set_auto_page_break(auto=True, margin=15)
        pdataset.add_page()
        pdataset.set_font("Arial", "B", 16)
        pdataset.cell(200, 10, "Model Evaluation Report", ln=True, align="C")

        # Add model overview
        pdataset.set_font("Arial", size=12)
        pdataset.multi_cell(0, 10, report_text)

        # Add visuals if available
        pdataset.add_page()
        pdataset.set_font("Arial", "B", 14)
        pdataset.cell(200, 10, "Visual Analysis", ln=True, align="C")

        roc_path = f"{self.reports_dir}/roc_curve.png"
        cm_path = f"reports/confusion_matrix.png"

        if os.path.exists(roc_path):
            pdataset.image(roc_path, x=30, y=50, w=150)
            pdataset.ln(85)  # Space after ROC curve

        if os.path.exists(cm_path):
            pdataset.image(cm_path, x=30, y=140, w=150)

        # Save final Pdataset
        pdataset_path = f"{self.reports_dir}/model_report.pdataset"
        pdataset.output(pdataset_path)
        print(f"✅ Full report saved as '{pdataset_path}'")

    def create_full_report(self, X_train, X_test, y_train, y_test, predictions, score):
        """Generate visuals, a text report, and a Pdataset."""
        self.create_visuals(X_test, y_test, predictions)
        report_text = self.generate_text_report(X_train, X_test, y_train, y_test, predictions, score)

        def clean_text(text):
            return re.sub(r'[^\x00-\x7F]+', ' ', text)
        self.save_report_as_pdataset(clean_text(report_text))

class DatasetScriptor(LLM_API, DatasetProcessor):
    def __init__(self, dataset, target_column):
        """Initialize with the dataset file and OpenAI API key."""
        
        # LLM_API initialization
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_message = "You are a data cleaning and processing expert. Your skills range from data type validation, missing value handling, text standardization, numeric range validation, and duplicate removal, etc. using Python and libraries like pandas, numpy, scikit-learn."
        self.messages = []

        self.dataset_path = f"datasets/{dataset}"
        self.dataset = pd.read_csv(self.dataset_path)
        self.current_script_path = ""
        self.cleaning_log = []
        self.backup_path = f"{self.dataset_path}.backup"

        #DatasetProcessor initialization
        self.df = None
        self.original_df = None
        self.analysis_results = {}
        self.backup_path = f"{self.dataset_path}.backup"
        self.target_column = target_column
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        # Load the dataset
        self._load_dataset()

        print(self.df.dtypes)

    def create_backup_DEPRECATED(self):
        """Create a backup of the original dataset."""
        self.dataset.to_csv(self.backup_path, index=False)
        self.cleaning_log.append(f"Backup created at {self.backup_path}")

    def validate_data_types_DEPRECATED(self):
        """Validate and convert data types appropriately."""
        type_changes = []
        for column in self.dataset.columns:
            try:
                # Try to convert to numeric first
                numeric_series = pd.to_numeric(self.dataset[column], errors='coerce')
                if not numeric_series.isna().all():  # If conversion was successful for some values
                    self.dataset[column] = numeric_series
                    type_changes.append(f"Converted {column} to numeric")
                else:
                    # Try to convert to datetime
                    try:
                        datetime_series = pd.to_datetime(self.dataset[column], errors='coerce')
                        if not datetime_series.isna().all():
                            self.dataset[column] = datetime_series
                            type_changes.append(f"Converted {column} to datetime")
                    except:
                        # Keep as string if neither conversion works
                        self.dataset[column] = self.dataset[column].astype(str)
            except Exception as e:
                self.cleaning_log.append(f"Error converting {column}: {str(e)}")
        return type_changes

    def handle_missing_values_DEPRECATED(self):
        """Handle missing values with appropriate strategies."""
        missing_values = self.dataset.isnull().sum()
        if missing_values.sum() > 0:
            for column in self.dataset.columns:
                if self.dataset[column].isnull().sum() > 0:
                    try:
                        if pd.api.types.is_numeric_dtype(self.dataset[column]):
                            # For numeric columns, use median (more robust than mean)
                            fill_value = self.dataset[column].median()
                            self.dataset[column].fillna(fill_value, inplace=True)
                            self.cleaning_log.append(f"Filled missing values in {column} with median: {fill_value}")
                        else:
                            # For categorical columns, use mode
                            fill_value = self.dataset[column].mode()[0]
                            self.dataset[column].fillna(fill_value, inplace=True)
                            self.cleaning_log.append(f"Filled missing values in {column} with mode: {fill_value}")
                    except Exception as e:
                        self.cleaning_log.append(f"Error handling missing values in {column}: {str(e)}")

    def standardize_text(self):
        """Standardize text formatting in string columns."""
        for column in self.dataset.select_dtypes(include=['object']).columns:
            try:
                # Convert to string and clean
                self.dataset[column] = self.dataset[column].astype(str)
                self.dataset[column] = self.dataset[column].str.lower()
                self.dataset[column] = self.dataset[column].str.strip()
                self.dataset[column] = self.dataset[column].str.replace(r'[^\w\s]', '', regex=True)
                self.cleaning_log.append(f"Standardized text in {column}")
            except Exception as e:
                self.cleaning_log.append(f"Error standardizing text in {column}: {str(e)}")

    def validate_numeric_ranges(self):
        """Validate and fix numeric ranges for common fields."""
        numeric_rules = {
            'age': (0, 120),
            'height': (0, 300),
            'weight': (0, float('inf')),
            'price': (0, float('inf')),
            'score': (0, 100)
        }
        
        for column in self.dataset.select_dtypes(include=['number']).columns:
            column_lower = column.lower()
            for rule_key, (min_val, max_val) in numeric_rules.items():
                if rule_key in column_lower:
                    try:
                        invalid_mask = (self.dataset[column] < min_val) | (self.dataset[column] > max_val)
                        if invalid_mask.any():
                            # Replace invalid values with median
                            median_val = self.dataset[column].median()
                            self.dataset.loc[invalid_mask, column] = median_val
                            self.cleaning_log.append(f"Fixed {invalid_mask.sum()} invalid values in {column} using median: {median_val}")
                    except Exception as e:
                        self.cleaning_log.append(f"Error validating numeric range in {column}: {str(e)}")

    def standardize_column_names_DEPRECATED(self):
        """Standardize column names."""
        try:
            new_columns = {col: col.lower().replace(' ', '_').replace('-', '_') for col in self.dataset.columns}
            self.dataset.rename(columns=new_columns, inplace=True)
            self.cleaning_log.append("Standardized column names")
        except Exception as e:
            self.cleaning_log.append(f"Error standardizing column names: {str(e)}")

    def remove_duplicates_DEPRECATED(self):
        """Remove duplicate rows while preserving the first occurrence."""
        try:
            initial_rows = len(self.dataset)
            self.dataset.drop_duplicates(inplace=True)
            removed_rows = initial_rows - len(self.dataset)
            if removed_rows > 0:
                self.cleaning_log.append(f"Removed {removed_rows} duplicate rows")
        except Exception as e:
            self.cleaning_log.append(f"Error removing duplicates: {str(e)}")

    def clean_dataset_DEPRECATED(self):
        """Clean the dataset using a comprehensive strategy."""
        try:
            # Create backup before any changes
            self.create_backup()
            
            # Execute cleaning steps in sequence
            self.cleaning_log.append("Starting dataset cleaning process")
            
            # Step 1: Standardize column names
            self.standardize_column_names()
            
            # Step 2: Validate and convert data types
            type_changes = self.validate_data_types()
            self.cleaning_log.extend(type_changes)
            
            # Step 3: Handle missing values
            self.handle_missing_values()
            
            # Step 4: Standardize text
            self.standardize_text()
            
            # Step 5: Validate numeric ranges
            self.validate_numeric_ranges()
            
            # Step 6: Remove duplicates
            self.remove_duplicates()
            
            # Save the cleaned dataset
            self.dataset.to_csv(self.dataset_path, index=False)
            self.cleaning_log.append("Dataset cleaning completed successfully")
            
            # Print cleaning summary
            print("\nCleaning Summary:")
            for log in self.cleaning_log:
                print(f"- {log}")
                
        except Exception as e:
            # Restore from backup if something goes wrong
            if os.path.exists(self.backup_path):
                self.dataset = pd.read_csv(self.backup_path)
                self.dataset.to_csv(self.dataset_path, index=False)
                print(f"Error during cleaning: {str(e)}")
                print("Dataset restored from backup")
            else:
                print(f"Critical error: {str(e)}")
                print("No backup available for restoration")

    def execute_script(self):
        """Execute the saved script."""
        if not self.current_script_path:
            print("No script available. Please generate and save the script first.")
            return

        try:
            with open(self.current_script_path, 'r') as f:
                script_content = f.read()
            exec(script_content)
            print("Script executed successfully!")
        except Exception as e:
            print(f"Error running script: {e}")

    def use_dataset_scriptor(self):
        # Read and print the content of the file
        file_path = "dataset_processor.py"

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                dataset_processor_script = file.read()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    
        file_script = self.prompt_whist(f"""Here is a dataset processor script called dataset_processor.py: {dataset_processor_script}. Please use the code and 
            functions in dataset_processor.py to clean this dataset: {self.dataset.head(10).to_string()}.
            Here is info on the dataset: 
                dataset.info(): {str(self.dataset.info())}
                dataset.shape: {self.dataset.shape}      
                dataset.columns: {self.dataset.columns.tolist()}              
                dataset.dtypes: {self.dataset.dtypes.to_dict()}
                dataset.info(): {str(self.dataset.info())}     
                dataset.tail(n=5): {self.dataset.tail(n=5).to_string()}
                dataset.sample(n=5): {self.dataset.sample(n=5).to_string()}
                dataset.describe(): {self.dataset.describe().to_string()}
                dataset.describe(include='object'): {self.dataset.describe(include='object').to_string()} 
                dataset.mean(): {self.dataset.mean(numeric_only=True).to_string()}
                dataset.median(): {self.dataset.median(numeric_only=True).to_string()}        
                dataset.std(): {self.dataset.std(numeric_only=True).to_string()}
                dataset.var(): {self.dataset.var(numeric_only=True).to_string()}            
                dataset.min(): {self.dataset.min(numeric_only=True).to_string()}
                dataset.max(): {self.dataset.max(numeric_only=True).to_string()}           
                dataset.mode(): {self.dataset.mode(numeric_only=True).to_string()}.
            Please provide the code for the dataset processor script to clean this dataset.

            Ensure to follow these rules:
            1) Use the code and functions in dataset_processor.py to clean the dataset.
            2) Refer to the dataset as "dataset" in the code.
            3) The target feature is {self.target_column}.
            4) DO NOT include any import statements or the class definition.
            5) Use '{self.dataset_path}' as the dataset path.
            6) DO NOT include any other text or explanation. Just provide the code.
            7) DO NOT include any decorators or text in the code. DO NOT USE "# " or "```python" in the code.)
            """)
        
        file_script = file_script.replace("```python", "").replace("```", "").replace("# ", "").replace("```", "")
        file_script = f"""import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset_processor import DatasetProcessor
import pandas as pd
import numpy as np
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

{file_script}

# Save the modified dataset
try:
    dataset.to_csv('{self.dataset_path.split('.')[0]}_cleaned.csv', index=False)
    print('Dataset updated successfully.')
except Exception as e:
    print('Dataset not saved manually:', str(e))"""
                      
        # Save the script to a file
        self.current_script_path = "scripts/generated_script_for_dataset_processing.py"
        os.makedirs(os.path.dirname(self.current_script_path), exist_ok=True)

        with open(self.current_script_path, "w", encoding="utf-8") as file:
            file.write(file_script)

        print("Saved to: ", self.current_script_path)
        
        self.execute_script(self)
    
        return self.dataset_path, self.dataset

predictor = AnalysisPredictor(dataset="20250314_144422_insurance.csv", target_class="charges")
problem_type = predictor.predict_problem()
bmi = BestModelIdentifier(problem_type)

X, y = bmi.process_csv("20250314_144422_insurance.csv", "charges")
bmi.fit_and_evaluate(X, y)

reporter = ReportCreator("20250314_144422_insurance.csv", bmi.best_model, problem_type)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pred, score = bmi.predict(X_test, y_test)
reporter.create_full_report(X_train, X_test, y_train, y_test, pred, score)


script_gen = DatasetScriptor('insurance.csv', target_column="charges")
#script_gen.use_dataset_scriptor(target_class="charges")
script_gen.clean_dataset()