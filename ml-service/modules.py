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
import subprocess

load_dotenv()

class LLM():
    def __init__(self, system_message):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_message = system_message

    def prompt(self, prompt):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": self.system_message}, {"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

    def prompt_with_history(self, prompt):
        self.messages.append([
                {
                    "role": "user",
                    "content": prompt
                }
            ])
        return prompt(self.messages)

class AnalysisPredictor(LLM):
    def __init__(self, dataset, target_class):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_message = "You are a data analyst expert."

        self.dataset = f"datasets/{dataset}"
        
        self.problem = None
        self.target_class = target_class
        self.model = None

        self.messages = []
        
        df = pd.read_csv(self.dataset)
        
        if not self.target_class.lower() in [l.lower() for l in df.columns]:
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
        
        prediction = super().prompt(message)
        
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
            
            llm = LLM("""You are an expert data analyst well trained in the ability to identify irrelevant features for model training."
                      Ensure to follow these rules: 1) Only include the names of the columns - no explanation is necessary, 
                      2) Ignore the target feature, 3) Organize the irrelevant feature names neatly with no other decorators or text.""")
            response = llm.prompt(prompt)
            
            irrelevant_columns = response.split("\n")

            return [col.strip() for col in irrelevant_columns if col.strip() in columns]
        
        df = pd.read_csv(f"datasets/{csv}")
        
        # Ask ChatGPT to check which columns are irrelevant
        sample_data = df.head(5).to_dict()
        irrelevant_columns = ask_chatgpt_to_identify_irrelevant_columns(df.columns.tolist(), sample_data)
        df.drop(columns=irrelevant_columns, errors='ignore', inplace=True)

        encoded_df = df.select_dtypes(include=[np.number]).copy()
    
        for col in df.select_dtypes(exclude=[np.number]).columns:
            if not col == target:
                unique_count = df[col].nunique()

                # Apply One-Hot Encoding if column is categorical (low cardinality)
                if unique_count <= threshold:
                    encoded_cols = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    encoded_df = pd.concat([encoded_df, encoded_cols], axis=1)

                # Apply Label Encoding if column has high cardinality
                else:
                    le = LabelEncoder()
                    encoded_df[col] = le.fit_transform(df[col].astype(str))
            else:
                 encoded_df[col] = df[target]

        return encoded_df[[c for c in encoded_df.columns if not c == target]], encoded_df[target]

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

    def save_report_as_pdf(self, report_text):
        """Generate and save the full report as a PDF."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Model Evaluation Report", ln=True, align="C")

        # Add model overview
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, report_text)

        # Add visuals if available
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Visual Analysis", ln=True, align="C")

        roc_path = f"{self.reports_dir}/roc_curve.png"
        cm_path = f"reports/confusion_matrix.png"

        if os.path.exists(roc_path):
            pdf.image(roc_path, x=30, y=50, w=150)
            pdf.ln(85)  # Space after ROC curve

        if os.path.exists(cm_path):
            pdf.image(cm_path, x=30, y=140, w=150)

        # Save final PDF
        pdf_path = f"{self.reports_dir}/model_report.pdf"
        pdf.output(pdf_path)
        print(f"✅ Full report saved as '{pdf_path}'")

    def create_full_report(self, X_train, X_test, y_train, y_test, predictions, score):
        """Generate visuals, a text report, and a PDF."""
        self.create_visuals(X_test, y_test, predictions)
        report_text = self.generate_text_report(X_train, X_test, y_train, y_test, predictions, score)

        def clean_text(text):
            return re.sub(r'[^\x00-\x7F]+', ' ', text)
        self.save_report_as_pdf(clean_text(report_text))

class DatasetScriptor():
    def __init__(self, dataset):
        """Initialize with the dataset file and OpenAI API key."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.dataset_path = f"datasets/{dataset}"
        self.df = pd.read_csv(self.dataset_path)
        self.current_script_path = ""
        self.cleaning_log = []
        self.backup_path = f"{self.dataset_path}.backup"

    def create_backup(self):
        """Create a backup of the original dataset."""
        self.df.to_csv(self.backup_path, index=False)
        self.cleaning_log.append(f"Backup created at {self.backup_path}")

    def validate_data_types(self):
        """Validate and convert data types appropriately."""
        type_changes = []
        for column in self.df.columns:
            try:
                # Try to convert to numeric first
                numeric_series = pd.to_numeric(self.df[column], errors='coerce')
                if not numeric_series.isna().all():  # If conversion was successful for some values
                    self.df[column] = numeric_series
                    type_changes.append(f"Converted {column} to numeric")
                else:
                    # Try to convert to datetime
                    try:
                        datetime_series = pd.to_datetime(self.df[column], errors='coerce')
                        if not datetime_series.isna().all():
                            self.df[column] = datetime_series
                            type_changes.append(f"Converted {column} to datetime")
                    except:
                        # Keep as string if neither conversion works
                        self.df[column] = self.df[column].astype(str)
            except Exception as e:
                self.cleaning_log.append(f"Error converting {column}: {str(e)}")
        return type_changes

    def handle_missing_values(self):
        """Handle missing values with appropriate strategies."""
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            for column in self.df.columns:
                if self.df[column].isnull().sum() > 0:
                    try:
                        if pd.api.types.is_numeric_dtype(self.df[column]):
                            # For numeric columns, use median (more robust than mean)
                            fill_value = self.df[column].median()
                            self.df[column].fillna(fill_value, inplace=True)
                            self.cleaning_log.append(f"Filled missing values in {column} with median: {fill_value}")
                        else:
                            # For categorical columns, use mode
                            fill_value = self.df[column].mode()[0]
                            self.df[column].fillna(fill_value, inplace=True)
                            self.cleaning_log.append(f"Filled missing values in {column} with mode: {fill_value}")
                    except Exception as e:
                        self.cleaning_log.append(f"Error handling missing values in {column}: {str(e)}")

    def standardize_text(self):
        """Standardize text formatting in string columns."""
        for column in self.df.select_dtypes(include=['object']).columns:
            try:
                # Convert to string and clean
                self.df[column] = self.df[column].astype(str)
                self.df[column] = self.df[column].str.lower()
                self.df[column] = self.df[column].str.strip()
                self.df[column] = self.df[column].str.replace(r'[^\w\s]', '', regex=True)
                self.cleaning_log.append(f"Standardized text in {column}")
            except Exception as e:
                self.cleaning_log.append(f"Error standardizing text in {column}: {str(e)}")

    def validate_numeric_ranges(self):
        """Validate and fix numeric ranges for common fields."""
        numeric_rules = {
            'age': (0, 120),
            'height': (0, 300),
            'weight': (0, 500),
            'price': (0, float('inf')),
            'score': (0, 100)
        }
        
        for column in self.df.select_dtypes(include=['number']).columns:
            column_lower = column.lower()
            for rule_key, (min_val, max_val) in numeric_rules.items():
                if rule_key in column_lower:
                    try:
                        invalid_mask = (self.df[column] < min_val) | (self.df[column] > max_val)
                        if invalid_mask.any():
                            # Replace invalid values with median
                            median_val = self.df[column].median()
                            self.df.loc[invalid_mask, column] = median_val
                            self.cleaning_log.append(f"Fixed {invalid_mask.sum()} invalid values in {column} using median: {median_val}")
                    except Exception as e:
                        self.cleaning_log.append(f"Error validating numeric range in {column}: {str(e)}")

    def standardize_column_names(self):
        """Standardize column names."""
        try:
            new_columns = {col: col.lower().replace(' ', '_').replace('-', '_') for col in self.df.columns}
            self.df.rename(columns=new_columns, inplace=True)
            self.cleaning_log.append("Standardized column names")
        except Exception as e:
            self.cleaning_log.append(f"Error standardizing column names: {str(e)}")

    def remove_duplicates(self):
        """Remove duplicate rows while preserving the first occurrence."""
        try:
            initial_rows = len(self.df)
            self.df.drop_duplicates(inplace=True)
            removed_rows = initial_rows - len(self.df)
            if removed_rows > 0:
                self.cleaning_log.append(f"Removed {removed_rows} duplicate rows")
        except Exception as e:
            self.cleaning_log.append(f"Error removing duplicates: {str(e)}")

    def clean_dataset(self):
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
            self.df.to_csv(self.dataset_path, index=False)
            self.cleaning_log.append("Dataset cleaning completed successfully")
            
            # Print cleaning summary
            print("\nCleaning Summary:")
            for log in self.cleaning_log:
                print(f"- {log}")
                
        except Exception as e:
            # Restore from backup if something goes wrong
            if os.path.exists(self.backup_path):
                self.df = pd.read_csv(self.backup_path)
                self.df.to_csv(self.dataset_path, index=False)
                print(f"Error during cleaning: {str(e)}")
                print("Dataset restored from backup")
            else:
                print(f"Critical error: {str(e)}")
                print("No backup available for restoration")

'''predictor = AnalysisPredictor(dataset="20250314_144422_insurance.csv", target_class="charges")
problem_type = predictor.predict_problem()
bmi = BestModelIdentifier(problem_type)

X, y = bmi.process_csv("20250314_144422_insurance.csv", "charges")
bmi.fit_and_evaluate(X, y)

reporter = ReportCreator("20250314_144422_insurance.csv", bmi.best_model, problem_type)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pred, score = bmi.predict(X_test, y_test)
reporter.create_full_report(X_train, X_test, y_train, y_test, pred, score)'''


script_gen = DatasetScriptor('my_file.csv')
script_gen.clean_dataset()