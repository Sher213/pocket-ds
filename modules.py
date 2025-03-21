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

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AnalysisPredictor():
	def __init__(self, dataset, target_class):
		self.dataset = f"datasets/{dataset}"
		
		self.problem = None
		self.target_class = target_class
		self.model = None

		self.messages = []
		
		df = pd.read_csv(self.dataset)
		
		if not self.target_class.lower() in [l.lower() for l in df.columns]:
			print("target class not in dataset")
			raise Exception
	
	def prompt(self, prompt):
		completion = client.chat.completions.create(
			model="gpt-4o",
			messages=[{"role": "user", "content": prompt}]
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
		
		prediction = self.prompt(message)
		print(prediction)
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

class BestModelIdentifier:
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
        df = pd.read_csv(f"datasets/{csv}")
        
        threshold = 50
        
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
                score = -score  # Minimize MSE
            
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
        self.problem_type = problem_type
        self.dataset = f"datasets/{dataset}"
        self.model = model

        self.report = None

        self.reports_dir = "datasets/"
    
    def create_visuals(self, X_test, y_test, predictions):
        """Generate and save visualizations for model evaluation."""
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
            plt.close()

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
        {list(score.keys())[0]}: {score[list(score.keys())[0]]}

        Include:
        - Overview of the model.
        - Explanation of feature importance (if available).
        - Analysis of evaluation metrics (accuracy, confusion matrix, ROC curve).
        - Recommendations for model improvement.

        Provide a detailed and well-structured report.
        """

        # Call OpenAI API
        completion = client.chat.completions.create(
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
        cm_path = f"{self.reports_dir}/confusion_matrix.png"

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
        self.save_report_as_pdf(report_text)
    

predictor = AnalysisPredictor(dataset="20250314_144422_insurance.csv", target_class="region")
problem_type = predictor.predict_problem()
bmi = BestModelIdentifier(problem_type)
reporter = ReportCreator("20250314_144422_insurance.csv", bmi.best_model, problem_type)

X, y = bmi.process_csv("20250314_144422_insurance.csv", "region")
bmi.fit_and_evaluate(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pred, score = bmi.predict(X_test, y_test)
reporter.create_full_report(X_train, X_test, y_train, y_test, pred, score)
