import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError, APIError
import httpx
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
from dataset_processor import DatasetProcessor

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set; cannot initialize GenAI client.")

client = genai.Client(api_key=API_KEY)

def ask_genai_to_identify_irrelevant_columns(columns, sample_data, target_column):
    prompt = f"""Given the following information about a dataset, identify the columns that are likely irrelevant for predicting the specified target variable in a machine learning model.

Dataset Information:

Columns: {columns}
Sample Data: {sample_data}
Target Column: {target_column}
Based on your expert data science knowledge, provide a list containing only the names of the columns you recommend for removal before model training.

Remember the following constraints:

List Only Removals: Your output should only include the names of the columns you want to remove. Do not list columns you intend to keep for training.
Encoded Column Handling: If any original categorical columns are present in the {columns} and the {sample_data} suggests they have been encoded (e.g., one-hot encoded), include the names of the original categorical columns in your removal list, not the encoded columns themselves **HINT: If the name of a column is "column_name_encoded", it is safe to remove "column_name" from {columns}.
Target Preservation: **DO NOT** include the name of the {target_column} in your list of columns to remove **UNLESS** it has been encoded. **IF THE TARGET COLUMN IS NOT ENCODED, DO NOT REMOVE IT.**
Provide your list of columns to remove here:"""
            
    llm = LLM_API("""As an expert data scientist skilled in identifying irrelevant features for model training, you will be provided with a list of columns. Your task is to identify and list only the names of the columns that should be removed before training a machine learning model.
Ensure you adhere to the following strict rules:
Output Format: Only include the names of the columns you want to remove. Do not include the names of columns you intend to use for training.
Target Preservation: Do not include the name of the target variable in your list of columns to remove.
Encoded Column Handling: If any original categorical columns have been encoded into new numerical columns, include the names of the original categorical columns in your list for removal. Do not remove the encoded columns themselves""")
    
    try:
        response = llm.prompt(prompt)
    except Exception as e:
        return ["Error getting target columns."]

    print("RESPONSE: ", response)

    # Convert string to list (and clean whitespace)
    remove_list = [ line.strip("[] \n'") for line in response.strip('[] \n').replace("```", "").split(',') ]

    return remove_list

class LLM_API:
    def __init__(self, system_message):
        self.system_message = system_message
        self.messages = []

    def prompt(self, prompt):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=self.system_message
                ),
                contents=prompt
            )
            return response.text
        except ClientError as e:
            # 4xx error: malformed request, bad input, auth issues, etc.
            print(f"[ClientError] {e.code} {e.status}: {e.message}")
            print("Full details JSON:", e.details)
            # You can also peek at the raw httpx.Response:
            print("Raw text:", e.response.text)

        except ServerError as e:
            # 5xx error: server-side problem
            print(f"[ServerError] {e.code} {e.status}: {e.message}")
            print("Details:", e.details)

        except APIError as e:
            # catch-all for any other APIError
            print(f"[APIError] {e.code} {e.status}: {e.message}")
            print("Details:", e.details)

        except httpx.HTTPError as e:
            # network / transport errors
            print("HTTPX transport error:", str(e))

    def prompt_whist(self, prompt):
        # Appending the user prompt to the message list
        self.messages.append(prompt)
        
        # Get the response from the API
        response = self.prompt(self.messages)
        
        # Return the response from the model
        return response

class DatasetLLM(LLM_API):
    def __init__(self, file_name, dataset):
        self.dataset = dataset
        self.file_name = file_name

        super().__init__(f"""You are a helpful, expert data analyst and information retriever. Your primary goal is to accurately and comprehensively answer user questions based on the provided dataset.

**Here's how you should operate:**

1.  **Understand the Dataset:** You will be provided with a description of a dataset, including its columns, data types, and potentially some context or instructions about its contents. Pay close attention to this description.

2.  **Identify Key Information:** When a user asks a question, identify the key entities, attributes, and relationships mentioned in their query that relate to the described dataset.

3.  **Formulate a Search Strategy (Internal):** Based on the identified key information, devise a mental strategy for how you would locate the answer within the (hypothetical) dataset. This might involve filtering by certain columns, comparing values, performing calculations, or identifying specific records.

4.  **Synthesize the Answer:** Once you have mentally "located" the relevant information, synthesize a clear and concise answer that directly addresses the user's question.

5.  **Provide Context (If Necessary):** If the answer requires additional context from the dataset to be fully understood, briefly include that context in your response.

6.  **Acknowledge Limitations (If Applicable):** If the question cannot be answered based on the provided dataset description, state this clearly and politely. Avoid making assumptions or bringing in outside information.

7.  **Maintain a Helpful and Respectful Tone:** Always strive to be helpful and respectful in your responses.
                         
8.  **Do not use code to provide a response, always provide directly what was asked for.""")

    def add_data(self, target_column, eda_report, model_report):
        self.target_column = target_column
        self.eda_report = eda_report
        self.model_report = model_report

    def prompt_whist(self, prompt, data_needed):
        full_prompt = f"{prompt}"
        
        if "eda_report" in data_needed:
            full_prompt += f"\n\nEDA Report:\n{self.eda_report}"
        if "model_report" in data_needed:
            full_prompt += f"\n\nModel Report:\n{self.model_report}"
        if "dataset" in data_needed:
            try:
                full_prompt += f"\n\nDataset:\n{self.dataset.to_string()}"
            except Exception as e:
                full_prompt += f"\n\nDataset:\n[Error loading dataset: {e}]"
        if "target_column" in data_needed:
            full_prompt += f"\n\nTarget Column:\n{self.target_column}"
        
        full_prompt += f"**If they ask for the file name**, give {self.file_name}."
        
        return super().prompt_whist(full_prompt)
    
class ModelLLM(LLM_API):
    def __init__(self, model_path):
        self.model_path = model_path

        super().__init__(f"""You are a highly skilled Data Scientist and Machine Learning Expert. Your primary function is to generate a Python script that, when executed, will load a pre-trained machine learning model located at the specified `{model_path}` and use it to predict an outcome based on user-provided input.

**Your operational guidelines are as follows:**

1.  **Identify the Model Type:** Based on the `{model_path}` (which might implicitly or explicitly suggest the model type, e.g., `.pkl` for scikit-learn, `.h5` for Keras, `.pt` for PyTorch), infer the likely machine learning library used to train the model.

2.  **Determine Necessary Libraries:** Import all Python libraries essential for loading the identified model type and processing input data. This will likely include libraries like `pickle`, `joblib`, `tensorflow`, `keras`, `torch`, `numpy`, and potentially `pandas` or `sklearn.preprocessing` depending on the expected input format and model requirements.

3.  **Implement the Loading Mechanism:** Write the code to load the model from the given `{model_path}`. Ensure robust error handling in case the file is not found or is corrupted.

4.  **Preprocess the User Input:** Numerically encode relevant string/object features in the user input using the encoding schemes (Label Encoding, One-Hot Encoding) derived from the dataset. 

5.  **Missing Features:** For any features present in the dataset but missing from the user input, impute the missing values using the most statistically appropriate method derived from the dataset (e.g., mean for numerical features, mode for categorical features). Ensure all expected features are present and filled to allow for model prediction.

4.  **Implement the Prediction Logic:** Write the code to feed the processed input data to the loaded model's prediction function (`predict`, `predict_proba`, etc.).

5.  **Format the Output:** Structure the output in a clear and understandable way. This might involve printing the predicted class label, probability scores, or a more descriptive interpretation of the prediction.

6:  **Executable:** Ensure your response is executable as is, this means there are no comments or instructions.

7:  **Process User Prompt:** The user will pass instructions. Use them to derive the features for the prediction.

**Example Scenario (Illustrative - Your script should be more general):**

If `{model_path}` suggests a scikit-learn model, your script might:

* Import `pickle` and `numpy`.
* Load the model using `pickle.load()`.
* You will already have the features and target column.
* Use `model.predict()` to get the prediction.
* Print the predicted class label and any other information asked for.

By following these guidelines, you will create a robust and well-documented script that enables prediction using the specified pre-trained model.""")

    def predict(self, prompt, dataset, features, target_column):
        """
        Generates a Python script that loads a pre-trained machine learning model,
        prompts the user for input based on the specified column names, and has the model
        automatically discern which inputs to use as features. THe features that cannot be
        automatically passed to the model are looked up in the dataset, and their encoded value is pulled.
        The script then performs a prediction for the target column and prints the results.

        The generated script performs these steps:
        1. Loads a pre-trained model from self.model_path.
        2. Pull encoded values for non-processable values.
        3. Any missing features that are not in the user input should be imputed to ensure all features are present to allow for model prediction.
        4. Uses the loaded model to predict the outcome for the given target column.
        5. Prints the prediction result, including any probabilities or additional model output.

        Parameters:
            prompt (str):
            dataset (DataFrame): 
            column_names (list or str): A list or comma-separated string of column names for which
                the user will provide input.
            target_column (str): The name of the target column that the model will predict.

        Returns:
            str: The captured output from executing the generated script, including any errors.
        """
        try:
            dataset = pd.read_csv(f"datasets/{dataset.split('.')[0]}_cleaned.csv")

            # Generate the Python script with detailed instructions.
            script = super().prompt(f"""Generate a Python script that performs the following: {prompt}

Ensure to:
1. **Load a pre-trained machine learning model** located at the path: `{self.model_path}`.
2. **Preprocess any non-numeric data** to match what the model expects. For example, if the model was trained on encoded string/object columns, ensure the same encoding is applied by referencing the dataset. 
3. **Fill any missing features** required for model prediction that are not provided in the user input. Use the full list of features expected by the model: {features}. For each missing feature, impute a value using an appropriate statistical method (mean, median, or mode) based on the dataset.
4. **Convert the final input into a pandas DataFrame**, ensuring each value is wrapped in a list to avoid scalar-related errors.
5. **Use the loaded model to predict** the outcome for the target column: `{target_column}`.
6. **Print the prediction result** clearly. If the model provides probabilities or other metadata, include them in the output.

**Important Considerations:**

* **Error Handling:** Add error handling for common issues such as file not found, invalid input types, and prediction errors.
* **Library Imports:** Import all required libraries (e.g., `pickle`, `joblib`, `tensorflow`, `torch`, `numpy`, `pandas`).
* **Executability:** Do not include any markdown, comments, or extra formatting—just pure Python code that can be executed as-is.

Model Features:
{features}

Dataset Sample:
{dataset}""")
            
            # Validate that a script was generated.
            if not script:
                raise ValueError("Script generation failed: No script was returned.")
            
            # Clean script
            script = script.replace("```python", "").replace("```", "")

            with open("scripts/generated_script.py", "w", encoding="utf-8") as file:
                file.write(script)

            import os
            import subprocess

            # Get the current working directory.
            current_dir = os.getcwd()

            # Build the paths.
            activate_script = os.path.join(current_dir, "myenv", "Scripts", "Activate.ps1")
            generated_script = os.path.join(current_dir, "scripts", "generated_script.py")

            # Construct a PowerShell command:
            # - NoProfile avoids loading extra profiles.
            # - ExecutionPolicy Bypass ensures that the activation script runs even if policies would normally block it.
            # - The command dot-sources the activation script (with a leading period and space), then runs the generated script.
            command = (
                f'powershell.exe -NoProfile -ExecutionPolicy Bypass -Command '
                f'"& {{ . \\"{activate_script}\\"; python \\"{generated_script}\\" }}"'
            )

            print("Executing command:")
            print(command)

            # Execute the command in PowerShell using subprocess.
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Build a robust error message by including the return code and stderr output.
            output = result.stdout
            if result.returncode != 0:
                output += f"\n[ERROR] Command exited with non-zero return code: {result.returncode}"
            if result.stderr:
                print("\nStandard error output:\n" + result.stderr)

            return output
        
        except Exception as e:
            print(f"Error generating or executing the prediction script: {e}")
            raise

class ReportCreator():
    def __init__(self, dataset_path, model=None, problem_type=None):
        self.problem_type = problem_type
        self.dataset_path = dataset_path

        self.model = model

        self.report = None

        self.reports_dir = "datasets/"

        self.df = pd.read_csv(self.dataset_path)
    
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
    
    def generate_eda_report_with_genai(self, output_path="reports/eda_report.txt"):
        """Generate a detailed text-based model evaluation report using Gemini API."""

        # Read the eda report from the file
        with open(output_path, "r") as file:
            eda_report = file.read()

        # Generate report with Gemini API
        prompt = f"""You are an expert data analyst. Generate a comprehensive report analyzing the provided dataset.

**Dataset:**

{self.df.sample(n=10).to_string()}

**Dataset EDA Statistics:**
{eda_report}

**Report Requirements:**

**1. Data Understanding and Preprocessing:**

* **Initial Overview:** Describe the dataset's structure, including the number of rows and columns, and the names and data types of each column.
* **Missing Value Analysis:** Identify any missing values in each column. Report the percentage of missing values for each column and suggest potential strategies for handling them (e.g., imputation, removal), without actually performing the imputation/removal.
* **Data Type Assessment:** Review the data types of each column and identify any potential inconsistencies or columns that might need type conversion for analysis.
* **Summary Statistics:** Provide descriptive statistics (mean, median, standard deviation, min, max, quartiles) for numerical columns and frequency counts for categorical columns. Highlight any initial observations or potential outliers based on these statistics.

**2. Exploratory Data Analysis (EDA):**

* **Univariate Analysis:** Analyze the distribution of individual variables.
    * For numerical variables, create histograms or box plots to visualize their distributions, identify skewness, and potential outliers. Describe the key characteristics of each distribution.
    * For categorical variables, create bar charts to visualize the frequency of each category. Describe the dominant categories and any imbalances.
* **Bivariate Analysis:** Explore the relationships between pairs of variables.
    * For numerical-numerical pairs, create scatter plots to visualize correlations. Calculate and report the Pearson correlation coefficient (if appropriate) and interpret the relationship.
    * For categorical-categorical pairs, create contingency tables and consider using stacked bar charts or grouped bar charts to visualize the relationship. You can also mention the possibility of using chi-squared tests for independence (without performing them).
    * For numerical-categorical pairs, create box plots or violin plots to compare the distribution of the numerical variable across different categories of the categorical variable. Describe any observed differences.
* **Multivariate Analysis (Optional but Encouraged):** If the dataset has more than a few relevant features, discuss potential multivariate relationships or interactions that might be worth investigating further. This could involve suggesting the use of techniques like pair plots or dimensionality reduction for visualization (without actually performing them).

**3. Task Definition and Evaluation Metrics:**

* **Infer the Potential Task:** Based on the features and potential target variables in the dataset, infer what a likely predictive modeling task could be (e.g., classification, regression, clustering). Clearly state the inferred task and the potential target variable(s).
* **Specify Relevant Evaluation Metrics:** Based on the inferred task, list and briefly explain the key evaluation metrics that would be appropriate for assessing the performance of a model trained on this data.
    * **For Regression:** Explain metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared.
    * **For Binary Classification:** Explain metrics like Accuracy, Precision, Recall, F1-score, and AUC-ROC.
    * **For Multiclass Classification:** Explain metrics like Accuracy, Precision (macro/micro/weighted), Recall (macro/micro/weighted), F1-score (macro/micro/weighted), and potentially the confusion matrix.
    * **For Clustering (if inferred):** Discuss metrics like Silhouette Score, Davies-Bouldin Index, or visual inspection.

**4. Potential Trends and Insights:**

* Based on your EDA, describe any potential trends or patterns you observe in the data. For example:
    * Are there any variables that seem strongly correlated?
    * Are there any noticeable differences in a numerical variable across different categories?
    * Are there any time-based trends (if the data includes a time component)?
    * Are there any unusual or unexpected distributions?
* Provide actionable insights based on these trends. What business questions could these insights help answer? What potential hypotheses could be formed for further investigation?

**5. Further Analysis and Next Steps:**

* Suggest potential avenues for further analysis or modeling based on your initial findings. This could include:
    * Specific feature engineering steps that might be beneficial.
    * Types of predictive models that could be explored.
    * Further data exploration techniques that could provide deeper understanding.
    * Considerations for data cleaning and preprocessing.

**Report Structure:**

Organize your report clearly with the following sections:

1.  **Introduction:** Briefly state the purpose of the report and the dataset being analyzed.
2.  **Data Understanding and Preprocessing**
3.  **Exploratory Data Analysis (EDA)**
4.  **Task Definition and Evaluation Metrics**
5.  **Potential Trends and Insights**
6.  **Further Analysis and Next Steps**
7.  **Conclusion:** Summarize the key findings and recommendations.

DO NOT USE ANY SPECIAL CHARACTERS. REMAIN COMPLAINT TO UTF-8 ENCODING and HELVETICA FONT.
"""

        # Call Gemini API
        llm = LLM_API("""You are an expert data analyst and Exploratory Data Analysis (EDA) specialist. Your primary function is to generate comprehensive and insightful data analysis reports. When presented with a dataset and a reporting objective, you will:

1. **Conduct a thorough Exploratory Data Analysis (EDA)** to understand the data's structure, quality, patterns, and potential issues. This includes:
    * **Examining data types and distributions of individual variables.**
    * **Identifying missing values, outliers, and inconsistencies.**
    * **Exploring relationships between variables using visualizations and statistical methods.**
    * **Formulating initial hypotheses and insights based on the EDA findings.**
2. **Structure the report logically** with clear sections such as:
    * **Introduction:** Briefly outlining the data and the reporting objective.
    * **Data Overview:** Describing the dataset's key characteristics and any data cleaning steps taken.
    * **Exploratory Data Analysis:** Presenting key findings from the EDA with relevant visualizations and statistical summaries. Explain the insights derived from each analysis.
    * **Insights and Observations:** Summarizing the main patterns, trends, and potential areas of interest identified during EDA.
    * **Limitations:** Acknowledging any limitations of the data or the analysis.
    * **(Optional) Recommendations/Next Steps:** Suggesting further analysis or actions based on the EDA findings.
3. **Utilize appropriate visualizations** (e.g., histograms, scatter plots, box plots, correlation matrices) to effectively communicate data characteristics and relationships. Ensure visualizations are clear, labeled, and accompanied by concise interpretations.
4. **Include relevant statistical summaries** (e.g., descriptive statistics, correlation coefficients) to support your visual findings.
5. **Maintain a clear, concise, and professional writing style**, explaining technical terms where necessary for the intended audience.
6. **Focus on uncovering meaningful insights** from the data through the EDA process and clearly articulating their implications in the report.""")
        
        report_text = llm.prompt(prompt)

        # Save text to file
        with open(output_path, "w", encoding='utf-8') as file:
            file.write(report_text)

        print(f"✅ genai Text report saved as '{output_path}'")
        return report_text

    def save_report_as_pdf(self, text_file_path, output_path="reports/model_report.pdf"):
        """Generate and save the full report as a PDF, supporting both EDA and model reports."""

        # Read file content
        with open(text_file_path, "r", encoding='utf-8') as file:
            report_text = file.read()

        # Decide title and visual section based on file type
        if "eda" in output_path.lower():
            title = "Exploratory Data Analysis Report"
            visual_title = "Data Visualizations"
            visuals = [
                f"{self.reports_dir}/eda_histograms.png",
                f"{self.reports_dir}/correlation_matrix.png"
            ]
        elif "model" in output_path.lower():
            title = "Model Evaluation Report"
            visual_title = "Model Performance Visuals"
            visuals = [
                f"{self.reports_dir}/roc_curve.png",
                f"{self.reports_dir}/confusion_matrix.png"
            ]
        else:
            title = "Analysis Report"
            visual_title = "Visual Analysis"
            visuals = []

        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", "", 14)
        pdf.cell(200, 10, title, ln=True, align="C")

        # Report content
        pdf.set_font("DejaVu", "", 14)
        pdf.multi_cell(0, 10, report_text)

        # Add visuals section
        if visuals:
            pdf.add_page()
            pdf.set_font("DejaVu", "", 14)
            pdf.cell(200, 10, visual_title, ln=True, align="C")

            for img_path in visuals:
                if os.path.exists(img_path):
                    pdf.ln(10)
                    pdf.image(img_path, x=30, w=150)

        # Save final PDF
        pdf.output(output_path)
        print(f"✅ PDF report saved as '{output_path}'")

    def generate_model_report_with_genai(self, model_report_path="reports/model_metrics.txt", output_path="reports/genai_model_report.txt"):
        """Generate a detailed model evaluation analysis report using Gemini API."""

        # Read the model evaluation report (metrics)
        with open(model_report_path, "r") as file:
            model_metrics = file.read()

        # Generate genai prompt
        prompt = f"""
    You are a senior machine learning engineer and data scientist. Given the model evaluation report below, generate a professional and structured analysis report.

    **Model Evaluation Metrics:**
    {model_metrics}

    **Dataset Sample:**
    {self.df.sample(n=10).to_string()}

    **Report Requirements:**

    1. **Model Overview:**
    - Infer the type of model used based on the metrics and dataset (classification, regression, etc.).
    - Comment on any known assumptions or use-cases of the inferred model type.

    2. **Metric Interpretation:**
    - Explain what each metric means (e.g., Accuracy, Precision, Recall, F1-score, AUC-ROC for classification; MAE, RMSE, R² for regression).
    - Analyze the values: Are they considered good? Are there signs of overfitting/underfitting?

    3. **Performance Evaluation:**
    - Comment on model strengths and weaknesses based on the reported metrics.
    - Mention if class imbalance could be a concern (based on classification metrics).
    - If a confusion matrix is available, discuss patterns (e.g., false positives/negatives).

    4. **Visual Insight (if applicable):**
    - Suggest interpretation for visual elements (ROC curve, confusion matrix, residual plots, etc.).

    5. **Model Improvement Suggestions:**
    - Provide recommendations for improving the model (e.g., hyperparameter tuning, feature engineering, different algorithms).
    - Discuss if more data or better quality data might help.

    6. **Next Steps:**
    - Suggest next actions (e.g., validation on unseen test data, model deployment, A/B testing, etc.).

    7. **Conclusion:**
    - Summarize findings and recommend whether the model is ready for production.

    Organize the final report in sections with headers like:
    - **Introduction**
    - **Metric Interpretation**
    - **Performance Analysis**
    - **Insights from Visuals**
    - **Recommendations for Improvement**
    - **Conclusion**

    DO NOT USE ANY SPECIAL CHARACTERS. REMAIN COMPLAINT TO UTF-8 ENCODING and HELVETICA FONT.
    """

        # Call Gemini API
        llm = LLM_API("""You are a highly knowledgeable data scientist and machine learning expert with expertise in statistical analysis, data preprocessing, model selection, and evaluation techniques. Your training includes extensive data up to October 2023, encompassing the latest advancements in machine learning algorithms, frameworks, and best practices.

Your primary goal is to generate comprehensive and insightful machine learning reports. When presented with a dataset and a prediction or classification task, you will:

1. **Conduct a thorough Exploratory Data Analysis (EDA)** to understand the data's characteristics, identify potential issues (missing values, outliers), and discover initial patterns relevant to the task.
2. **Perform appropriate data preprocessing techniques**, justifying your choices (e.g., handling missing data, feature scaling, encoding categorical variables).
3. **Select and implement relevant machine learning models**, explaining the rationale behind your model selection and considering multiple options where appropriate.
4. **Apply rigorous model evaluation techniques**, including appropriate metrics and validation strategies (e.g., cross-validation), to assess model performance and generalization ability.
5. **Structure the report logically** with clear sections such as:
    * **Introduction:** Defining the problem, the dataset, and the objective.
    * **Exploratory Data Analysis:** Presenting key EDA findings with visualizations and statistical summaries relevant to the modeling task.
    * **Data Preprocessing:** Detailing the steps taken to prepare the data for modeling and the reasoning behind these steps.
    * **Model Selection and Implementation:** Describing the models considered, the chosen model(s), and the implementation process.
    * **Model Evaluation:** Presenting the evaluation results, including relevant metrics and interpretations.
    * **Results and Discussion:** Summarizing the model performance, discussing the key findings, and interpreting the results in the context of the problem.
    * **Limitations and Future Work:** Acknowledging any limitations of the data or the modeling process and suggesting potential future improvements.
6. **Utilize clear visualizations and statistical summaries** throughout the report to support your analysis and findings.
7. **Maintain a clear, concise, and professional writing style**, explaining technical concepts where necessary.
8. **Focus on providing actionable insights** derived from the machine learning process and clearly articulating their implications.""")
        
        report_text = llm.prompt(prompt)

        # Save to file
        with open(output_path, "w", encoding= 'utf-8') as file:
            file.write(report_text)

        print(f"✅ genai Model report saved as '{output_path}'")
        return report_text

    def create_full_report(self, X_train, X_test, y_train, y_test, predictions, score):
        """Generate visuals, a text report, and a pdf."""
        self.create_visuals(X_test, y_test, predictions)
        report_text = self.generate_text_report(X_train, X_test, y_train, y_test, predictions, score)

        def clean_text(text):
            return re.sub(r'[^\x00-\x7F]+', ' ', text)
        self.save_report_as_pdf(clean_text(report_text))

class DatasetScriptor(LLM_API, DatasetProcessor):
    def __init__(self, dataset, target_column):
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
        self.cleaned_dataset_path = self.dataset_path.replace(".csv", "_cleaned.csv")
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

'''predictor = AnalysisPredictor(dataset="20250314_144422_insurance.csv", target_class="charges")
problem_type = predictor.predict_problem()
bmi = BestModelIdentifier(problem_type)

X, y = bmi.process_csv("20250314_144422_insurance.csv", "charges")
bmi.fit_and_evaluate(X, y)

reporter = ReportCreator("20250314_144422_insurance.csv", bmi.best_model, problem_type)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pred, score = bmi.predict(X_test, y_test)
reporter.create_full_report(X_train, X_test, y_train, y_test, pred, score)'''

#script_gen = DatasetScriptor('insurance.csv', target_column="charges")
#script_gen.use_dataset_scriptor(target_class="charges")
#script_gen.clean_dataset()