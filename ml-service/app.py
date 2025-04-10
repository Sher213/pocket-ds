import difflib
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import os
import logging
from dataset_processor_ai import LLM_API, DatasetLLM, ModelLLM, DatasetScriptor, ReportCreator, ask_gpt_to_identify_irrelevant_columns
from auto_modeler import AutoModeler

# ===================== Setup Logging =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ===================== App Setup =====================
app = FastAPI(title="Pocket Data Scientist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports/visualizations", exist_ok=True)
os.makedirs("logs", exist_ok=True)

app.mount("/reports", StaticFiles(directory="reports"), name="reports")

processing_results: Dict[str, dict] = {}

dLLM = None
# ===================== Models =====================

class DatasetMetadata(BaseModel):
    filename: str
    rows: int
    columns: int
    column_types: Dict[str, str]
    missing_values: Dict[str, int]

class ModelingConfig(BaseModel):
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    tune_hyperparameters: bool = True

# ===================== Helpers =====================
def find_target_column(user_input: str, df_columns: List[str], cutoff: float = 0.6) -> Optional[str]:
    user_input_clean = user_input.strip().lower()
    column_map = {col: col.strip().lower() for col in df_columns}

    for orig_col, clean_col in column_map.items():
        if clean_col == user_input_clean:
            return orig_col

    best_matches = difflib.get_close_matches(user_input_clean, column_map.values(), n=1, cutoff=cutoff)
    if best_matches:
        for orig_col, clean_col in column_map.items():
            if clean_col == best_matches[0]:
                return orig_col

    return None

def generate_preview(file_path: str):
    try:
        df = pd.read_csv(file_path)
        preview = df.head().to_html()
        return preview
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        return str(e)

UPLOAD_DIRECTORY = "datasets"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# ===================== Endpoints =====================

@app.post("/dataset/llm")
async def prompt_llm(data: dict = Body(...)):
    decider = LLM_API("""You are an expert system for determining the appropriate API for a given Data Science and Machine Learning task. Your decision is based on whether the user's prompt can be directly answered using the provided dataset alone, or if it necessitates the application of a pre-existing machine learning model.

**Decision Criteria:**

* **Dataset API (Output: 1):** The user's prompt requests information retrieval, aggregation, filtering, or basic analysis that can be performed solely by querying or manipulating the dataset. No predictive modeling or complex transformations beyond standard data operations are required.

* **Model API (Output: 2):** The user's prompt requires the application of a machine learning model to generate a prediction, classification, embedding, or other model-driven output. The dataset serves as input to the model through the API.

**Your Task:**

Analyze the user's prompt and determine whether a dataset API or a model API is the appropriate tool to fulfill the request. Output your decision as a single integer: `1` for Dataset API, `2` for Model API.

**Constraints:**

* You must output only the integer `1` or `2`.
* Do not provide any explanations or justifications for your decision.

**Example Scenarios (for internal reasoning, do not output these):**

* **Prompt:** "What is the average age of customers in the dataset?" (Dataset API needed - Answer: 1)
* **Prompt:** "Predict the probability of churn for customer ID 123." (Model API needed - Answer: 2)
* **Prompt:** "List all products with a price greater than $100." (Dataset API needed - Answer: 1)
* **Prompt:** "Generate a customer segment for the following user features: age 35, income $50,000." (Model API needed - Answer: 2)""")
    
    prompt = data.get("prompt")
    filename = data.get("filename")
    target_column = data.get('target_column')
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    if filename not in processing_results:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    logger.info(f"Using file {filename} for LLM Chat.")

    try:
        decision = decider.prompt(prompt)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Error getting decision from LLM.")

    logger.info(f"Decision for LLM API was to use: {decision}.")

    try:
        if '1' in decision:
            model_content = ""
            eda_content = ""
            
            # Read model report if available
            model_report_path = processing_results[filename].get("model_report_path")
            if model_report_path:
                with open(model_report_path, "r", encoding="utf-8") as file:
                    model_content = file.read()
                
                logger.info("Read Model Report.")
            
            # Read EDA report text if available
            eda_report_path_txt = processing_results[filename].get("eda_report_path_txt")
            if eda_report_path_txt:
                with open(eda_report_path_txt, "r", encoding="utf-8") as file:
                    eda_content = file.read()

                    logger.info("Read EDA Report.")
            
            data_needed_decider = LLM_API(f"""You are a data and analyst report expert. Your task is to analyze a given natural language prompt and determine if the Exploratory Data Analysis (EDA) report, the Model report, or the Dataset (or any combination thereof) are required to fulfill the request in the prompt.

Your output MUST be a list containing only the names of the required items, in string format, such as ["eda", "model", "dataset", "target_column"]. Do not include any other text or explanation in your response.

For example, if the prompt requires an EDA report and the dataset, your output should be:
["eda", "dataset"]

If the prompt only requires a model report, your output should be:
["model"]

And so on. Remember to only output the list and nothing else.""")
            
            data_needed = data_needed_decider.prompt_whist(f"""{prompt} 
Use the provided data (considering 'dataset', 'model report', 'eda report', and 'target_column'). If the number of features in the input does not match the number of columns expected based on the dataset schema, impute the missing columns with a suitable filler value using the dataset.""")

            logger.info(f"Decided on needing: {data_needed}.")

            # Process the prompt using the DatasetLLM functionality
            dLLM.add_data(target_column, eda_content, model_content)
            response = dLLM.prompt_whist(prompt, data_needed)

            logger.info(f"Response: {response}.")

            return {"prompt": prompt, "response": response}
        elif '2' in decision:
            model_path = processing_results[filename]["model_path"]
            dataset = filename
            target_column = processing_results[filename]["target_class"]

            if not model_path:
                raise HTTPException(status_code=404, detail="Model not trained. Could not find the model file.")
            if dataset is None or len(dataset) == 0:
                raise HTTPException(status_code=404, detail="Error getting dataset.")
            if not target_column:
                raise HTTPException(status_code=404, detail="Error getting target column.")

            modelLLM = ModelLLM(model_path)
            features = processing_results[filename]["model_features"]

            logger.info(f"Loaded ModelLLM. Asking for prediction on model: {model_path}.")
            logger.info(f"Features passed: {features}.")

            response = modelLLM.predict(prompt, dataset, features, target_column)

            logger.info(f"Prediction: {response}.")

            return {"prompt": prompt, "response": response}

    except Exception as e:
        logger.exception("Error processing LLM prompt.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/upload")
async def upload_file(file: UploadFile = File(...), target_column: str = Form(...)):
    global dLLM
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)

        logger.info(f"Uploading file: {filename}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        df = pd.read_csv(file_path)
        df = df.astype(str)
        df.to_csv(file_path, index=False)

        matched_column = find_target_column(target_column, df.columns.tolist())
        logger.info(f"Matched target column: {matched_column}")

        if matched_column is None:
            return JSONResponse(status_code=400, content={
                "detail": f"Target column '{target_column}' not found or recognized in dataset."
            })

        processor = DatasetScriptor(filename, matched_column)

        processing_results[filename] = {
            "processor": processor,
            "target_class": matched_column,
            "metadata": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
        }

        logger.info(f"File {filename} uploaded and processed successfully.")

        dLLM = DatasetLLM(df)

        logger.info(f"Created Dataset LLM for {df.head()}.")

        return {
            "filename": filename,
            "message": "File uploaded successfully",
            "preview": generate_preview(file_path),
            "metadata": processing_results[filename]["metadata"]
        }

    except Exception as e:
        logger.exception("Exception during file upload")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/dataset/clean")
async def clean_dataset(data: Dict):
    filename = data.get("filename")
    target_class = data.get("target_class")
    use_gpt = data.get("use_gpt", True)

    logger.info(f"Initiating cleaning for: {filename}")

    if not filename or not target_class:
        logger.warning("Missing filename or target_class")
        raise HTTPException(status_code=400, detail="Filename and target_class are required")

    if filename not in processing_results:
        logger.warning("Dataset not found in processing results")
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = processing_results[filename]["processor"]
    results, target_class = processor.clean_dataset()

    # update target class name
    processing_results[filename]["target_class"] = target_class

    logger.info(f"Updated target class to: {processing_results[filename]['target_class']}")

    cleaned_path = processor.save_cleaned_dataset()
    visualizations = processor.generate_visualizations()
    eda_report_pdf_path, eda_report_txt_path = processor.generate_eda_report()

    if use_gpt:
        logger.info("Generating GPT-based EDA report")
        report_creator = ReportCreator(cleaned_path)
        report_creator.generate_eda_report_with_gpt(eda_report_txt_path)
        report_creator.save_report_as_pdf(eda_report_txt_path, eda_report_pdf_path)

    logger.info(f"Cleaning complete for: {filename}")

    processing_results[filename]["eda_report_path_txt"] = eda_report_txt_path

    return {
        "message": "Dataset cleaned successfully",
        "cleaned_dataset_path": cleaned_path,
        "visualization_paths": visualizations,
        "report_path": eda_report_pdf_path,
        "cleaning_log": processor.cleaning_log,
        "results": results
    }

@app.post("/dataset/model")
async def train_model(filename: str = Query(...), config: ModelingConfig = Body(...), use_gpt_model_report: bool = False):
    logger.info(f"Starting model training for: {filename}")

    target_column = processing_results[filename]["target_class"]

    logger.info(f"Target Column is {target_column}")

    if filename not in processing_results:
        logger.warning("Dataset not found in processing results")
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = processing_results[filename]["processor"]
    df = pd.read_csv(processor.cleaned_dataset_path)

    logger.info("Identifying irrelevant columns using GPT")
    columns_to_remove = ask_gpt_to_identify_irrelevant_columns(
        df.columns.tolist(), df.sample(n=10), target_column
    )

    logger.info(f"Columns to remove: {columns_to_remove}")

    processing_results[filename]["columns_removed"] = columns_to_remove

    df = df.drop(columns=columns_to_remove)
    processor.df = df

    X_train, X_test, y_train, y_test = processor.prepare_for_modeling(
        target_column=target_column,
        test_size=config.test_size,
        random_state=config.random_state
    )

    logger.info(f"Used: {X_train.columns} features to train model.")

    processing_results[filename]["model_features"] = X_train.columns

    modeler = AutoModeler(X_train, y_train, X_test, y_test)
    results = modeler.auto_model(tune_hyperparameters=config.tune_hyperparameters)
    visualization_paths = modeler.generate_visualizations()

    report_path_txt = modeler.generate_model_report_txt()
    report_path = modeler.generate_model_pdf_report(report_path_txt)

    logger.info("Model training completed")

    if use_gpt_model_report:
        logger.info("Generating GPT-based model report")
        reporter = ReportCreator(processor.cleaned_dataset_path)
        reporter.generate_model_report_with_gpt(report_path_txt, report_path_txt)
        reporter.save_report_as_pdf(report_path_txt, report_path)

    processing_results[filename]["model_report_path_txt"] = report_path_txt
    # --- Add this line to store the model file path in processing_results ---
    processing_results[filename]["model_path"] = results["model_path"]

    return {
        "message": "Model training complete",
        "columns_removed": columns_to_remove,
        "best_model_name": results["best_model_name"],
        "best_score": results["best_score"],
        "tuning_results": results["tuning_results"],
        "model_path": results["model_path"],
        "report_path": report_path,
        "visualization_paths": visualization_paths,
        "training_log": results["training_log"]
    }

@app.get("/model/download/{filename}")
async def download_model(filename: str):
    # Check that the dataset exists in our processing results
    if filename not in processing_results:
        raise HTTPException(status_code=404, detail="Dataset not found")
    # Get the stored model file path from the processing results
    model_path = processing_results[filename].get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Trained model file not found")
    return FileResponse(
        model_path,
        filename=os.path.basename(model_path),
        media_type="application/octet-stream"
    )

@app.get("/dataset/report/{filename}")
async def get_eda_report(filename: str):
    report_path = "reports/eda_report.pdf"
    if not os.path.exists(report_path):
        logger.warning(f"EDA report not found for: {filename}")
        raise HTTPException(status_code=404, detail="EDA report not found")
    return FileResponse(report_path, filename="eda_report.pdf")

@app.get("/model/report/{filename}")
async def get_model_report(filename: str):
    report_path = "reports/model_report.pdf"
    if not os.path.exists(report_path):
        logger.warning(f"Model report not found for: {filename}")
        raise HTTPException(status_code=404, detail="Model report not found")
    return FileResponse(report_path, filename="model_report.pdf")

@app.get("/dataset/visualization/{filename}/{viz_type}")
async def get_dataset_visualization(filename: str, viz_type: str):
    viz_dir = "reports/visualizations/"
    if not os.path.exists(viz_dir):
        logger.error("Visualization directory not found")
        raise HTTPException(status_code=500, detail="Visualization directory not found")

    all_files = os.listdir(viz_dir)
    html_files = [f for f in all_files if f.endswith(".html")]

    if viz_type == "all":
        matched = html_files
    elif viz_type == "correlation":
        matched = [f for f in html_files if "scatter" in f]
    elif viz_type == "distribution":
        matched = [f for f in html_files if "distributions" in f]
    elif viz_type == "boxplot":
        matched = [f for f in html_files if "bar_chart" in f]
    elif viz_type == "correlation heatmap":
        matched = [f for f in html_files if "correlation" in f]
    else:
        logger.warning(f"Invalid viz_type requested: {viz_type}")
        raise HTTPException(status_code=400, detail=f"Invalid visualization type: {viz_type}")

    if not matched:
        logger.warning(f"No {viz_type} visualizations found")
        raise HTTPException(status_code=404, detail=f"No {viz_type} visualizations found")

    return JSONResponse(content={"links": [f"/reports/visualizations/{f}" for f in matched]})

@app.get("/model/visualization/{filename}/{viz_type}")
async def get_model_visualization(filename: str, viz_type: str):
    viz_dir = "reports/visualizations/"
    if not os.path.exists(viz_dir):
        logger.error("Visualization directory not found")
        raise HTTPException(status_code=500, detail="Visualization directory not found")

    html_files = [f for f in os.listdir(viz_dir) if f.endswith(".html")]

    if viz_type == "all":
        matched_files = html_files
    elif viz_type == "actual vs predicted":
        matched_files = [f for f in html_files if "actual_vs_predicted" in f]
    elif viz_type == "residuals":
        matched_files = [f for f in html_files if "residuals" in f]
    elif viz_type == "feature importance":
        matched_files = [f for f in html_files if "importance" in f]
    else:
        matched_files = [f for f in html_files if f"{viz_type.replace(' ', '_')}" in f]

    if not matched_files:
        logger.warning(f"No visualizations found for type: {viz_type}")
        raise HTTPException(status_code=404, detail=f"No visualizations found for type: {viz_type}")

    return JSONResponse(content={"links": [f"/reports/visualizations/{f}" for f in matched_files]})

# ===================== Run Server =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
