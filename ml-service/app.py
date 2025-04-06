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
from dataset_processor_ai import DatasetScriptor, ReportCreator, ask_gpt_to_identify_irrelevant_columns
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

@app.post("/dataset/upload")
async def upload_file(file: UploadFile = File(...), target_column: str = Form(...)):
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
    df = df.drop(columns=columns_to_remove)
    processor.df = df

    X_train, X_test, y_train, y_test = processor.prepare_for_modeling(
        target_column=target_column,
        test_size=config.test_size,
        random_state=config.random_state
    )

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

    return {
        "message": "Model training complete",
        "best_model_name": results["best_model_name"],
        "best_score": results["best_score"],
        "model_path": results["model_path"],
        "report_path": report_path,
        "visualization_paths": visualization_paths,
        "training_log": results["training_log"]
    }

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
