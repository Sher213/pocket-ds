import difflib
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import pandas as pd
from typing import Dict
from datetime import datetime
import os
from dataset_processor_ai import DatasetScriptor, ReportCreator, ask_gpt_to_identify_irrelevant_columns
from auto_modeler import AutoModeler
from typing import List, Optional

app = FastAPI(title="Pocket Data Scientist API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory Setup
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports/visualizations", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Mount the 'reports' directory to be accessible via the '/reports' URL path
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# In-memory cache to store processing results
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
    """
    Find the most likely target column from a list of DataFrame columns,
    even if user input is not exact.
    """
    user_input_clean = user_input.strip().lower()
    column_map = {col: col.strip().lower() for col in df_columns}

    # Exact match
    for orig_col, clean_col in column_map.items():
        if clean_col == user_input_clean:
            return orig_col

    # Fuzzy match
    best_matches = difflib.get_close_matches(user_input_clean, column_map.values(), n=1, cutoff=cutoff)
    if best_matches:
        for orig_col, clean_col in column_map.items():
            if clean_col == best_matches[0]:
                return orig_col

    return None

def generate_preview(file_path: str):
    try:
        df = pd.read_csv(file_path)
        preview = df.head().to_html()  # Convert first few rows to HTML table
        return preview
    except Exception as e:
        return str(e)

# ===================== Endpoints =====================

UPLOAD_DIRECTORY = "datasets"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.post("/dataset/upload")
async def upload_file(file: UploadFile = File(...), target_column: str = Form(...)):
    try:
        # Generate a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)

        # Save uploaded file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load and clean the dataset
        df = pd.read_csv(file_path)
        df = df.astype(str)  # Optional: standardize types
        df.to_csv(file_path, index=False)  # Save cleaned version

        # Find target column
        matched_column = find_target_column(target_column, df.columns.tolist())
        print(f"Matched column: {matched_column}")
        if matched_column is None:
            return JSONResponse(status_code=400, content={
                "detail": f"Target column '{target_column}' not found or recognized in dataset."
            })

        # Create DatasetProcessor (assuming this class is defined elsewhere)
        processor = DatasetScriptor(filename, matched_column)

        # Store processor and metadata
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

        # Return response
        return {
            "filename": filename,
            "message": "File uploaded successfully",
            "preview": generate_preview(file_path),
            "metadata": processing_results[filename]["metadata"]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# Expecting filename and target_class in the request body (Recommended)
@app.post("/dataset/clean")
async def clean_dataset(data: Dict):
    filename = data.get("filename")
    target_class = data.get("target_class")
    use_gpt = data.get("use_gpt", True)

    if not filename or not target_class:
        raise HTTPException(status_code=400, detail="Filename and target_class are required")

    if filename not in processing_results:
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = processing_results[filename]["processor"]
    results = processor.clean_dataset()
    cleaned_path = processor.save_cleaned_dataset()
    visualizations = processor.generate_visualizations()
    eda_report_pdf_path, eda_report_txt_path = processor.generate_eda_report()

    # Conditionally generate GPT summary
    if use_gpt:
        report_creator = ReportCreator(cleaned_path)
        report_creator.generate_eda_report_with_gpt(eda_report_txt_path)
        report_creator.save_report_as_pdf(eda_report_txt_path, eda_report_pdf_path)

    return {
        "message": "Dataset cleaned successfully",
        "cleaned_dataset_path": cleaned_path,
        "visualization_paths": visualizations,
        "report_path": eda_report_pdf_path,
        "cleaning_log": processor.cleaning_log
    }

# Expecting filename as query parameter and config in the request body
@app.post("/dataset/model")
async def train_model(filename: str = Query(...), config: ModelingConfig = Body(...)):
    if filename not in processing_results:
        print("Processing results:", processing_results)  # Debugging line
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = processing_results[filename]["processor"]
    df = pd.read_csv(processor.cleaned_dataset_path)  # Load cleaned dataset

    columns_to_remove = ask_gpt_to_identify_irrelevant_columns(
        df.columns.tolist(), df.sample(n=min(10, len(df))), config.target_column
    )

    # Drop the irrelevant columns before training
    df = df.drop(columns=columns_to_remove)

    # Update processor's cleaned dataframe
    processor.df = df

    processor = processing_results[filename]["processor"]
    X_train, X_test, y_train, y_test = processor.prepare_for_modeling(
        target_column=config.target_column,
        test_size=config.test_size,
        random_state=config.random_state
    )

    modeler = AutoModeler(X_train, y_train, X_test, y_test)

    results = modeler.auto_model(tune_hyperparameters=config.tune_hyperparameters)

    return {
        "message": "Model training complete",
        "best_model_name": results["best_model_name"],
        "best_score": results["best_score"],
        "model_path": results["model_path"],
        "report_path": results["report_path"],
        "visualization_paths": results["visualization_paths"],
        "training_log": results["training_log"]
    }

@app.get("/dataset/report/{filename}")
async def get_eda_report(filename: str):
    report_path = "reports/eda_report.pdf"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="EDA report not found")
    return FileResponse(report_path, filename="eda_report.pdf")

@app.get("/model/report/{filename}")
async def get_model_report(filename: str):
    report_path = "reports/model_report.pdf"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Model report not found")
    return FileResponse(report_path, filename="model_report.pdf")

@app.get("/dataset/visualization/{filename}/{viz_type}")
async def get_dataset_visualization(filename: str, viz_type: str):
    viz_dir = "reports/visualizations/"
    if not os.path.exists(viz_dir) or not os.path.isdir(viz_dir):
        raise HTTPException(status_code=500, detail="Visualization directory not found")

    all_files = os.listdir(viz_dir)
    html_files = [f for f in all_files if f.endswith(".html")]
    visualization_links = []

    if viz_type == "all":
        visualization_links = [f"/reports/visualizations/{f}" for f in html_files]
    elif viz_type == "correlation":
        visualization_links = [f"/reports/visualizations/{f}" for f in html_files if "scatter" in f]
    elif viz_type == "distribution":
        visualization_links = [f"/reports/visualizations/{f}" for f in html_files if "distributions" in f]
    elif viz_type == "boxplot":
        visualization_links = [f"/reports/visualizations/{f}" for f in html_files if "bar_chart" in f]
    else:
        raise HTTPException(status_code=400, detail=f"Invalid visualization type: {viz_type}")

    if not visualization_links:
        raise HTTPException(status_code=404, detail=f"No {viz_type} visualizations found")

    return JSONResponse(content={"links": visualization_links})

@app.get("/model/visualization/{filename}/{viz_type}")
async def get_model_visualization(filename: str, viz_type: str):
    viz_dir = "reports/visualizations/"
    if not os.path.exists(viz_dir) or not os.path.isdir(viz_dir):
        raise HTTPException(status_code=500, detail="Visualization directory not found")

    all_files = os.listdir(viz_dir)
    html_files = [f for f in all_files if f.endswith(".html")]
    # Determine which files to return based on viz_type
    if viz_type == "all":
        matched_files = html_files
    elif viz_type == "actual vs predicted":
        matched_files = [f for f in html_files if "actual_vs_predicted" in f]
    elif viz_type == "residuals":
        matched_files = [f for f in html_files if "residuals" in f]
    elif viz_type == "feature importance":
        matched_files = [f for f in html_files if "importance" in f]
    else:
        # Assume it's a specific file, like "confusion_matrix" or a specific plot
        specific_path = os.path.join(viz_dir, f"{viz_type}.html")
        if not os.path.exists(specific_path):
            raise HTTPException(status_code=404, detail="Model visualization not found")
        return FileResponse(specific_path, filename=f"{viz_type}.html")

    if not matched_files:
        raise HTTPException(status_code=404, detail=f"No visualizations found for type: {viz_type}")

    visualization_links = [f"/reports/visualizations/{f}" for f in matched_files]
    return JSONResponse(content={"links": visualization_links})

# ===================== Run Server =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)