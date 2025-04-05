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
from dataset_processor_ai import DatasetScriptor
from auto_modeler import AutoModeler

app = FastAPI(title="Pocket Data Scientist API")

# Mount the 'reports' directory to be accessible via the '/reports' URL path
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

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

# ===================== Endpoints =====================

UPLOAD_DIRECTORY = "datasets"

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@app.post("/dataset/upload")
async def upload_file(file: UploadFile = File(...), target_column: str = Form(...)):
    try:
        # Generate a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)

        # Save the file to the specified path
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load the dataset to preprocess column types
        df = pd.read_csv(file_path)

        # Convert all columns to strings (can adjust as necessary)
        df = df.astype(str)

        # Save the cleaned file
        df.to_csv(file_path, index=False)

        # Create a DatasetProcessor instance
        processor = DatasetScriptor(filename, target_column)

        # Store the processor instance, target_class, and metadata in processing_results
        processing_results[filename] = {
            "processor": processor,
            "target_class": target_column,
            "metadata": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
        }

        # Return metadata and preview
        return {
            "filename": filename,
            "message": "File uploaded successfully",
            "preview": generate_preview(file_path),
            "metadata": processing_results[filename]["metadata"]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

def generate_preview(file_path: str):
    try:
        df = pd.read_csv(file_path)
        preview = df.head().to_html()  # Convert first few rows to HTML table
        return preview
    except Exception as e:
        return str(e)

# Expecting filename and target_class in the request body (Recommended)
@app.post("/dataset/clean")
async def clean_dataset(data: Dict):
    filename = data.get("filename")
    target_class = data.get("target_class")
    if not filename or not target_class:
        raise HTTPException(status_code=400, detail="Filename and target_class are required in the request body")

    if filename not in processing_results:
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = processing_results[filename]["processor"]
    results = processor.clean_dataset()
    cleaned_path = processor.save_cleaned_dataset()
    visualizations = processor.generate_visualizations()
    eda_report = processor.generate_eda_report()

    return {
        "message": "Dataset cleaned successfully",
        "cleaned_dataset_path": cleaned_path,
        "visualization_paths": visualizations,
        "report_path": eda_report,
        "cleaning_log": processor.cleaning_log
    }

# Expecting filename as query parameter and config in the request body
@app.post("/dataset/model")
async def train_model(filename: str = Query(...), config: ModelingConfig = Body(...)):
    if filename not in processing_results:
        print("Processing results:", processing_results)  # Debugging line
        raise HTTPException(status_code=404, detail="Dataset not found")

    processor = processing_results[filename]["processor"]
    X_train, y_train, X_test, y_test = processor.prepare_for_modeling(
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
    path = f"reports/visualizations/{viz_type}.html"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model visualization not found")
    return FileResponse(path, filename=f"{viz_type}.html")

# ===================== Run Server =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)