from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import uvicorn
from dataset_processor_ai import DatasetScriptor
from auto_modeler import AutoModeler

app = FastAPI(title="Pocket Data Scientist API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("ml-service/datasets", exist_ok=True)
os.makedirs("ml-service/models", exist_ok=True)
os.makedirs("ml-service/reports", exist_ok=True)
os.makedirs("ml-service/logs", exist_ok=True)

# Store processing results
processing_results = {}

class DatasetMetadata(BaseModel):
    """Metadata for uploaded dataset."""
    filename: str
    rows: int
    columns: int
    column_types: Dict[str, str]
    missing_values: Dict[str, int]

class CleaningConfig(BaseModel):
    """Configuration for data cleaning."""
    handle_missing: bool = True
    remove_outliers: bool = True
    encode_categorical: bool = True
    extract_datetime: bool = True
    standardize_names: bool = True
    remove_duplicates: bool = True

class ModelingConfig(BaseModel):
    """Configuration for automated modeling."""
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    tune_hyperparameters: bool = True

@app.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)) -> DatasetMetadata:
    """
    Upload a dataset file (CSV, Excel, or JSON).
    
    Args:
        file: The dataset file to upload
        
    Returns:
        Dataset metadata
    """
    try:
        # Save the uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"ml-service/datasets/{timestamp}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the dataset
        processor = DatasetScriptor(file_path)
        
        # Generate metadata
        metadata = DatasetMetadata(
            filename=file.filename,
            rows=len(processor.df),
            columns=len(processor.df.columns),
            column_types=processor.column_types,
            missing_values=processor.df.isnull().sum().to_dict()
        )
        
        # Store processor for later use
        processing_results[file.filename] = {
            "processor": processor,
            "file_path": file_path
        }
        
        return metadata
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/clean")
async def clean_dataset(filename: str, target_class: str) -> Dict[str, Any]:
    """
    Clean the uploaded dataset.
    
    Args:
        filename: Name of the uploaded file
        config: Cleaning configuration
        
    Returns:
        Cleaning results
    """
    try:
        if filename not in processing_results:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        processor = processing_results[filename]["processor"]
        
        # Clean the dataset
        dataset_path, dataset = processor.use_dataset_scriptor(target_class=target_class)
        
        # Generate visualizations
        visualization_paths = processor.generate_visualizations()
        
        return {
            "message": "Dataset cleaned successfully",
            "cleaned_dataset_path": dataset_path,
            "visualization_paths": visualization_paths,
            "report_path": "reports/eda_report.pdf",
            "cleaning_log": processor.cleaning_log
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/model")
async def train_model(filename: str, config: ModelingConfig) -> Dict[str, Any]:
    """
    Train models on the cleaned dataset.
    
    Args:
        filename: Name of the uploaded file
        config: Modeling configuration
        
    Returns:
        Modeling results
    """
    try:
        if filename not in processing_results:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        processor = processing_results[filename]["processor"]
        
        # Prepare data for modeling
        X_train, y_train, X_test, y_test = processor.prepare_for_modeling(
            target_column=config.target_column,
            test_size=config.test_size,
            random_state=config.random_state
        )
        
        # Initialize auto modeler
        modeler = AutoModeler(X_train, y_train, X_test, y_test)
        
        # Train and evaluate models
        results = modeler.auto_model(tune_hyperparameters=config.tune_hyperparameters)
        
        return {
            "message": "Models trained successfully",
            "best_model_name": results["best_model_name"],
            "best_score": results["best_score"],
            "model_path": results["model_path"],
            "report_path": results["report_path"],
            "visualization_paths": results["visualization_paths"],
            "training_log": results["training_log"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/report/{filename}")
async def get_report(filename: str) -> FileResponse:
    """
    Get the EDA report for a dataset.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        EDA report file
    """
    try:
        if filename not in processing_results:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        report_path = f"ml-service/reports/eda_report.pdf"
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(report_path, filename="eda_report.pdf")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/visualization/{filename}/{viz_type}")
async def get_visualization(filename: str, viz_type: str) -> FileResponse:
    """
    Get a visualization for a dataset.
    
    Args:
        filename: Name of the uploaded file
        viz_type: Type of visualization
        
    Returns:
        Visualization file
    """
    try:
        if filename not in processing_results:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        viz_path = f"ml-service/reports/visualizations/{viz_type}.html"
        
        if not os.path.exists(viz_path):
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        return FileResponse(viz_path, filename=f"{viz_type}.html")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/report/{filename}")
async def get_model_report(filename: str) -> FileResponse:
    """
    Get the model report for a dataset.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        Model report file
    """
    try:
        if filename not in processing_results:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        report_path = f"ml-service/reports/model_report.pdf"
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(report_path, filename="model_report.pdf")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/visualization/{filename}/{viz_type}")
async def get_model_visualization(filename: str, viz_type: str) -> FileResponse:
    """
    Get a model visualization for a dataset.
    
    Args:
        filename: Name of the uploaded file
        viz_type: Type of visualization
        
    Returns:
        Visualization file
    """
    try:
        if filename not in processing_results:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        viz_path = f"ml-service/reports/visualizations/{viz_type}.html"
        
        if not os.path.exists(viz_path):
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        return FileResponse(viz_path, filename=f"{viz_type}.html")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 