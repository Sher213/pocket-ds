from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import joblib
import os
from datetime import datetime

app = FastAPI(title="Pocket Data Scientist ML Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing models and datasets
os.makedirs("models", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

class DatasetMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None

class TrainingRequest(BaseModel):
    dataset_id: str
    model_type: str
    target_column: str
    feature_columns: List[str]
    test_size: float = 0.2
    random_state: Optional[int] = None
    hyperparameters: Dict[str, Union[str, int, float]] = {}

class PredictionRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Union[str, int, float]]]

@app.post("/dataset/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),  # Extract name as a form field
    description: str = Form(...)  # Extract description as a form field
):
    try:
        # Save the file
        file_path = f"datasets/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Read and validate the dataset
        df = pd.read_csv(file_path)  # Add support for other formats if needed

        return {
            "message": "Dataset uploaded successfully",
            "dataset_id": os.path.basename(file_path),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "preview": df.head().to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train")
async def train_model(request: TrainingRequest):
    try:
        # Load dataset
        dataset_path = f"datasets/{request.dataset_id}"
        df = pd.read_csv(dataset_path)

        # Prepare data
        X = df[request.feature_columns]
        y = df[request.target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=request.test_size,
            random_state=request.random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = None
        if request.model_type == "linear_regression":
            model = LinearRegression(**request.hyperparameters)
        elif request.model_type == "logistic_regression":
            model = LogisticRegression(**request.hyperparameters)
        elif request.model_type == "random_forest":
            model = RandomForestRegressor(**request.hyperparameters)
        elif request.model_type == "xgboost":
            model = xgb.XGBRegressor(**request.hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {request.model_type}")

        model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        # Save model and scaler
        model_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.model_type}"
        model_path = f"models/{model_id}"
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(model, f"{model_path}/model.joblib")
        joblib.dump(scaler, f"{model_path}/scaler.joblib")
        
        # Save model metadata
        metadata = {
            "feature_columns": request.feature_columns,
            "target_column": request.target_column,
            "model_type": request.model_type,
            "hyperparameters": request.hyperparameters,
            "metrics": {
                "train_score": train_score,
                "test_score": test_score
            }
        }
        
        return {
            "model_id": model_id,
            "metrics": metadata["metrics"],
            "feature_importance": getattr(model, "feature_importances_", None)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{model_id}")
async def predict(model_id: str, request: PredictionRequest):
    try:
        # Load model and scaler
        model_path = f"models/{model_id}"
        model = joblib.load(f"{model_path}/model.joblib")
        scaler = joblib.load(f"{model_path}/scaler.joblib")

        # Prepare input data
        input_df = pd.DataFrame(request.data)
        input_scaled = scaler.transform(input_df)

        # Make predictions
        predictions = model.predict(input_scaled)

        return {
            "predictions": predictions.tolist(),
            "model_id": model_id
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def list_models():
    try:
        models = []
        for model_id in os.listdir("models"):
            model_path = f"models/{model_id}"
            if os.path.isdir(model_path):
                # Load model metadata
                model = joblib.load(f"{model_path}/model.joblib")
                models.append({
                    "id": model_id,
                    "type": type(model).__name__,
                    "created": datetime.fromtimestamp(
                        os.path.getctime(f"{model_path}/model.joblib")
                    ).isoformat()
                })
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 