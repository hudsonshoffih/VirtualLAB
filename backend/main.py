from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import asyncio
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from .algorithms import (
    perform_eda,
    train_linear_regression,
    train_logistic_regression,
    train_svm,
    train_random_forest,
    perform_kmeans,
    perform_pca
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeExecutionRequest(BaseModel):
    code: str
    algorithm: str
    dataset: str
    parameters: Dict[str, Any]

class AlgorithmRequest(BaseModel):
    algorithm: str
    dataset: str
    parameters: Dict[str, Any]

# In-memory cache for datasets and results
cache = {}

@app.post("/api/execute")
async def execute_code(request: CodeExecutionRequest):
    try:
        # Create a secure execution environment
        local_vars = {}
        global_vars = {
            'np': np,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
        }

        # Execute the code
        exec(request.code, global_vars, local_vars)

        # Get the results
        figures = []
        for var in local_vars.values():
            if isinstance(var, (px.Figure, go.Figure)):
                figures.append(var.to_json())

        return {
            "success": True,
            "figures": figures,
            "variables": {
                k: str(v) for k, v in local_vars.items()
                if not k.startswith('_') and not callable(v)
            }
        }
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            
            # Process the algorithm request
            result = await process_algorithm(data)
            
            # Send back the results
            await websocket.send_json(result)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

async def process_algorithm(data: Dict[str, Any]):
    algorithm = data.get("algorithm")
    dataset = data.get("dataset")
    parameters = data.get("parameters", {})

    # Load or get dataset from cache
    df = load_dataset(dataset)

    if algorithm == "eda":
        return await perform_eda(df, parameters)
    elif algorithm == "linear_regression":
        return await train_linear_regression(df, parameters)
    elif algorithm == "logistic_regression":
        return await train_logistic_regression(df, parameters)
    elif algorithm == "svm":
        return await train_svm(df, parameters)
    elif algorithm == "random_forest":
        return await train_random_forest(df, parameters)
    elif algorithm == "kmeans":
        return await perform_kmeans(df, parameters)
    elif algorithm == "pca":
        return await perform_pca(df, parameters)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algorithm}")

def load_dataset(dataset_name: str) -> pd.DataFrame:
    # Check cache first
    if dataset_name in cache:
        return cache[dataset_name]

    # Load built-in datasets
    if dataset_name == "iris":
        df = sns.load_dataset("iris")
    elif dataset_name == "boston":
        from sklearn.datasets import load_boston
        data = load_boston()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}")

    # Cache the dataset
    cache[dataset_name] = df
    return df

@app.get("/api/datasets")
async def get_available_datasets():
    return {
        "datasets": [
            {
                "name": "iris",
                "description": "Classic iris flowers dataset",
                "type": "classification",
                "features": 4,
                "samples": 150
            },
            {
                "name": "boston",
                "description": "Boston housing prices dataset",
                "type": "regression",
                "features": 13,
                "samples": 506
            },
            {
                "name": "breast_cancer",
                "description": "Breast cancer wisconsin dataset",
                "type": "classification",
                "features": 30,
                "samples": 569
            }
        ]
    }

@app.get("/api/algorithms")
async def get_available_algorithms():
    return {
        "algorithms": [
            {
                "name": "eda",
                "description": "Exploratory Data Analysis",
                "parameters": [
                    {"name": "plot_type", "type": "string", "options": ["histogram", "boxplot", "scatter", "correlation"]},
                    {"name": "features", "type": "array", "description": "Features to analyze"}
                ]
            },
            {
                "name": "linear_regression",
                "description": "Linear Regression",
                "parameters": [
                    {"name": "test_size", "type": "float", "default": 0.2},
                    {"name": "random_state", "type": "integer", "default": 42}
                ]
            },
            # Add other algorithms...
        ]
    }

