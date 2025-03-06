from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import traceback
from io import StringIO, BytesIO
import base64
import pickle
import tempfile
from contextlib import redirect_stdout

# Import commonly used ML libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Machine learning imports
from sklearn import (
    linear_model, ensemble, tree, neighbors, svm, 
    neural_network, model_selection, metrics, preprocessing,
    cluster, decomposition, feature_extraction, manifold
)
import xgboost
import lightgbm
import tensorflow as tf
import torch
import joblib

app = Flask(__name__)
CORS(app)

# Session storage to maintain state between cells
SESSIONS = {}
MAX_MEMORY_MB = 500  # Memory limit per session (MB)

def get_session(session_id):
    """Get or create a new session environment"""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            'env': create_execution_environment(),
            'history': [],
            'temp_dir': tempfile.mkdtemp()
        }
    return SESSIONS[session_id]

def create_execution_environment():
    """Create a safe execution environment with ML libraries"""
    return {
        "__builtins__": __builtins__,
        # Data manipulation
        "np": np,
        "pd": pd,
        # Visualization
        "plt": plt,
        "sns": sns,
        # Stats
        "stats": stats,
        # Scikit-learn
        "linear_model": linear_model,
        "ensemble": ensemble,
        "tree": tree,
        "neighbors": neighbors,
        "svm": svm,
        "neural_network": neural_network,
        "model_selection": model_selection,
        "metrics": metrics,
        "preprocessing": preprocessing,
        "cluster": cluster,
        "decomposition": decomposition,
        "feature_extraction": feature_extraction,
        "manifold": manifold,
        # Other ML frameworks
        "xgboost": xgboost,
        "lightgbm": lightgbm,
        "tf": tf,
        "torch": torch,
        "joblib": joblib,
        # Common functions
        "train_test_split": model_selection.train_test_split
    }

def check_memory_usage(session_id):
    """Check if session memory usage exceeds limits"""
    session = SESSIONS.get(session_id)
    if not session:
        return False
        
    # Estimate memory usage (not perfect but gives some protection)
    session_size = 0
    for var_name, var in session['env'].items():
        if var_name.startswith('__'):
            continue
        try:
            if hasattr(var, 'nbytes'):
                session_size += var.nbytes
            elif isinstance(var, pd.DataFrame):
                session_size += var.memory_usage(deep=True).sum()
        except:
            pass
            
    return session_size > (MAX_MEMORY_MB * 1024 * 1024)

def execute_python_code(code, session_id):
    """Execute Python code and maintain state between executions"""
    session = get_session(session_id)
    exec_globals = session['env']
    
    # Configure matplotlib for non-interactive backend
    plt.switch_backend('Agg')
    plt.close('all')
    
    # Create output buffers
    output_buffer = StringIO()
    
    # Variables to capture results
    plot_data = None
    df_html = None
    error = None
    
    try:
        # Execute code with context
        with redirect_stdout(output_buffer):
            exec(code, exec_globals)
        
        # Capture DataFrame outputs
        for var_name, var in exec_globals.items():
            if var_name.startswith('__') or var_name in session['env']:
                continue
                
            if isinstance(var, pd.DataFrame):
                # Take the first dataframe found or largest one
                if df_html is None or (var.shape[0] * var.shape[1]) > 0:
                    df_html = var.head(100).to_html(classes="table table-striped table-hover")
        
        # Capture matplotlib plots
        if plt.get_fignums():
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close('all')
        
        # Check for memory limits
        if check_memory_usage(session_id):
            # If exceeded, we could implement cleanup strategies here
            pass
            
        # Save execution history
        session['history'].append(code)
        
        return output_buffer.getvalue(), error, df_html, plot_data
    
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        plt.close('all')
        return output_buffer.getvalue(), error, None, None
    
    finally:
        output_buffer.close()

@app.route('/api/execute', methods=['POST'])
def execute_code():
    """API endpoint to execute code"""
    data = request.json
    cells = data.get('cells', [])
    session_id = data.get('session_id', 'default')
    
    results = []
    for cell in cells:
        cell_id = cell.get("id")
        code = cell.get("code", "")
        
        output, error, df_html, plot_data = execute_python_code(code, session_id)
        
        results.append({
            "cell_id": cell_id,
            "output": output,
            "error": error,
            "table_html": df_html,
            "plot": plot_data
        })
    
    return jsonify(results)

@app.route('/api/reset_session', methods=['POST'])
def reset_session():
    """Reset a specific session or all sessions"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id:
        if session_id in SESSIONS:
            # Clean up temp files
            try:
                if os.path.exists(SESSIONS[session_id]['temp_dir']):
                    for file in os.listdir(SESSIONS[session_id]['temp_dir']):
                        os.remove(os.path.join(SESSIONS[session_id]['temp_dir'], file))
                    os.rmdir(SESSIONS[session_id]['temp_dir'])
            except:
                pass
            
            # Delete session
            del SESSIONS[session_id]
            return jsonify({"status": "success", "message": f"Session {session_id} reset"})
        return jsonify({"status": "error", "message": "Session not found"})
    else:
        # Reset all sessions
        for sid in list(SESSIONS.keys()):
            try:
                if os.path.exists(SESSIONS[sid]['temp_dir']):
                    for file in os.listdir(SESSIONS[sid]['temp_dir']):
                        os.remove(os.path.join(SESSIONS[sid]['temp_dir'], file))
                    os.rmdir(SESSIONS[sid]['temp_dir'])
            except:
                pass
            
        SESSIONS.clear()
        return jsonify({"status": "success", "message": "All sessions reset"})

@app.route('/api/get_variables', methods=['POST'])
def get_variables():
    """Get variable information for the current session"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    session = get_session(session_id)
    variables = {}
    
    for var_name, var in session['env'].items():
        # Skip built-in and library references
        if var_name.startswith('__') or var_name in create_execution_environment():
            continue
            
        var_info = {
            "type": type(var).__name__
        }
        
        # Add shape for arrays and dataframes
        if hasattr(var, 'shape'):
            var_info["shape"] = str(var.shape)
        elif isinstance(var, list):
            var_info["length"] = len(var)
            
        # Add more specific info based on type
        if isinstance(var, pd.DataFrame):
            var_info["columns"] = var.columns.tolist()
            var_info["dtypes"] = {col: str(dtype) for col, dtype in var.dtypes.items()}
        
        variables[var_name] = var_info
    
    return jsonify({"variables": variables})

@app.route('/api/export_notebook', methods=['POST'])
def export_notebook():
    """Export session history as Jupyter notebook format"""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    session = get_session(session_id)
    
    # Create a simple Jupyter notebook structure
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": i + 1,
                "metadata": {},
                "source": code.split('\n'),
                "outputs": []
            }
            for i, code in enumerate(session['history'])
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return jsonify(notebook)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)