from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import warnings
from io import StringIO
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, make_blobs, load_breast_cancer

app = Flask(__name__)
CORS(app)

# Store execution contexts for each session
execution_contexts = {}

def execute_python_code(code, algorithm=None, session_id=None, preserve_context=False):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    # Suppress matplotlib warnings about non-GUI backend
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    plt.figure()  # Create a new figure
    
    try:
        # Define a safe execution environment with algorithm-specific imports
        if preserve_context and session_id in execution_contexts:
            # Use existing context
            exec_globals = execution_contexts[session_id]
        else:
            # Create new context
            exec_globals = {
                "__builtins__": __builtins__, 
                "np": np, 
                "pd": pd,
                "plt": plt,
                "sns": sns,
                "train_test_split": train_test_split,
                "StandardScaler": StandardScaler,
                "confusion_matrix": confusion_matrix,
                "classification_report": classification_report,
                "silhouette_score": silhouette_score,
                "warnings": warnings,
                "load_iris": load_iris,
                "make_blobs": make_blobs,
                "load_breast_cancer": load_breast_cancer
            }
            
            # Add algorithm-specific libraries
            if algorithm == 'linear-regression':
                exec_globals["LinearRegression"] = LinearRegression
            elif algorithm == 'svm':
                exec_globals["SVC"] = SVC
            elif algorithm == 'kmeans':
                exec_globals["KMeans"] = KMeans
        
        # Add this line to the beginning of the code to suppress warnings
        modified_code = "warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')\n" + code
        
        exec(modified_code, exec_globals)
        
        # Store the execution context for future use if session_id is provided
        if session_id:
            execution_contexts[session_id] = exec_globals
        
        # Capture DataFrame output if one exists
        df_html = None
        for var_name, var in exec_globals.items():
            if isinstance(var, pd.DataFrame) and var_name != 'pd':
                df_html = var.to_html(classes="table table-striped table-hover")
                break
        
        # Capture plot if one was created
        plot_data = None
        if plt.get_fignums():
            plt.tight_layout()
            from io import BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()  # Close the figure to free memory
        
        sys.stdout = old_stdout
        
        return redirected_output.getvalue(), None, df_html, plot_data
        
    except Exception as e:
        sys.stdout = old_stdout
        plt.close()  # Close any open figures
        return None, str(e), None, None

@app.route('/api/execute', methods=['POST'])
def execute_code():
    data = request.json
    cells = data.get('cells', [])
    algorithm = data.get('algorithm')
    session_id = data.get('session_id', request.remote_addr)  # Use IP as default session ID
    
    results = []
    for cell in cells:
        preserve_context = cell.get('preserveContext', False)
        output, error, df_html, plot_data = execute_python_code(
            cell["code"], 
            algorithm, 
            session_id, 
            preserve_context
        )
        
        results.append({
            "output": output,
            "error": error,
            "table_html": df_html,
            "plot": plot_data
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

