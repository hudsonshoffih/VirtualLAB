from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from io import StringIO, BytesIO
import base64


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

def execute_python_code(code):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    # For capturing plots
    plt.switch_backend('Agg')
    plot_data = None

    try:
        # Define a safe execution environment
        exec_globals = {
            "__builtins__": __builtins__, 
            "np": np, 
            "pd": pd,
            "plt": plt,
            "train_test_split": train_test_split,
            "LinearRegression": LinearRegression
        }

        exec(code, exec_globals)

        # Capture DataFrame output if one exists
        df_html = None
        for var_name, var in exec_globals.items():
            if isinstance(var, pd.DataFrame) and var_name not in ['np', 'pd']:
                df_html = var.to_html(classes="table table-striped")
                break
        
        # Capture matplotlib plots if any
        if plt.get_fignums():
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close('all')  # Close all figures
        
        sys.stdout = old_stdout

        return redirected_output.getvalue(), None, df_html, plot_data

    except Exception as e:
        sys.stdout = old_stdout
        plt.close('all')  # Close all figures in case of error
        return None, str(e), None, None

@app.route('/api/execute', methods=['POST'])
def execute_code():
    cells = request.json.get('cells', [])

    results = []
    for cell in cells:
        output, error, df_html, plot_data = execute_python_code(cell["code"])
        results.append({
            "output": output,
            "error": error,
            "table_html": df_html,
            "plot": plot_data
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

