# eval_container/main.py

import io
import asyncio
import base64
from functools import wraps
from client_utils import get_file
from fastapi import FastAPI, Request                                            # type: ignore
from fastapi.responses import HTMLResponse                                      # type: ignore
from fastapi.templating import Jinja2Templates                                  # type: ignore
from starlette.concurrency import run_in_threadpool                             # type: ignore
import numpy as np                                                              # type: ignore
import pandas as pd                                                             # type: ignore
import pyarrow.parquet as pq                                                    # type: ignore
import pyarrow as pa                                                            # type: ignore
import mlflow                                                                   # type: ignore
import matplotlib.pyplot as plt                                                 # type: ignore
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas     # type: ignore

from metrics import *

# Data import handling
# TRAINING PERFORMANCE METRICS
# INFERRED DATA
FASTAPI_URL = "http://fastapi-app:8000"
BUCKET = "dataset"
OBJECT_NAME = "PobleSec_test.csv"
MLFLOW_TRACKING_URI = "localhost:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Pickle stream vs Parquet???

y_true_bytes = get_file(FASTAPI_URL, BUCKET, OBJECT_NAME)
if isinstance(y_true_bytes, io.BytesIO):
    df_true = pd.read_csv(y_true_bytes, parse_dates=["time"])

# y_true_bytes = get_file(FASTAPI_URL, "processed-data", "test_processed_data.parquet")
# table = pq.read_table(source=y_true_bytes)
# df_true = table.to_pandas()

parquet_bytes = get_file(FASTAPI_URL, "predictions", "LSTM.parquet")
table = pq.read_table(source=parquet_bytes)
df_pred = table.to_pandas()

print(df_true.index)
print(df_true)
print(df_pred)

df_true = df_true.iloc[:df_pred.shape[0], :]

supported = ["LSTM"] # supported models

# for model in models:
#     if model in supported:
#         eval_forecast(inferences[model])


# Create an instance of FastAPI
app = FastAPI()

# This will look for a directory named "templates" in the same directory as main.py
templates = Jinja2Templates(directory="templates")

def plot_to_base64(func):
    """
    A decorator that takes a plot-generating function,
    and handles the boilerplate of saving it to a BytesIO buffer,
    encoding it to base64, and closing the figure.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the original plot function to get the figure and title
        fig, title = func(*args, **kwargs)
        
        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        
        # Encode the buffer's content to a base64 string
        data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # IMPORTANT: Close the figure to free up memory
        plt.close(fig)
        
        # Return the formatted dictionary
        return {"title": title, "data": data}
    return wrapper

# Decorate the metrics plotting functions (Doing it here so the metrics function maintain figure return)
_plot_predictions = plot_to_base64(plot_predictions)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    This is the main endpoint for the application
    """

    # Adjustable graph generator
    PLOTS = {
        "down" : {
            "title": "Comparison of Actual vs Predicted Down",
            "generator": lambda: _plot_predictions(df_true.index, df_true["down"], df_pred["down"], "Comparison of Actual vs Predicted Down")
        },
        "up" : {
            "title": "Comparison of Actual vs Predicted Up",
            "generator": lambda: _plot_predictions(df_true.index, df_true["up"], df_pred["up"], "Comparison of Actual vs Predicted Up")
        },
        "rnti_count" : {
            "title": "Comparison of Actual vs Predicted RNTI Count",
            "generator": lambda: _plot_predictions(df_true.index, df_true["rnti_count"], df_pred["rnti_count"], "Comparison of Actual vs Predicted RNTI Count")
        },
    }

    plot_functions = [plot_info["generator"] for plot_info in PLOTS.values()]
    
    # create plots in parallel
    tasks = [run_in_threadpool(func) for func in plot_functions] # type: ignore
    plot_results = await asyncio.gather(*tasks)
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, 
         "title": "Evaluation Dashboard", 
         "plots": plot_results},
    )
