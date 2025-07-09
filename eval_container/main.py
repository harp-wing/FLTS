# eval_container/main.py

import io
import asyncio
import base64
from functools import wraps
from fastapi import FastAPI, Request                                            # type: ignore
from fastapi.responses import HTMLResponse                                      # type: ignore
from fastapi.templating import Jinja2Templates                                  # type: ignore
from starlette.concurrency import run_in_threadpool                             # type: ignore
import numpy as np                                                              # type: ignore
import pandas as pd                                                             # type: ignore
import matplotlib.pyplot as plt                                                 # type: ignore
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas     # type: ignore

from metrics import *

# Data import handling
# TRAINING PERFORMANCE METRICS
# INFERRED DATA

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
_test_plot = plot_to_base64(test_plot)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    This is the main endpoint for the application
    """

    # Adjustable graph generator
    plot_functions = [
        _test_plot
    ]
    
    # create plots in parallel
    tasks = [run_in_threadpool(func) for func in plot_functions] # type: ignore
    plot_results = await asyncio.gather(*tasks)
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "plots": plot_results},
    )
