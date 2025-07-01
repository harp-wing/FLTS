# eda_container/main.py

import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from stats import *

DATA_PATH = "/app/data/dataset"

# Non-interactive backend for headless environments
matplotlib.use('Agg')

# Load the dataset
file_path = os.path.join(DATA_PATH, 'ElBorn.csv')
df = pd.read_csv(file_path)

def table_style(df):
    """
    Apply styling to df_stats for better readability.
    """

    formatters = {
        'mean': '{:.2f}',
        'stdev': '{:.2f}',
        'median': '{:.2f}',
        'mode': '{:.2f}',
        'gmean': '{:.2f}',
        'variance': '{:.2f}',
        'skewness': '{:.3f}',
        'kurtosis': '{:.3f}'
    }

    styler = df.style

    # Iterate through the formatters and apply them to the correct row subset
    for statistic, fmt_string in formatters.items():
        styler = styler.format(
            fmt_string, 
            subset=pd.IndexSlice[statistic, :] # Selects the row by its label
        )

    table_html = (styler
                   .set_properties(**{'border': '1px solid black', 'text-align': 'right'})
                   .set_table_attributes('class="min-w-full divide-y divide-gray-200"')
                   .to_html()
                )
    
    return table_html

table = table_style(stat_analyze(df))
nan_eval = eval_nans(df)


# Create the FastAPI app instance
app = FastAPI(
    title="Matplotlib Plot Server",
    description="An API to dynamically generate and serve Matplotlib plots."
)

# --- A dictionary to map plot names to their creation functions ---
PLOTS = {
    'distance_correlation': {
        'title':'Distance Correlation Pair Plot',
        'generator': corrplot,
    },
    'pca_plot': {
        'title': 'PCA Plot',
        'generator': pca_plot,
    },
    'pacf_plot': {
        'title': 'Partial Autocorrelation Plot',
        'generator': pacf_plot,
    }
}


# --- A single, dynamic endpoint for generating plots ---
@app.get("/plot/{plot_slug}.png")
def plot_png(plot_slug: str):
    """
    Generates and returns a specific plot based on its name.
    """

    plot_info = PLOTS.get(plot_slug)

    if not plot_info:
        raise HTTPException(status_code=404, detail="Plot not found")

    # Look up the plot generator function from our dictionary
    generator_func = plot_info['generator']

    try:
        fig = generator_func(df)
    except Exception as e:
        print(f"Error generating plot {plot_slug}")
        exec.traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")
    

    # Convert plot to PNG in memory
    with io.BytesIO() as buf:
        FigureCanvas(fig).print_png(buf)
        image_bytes = buf.getvalue()
    
    # Return the image as a StreamingResponse
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

# --- Main HTML page to display plots ---
@app.get("/", response_class=HTMLResponse)
def index():
    """
    Serves the main HTML page which displays links to the available plots.
    """

    # Dynamically build the list of images to display
    img_tags = ""
    for slug, plot_info in PLOTS.items():
        title = plot_info['title']
        img_tags += f"""
            <h2>{title}</h2>
            <p>Available at: <code>/plot/{slug}.png</code></p>
            <img src="/plot/{slug}.png" alt="Plot of {title}" style="border: 1px solid #ccc; max-width: 1000px;">
            <hr>
        """

    html_content = f"""
    <html>
        <head>
            <title>Statistics Dashboard</title>
            <style>
                body {{ font-family: sans-serif; padding: 2em; }}
                code {{ background-color: #eee; padding: 3px 5px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <h1>Statistics Dashboard</h1>
            <p>Various properties of the dataset are visualized below.</p>
            <h2>Summary of Data Quality</h2>
            {nan_eval}
            <h2>Feature Statistics</h2>
            {table}
            {img_tags}
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)