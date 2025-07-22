# nonML container

import numpy as np
import pandas as pd
from statsforecast.models import (
    AutoARIMA, # AutoRegressive Integrated Moving Average
    AutoETS, # Exponential Smoothing
    AutoTBATS # Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend components, Seasonal components
    # Consider also AutoMFLES
)
from greykite.framework.templates.autogen.forecast_config import ForecastConfig, MetadataParam
from greykite.framework.templates.forecaster import Forecaster


# -------------------- Statsforecast Models -------------------


# ---------------------- Greykite Models ----------------------
HORIZON = 30 # days

# Generate sample data
data = pd.DataFrame({
    "ts": pd.to_datetime(pd.date_range("2022-01-01", periods=180, freq="D")),
    "y": np.arange(180) + np.sin(np.arange(180) / 20) * 20 + np.random.randn(180) * 5 + 200
})

# Define metadata: Tell Greykite about the data columns and frequency
metadata = MetadataParam(
    time_col="ts",      # The column with timestamps
    value_col="y",      # The column with values to forecast
    freq="D"            # The frequency of the data (D for daily)
)

forecaster = Forecaster()
slkte_result = forecaster.run_forecast_config(
    df=data, 
    config=ForecastConfig(
        model_template="SILVERKITE",
        forecast_horizon=HORIZON,
        metadata_param=metadata
    )
)
proph_result = forecaster.run_forecast_config(
    df=data, 
    config=ForecastConfig(
        model_template="PROPHET",
        forecast_horizon=HORIZON,
        metadata_param=metadata
    )
)

print(np.__version__)
print(pd.__version__)
print(statsforecast.__version__)
print(greykite.__version__)


print("--- Silverkite Model Summary ---")
print(slkte_result.summary())
# slkte_result.plot()
print("\n--- Prophet Model Summary ---")
print(proph_result.summary())
# proph_result.plot()