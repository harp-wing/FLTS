import pandas as pd
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.pyfunc import PythonModel
from prophet import Prophet
from typing import List, Dict, Any, Optional, Union, Tuple
from prophet.diagnostics import cross_validation, performance_metrics
from statsforecast import StatsForecast
from statsforecast.models import (                                                                                              # type: ignore
    AutoARIMA, # AutoRegressive Integrated Moving Average
    AutoETS, # Exponential Smoothing
    AutoTheta,
    AutoMFLES,
    AutoTBATS # Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend components, Seasonal components
)

class ProphetMultiFeatureModel(PythonModel):
    """
    Wrapper class to bundle multiple Prophet models for different features
    """
    
    def __init__(self):
        self.models: Dict[str, Prophet] = {}
        self.feature_columns: List[str] = []
        self.model_params: Dict[str, Any] = {}
        self.ds_column: str = "ds"
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str], model_params: Dict[str, Any]):
        """
        Fit Prophet models for each feature column
        
        Args:
            df: DataFrame with 'ds' column and feature columns
            feature_columns: List of column names to forecast
            model_params: Dictionary of parameters for Prophet model
        """
        self.feature_columns = feature_columns
        self.model_params = model_params
        
        for column in feature_columns:
            if column == self.ds_column:
                continue
                
            # Prepare data for this feature
            prophet_df = df[[self.ds_column, column]].rename(columns={column: "y"})
            prophet_df = prophet_df.dropna()  # Prophet requires no missing values
            
            # Create Prophet model with parameters
            model = Prophet(
                growth=model_params.get("growth", "linear"),
                n_changepoints=model_params.get("n_changepoints", 25),
                changepoint_range=model_params.get("changepoint_range", 0.8),
                yearly_seasonality=model_params.get("yearly_seasonality", "auto"),
                weekly_seasonality=model_params.get("weekly_seasonality", "auto"),
                daily_seasonality=model_params.get("daily_seasonality", "auto"),
                seasonality_mode=model_params.get("seasonality_mode", "additive"),
                seasonality_prior_scale=model_params.get("seasonality_prior_scale", 10.0),
                holidays_prior_scale=model_params.get("holidays_prior_scale", 10.0),
                changepoint_prior_scale=model_params.get("changepoint_prior_scale", 0.05)
            )
            
            # Add country holidays if specified
            if model_params.get("country"):
                model.add_country_holidays(country_name=model_params["country"])
            
            # Fit the model
            model.fit(prophet_df)
            self.models[column] = model
    
    def predict(self, context, model_input, params=None):
        """
        Generate predictions for all features
        
        Args:
            context: MLflow context (unused)
            model_input: DataFrame with 'ds' column for prediction dates
            params: Optional parameters (unused)
            
        Returns:
            DataFrame with predictions for all features
        """
        if isinstance(model_input, pd.DataFrame):
            future_df = model_input
        else:
            # Handle other input types if needed
            future_df = pd.DataFrame(model_input)
        
        predictions = {}
        predictions[self.ds_column] = future_df[self.ds_column]
        
        for column in self.feature_columns:
            if column == self.ds_column:
                continue
                
            if column in self.models:
                forecast = self.models[column].predict(future_df[[self.ds_column]])
                predictions[f"{column}_yhat"] = forecast["yhat"]
                predictions[f"{column}_yhat_lower"] = forecast["yhat_lower"]
                predictions[f"{column}_yhat_upper"] = forecast["yhat_upper"]
        
        return pd.DataFrame(predictions)
    
    def get_cross_validation_metrics(self, df: pd.DataFrame, horizon: str) -> Dict[str, float]:
        """
        Perform cross-validation for all features and return aggregated metrics
        """
        all_metrics = {}
        
        for column in self.feature_columns:
            if column == self.ds_column or column not in self.models:
                continue
                
            try:
                prophet_df = df[[self.ds_column, column]].rename(columns={column: "y"})
                prophet_df = prophet_df.dropna()
                
                cv_results = cross_validation(self.models[column], horizon=horizon)
                df_p = performance_metrics(cv_results)
                
                if df_p is not None:
                    all_metrics[f"{column}_mae"] = df_p['mae'].mean()
                    all_metrics[f"{column}_mape"] = df_p['mape'].mean()
                    all_metrics[f"{column}_rmse"] = df_p['rmse'].mean()
                else:
                    all_metrics[f"{column}_mae"] = None
                    all_metrics[f"{column}_mape"] = None
                    all_metrics[f"{column}_rmse"] = None
            except Exception as e:
                print(f"Cross-validation failed for {column}: {e}")
                all_metrics[f"{column}_mae"] = None
        
        return all_metrics


class StatsForecastMultiFeatureModel(PythonModel):
    """
    Wrapper class to bundle multiple StatsForecast models for different features
    """
    
    def __init__(self):
        self.models: Dict[str, StatsForecast] = {}
        self.feature_columns: List[str] = []
        self.model_params: Dict[str, Any] = {}
        self.model_type: str = ""
        self.exog_df: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame, feature_columns: List[str], model_params: Dict[str, Any], exog_columns: List[str]=[]):
        """
        Fit StatsForecast models for each feature column
        
        Args:
            df: DataFrame with 'ds', 'unique_id' columns and feature columns
            feature_columns: List of column names to forecast
            exog_columns: List of exogenous feature columns
            model_params: Dictionary of parameters for StatsForecast models
            freq: Frequency string for StatsForecast
            model_type: Type of model (AUTOARIMA, AUTOETS, etc.)
        """
        self.feature_columns = feature_columns
        self.model_params = model_params
        self.model_type = model_params.get("model_type", "undefined")
        self.exog_df = df[exog_columns] if exog_columns else None
        
        # Create model based on type and parameters
        season_length = model_params.get("season_length", [1])
        sl = season_length[0] if isinstance(season_length, list) else season_length
        
        models_dict = {
            "AUTOARIMA": AutoARIMA(season_length=sl),
            "AUTOETS": AutoETS(season_length=sl),
            "AUTOTHETA": AutoTheta(season_length=sl),
            "AUTOMFLES": AutoMFLES(season_length=season_length, test_size=model_params.get("output_sequence_length", 1)),
            "AUTOTBATS": AutoTBATS(season_length=season_length)
        }
        
        for column in feature_columns:
            if column in ["ds", "unique_id"]:
                continue
            
            print(f"Fitting {self.model_type} model for feature: {column}")

            # Prepare data for this feature
            feature_df = df[["ds", "unique_id", column] + exog_columns].rename(columns={column: "y"})
            
            # Apply downsampling if specified
            if model_params.get("downsampling", "0") != "0":
                feature_df = feature_df.set_index("ds").resample(
                    model_params["downsampling"]).mean().reset_index()
            
            # Create and fit StatsForecast model
            sf = StatsForecast(
                models=[models_dict[self.model_type]],
                freq=model_params.get("frequency", 0),
                n_jobs=-1,
            )

            sf.fit(feature_df) 
            self.models[column] = sf
    
    def predict(self, context, model_input, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate predictions for all features.

        Args:
            context: MLflow context (unused).
            model_input:
                - "h": forecast horizon (int, default=1)
                - "exog": exogenous DataFrame (optional but necessary if the model was fitted with exogenous variables)
                - "level": list of confidence intervals (optional)
            params: Optional extra parameters.

        Returns:
            DataFrame with forecasts for all feature columns.
        """
        # Extract parameters
        h: int = 1
        X: Optional[pd.DataFrame] = None
        level: Optional[List[int]] = None

        if isinstance(model_input, dict):
            h = model_input.get("h", 1)
            X = model_input.get("X", None)
            if X is not None:
                X = X.copy()
            level = model_input.get("level", None)

        if params:
            if h == 1 and "h" in params:
                h = params["h"]
            if level is None and "level" in params:
                level = params["level"]

        if not level:
            level = None

        if X is not None:     
            X.loc[:, "unique_id"] = "1"
            X.index.rename("ds", inplace=True)
            X = X.reset_index()

        all_predictions = []

        for column in self.feature_columns:
            if column in ["ds", "unique_id"] or column not in self.models:
                continue

            forecast: pd.DataFrame = self.models[column].predict(h=h, X_df=X, level=level)  # type: ignore

        # Extract the forecast series and align on time index
            match = next((c for c in forecast.columns if c.lower() == self.model_type.lower()), None)
            if match:
                forecast = forecast.rename(columns={match: column})

            series = forecast[forecast.columns.difference(["unique_id"], sort=False)]

            series = series.set_index("ds")
            all_predictions.append(series)

        if all_predictions:
            # Join on the "ds" index to ensure time alignment
            df_predictions = pd.concat(all_predictions, axis=1)

            # Reorder columns to match the original input DataFrame
            df_predictions = df_predictions[self.feature_columns]

            return df_predictions
        else:
            return pd.DataFrame()

    def get_signature(self) -> ModelSignature: # Defining a signature is kind of a bait, mlflow is REALLY restrictive about it but leaving this here just in case
        """
        Get the model signature for MLflow logging.
        
        Returns:
            ModelSignature: Signature of the model inputs and outputs.
        """
        from mlflow.models import infer_signature

        example_input = {"h:": 10, "X": pd.DataFrame(), "level": [95]}

        # if self.exog_df is not None:
        #     example_input = (self.exog_df.head(1).copy(), {"h": 10})
        # else:
        #     example_input = (pd.DataFrame(), {"h": 10})
        # return infer_signature(example_input, pd.DataFrame(columns=self.feature_columns))

        # Create example input as single DataFrame with metadata
        if self.exog_df is not None:
            example_input["X"] = self.exog_df.head(1).copy()
    
        return infer_signature(example_input, pd.DataFrame(columns=self.feature_columns))
