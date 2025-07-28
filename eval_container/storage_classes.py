from typing import Dict, Optional
import pandas as pd
import threading

class PredictionStore:
    """Thread-safe storage for prediction results from multiple inference containers"""
    def __init__(self):
        self._predictions: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def add_prediction(self, model_name: str, df_pred: pd.DataFrame, metadata: Dict):
        with self._lock:
            self._predictions[model_name] = df_pred
            self._metadata[model_name] = metadata
            print(f"Added prediction for model: {model_name}")
    
    def get_predictions(self) -> Dict[str, pd.DataFrame]:
        with self._lock:
            return self._predictions.copy()
    
    def get_metadata(self) -> Dict[str, Dict]:
        with self._lock:
            return self._metadata.copy()
    
    def get_latest_prediction(self) -> Optional[pd.DataFrame]:
        with self._lock:
            if self._predictions:
                # Return the most recently added prediction
                # Ensure 'timestamp' key exists or handle its absence
                latest_key = max(self._metadata.keys(), key=lambda k: self._metadata[k].get('timestamp', 0))
                return self._predictions[latest_key]
            return None
        
class TruthStore:
    """Thread-safe storage for ground truth data"""
    def __init__(self):
        self._ground_truth: Optional[pd.DataFrame] = None
        self._metadata: Dict = {}
        self._lock = threading.Lock()
    
    def set_ground_truth(self, df_true: pd.DataFrame, metadata: Dict):
        with self._lock:
            self._ground_truth = df_true
            self._metadata = metadata
            print(f"Ground truth data loaded: shape {df_true.shape}")
    
    def get_ground_truth(self) -> Optional[pd.DataFrame]:
        with self._lock:
            return self._ground_truth.copy() if self._ground_truth is not None else None
    
    def get_metadata(self) -> Dict:
        with self._lock:
            return self._metadata.copy()