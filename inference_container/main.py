# # predict_container/main.py

# import pandas as pd
# import numpy as np
# import os

# from predict import predict, load_model
# from shared.data_utils import to_Xy, generate_time_lags, remove_identifiers, to_timeseries_rep

# def run_inference():
#     df = pd.read_csv("../data/ElBorn.csv", parse_dates=["time"])

#     num_lags = 10
#     targets = ["down", "up"]

#     X_df, y_df = to_Xy(df, targets=targets)
#     X_df = generate_time_lags(X_df, num_lags)
#     y_df = generate_time_lags(y_df, num_lags, is_y=True)
#     X_df, y_df = remove_identifiers(X_df, y_df)

#     num_features = X_df.shape[1] // num_lags
#     X_np = to_timeseries_rep(X_df.to_numpy(), num_lags=num_lags, num_features=num_features)
#     y_np = y_df.to_numpy()

#     model = load_model("../outputs/models/lstm.pt", input_dim=num_features, output_dim=y_np.shape[1], num_lags=num_lags)

#     y_pred = predict(model, X_np)

#     os.makedirs("../outputs/predictions", exist_ok=True)
#     np.save("../outputs/predictions/elborn_preds.npy", y_pred)
#     print("✅ Predictions saved to outputs/predictions/elborn_preds.npy")

# if __name__ == "__main__":
#     run_inference()


# inference_container/main.py

import numpy as np
import mlflow
import io
from client_utils import get_file, post_file
import os

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://fastapi_service:8000")

# === Pull test data from MinIO ===
X_test_bytes = get_file(GATEWAY_URL, "preprocessed", "X_test.npy")
X_test = np.load(X_test_bytes)

# === Load model from MLflow ===
mlflow.set_tracking_uri("http://mlflow:5000")
model_uri = "models:/flts-lstm/1"
model = mlflow.pyfunc.load_model(model_uri)

# === Predict ===
y_pred = model.predict(X_test)

# === Push predictions to MinIO ===
preds_bytes = io.BytesIO()
np.save(preds_bytes, y_pred)
preds_bytes.seek(0)
post_file(GATEWAY_URL, "predictions", "elborn_preds.npy", preds_bytes.read())
print("Predictions pushed to MinIO")
