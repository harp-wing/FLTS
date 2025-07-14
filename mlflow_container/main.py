import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
experiments = mlflow.search_experiments()
print("Experiments:", [e.name for e in experiments])
