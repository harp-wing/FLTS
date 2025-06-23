# eda_container/main.py

import pandas as pd
from explore import describe_data, plot_feature_trends, correlation_matrix

def run_eda(csv_path="data/ElBorn.csv", output_dir="outputs/eda"):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    describe_data(df)
    plot_feature_trends(df, output_dir=output_dir)
    correlation_matrix(df, output_path=f"{output_dir}/correlation_matrix.png")

if __name__ == "__main__":
    run_eda()
