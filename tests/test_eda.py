import pandas as pd
from eda_container.explore import describe_data, plot_feature_trends, correlation_matrix

def test_eda_on_sample_data(tmp_path):
    df = pd.read_csv("data/ElBorn.csv", parse_dates=["time"])

    describe_data(df)  # smoke test: should not raise
    plot_feature_trends(df, output_dir=tmp_path / "eda")
    correlation_matrix(df, output_path=tmp_path / "eda" / "corr.png")

    assert (tmp_path / "eda" / "down_trend.png").exists()
    assert (tmp_path / "eda" / "corr.png").exists()
