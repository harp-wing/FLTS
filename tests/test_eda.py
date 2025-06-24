import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda_container.explore import describe_data, plot_feature_trends, correlation_matrix

def test_describe_data_smoke(capsys):
    df = pd.read_csv("data/ElBorn.csv", parse_dates=["time"])
    describe_data(df)
    out = capsys.readouterr().out
    assert "Basic Info" in out
    assert "Descriptive Statistics" in out
    assert "Date Range" in out

def test_plot_feature_trends(tmp_path):
    df = pd.read_csv("data/ElBorn.csv", parse_dates=["time"])
    output_dir = tmp_path / "eda"

    plot_feature_trends(df, output_dir=output_dir)

    expected_files = ["down_trend.png", "up_trend.png", "rnti_count_trend.png"]
    for file in expected_files:
        assert (output_dir / file).exists()

def test_correlation_matrix_output(tmp_path):
    df = pd.read_csv("data/ElBorn.csv", parse_dates=["time"])
    out_path = tmp_path / "corr.png"

    correlation_matrix(df, output_path=out_path)
    assert out_path.exists()

def test_all_plots_run_without_error():
    df = pd.read_csv("data/ElBorn.csv", parse_dates=["time"])
    describe_data(df)
    plot_feature_trends(df)
    correlation_matrix(df)
