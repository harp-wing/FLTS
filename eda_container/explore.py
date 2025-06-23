# eda_container/explore.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import os
import os

sns.set_style('darkgrid')
sns.set_theme(rc={'figure.figsize': (14, 8)})

def describe_data(df: pd.DataFrame):
    print("🔍 Basic Info:")
    print(df.info())
    print("\n📊 Descriptive Statistics:")
    print(df.describe())
    print("\n🕒 Date Range:", df['time'].min(), "to", df['time'].max())

def plot_feature_trends(df: pd.DataFrame, features=None, output_dir="outputs/eda"):
    os.makedirs(output_dir, exist_ok=True)

    if features is None:
        features = ['down', 'up', 'rnti_count']

    for feature in features:
        plt.figure()
        sns.lineplot(x="time", y=feature, data=df)
        plt.title(f"Trend over Time: {feature}")
        plt.xlabel("Time")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{feature}_trend.png"))
        plt.close()

def correlation_matrix(df: pd.DataFrame, output_path="outputs/eda/correlation_matrix.png"):
    plt.figure(figsize=(12, 10))
    corr = df.drop(columns=['time']).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("📈 Correlation Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
