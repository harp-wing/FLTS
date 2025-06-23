# # basic_pipeline/data/load_data.py

# import pandas as pd
# from sklearn.model_selection import train_test_split

# def load_dataset(path: str, test_size: float = 0.2, random_state: int = 42):
#     """
#     Load a time-series dataset from CSV and split into train/test sets chronologically.
#     """
#     df = pd.read_csv(path, parse_dates=["time"])
#     df = df.sort_values("time")

#     # Since this is time-series, we split chronologically, not randomly
#     split_index = int(len(df) * (1 - test_size))
#     train_df = df.iloc[:split_index]
#     test_df = df.iloc[split_index:]

#     return train_df, test_df


pass