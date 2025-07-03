from data_utils import *

PATH = "/app/data/dataset"
full_data_path = PATH + "/full_dataset.csv"

df = read_data(full_data_path)
df = handle_nans(df)
df = scale_data(df)
# bin_outliers(df)
df = generate_lags(df, n_lags=5)
df = time_to_feature(df)