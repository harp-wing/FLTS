from data_utils import *
import pyarrow as pa # type: ignore
import pyarrow.parquet as pq # type: ignore

PATH = "/app/data/dataset/"
full_data_path = PATH + "full_dataset.csv"

df = read_data(full_data_path)
df = handle_nans(df)
df = scale_data(df)
# bin_outliers(df)
df = generate_lags(df, n_lags=5)
df = time_to_feature(df)

out_table = pa.Table.from_pandas(df, preserve_index=True)
pq.write_table(out_table, PATH + "processed_data.parquet")