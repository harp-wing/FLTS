# train_container/main.py
import os
import sys
import pandas as pd # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
import time
import pyarrow.parquet as pq # type: ignore

from train import train_model

PATH = "/app/data/dataset/"

in_table = pq.read_pandas(PATH + "processed_data.parquet")
df = in_table.to_pandas()

print(df)



