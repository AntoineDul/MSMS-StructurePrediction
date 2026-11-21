"""
import kagglehub

# Download latest version
path = kagglehub.competition_download('enveda-CASMI26-molecule-id-mass-spectra')

print("Path to competition files:", path)
"""

# import pandas as pd

# df = pd.read_parquet("data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet")

# print(df.head())

import pyarrow.parquet as pq

train_file = pq.ParquetFile("data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet")
test_file = pq.ParquetFile("data/enveda-CASMI26-molecule-id-mass-spectra/test.parquet")

batch = next(train_file.iter_batches(batch_size=5))
df = batch.to_pandas()


print(df.head())

print("Schema : ", train_file.schema)
print("Rows:", train_file.metadata.num_rows)
print("Row groups:", train_file.num_row_groups)
print("Unique adducts:", df['adduct'].unique())

print("---------------\n\n Test file info \n\n-----------------")

print(test_file.schema)
print("Rows:", test_file.metadata.num_rows)
print("Row groups:", test_file.num_row_groups)