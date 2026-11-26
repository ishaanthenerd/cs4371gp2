import os
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

PANDAS_FILEPATH = 'T1_seattle_incidents.csv'
PARQUET_FILEPATH = 'T1_seattle_incidents.parquet'
FINAL_PARQUET_FILEPATH = 'neighbourhood_month.parquet'

# generate pandas dataframe

pandas_filepath_present = False
directory = Path('.')
for file in directory.iterdir():
    if file.is_file() and file.name == PANDAS_FILEPATH:
        pandas_filepath_present = True
        break

if not pandas_filepath_present:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path)
    token = os.getenv("SOCRATA_APP_TOKEN")
    url = "https://data.seattle.gov/resource/tazs-3rd5.json"
    headers = {"X-App-Token": token}

    limit = 50000
    offset = 0
    all_data = []

    while True:
        params = {"$limit": limit, "$offset": offset}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        all_data.extend(batch)
        offset += limit
        print(f"Retrieved {len(all_data)} records")

    df = pd.DataFrame(all_data)
    df.to_csv(PANDAS_FILEPATH)

# generate parquet

parquet_filepath_present = False
directory = Path('.')
for file in directory.iterdir():
    if file.is_file() and file.name == PARQUET_FILEPATH:
        parquet_filepath_present = True
        break

if not parquet_filepath_present:
    chunk_size = 100_000
    input_path = Path(PANDAS_FILEPATH)
    output_path = Path(PARQUET_FILEPATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0

    try:
        for chunk in pd.read_csv(input_path, chunksize=chunk_size):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy"
                )
            writer.write_table(table)
            total_rows += len(chunk)
            print(f"Processed {total_rows} records")
    finally:
        if writer is not None:
            writer.close()

    print(f"Conversion complete: {output_path}")
    print(f"Total records: {total_rows}")

# process with spark

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet(PARQUET_FILEPATH)
starting_rows = df.count()
remaining_rows = starting_rows
print(f"Starting rows: {starting_rows}")

def print_row_status(step_name):
    global remaining_rows
    new_count = df.count()
    removed = remaining_rows - new_count
    remaining_rows = new_count
    print(
        f"{step_name}: "
        f"Removed {removed} | Remaining {remaining_rows}"
    )

# 1. Remove duplicate rows
df = df.distinct()
print_row_status("Duplicate removal")

# TODO: check all possible dates, see if other null types exist
# 2. Remove rows without a date
df = df.filter(col("offense_date").isNotNull())
print_row_status("Missing date removal")

# TODO: check all possible neighborhoods, see if other null types exist
# 3. Remove rows without a neighborhood
df = df.filter(
    (col("neighborhood").isNotNull()) &
    (col("neighborhood") != "") &
    (col("neighborhood") != "-")
)
print_row_status("Missing neighborhood removal")

# TODO: check all possible coordinates, see if other null types exist
# 4. Remove rows with invalid coordinates
df = df.filter(
    (col("latitude") != "REDACTED") &
    (col("longitude") != "REDACTED")
)
df = df \
    .withColumn("latitude", col("latitude").cast("double")) \
    .withColumn("longitude", col("longitude").cast("double"))
df = df.filter(
    col("latitude").isNotNull() &
    col("longitude").isNotNull() &
    col("latitude").between(-90, 90) &
    col("longitude").between(-180, 180)
)
print_row_status("Invalid coordinate removal")

# 5. Keep only incidents within Seattle's bounding box
df = df.filter(
    col("latitude").between(47.4922, 47.7360) &
    col("longitude").between(-122.4318, -122.2377)
)
print_row_status("Outside Seattle removal")
df.write.mode("overwrite").parquet(PARQUET_FILEPATH)

# add category column (type of crime) using nibrs_crime_against_category

# NOTE: first filter rows with "NOT_A_CRIME" + other failure points if any
# find all possible values, print them, group them into person/property/other

# summarize data: one row per neighborhood-year-month combination
# this is a map-reduce

# output to final parquet 
# use df.coalesce(1).write.mode("overwrite").parquet(...) to combine into one file