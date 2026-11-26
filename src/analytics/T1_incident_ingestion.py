import os
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dotenv import load_dotenv

PANDAS_FILEPATH = 'T1_seattle_incidents.csv'
PARQUET_FILEPATH = 'T1_seattle_incidents.parquet'

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
            print(f"Processed {total_rows:,} records")
    finally:
        if writer is not None:
            writer.close()

    print(f"Conversion complete: {output_path}")
    print(f"Total records: {total_rows:,}")