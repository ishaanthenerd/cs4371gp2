
import os
import json
import time
import requests

from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from pyspark.sql import SparkSession


# ==========================================
# 1. Configuration
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")

APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")

API_URL = "https://data.seattle.gov/resource/tazs-3rd5.json"

BATCH_SIZE = 5000

DATE_FILTER = (
    "offense_date >= '2022-01-01T00:00:00' "
    "AND offense_date < '2025-01-01T00:00:00'"
)

CACHE_DIR = ROOT_DIR / "cache" / "incidents"
RAW_DIR = ROOT_DIR / "data" / "raw"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "X-App-Token": APP_TOKEN
}


# ==========================================
# 2. API requests and caching
# ==========================================

def fetch_data(params, filename):
    cache_file = CACHE_DIR / filename

    # reuse old data

    if cache_file.exists():
        cached = json.loads(
            cache_file.read_text(encoding="utf-8")
        )
        return cached["data"]

    if not APP_TOKEN:
        raise RuntimeError("SOCRATA_APP_TOKEN is missing")

    # minimum one-second wait between API requests

    time.sleep(1)

    response = requests.get(
        API_URL,
        headers=HEADERS,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    cached = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": API_URL,
        "parameters": params,
        "data": data
    }

    # write to temp file

    temp_file = cache_file.with_suffix(".tmp")

    temp_file.write_text(
        json.dumps(cached, indent=2),
        encoding="utf-8"
    )

    temp_file.replace(cache_file)

    return data


# ==========================================
# 3. Count the incident records
# ==========================================

def get_total_records():

    params = {
        "$select": "count(*) AS total_records",
        "$where": DATE_FILTER
    }

    data = fetch_data(params, "record_count.json")

    return int(data[0]["total_records"])


# ==========================================
# 4. Download all incident records
# ==========================================

def download_incidents():

    total_records = get_total_records()

    print(f"Expected records: {total_records:,}")

    downloaded_records = 0
    offset = 0
    batch_number = 1

    jsonl_path = RAW_DIR / "seattle_incidents.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as output:

        while True:

            params = {
                "$limit": BATCH_SIZE,
                "$offset": offset,
                "$where": DATE_FILTER,
                "$order": "offense_date ASC, offense_id ASC, :id ASC"
            }

            filename = f"batch_{offset:07d}.json"

            records = fetch_data(params, filename)

            if not records:
                break

            for record in records:

                output.write(json.dumps(record) + "\n")

            downloaded_records += len(records)

            print(
                f"Batch {batch_number}: "
                f"{len(records):,} records | "
                f"Total: {downloaded_records:,}"
            )

            if len(records) < BATCH_SIZE:
                break

            offset += len(records)
            batch_number += 1

    print(f"\nExpected records: {total_records:,}")
    print(f"Downloaded records: {downloaded_records:,}")

    if downloaded_records != total_records:

        raise RuntimeError(
            "Record count mismatch. "
            "Investigate before continuing."
        )

    print("Download count verification passed.")

    return jsonl_path, downloaded_records


# ==========================================
# 5. Convert raw JSON records into Parquet
# ==========================================

def save_raw_parquet(jsonl_path, expected_count):

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("SafeStay-IncidentIngestion")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        df = spark.read.json(str(jsonl_path))

        raw_count = df.count()

        print(f"\nSpark loaded {raw_count:,} records")

        if raw_count != expected_count:

            raise RuntimeError("Spark input count mismatch")

        parquet_path = RAW_DIR / "seattle_incidents.parquet"

        df.write.mode("overwrite").parquet(
            str(parquet_path)
        )

        verified_count = spark.read.parquet(
            str(parquet_path)
        ).count()

        print(f"Parquet record count: {verified_count:,}")

        if verified_count != raw_count:

            raise RuntimeError("Parquet count mismatch")

        print("Parquet verification passed.")
        print(f"Saved to: {parquet_path}")

        print("\nDataset schema:")
        df.printSchema()

    finally:

        spark.stop()


# ==========================================
# 6. Run the ingestion pipeline
# ==========================================

def main():

    print("Starting SafeStay Seattle data ingestion\n")

    jsonl_path, downloaded_count = download_incidents()

    save_raw_parquet(jsonl_path, downloaded_count)

    print("\nTask 1 ingestion completed successfully.")

if __name__ == "__main__":

    main()
