
import os
import sys
import json
import time
import requests

from pathlib import Path
from datetime import datetime, timezone
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import BooleanType

os.environ["PYSPARK_PYTHON"] = sys.executable

# ==========================================
# 1. Configuration
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT_DIR / "data/raw/seattle_incidents.parquet"
OUTPUT_DIR = ROOT_DIR / "outputs"
CACHE_DIR = ROOT_DIR / "cache/incidents"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_URL = (
    "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/"
    "arcgis/rest/services/Seattle_Area_Polygon/"
    "FeatureServer/0/query"
)


# ==========================================
# 2. Download and cache Seattle boundary
# ==========================================

def get_city_boundary():

    cache_file = CACHE_DIR / "seattle_boundary.json"

    if cache_file.exists():

        cached = json.loads(
            cache_file.read_text(encoding="utf-8")
        )
        data = cached["data"]

    else:

        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "geojson"
        }

        time.sleep(1)

        response = requests.get(
            BOUNDARY_URL,
            params=params,
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        if "error" in data or not data.get("features"):

            raise RuntimeError("Invalid city boundary response")

        cached = {
            "retrieved_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "source": BOUNDARY_URL,
            "parameters": params,
            "data": data
        }

        cache_file.write_text(
            json.dumps(cached, indent=2),
            encoding="utf-8"
        )

    if data.get("exceededTransferLimit"):

        raise RuntimeError("Incomplete city boundary response")

    polygons = []

    for feature in data["features"]:

        geometry = feature["geometry"]

        if geometry["type"] == "Polygon":

            polygons.append(geometry["coordinates"])

        elif geometry["type"] == "MultiPolygon":

            polygons.extend(geometry["coordinates"])

    if not polygons:

        raise RuntimeError("No boundary polygons found")

    return polygons


# ==========================================
# 3. Point-in-polygon calculation
# ==========================================

def point_in_ring(x, y, ring):

    inside = False

    j = len(ring) - 1

    for i in range(len(ring)):

        xi, yi = ring[i][:2]
        xj, yj = ring[j][:2]

        if (yi > y) != (yj > y):

            intersection = (
                (xj - xi) * (y - yi) / (yj - yi) + xi
            )

            if x < intersection:
                inside = not inside

        j = i

    return inside


def inside_city(longitude, latitude, polygons):

    if longitude is None or latitude is None:
        return False

    for polygon in polygons:

        # First ring is the exterior boundary

        if point_in_ring(longitude, latitude, polygon[0]):

            # Remaining rings represent holes

            in_hole = any(
                point_in_ring(longitude, latitude, hole)
                for hole in polygon[1:]
            )

            if not in_hole:
                return True

    return False


# ==========================================
# 4. Clean incident records with Spark
# ==========================================

def clean_incidents(df, spark, polygons):

    cleaning_log = []

    df = df.cache()
    current_count = df.count()

    cleaning_log.append(
        f"Initial raw records: {current_count:,}"
    )

    def apply_step(new_df, description):

        nonlocal df, current_count

        new_df = new_df.cache()
        new_count = new_df.count()

        removed = current_count - new_count

        message = (
            f"{description}: "
            f"before={current_count:,}, "
            f"removed={removed:,}, "
            f"remaining={new_count:,}"
        )

        print(message)

        cleaning_log.append(message)

        df.unpersist()

        df = new_df
        current_count = new_count

    # removing duplicates

    apply_step(
        df.dropDuplicates(["offense_id"]),
        "Duplicate removal"
    )

    # converting date into timestamp

    dated_df = df.withColumn(
        "incident_timestamp",
        F.to_timestamp("offense_date")
    )

    # removing missing or invalid dates

    apply_step(
        dated_df.filter(
            F.col("incident_timestamp").isNotNull()
        ),
        "Missing or invalid dates"
    )

    # verifying date range

    apply_step(
        df.filter(
            (F.col("incident_timestamp") >= "2022-01-01") &
            (F.col("incident_timestamp") < "2025-01-01")
        ),
        "Outside requested date range"
    )

    # normalizing neighborhood names

    normalized_df = df.withColumn(
        "neighborhood",
        F.upper(F.trim(F.col("neighborhood")))
    )

    # removing missing neighborhoods

    apply_step(
        normalized_df.filter(
            F.col("neighborhood").isNotNull() &
            (~F.col("neighborhood").isin(
                "", "-", "UNKNOWN", "NULL", "N/A"
            ))
        ),
        "Missing or placeholder neighborhoods"
    )

    # converting coordinate strings into numbers

    numeric_df = (
        df.withColumn(
            "latitude",
            F.col("latitude").cast("double")
        )
        .withColumn(
            "longitude",
            F.col("longitude").cast("double")
        )
    )

    # removing invalid geographic coordinates

    apply_step(
        numeric_df.filter(
            F.col("latitude").isNotNull() &
            F.col("longitude").isNotNull() &
            (~F.isnan("latitude")) &
            (~F.isnan("longitude")) &
            F.col("latitude").between(-90, 90) &
            F.col("longitude").between(-180, 180)
        ),
        "Invalid coordinates"
    )

    # removing out-of-jurisdiction neighborhood label

    apply_step(
        df.filter(F.col("neighborhood") != "OOJ"),
        "Out-of-jurisdiction records"
    )

    # calculating city bounding box from actual geometry

    boundary_points = [
        point
        for polygon in polygons
        for ring in polygon
        for point in ring
    ]

    min_lon = min(p[0] for p in boundary_points)
    max_lon = max(p[0] for p in boundary_points)
    min_lat = min(p[1] for p in boundary_points)
    max_lat = max(p[1] for p in boundary_points)

    # removing out-of-city coordinates

    apply_step(
        df.filter(
            F.col("latitude").between(min_lat, max_lat) &
            F.col("longitude").between(min_lon, max_lon)
        ),
        "Outside Seattle bounding box"
    )

    # Broadcast boundary to Spark workers

    boundary = spark.sparkContext.broadcast(polygons)

    is_inside = F.udf(
        lambda lon, lat: inside_city(
            lon, lat, boundary.value
        ),
        BooleanType()
    )

    # checking each coordinate only once

    locations = (
        df.select("longitude", "latitude")
        .distinct()
        .withColumn(
            "inside_seattle",
            is_inside(
                F.col("longitude"),
                F.col("latitude")
            )
        )
        .filter(F.col("inside_seattle"))
        .drop("inside_seattle")
    )

    # removing points outside Seattle's polygon

    apply_step(
        df.join(
            locations,
            ["longitude", "latitude"],
            "left_semi"
        ),
        "Outside Seattle polygon"
    )

    return df, cleaning_log


# ==========================================
# 5. Classify offenses
# ==========================================

def classify_incidents(df):

    classification = F.upper(
        F.trim(F.col("nibrs_crime_against_category"))
    )

    df = df.withColumn(
        "category",

        F.when(
            classification == "PERSON",
            "crimes against persons"
        )
        .when(
            classification == "PROPERTY",
            "crimes against property"
        )
        .otherwise("other")
    )

    df = (
        df.withColumn(
            "year",
            F.year("incident_timestamp")
        )
        .withColumn(
            "month",
            F.month("incident_timestamp")
        )
        .withColumn(
            "offense_type",
            F.when(
                F.col("offense_sub_category").isNull() |
                (F.trim("offense_sub_category") == "") |
                (F.trim("offense_sub_category") == "-"),
                "UNSPECIFIED"
            ).otherwise(
                F.upper(F.trim("offense_sub_category"))
            )
        )
    )

    return df


# ==========================================
# 6. Aggregate neighborhood statistics
# ==========================================

def aggregate_incidents(df):

    keys = ["neighborhood", "year", "month"]

    monthly = df.groupBy(*keys).agg(

        F.count("*").alias("total_incidents"),

        F.sum(
            F.when(
                F.col("category") == "crimes against persons",
                1
            ).otherwise(0)
        ).alias("persons_incidents"),

        F.sum(
            F.when(
                F.col("category") == "crimes against property",
                1
            ).otherwise(0)
        ).alias("property_incidents"),

        F.sum(
            F.when(
                F.col("category") == "other",
                1
            ).otherwise(0)
        ).alias("other_incidents")
    )

    # Count each offense type within each group
    offense_counts = (
        df.filter(F.col("offense_type") != "UNSPECIFIED")
        .groupBy(*keys, "offense_type")
        .count()
    )

    # ranking by frequency
    ranking_window = (
        Window.partitionBy(*keys)
        .orderBy(
            F.desc("count"),
            F.asc("offense_type")
        )
    )

    ranked = offense_counts.withColumn(
        "offense_rank",
        F.row_number().over(ranking_window)
    )

    # converting top three offense types into cols
    top_offenses = (
        ranked.filter(F.col("offense_rank") <= 3)
        .groupBy(*keys)
        .agg(
            F.max(
                F.when(
                    F.col("offense_rank") == 1,
                    F.col("offense_type")
                )
            ).alias("top_offense_1"),

            F.max(
                F.when(
                    F.col("offense_rank") == 2,
                    F.col("offense_type")
                )
            ).alias("top_offense_2"),

            F.max(
                F.when(
                    F.col("offense_rank") == 3,
                    F.col("offense_type")
                )
            ).alias("top_offense_3")
        )
    )

    result = monthly.join(
        top_offenses,
        keys,
        "left"
    )

    return result.orderBy(*keys)


# ==========================================
# 7. Validate and save results
# ==========================================

def save_results(cleaned, monthly, cleaning_log, spark):

    cleaned_count = cleaned.count()

    aggregated_count = monthly.agg(
        F.sum("total_incidents")
    ).first()[0]

    if cleaned_count != aggregated_count:
        raise RuntimeError(
            "Aggregated incident count does not match cleaned data"
        )

    invalid_totals = monthly.filter(
        F.col("total_incidents") != (
            F.col("persons_incidents") +
            F.col("property_incidents") +
            F.col("other_incidents")
        )
    ).count()

    if invalid_totals != 0:
        raise RuntimeError("Category total validation failed")

    neighborhood_count = (
        cleaned.select("neighborhood")
        .distinct()
        .count()
    )

    date_range = cleaned.agg(
    F.date_format(
        F.min("incident_timestamp"),
        "yyyy-MM-dd HH:mm:ss"
    ).alias("earliest"),
    F.date_format(
        F.max("incident_timestamp"),
        "yyyy-MM-dd HH:mm:ss"
    ).alias("latest")
).first()

    parquet_path = (
        OUTPUT_DIR / "neighbourhood_month.parquet"
    )

    monthly.write.mode("overwrite").parquet(
        str(parquet_path)
    )

    monthly_rows = monthly.count()

    saved_df = spark.read.parquet(str(parquet_path))
    saved_rows = saved_df.count()

    saved_total = saved_df.agg(
        F.sum("total_incidents")
    ).first()[0]

    if saved_rows != monthly_rows or saved_total != cleaned_count:
        raise RuntimeError("Saved Parquet validation failed")

    summary = [
        "SAFESTAY SEATTLE - TASK 1 SUMMARY",
        "=" * 45,
        "",
        "DATA CLEANING",
        *cleaning_log,
        "",
        f"Final cleaned records: {cleaned_count:,}",
        f"Neighborhoods: {neighborhood_count}",
        f"Aggregated rows: {monthly_rows:,}",
        f"Earliest incident: {date_range['earliest']}",
        f"Latest incident: {date_range['latest']}",
        "",
        "CATEGORY MAPPING",
        "PERSON -> crimes against persons",
        "PROPERTY -> crimes against property",
        "All remaining values -> other",
        "",
        "METHODOLOGY",
        "Counts represent individual reported offense records.",
        "Duplicates identified using offense_id.",
        "Monthly statistics use offense_date.",
        "Missing offense dates are excluded by the API date filter.",
        "NOT_A_CRIME records retained under other.",
        "UNKNOWN and placeholder neighborhoods excluded.",
        "OOJ records excluded as out-of-jurisdiction.",
        "Coordinates checked against Seattle Area Polygon.",
        "Boundary is generalized and extends past parts of shoreline.",
        "Common offense types ranked by count, then alphabetically.",
        "",
        "DATA SOURCES",
        "https://data.seattle.gov/resource/tazs-3rd5.json",
        BOUNDARY_URL,
        "",
        "VALIDATION",
        "Cleaned and aggregated incident counts match.",
        "All three category totals sum to total incidents.",
        "Saved Parquet verification passed."
    ]

    summary_path = OUTPUT_DIR / "t1_summary.txt"

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("\n" + "\n".join(summary))

    print("\n--- MONTHLY DATA PREVIEW ---")

    saved_df.orderBy(
        "neighborhood", "year", "month"
    ).show(15, truncate=False)

    print("\nTask 1 outputs saved successfully.")


# ==========================================
# 8. Run Task 1 processing
# ==========================================

def main():

    print("Starting SafeStay Spark processing\n")

    polygons = get_city_boundary()

    print("Seattle boundary loaded successfully.")

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("SafeStay-SparkProcessing")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.python.worker.reuse", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        raw = spark.read.parquet(str(RAW_PATH))

        cleaned, cleaning_log = clean_incidents(
            raw, spark, polygons
        )

        classified = classify_incidents(cleaned).cache()

        print("\n--- FINAL CATEGORY COUNTS ---")

        classified.groupBy("category").count().show(
            truncate=False
        )

        monthly = aggregate_incidents(classified).cache()

        save_results(
            classified,
            monthly,
            cleaning_log,
            spark
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
