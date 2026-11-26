
import calendar

import pandas as pd

from pathlib import Path


# ==========================================
# 1. Configuration
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT_DIR / "outputs/neighbourhood_month.parquet"
)

OUTPUT_DIR = ROOT_DIR / "outputs"

YEARS = [2022, 2023, 2024]

COUNT_COLUMNS = [
    "total_incidents",
    "persons_incidents",
    "property_incidents",
    "other_incidents"
]


# ==========================================
# 2. Load and validate Task 1 data
# ==========================================

def load_monthly_data():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Task 1 output not found. "
            "Run spark_processing.py first."
        )

    df = pd.read_parquet(INPUT_PATH)

    required_columns = [
        "neighborhood",
        "year",
        "month",
        *COUNT_COLUMNS
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    if df.duplicated(
        ["neighborhood", "year", "month"]

    ).any():
        
        raise ValueError(
            "Duplicate neighborhood-month groups found"
        )

    if not df["year"].isin(YEARS).all():

        raise ValueError("Unexpected years in dataset")

    if not df["month"].isin(range(1, 13)).all():

        raise ValueError("Invalid month values")

    if df[required_columns].isna().any().any():

        raise ValueError("Unexpected missing values")

    if (df[COUNT_COLUMNS] < 0).any().any():

        raise ValueError("Negative incident counts found")

    category_total = (
        df["persons_incidents"]
        + df["property_incidents"]
        + df["other_incidents"]
    )

    if not category_total.eq(df["total_incidents"]).all():

        raise ValueError("Incident category totals do not match")

    return df


# ==========================================
# 3. Fill missing neighborhood-month groups
# ==========================================

def complete_monthly_data(df):

    neighborhoods = sorted(
        df["neighborhood"].unique()
    )

    full_index = pd.MultiIndex.from_product(
        [
            neighborhoods,
            YEARS,
            range(1, 13)
        ],
        names=["neighborhood", "year", "month"]
    )

    complete_df = (
        df.set_index(["neighborhood", "year", "month"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    if (
        complete_df["total_incidents"].sum()
        != df["total_incidents"].sum()
    ):
        raise RuntimeError(
            "Incident totals changed during completion"
        )

    return complete_df


# ==========================================
# 4. Calculate neighborhood baselines
# ==========================================

def get_neighborhood_averages():

    df = complete_monthly_data(
        load_monthly_data()
    )

    averages = (
        df.groupby("neighborhood", as_index=False)
        ["total_incidents"]
        .mean()
        .rename(columns={
            "total_incidents": "overall_monthly_average"
        })
    )

    return averages


# ==========================================
# 5. Calculate outlook for a selected month
# ==========================================

def get_monthly_outlook(month):

    if (
        not isinstance(month, int)
        or isinstance(month, bool)
        or not 1 <= month <= 12
    ):
        
        raise ValueError("Month must be an integer from 1 to 12")

    df = complete_monthly_data(
        load_monthly_data()
    )

    selected = df[df["month"] == month]

    outlook = (
        selected.groupby("neighborhood", as_index=False)
        [COUNT_COLUMNS]
        .mean()
        .rename(columns={
            "total_incidents": "monthly_average",
            "persons_incidents": "avg_persons_incidents",
            "property_incidents": "avg_property_incidents",
            "other_incidents": "avg_other_incidents"
        })
    )

    # Avg total incidents per month (all years)

    baseline = (
        df.groupby("neighborhood", as_index=False)
        ["total_incidents"]
        .mean()
        .rename(columns={
            "total_incidents": "overall_monthly_average"
        })
    )

    outlook = outlook.merge(
        baseline,
        on="neighborhood",
        validate="one_to_one"
    )

    # Avg citywide incidents (selected month)

    city_total = outlook["monthly_average"].sum()

    if city_total <= 0:

        raise ValueError(
            "No citywide incident records for this month"
        )

    # Neighborhood's % of citywide incidents

    outlook["city_share_pct"] = (
        outlook["monthly_average"] / city_total * 100
    )

    # Avg / baseline (selected month)

    outlook["seasonal_ratio"] = (
        outlook["monthly_average"]
        / outlook["overall_monthly_average"].replace(0, float("nan"))
    )

    # % increase or decrease

    outlook["seasonal_change_pct"] = (
        outlook["seasonal_ratio"] - 1
    ) * 100

    outlook["month"] = month

    # Validate the city shares
    
    if abs(outlook["city_share_pct"].sum() - 100) > 1e-8:
        raise RuntimeError(
            "City shares do not add up to 100%"
        )

    # Validate the category avg

    category_average = (
        outlook["avg_persons_incidents"]
        + outlook["avg_property_incidents"]
        + outlook["avg_other_incidents"]
    )

    if (
        (category_average - outlook["monthly_average"])
        .abs().max() > 1e-8
    ):
        raise RuntimeError(
            "Category averages do not match monthly totals"
        )

    return outlook.sort_values(
        ["monthly_average", "neighborhood"],
        ascending=[False, True]
    ).reset_index(drop=True)


# ==========================================
# 6. Create the three ranked lists
# ==========================================

def get_ranked_lists(month):

    outlook = get_monthly_outlook(month)

    most = outlook.sort_values(
        ["monthly_average", "neighborhood"],
        ascending=[False, True]
    ).head(5).copy()

    seasonal = (
        outlook.dropna(subset=["seasonal_ratio"])
        .sort_values(
            ["seasonal_ratio", "neighborhood"],
            ascending=[False, True]
        )
        .head(5)
        .copy()
    )

    fewest = outlook.sort_values(
        ["monthly_average", "neighborhood"],
        ascending=[True, True]
    ).head(5).copy()

    most["ranking_type"] = "most_incidents"
    seasonal["ranking_type"] = "largest_seasonal_ratio"
    fewest["ranking_type"] = "fewest_incidents"

    for table in [most, seasonal, fewest]:
        table["rank"] = range(1, len(table) + 1)

    ranked = pd.concat(
        [most, seasonal, fewest],
        ignore_index=True
    )

    columns = [
        "month",
        "ranking_type",
        "rank",
        "neighborhood",
        "monthly_average",
        "avg_persons_incidents",
        "avg_property_incidents",
        "avg_other_incidents",
        "overall_monthly_average",
        "city_share_pct",
        "seasonal_ratio",
        "seasonal_change_pct"
    ]

    return ranked[columns]


# ==========================================
# 7. Save required CSV output
# ==========================================

def save_monthly_outlook(month):

    ranked = get_ranked_lists(month)

    month_name = calendar.month_name[month].lower()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR / f"t2_outlook_{month_name}.csv"
    )

    # Round to 2 decimal places and save to CSV

    ranked.round(2).to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path.name}")
    print(f"Rows: {len(ranked)}")

    return ranked.round(2)


# ==========================================
# 8. Run Task 2 analytics
# ==========================================

def main():

    print("Starting SafeStay Task 2 analytics")

    raw = load_monthly_data()
    complete = complete_monthly_data(raw)

    print(f"\nOriginal rows: {len(raw):,}")
    print(f"Completed rows: {len(complete):,}")
    print(f"Missing groups filled: {len(complete) - len(raw)}")
    print(f"Neighborhoods: {raw['neighborhood'].nunique()}")
    print(f"Total incidents: {complete['total_incidents'].sum():,}")

    for month in [1, 7]:

        print("\n" + "=" * 65)
        print(calendar.month_name[month].upper())
        print("=" * 65)

        ranked = save_monthly_outlook(month)

        for category, group in ranked.groupby(
            "ranking_type", sort=False
        ):

            print(
                "\n--- "
                + category.replace("_", " ").upper()
                + " ---"
            )

            print(
                group[
                    [
                        "rank",
                        "neighborhood",
                        "monthly_average",
                        "avg_persons_incidents",
                        "avg_property_incidents",
                        "city_share_pct",
                        "seasonal_ratio"
                    ]
                ].to_string(index=False)
            )

    print("\nTask 2 numerical analytics completed.")


if __name__ == "__main__":
    main()
