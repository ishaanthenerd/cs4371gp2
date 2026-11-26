
import calendar
import pandas as pd

from pathlib import Path

from src.analytics.neighborhood_outlook import (
    load_monthly_data,
    complete_monthly_data
)

ROOT = Path(__file__).resolve().parents[2]

TEMP_PATH = ROOT / "data/raw/city_temperature.csv"

OUTPUT_PATH = ROOT / "outputs/t6_best_months.csv"


# Seattle's monthly avg temp

def get_monthly_temperatures():

    df = pd.read_csv(TEMP_PATH)

    df = df[
        (df["City"] == "Seattle")
        & (df["State"] == "Washington")
        & (df["Year"].between(2020, 2024))
    ].copy()

    df["AvgTemperature"] = pd.to_numeric(
        df["AvgTemperature"], errors="coerce"
    )

    # removing missing temperatures and invalid markers

    df = df.dropna(subset=["AvgTemperature"])
    df = df[df["AvgTemperature"] != -99]

    print("Valid Seattle temperature records:", len(df))

    monthly = (
        df.groupby("Month")["AvgTemperature"]
        .mean()
        .reset_index()
        .rename(columns={
            "Month": "month",
            "AvgTemperature": "avg_temperature_f"
        })
    )

    monthly["temperature_deviation"] = (
        monthly["avg_temperature_f"] - 70
    ).abs()

    return monthly


# average citywide incidents (each month)

def get_monthly_incidents():

    df = complete_monthly_data(load_monthly_data())

    city_totals = (
        df.groupby(["year", "month"])["total_incidents"]
        .sum()
        .reset_index()
    )

    monthly = (
        city_totals.groupby("month")["total_incidents"]
        .mean()
        .reset_index()
        .rename(columns={
            "total_incidents": "avg_monthly_incidents"
        })
    )

    return monthly


# between 0 and 1

def normalize(values):
    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return values * 0

    return (values - minimum) / (maximum - minimum)


# Combine temperatures and incidents to rank months

def get_best_months():
    temperatures = get_monthly_temperatures()
    incidents = get_monthly_incidents()

    df = temperatures.merge(incidents, on="month")

    if len(df) != 12:
        raise ValueError("Expected data for all 12 months")

    df["normalized_temperature"] = normalize(
        df["temperature_deviation"]
    )

    df["normalized_incidents"] = normalize(
        df["avg_monthly_incidents"]
    )

    # Temperature: 60%, reported incidents: 40%

    df["score"] = 100 * (
        1
        - 0.6 * df["normalized_temperature"]
        - 0.4 * df["normalized_incidents"]
    )

    df["month_name"] = df["month"].apply(
        lambda month: calendar.month_name[month]
    )

    df = df.sort_values(
        ["score", "month"],
        ascending=[False, True]
    ).reset_index(drop=True)

    df["rank"] = range(1, 13)
    df["recommended"] = df["rank"] <= 3

    return df[
        [
            "rank",
            "month",
            "month_name",
            "avg_temperature_f",
            "temperature_deviation",
            "avg_monthly_incidents",
            "normalized_temperature",
            "normalized_incidents",
            "score",
            "recommended"
        ]
    ]


def main():

    print("Starting SafeStay Task 6\n")

    results = get_best_months()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.round(2).to_csv(OUTPUT_PATH, index=False)

    print("\n--- ALL MONTHS RANKED ---")

    print(results.round(2).to_string(index=False))

    print("\n--- TOP THREE RECOMMENDED MONTHS ---")

    print(
        results.head(3)[
            [
                "rank",
                "month_name",
                "avg_temperature_f",
                "avg_monthly_incidents",
                "score"
            ]
        ].round(2).to_string(index=False)
    )

    print("\nSaved:", OUTPUT_PATH.name)
    print("Task 6 completed successfully.")


if __name__ == "__main__":
    main()
