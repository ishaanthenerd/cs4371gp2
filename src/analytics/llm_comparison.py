
import os
import json
import time
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

# Input: Task 1 aggregated dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "outputs" / "neighbourhood_month.parquet"

# Output files
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANSWERS_FILE = OUTPUT_DIR / "t3_llm_answers.json"
COMPARISON_FILE = OUTPUT_DIR / "t3_comparison.csv"
METRICS_FILE = OUTPUT_DIR / "t3_metrics.txt"

# API response cache
CACHE_DIR = Path("cache/task3")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Dataset columns
NEIGHBORHOOD_COL = "neighborhood"
YEAR_COL = "year"
MONTH_COL = "month"
INCIDENTS_COL = "total_incidents"

# Models
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Historical period
START_YEAR = 2022
END_YEAR = 2024

# API throttling
REQUEST_DELAY = 1.1


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def load_actual_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file or directory not found: {INPUT_FILE}"
        )

    # Spark may have written multiple part-*.parquet files.
    parquet_files = list(INPUT_FILE.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found inside: {INPUT_FILE}"
        )

    print(f"Reading {len(parquet_files)} Parquet file(s) from {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE, engine="pyarrow")

    required = {
        NEIGHBORHOOD_COL,
        YEAR_COL,
        MONTH_COL,
        INCIDENTS_COL
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    df = df.copy()

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[MONTH_COL] = pd.to_numeric(df[MONTH_COL], errors="coerce")
    df[INCIDENTS_COL] = pd.to_numeric(
        df[INCIDENTS_COL], errors="coerce"
    )

    df = df.dropna(subset=[
        NEIGHBORHOOD_COL,
        YEAR_COL,
        MONTH_COL,
        INCIDENTS_COL
    ])

    df[NEIGHBORHOOD_COL] = (
        df[NEIGHBORHOOD_COL].astype(str).str.strip()
    )

    df = df[
        df[YEAR_COL].between(START_YEAR, END_YEAR)
        & df[MONTH_COL].between(1, 12)
    ]

    # Combine duplicate neighborhood/year/month entries
    monthly = (
        df.groupby(
            [NEIGHBORHOOD_COL, YEAR_COL, MONTH_COL],
            as_index=False
        )[INCIDENTS_COL]
        .sum()
    )

    neighborhoods = sorted(
        monthly[NEIGHBORHOOD_COL].unique().tolist()
    )

    # Build a complete 36-month grid.
    # Missing neighborhood-month combinations count as zero.
    full_index = pd.MultiIndex.from_product(
        [
            neighborhoods,
            range(START_YEAR, END_YEAR + 1),
            range(1, 13)
        ],
        names=[NEIGHBORHOOD_COL, YEAR_COL, MONTH_COL]
    )

    monthly = (
        monthly
        .set_index([
            NEIGHBORHOOD_COL,
            YEAR_COL,
            MONTH_COL
        ])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Average incidents per month across all 36 months
    actual = (
        monthly.groupby(NEIGHBORHOOD_COL)[INCIDENTS_COL]
        .mean()
        .reset_index()
        .rename(columns={
            INCIDENTS_COL: "actual_monthly_average"
        })
    )

    print(f"Loaded {len(neighborhoods)} neighborhoods.")
    print(f"Historical period: {START_YEAR}-{END_YEAR}")

    return neighborhoods, actual


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are evaluating reported police incident levels across Seattle
neighborhoods.

Your task is to estimate the number of reported incidents per month
for each neighborhood and assign a relative rating.

Rating scale:
1 = Very few reported incidents
2 = Relatively few reported incidents
3 = Moderate reported incidents
4 = Relatively many reported incidents
5 = Very many reported incidents

Important rules:
- Return every neighborhood provided.
- Do not invent additional neighborhoods.
- Ratings must be integers from 1 to 5.
- Incident estimates must be nonnegative numbers.
- Estimates should represent reported incidents per month.
- Do not describe neighborhoods as dangerous or safe.
- Return valid JSON only.
"""

def build_user_prompt(neighborhoods, actual=None):

    prompt = f"""
Estimate the relative reported incident levels for these
Seattle neighborhoods:

{json.dumps(neighborhoods, indent=2)}
"""

    if actual is not None:
        prompt += f"""

Here are the actual historical monthly averages of reported
incidents for each neighborhood, calculated from 2022-2024:

{actual.to_json(orient="records", indent=2)}

Use these figures to inform your ratings and estimates.
"""

    prompt += """

Return JSON in exactly this format:

{
    "predictions": [
        {
            "neighborhood": "Example Neighborhood",
            "rating": 3,
            "estimated_incidents_per_month": 125.5
        }
    ]
}
"""

    return prompt


# ============================================================
# API RESPONSE CACHE
# ============================================================

def get_cache_path(provider, model, system_prompt, user_prompt):

    content = json.dumps({
        "provider": provider,
        "model": model,
        "system": system_prompt,
        "user": user_prompt
    }, sort_keys=True)

    key = hashlib.sha256(content.encode()).hexdigest()

    return CACHE_DIR / f"{key}.json"


def load_cached_response(cache_path):

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def save_cached_response(cache_path, response):

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)


# ============================================================
# LLM API CALLS
# ============================================================

last_request_time = {
    "groq": 0,
    "gemini": 0
}


def enforce_delay(provider):

    elapsed = time.time() - last_request_time[provider]

    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)


def call_groq(system_prompt, user_prompt):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"}
        },
        timeout=180
    )

    response.raise_for_status()

    raw = response.json()

    return {
        "text": raw["choices"][0]["message"]["content"],
        "raw_response": raw
    }


def call_gemini(system_prompt, user_prompt):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            response_mime_type="application/json"
        )
    )

    return {
        "text": response.text,
        "raw_response": {
            "text": response.text,
            "model": GEMINI_MODEL
        }
    }


def call_model(provider, system_prompt, user_prompt):

    model = GROQ_MODEL if provider == "groq" else GEMINI_MODEL

    cache_path = get_cache_path(
        provider, model, system_prompt, user_prompt
    )

    cached = load_cached_response(cache_path)

    if cached is not None:
        print(f"Using cached response: {provider}")
        return cached["text"]

    print(f"Calling {provider} ({model})...")

    enforce_delay(provider)

    if provider == "groq":
        result = call_groq(system_prompt, user_prompt)
    elif provider == "gemini":
        result = call_gemini(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    last_request_time[provider] = time.time()

    # Save the response and retrieval timestamp
    save_cached_response(cache_path, {
        "provider": provider,
        "model": model,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "text": result["text"],
        "raw_response": result["raw_response"]
    })

    return result["text"]


# ============================================================
# JSON VALIDATION
# ============================================================

def parse_predictions(response_text):

    # Handle occasional Markdown code fences
    cleaned = response_text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    result = json.loads(cleaned)

    if not isinstance(result, dict):
        raise ValueError("Response must be a JSON object.")

    predictions = result.get("predictions")

    if not isinstance(predictions, list):
        raise ValueError("Missing predictions array.")

    validated = []

    for item in predictions:

        if not isinstance(item, dict):
            raise ValueError("Prediction must be a JSON object.")

        name = item.get("neighborhood")
        rating = item.get("rating")
        estimate = item.get("estimated_incidents_per_month")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid neighborhood name.")

        if isinstance(rating, bool) or not isinstance(rating, int):
            raise ValueError(f"Invalid rating for {name}.")

        if not 1 <= rating <= 5:
            raise ValueError(f"Rating outside 1-5 for {name}.")

        if isinstance(estimate, bool) or not isinstance(
            estimate, (int, float)
        ):
            raise ValueError(f"Invalid estimate for {name}.")

        if not pd.notna(estimate) or estimate < 0:
            raise ValueError(f"Negative or invalid estimate for {name}.")

        validated.append({
            "neighborhood": name.strip(),
            "rating": rating,
            "estimated_incidents_per_month": float(estimate)
        })

    return validated


# ============================================================
# RUN EXPERIMENTS
# ============================================================

def run_experiment(experiment, provider, neighborhoods, actual):

    print(f"\nRunning {experiment} ({provider})")

    # E1 and E2 receive no actual data
    with_data = experiment in ["E3", "E4"]

    user_prompt = build_user_prompt(
        neighborhoods,
        actual if with_data else None
    )

    response_text = call_model(
        provider,
        SYSTEM_PROMPT,
        user_prompt
    )

    predictions = parse_predictions(response_text)

    return {
        "experiment": experiment,
        "provider": provider,
        "model": GROQ_MODEL if provider == "groq" else GEMINI_MODEL,
        "with_data": with_data,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
        "raw_text": response_text
    }


# ============================================================
# METRICS
# ============================================================

def normalize_name(name):
    return " ".join(str(name).strip().casefold().split())


def calculate_metrics(result, actual):

    predictions = result["predictions"]

    actual_lookup = {
        normalize_name(row[NEIGHBORHOOD_COL]): row[
            "actual_monthly_average"
        ]
        for _, row in actual.iterrows()
    }

    actual_names = set(actual_lookup)

    prediction_lookup = {}

    for prediction in predictions:
        key = normalize_name(prediction["neighborhood"])

        if key in prediction_lookup:
            raise ValueError(
                f"Duplicate neighborhood in response: "
                f"{prediction['neighborhood']}"
            )

        prediction_lookup[key] = prediction

    predicted_names = set(prediction_lookup)

    missing = sorted(actual_names - predicted_names)
    invented = sorted(predicted_names - actual_names)

    common = actual_names & predicted_names

    comparison = pd.DataFrame([
        {
            "neighborhood": name,
            "rating": prediction_lookup[name]["rating"],
            "estimated": prediction_lookup[name][
                "estimated_incidents_per_month"
            ],
            "actual": actual_lookup[name]
        }
        for name in common
    ])

    # Spearman correlation.
    # Pandas uses average ranks for tied values.
    if len(comparison) >= 2:
        spearman = comparison["rating"].corr(
            comparison["actual"],
            method="spearman"
        )
    else:
        spearman = float("nan")

    # Actual top five
    actual_top5 = set(
        actual.sort_values(
            ["actual_monthly_average", NEIGHBORHOOD_COL],
            ascending=[False, True]
        ).head(5)[NEIGHBORHOOD_COL].map(normalize_name)
    )

    # Predicted top five:
    # rating descending, estimate descending, alphabetical
    predicted_top5 = set(
        sorted(
            prediction_lookup.keys(),
            key=lambda name: (
                -prediction_lookup[name]["rating"],
                -prediction_lookup[name][
                    "estimated_incidents_per_month"
                ],
                name
            )
        )[:5]
    )

    overlap = len(actual_top5 & predicted_top5)

    # Absolute percentage error
    # Actual values of zero are excluded because percentage
    # error is undefined when the denominator is zero.
    nonzero_actual = comparison[comparison["actual"] > 0].copy()

    if len(nonzero_actual) > 0:
        nonzero_actual["percentage_error"] = (
            (
                nonzero_actual["estimated"]
                - nonzero_actual["actual"]
            ).abs()
            / nonzero_actual["actual"]
            * 100
        )

        median_error = nonzero_actual[
            "percentage_error"
        ].median()
    else:
        median_error = float("nan")

    return {
        "experiment": result["experiment"],
        "provider": result["provider"],
        "model": result["model"],
        "with_data": result["with_data"],
        "spearman_correlation": spearman,
        "top5_overlap": overlap,
        "median_percentage_error": median_error,
        "missing_count": len(missing),
        "missing_neighborhoods": missing,
        "invented_count": len(invented),
        "invented_neighborhoods": invented,
        "zero_actual_excluded": len(comparison) - len(nonzero_actual),
        "evaluated_count": len(common)
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_answers(results):

    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {ANSWERS_FILE}")


def save_comparison(metrics):

    df = pd.DataFrame(metrics)

    df.to_csv(COMPARISON_FILE, index=False)

    print(f"Saved: {COMPARISON_FILE}")


def save_metrics(metrics):

    with open(METRICS_FILE, "w", encoding="utf-8") as f:

        f.write("TASK 3: LLM COMPARISON RESULTS\n")
        f.write("=" * 55 + "\n\n")

        f.write("Evaluation methodology:\n")
        f.write("- Spearman correlation uses average ranks for ties.\n")
        f.write("- Top-five ties are broken by estimate, then alphabetically.\n")
        f.write("- Percentage error is absolute percentage error.\n")
        f.write("- Actual averages of zero are excluded from median percentage error.\n")
        f.write("- Missing and invented neighborhoods are matched case-insensitively.\n\n")

        for metric in metrics:

            f.write(f"\n{metric['experiment']} - {metric['provider'].upper()}\n")
            f.write("-" * 45 + "\n")

            f.write(
                f"Model: {metric['model']}\n"
            )

            f.write(
                f"Data supplied: {metric['with_data']}\n"
            )

            f.write(
                f"Spearman correlation: "
                f"{metric['spearman_correlation']:.4f}\n"
                if pd.notna(metric["spearman_correlation"])
                else "Spearman correlation: N/A\n"
            )

            f.write(
                f"Top-five overlap: {metric['top5_overlap']}/5\n"
            )

            f.write(
                f"Median percentage error: "
                f"{metric['median_percentage_error']:.2f}%\n"
                if pd.notna(metric["median_percentage_error"])
                else "Median percentage error: N/A\n"
            )

            f.write(
                f"Evaluated neighborhoods: {metric['evaluated_count']}\n"
            )

            f.write(
                f"Missing neighborhoods: "
                f"{metric['missing_neighborhoods']}\n"
            )

            f.write(
                f"Invented neighborhoods: "
                f"{metric['invented_neighborhoods']}\n"
            )

            f.write(
                f"Zero-actual neighborhoods excluded from error: "
                f"{metric['zero_actual_excluded']}\n"
            )

    print(f"Saved: {METRICS_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    neighborhoods, actual = load_actual_data()

    results = []
    metrics = []

    experiments = [
        ("E1", "groq"),
        ("E2", "gemini"),
        ("E3", "groq"),
        ("E4", "gemini")
    ]

    for experiment, provider in experiments:

        result = run_experiment(
            experiment,
            provider,
            neighborhoods,
            actual
        )

        metric = calculate_metrics(result, actual)

        results.append(result)
        metrics.append(metric)

        # Save incrementally in case a later API call fails
        save_answers(results)
        save_comparison(metrics)
        save_metrics(metrics)

    print("\nAll four experiments completed successfully.")

    print("\nFinal comparison:")
    print(
        pd.DataFrame(metrics)[[
            "experiment",
            "provider",
            "spearman_correlation",
            "top5_overlap",
            "median_percentage_error",
            "missing_count",
            "invented_count"
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()