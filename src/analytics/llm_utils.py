
import os
import re
import json
import time
import hashlib
import requests

from decimal import Decimal
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv


# ==========================================
# 1. Configuration
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = "openai/gpt-oss-120b"

CACHE_DIR = ROOT_DIR / "cache" / "llm"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Communicate with Groq
# ==========================================

def call_llm(system_prompt, user_prompt):

    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }

    # unique cache identifier

    cache_key = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    cache_file = CACHE_DIR / f"{cache_key}.json"

    # reuse previously cached responses

    if cache_file.exists():

        cached = json.loads(
            cache_file.read_text(encoding="utf-8")
        )

        print("LLM response loaded from cache")

        return cached["response"]["choices"][0]["message"]["content"]

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # throttling requirement

    time.sleep(1)

    response = requests.post(
        GROQ_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    retrieved_at = datetime.now(timezone.utc).isoformat()

    try:
        response_data = response.json()
    except ValueError:
        response_data = {"raw_text": response.text}

    cached = {
        "retrieved_at": retrieved_at,
        "source": GROQ_URL,
        "request": payload,
        "status_code": response.status_code,
        "response": response_data
    }

    # Save unsuccessful API responses separately

    if not response.ok:

        error_id = time.time_ns()

        error_file = CACHE_DIR / f"error_{error_id}.json"

        error_file.write_text(
            json.dumps(cached, indent=2),
            encoding="utf-8"
        )

        response.raise_for_status()

    # Save successful responses
    
    temp_file = cache_file.with_suffix(".tmp")

    temp_file.write_text(
        json.dumps(cached, indent=2),
        encoding="utf-8"
    )

    temp_file.replace(cache_file)

    print("Groq API status:", response.status_code)

    return response_data["choices"][0]["message"]["content"]


# ==========================================
# 3. Extract numerical values
# ==========================================

def extract_numbers(text):

    pattern = (
        r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?(?:[eE][+-]?\d+)?"
    )

    matches = re.findall(pattern, str(text))

    numbers = set()

    for value in matches:

        normalized = value.replace(",", "")

        numbers.add(Decimal(normalized))

    return numbers


# ==========================================
# 4. Validate generated numbers
# ==========================================

def validate_numbers(answer, source_data):

    allowed_numbers = extract_numbers(source_data)

    answer_numbers = extract_numbers(answer)

    unsupported = answer_numbers - allowed_numbers

    passed = len(unsupported) == 0

    return passed, sorted(unsupported)
