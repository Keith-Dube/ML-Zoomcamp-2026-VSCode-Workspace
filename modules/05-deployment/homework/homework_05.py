# %%
# ML Zoomcamp 2026 - Homework 5: Deployment

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE_DIR))

from model import normalize_record

MODEL_PATH = BASE_DIR / "pipeline.bin"
DEFAULTS_PATH = BASE_DIR / "feature_defaults.json"
METADATA_PATH = BASE_DIR / "model_metadata.json"

EXPECTED_SHA256 = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"

API_URL = "http://localhost:9696"


# %%
# Question 1 - uv version
# My installed version: 0.12.17

# Question 2 - scikit-learn version
# Reference environment: 1.7.2


# %%
# Question 3 - Lead data

lead_q3 = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}


# %%
# Question 4 and 6 - Lead data

lead_q4 = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74,
}


# %%
# Verify the model file


def verify_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    if model_hash != EXPECTED_SHA256:
        raise ValueError("Model checksum does not match the official 2026 release")

    print("Model checksum verified")

    return True


# %%
# Question 3 - Load the model and make a prediction


def model_prediction():

    verify_model()

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    with DEFAULTS_PATH.open("r", encoding="utf-8") as file:
        defaults = json.load(file)

    # Prepare the lead using the official model rules
    # Confirm the function signature in model.py.
    lead = normalize_record(lead_q3)

    probability = float(model.predict_proba([lead])[0, 1])

    print("\nQuestion 3")
    print("Conversion probability:", round(probability, 3))

    return probability


# %%
# Send a request to the prediction API


def send_request(lead):

    request = Request(
        f"{API_URL}/predict",
        data=json.dumps(lead).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=15) as response:
        result = json.load(response)

    return result


# %%
# Check API health


def check_health():

    with urlopen(f"{API_URL}/health", timeout=15) as response:
        health = json.load(response)

    if health.get("status") != "ok":
        raise ValueError("API health check failed")

    # Check the model checksum if it is provided.
    model_hash = health.get("model_sha256", health.get("model_checksum"))

    if model_hash is not None and model_hash != EXPECTED_SHA256:
        raise ValueError("API model checksum does not match")

    print("API health check passed")
    print("Health response:", health)

    return health


# %%
# Question 4 and 6 - API prediction


def api_prediction(question):

    check_health()

    result = send_request(lead_q4)

    if "conversion_probability" not in result:
        raise ValueError("API response is missing conversion_probability")

    if "conversion" not in result:
        raise ValueError("API response is missing conversion")

    probability = result["conversion_probability"]

    if not isinstance(probability, (int, float)):
        raise TypeError("Conversion probability must be numeric")

    if not 0 <= probability <= 1:
        raise ValueError("Conversion probability must be between 0 and 1")

    if not isinstance(result["conversion"], bool):
        raise TypeError("Conversion must be a boolean")

    print(f"\nQuestion {question}")
    print("Conversion probability:", round(probability, 3))
    print("Conversion:", result["conversion"])

    return probability


# %%
# Run the selected homework question


def main():

    parser = argparse.ArgumentParser(description="ML Zoomcamp 2026 - Homework 5")

    parser.add_argument("--question", choices=["3", "4", "6"], default="3")

    args = parser.parse_args()

    try:
        if args.question == "3":
            print("\nQuestion 3 - Local Model Prediction")
            model_prediction()

        elif args.question == "4":
            print("\nQuestion 4 - FastAPI Prediction")
            api_prediction(4)

        elif args.question == "6":
            print("\nQuestion 6 - Docker Prediction")
            api_prediction(6)

    except (HTTPError, URLError) as error:
        print("\nCould not connect to the prediction API.")
        print("Make sure FastAPI or Docker is running on port 9696.")
        print("Error:", error)

    except Exception as error:
        print("\nAn error occurred:")
        print(error)
        raise


if __name__ == "__main__":
    main()


# %%
# Question 5
# Docker base image: python:3.11.15-slim-bookworm

# Question 6
# The Docker prediction should match Question 4.
