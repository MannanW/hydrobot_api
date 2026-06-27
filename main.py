# =====================================================================
# HydroBot API — FastAPI service
#
# Contract:
#   - Auth: X-API-Key header must match env var HYDROBOT_API_KEY
#   - GET  /health   -> { "status": "ok" }
#   - POST /analyse  -> { user_id, crop, scope } -> dashboard JSON
#
# This file currently returns MOCK / PLACEHOLDER data shaped exactly
# like the real output of hydrobotv4.py (data quality report, model
# performance, golden recipe + optimizer + top-5 alternatives), so the
# frontend can be built against a stable contract before the real
# training pipeline is wired in.
#
# Search "REAL LOGIC GOES HERE" to find the one spot you'll replace
# with an actual call into hydrobotv4.py's logic.
# =====================================================================

from __future__ import annotations

import os
import random
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="HydroBot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_ENV_VAR = "HYDROBOT_API_KEY"


def verify_api_key(x_api_key: str | None) -> None:
    expected = os.environ.get(API_KEY_ENV_VAR)
    if not expected or not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ─────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────
class AnalyseRequest(BaseModel):
    user_id: str
    crop: str
    scope: Literal["private", "global"]


# Feature columns used across the dummy data, matching hydrobotv4.py's
# ALL_FEATURES list (intersected with whatever a crop sheet contains).
FEATURE_COLS = [
    "Day",
    "Seed density",
    "Seed soaking time",
    "Biofertilizer innoculation",
    "Cocopeat",
    "Harvest time",
    "Blackout duration",
    "Nutrient EC",
    "nutrient spray start day",
    "media thickness",
]

BIO_VALUES = ["Water", "Trichoderma"]


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyse")
def analyse(
    body: AnalyseRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    verify_api_key(x_api_key)

    # >>> REAL LOGIC GOES HERE <<<
    # Replace everything below this line with:
    #   1. Fetch rows for body.user_id (+ globally shared rows if
    #      body.scope == "global") for body.crop from your DB.
    #   2. Run the equivalent of hydrobotv4.py's STEP 3 (clean/build
    #      training frame), STEP 5/6 (train + evaluate RandomForest
    #      models for Weight & Height), STEP 9/10 (golden recipe
    #      optimizer + top-5 alternatives).
    #   3. Return the same JSON shape produced below.

    rng = random.Random(f"{body.user_id}:{body.crop}:{body.scope}")

    total_rows = rng.randint(180, 480)
    rows_with_data = total_rows - rng.randint(5, 30)
    missing_targets = total_rows - rows_with_data
    pre_emergence_rows = rng.randint(0, 15)

    data_quality_summary: dict[str, Any] = {
        "crop": body.crop,
        "scope": body.scope,
        "rows_used": rows_with_data,
        "total_rows_in_sheet": total_rows,
        "missing_targets": missing_targets,
        "pre_emergence_rows": pre_emergence_rows,
        "feature_list": FEATURE_COLS,
        "min_rows_required": 20,
        "status": "ok" if rows_with_data >= 20 else "insufficient_data",
    }

    def fake_metrics(base_r2: float) -> dict[str, float]:
        r2 = round(base_r2 + rng.uniform(-0.03, 0.03), 4)
        mae = round(rng.uniform(0.05, 0.4), 3)
        rmse = round(mae * rng.uniform(1.1, 1.6), 3)
        pct_error = round(rng.uniform(3.0, 15.0), 1)
        return {"r2": r2, "mae": mae, "rmse": rmse, "pct_error": pct_error}

    model_performance: dict[str, Any] = {
        "weight": fake_metrics(0.93),
        "height": fake_metrics(0.90),
        "cv_folds": 5,
    }

    def random_recipe() -> dict[str, Any]:
        return {
            "Day": rng.randint(1, 14),
            "Seed density": round(rng.uniform(10, 40), 1),
            "Seed soaking time": rng.choice([0, 4, 8, 12]),
            "Biofertilizer innoculation": rng.choice(BIO_VALUES),
            "Cocopeat": rng.choice([60, 70, 80, 90, 100]),
            "Harvest time": rng.choice([7, 8, 9, 10, 12, 14]),
            "Blackout duration": rng.choice([0, 24, 48, 72]),
            "Nutrient EC": rng.choice([0, 1, 2, 3]),
            "nutrient spray start day": rng.randint(0, 5),
            "media thickness": rng.choice([1, 1.5, 2, 2.5]),
        }

    best_recipe = random_recipe()
    pred_w = round(rng.uniform(15, 60), 2)
    pred_h = round(rng.uniform(5, 18), 2)

    alternatives: list[dict[str, Any]] = []
    for _ in range(5):
        alt_recipe = random_recipe()
        alt_w = round(rng.uniform(10, 55), 2)
        alt_h = round(rng.uniform(4, 17), 2)
        score = round(rng.uniform(0.5, 0.99), 4)
        alternatives.append(
            {
                "recipe": alt_recipe,
                "predicted_weight_g": alt_w,
                "predicted_height_cm": alt_h,
                "composite_score": score,
            }
        )
    alternatives.sort(key=lambda r: r["composite_score"], reverse=True)

    total_combos = rng.randint(50_000, 500_000)
    valid_combos = int(total_combos * rng.uniform(0.6, 0.9))

    golden_recipe_and_optimizer: dict[str, Any] = {
        "best_recipe": best_recipe,
        "expected_weight_g": pred_w,
        "expected_weight_percentile": round(rng.uniform(70, 99), 1),
        "expected_height_cm": pred_h,
        "expected_height_percentile": round(rng.uniform(70, 99), 1),
        "composite_score": round(rng.uniform(0.85, 0.99), 4),
        "scoring_weights": {"weight": 0.7, "height": 0.3},
        "top_5_alternatives": alternatives,
        "combinations_explored": {
            "total": total_combos,
            "biologically_valid": valid_combos,
            "filtered_out": total_combos - valid_combos,
        },
    }

    return {
        "user_id": body.user_id,
        "crop": body.crop,
        "scope": body.scope,
        "data_quality_summary": data_quality_summary,
        "model_performance": model_performance,
        "golden_recipe_and_optimizer": golden_recipe_and_optimizer,
    }