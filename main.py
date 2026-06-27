# =====================================================================
# HydroBot API — FastAPI service
#
# Contract:
#   - Auth: X-API-Key header must match env var HYDROBOT_API_KEY
#   - GET  /health   -> { "status": "ok" }
#   - POST /analyse  -> { user_id, crop, scope } -> dashboard JSON
#
# ML logic lives in engine/hydrobot.py — this file is transport only:
# auth, request/response validation, error translation to HTTP codes.
# =====================================================================

from __future__ import annotations

import os
import sys

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional

# Allow `from engine.hydrobot import ...` whether this file is run as
# `uvicorn api.main:app` from the project root, or some other layout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.hydrobot import (  # noqa: E402
    CropNotFoundError,
    InsufficientDataError,
    run_analysis,
)

# ─────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="HydroBot API", version="2.0.0")

# Restrict CORS to your real frontend origin(s). Set FRONTEND_ORIGINS
# as a comma-separated env var, e.g.:
#   FRONTEND_ORIGINS=https://your-app.lovable.app,https://your-domain.com
_origins_env = os.environ.get("FRONTEND_ORIGINS", "")
allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

if not allowed_origins:
    # Fail loudly rather than silently falling back to "*" in production.
    # For local dev, set FRONTEND_ORIGINS=http://localhost:5173 (or similar).
    allowed_origins = ["http://localhost:5173", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

API_KEY_ENV_VAR = "HYDROBOT_API_KEY"


# ─────────────────────────────────────────────────────────────────────
# Auth dependency
# ─────────────────────────────────────────────────────────────────────
def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.environ.get(API_KEY_ENV_VAR)
    if not expected or not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ─────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────
class AnalyseRequest(BaseModel):
    user_id: str
    crop: str
    scope: Literal["private", "global"]


# ─────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str


class DataQualitySummary(BaseModel):
    crop: str
    scope: Literal["private", "global"]
    rows_used: int
    total_rows_in_sheet: int
    missing_targets: int
    pre_emergence_rows: int
    feature_list: list[str]
    min_rows_required: int
    status: str


class ModelMetrics(BaseModel):
    r2: float
    mae: float
    rmse: float
    pct_error: Optional[float] = None


class ModelPerformance(BaseModel):
    weight: ModelMetrics
    height: ModelMetrics
    cv_folds: int


class RecipeAlternative(BaseModel):
    recipe: dict[str, str | int | float]
    predicted_weight_g: float
    predicted_height_cm: float
    composite_score: float


class ScoringWeights(BaseModel):
    weight: float
    height: float


class CombinationsExplored(BaseModel):
    total: int
    biologically_valid: int
    filtered_out: int


class GoldenRecipeAndOptimizer(BaseModel):
    best_recipe: dict[str, str | int | float]
    expected_weight_g: float
    expected_weight_percentile: float
    expected_height_cm: float
    expected_height_percentile: float
    composite_score: float
    scoring_weights: ScoringWeights
    top_5_alternatives: list[RecipeAlternative]
    combinations_explored: CombinationsExplored


class AnalyseResponse(BaseModel):
    user_id: str
    crop: str
    scope: Literal["private", "global"]
    data_quality_summary: DataQualitySummary
    model_performance: ModelPerformance
    golden_recipe_and_optimizer: GoldenRecipeAndOptimizer


class ErrorResponse(BaseModel):
    detail: str


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/analyse",
    response_model=AnalyseResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        404: {"model": ErrorResponse, "description": "Crop not found"},
        422: {"model": ErrorResponse, "description": "Insufficient training data for this crop"},
    },
    dependencies=[Depends(verify_api_key)],
)
def analyse(body: AnalyseRequest) -> AnalyseResponse:
    try:
        result = run_analysis(
            crop=body.crop,
            scope=body.scope,
            user_id=body.user_id,
        )
    except CropNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AnalyseResponse(
        user_id=body.user_id,
        crop=body.crop,
        scope=body.scope,
        data_quality_summary=result["data_quality_summary"],
        model_performance=result["model_performance"],
        golden_recipe_and_optimizer=result["golden_recipe_and_optimizer"],
    )
