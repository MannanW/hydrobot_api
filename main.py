# =====================================================================
# HydroBot API — FastAPI service
#
# Contract:
#   - Auth: X-API-Key header must match env var HYDROBOT_API_KEY
#   - GET  /health      -> { "status": "ok" }
#   - POST /analyse     -> { user_id, crop, scope, rows: [...] }
#                          -> { job_id, status: "pending" | "done" }
#   - GET  /analyse/{job_id} -> { status, result, error }
#
# As of v4: training rows are sent directly in the POST /analyse body
# (fetched by the caller's backend, which owns the database and its
# row-level security — see TrainingRow below for the exact shape).
# This API never reads a local data file or holds a DB credential.
#
# ML logic lives in engine/hydrobot.py — this file is transport only:
# auth, request/response validation, job queue, error translation.
# =====================================================================

from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow `from engine.hydrobot import ...` regardless of where this
# file sits in the repo (root, api/, or anywhere else) and regardless
# of the launch cwd. This has broken twice already from a fixed
# "walk up N levels" assumption that silently went stale the moment
# the folder layout changed — so instead of hardcoding a depth, walk
# up from this file's directory until we find a directory that
# actually contains an `engine` package, and use THAT as repo_root.
def _find_repo_root(start: str) -> str:
    current = start
    for _ in range(6):  # six levels is far more than this repo should ever need
        if os.path.isdir(os.path.join(current, "engine")):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # reached filesystem root, stop
            break
        current = parent
    # Fall back to the starting directory if `engine/` was never found —
    # this preserves the old (sometimes-correct) behavior rather than
    # silently doing nothing, and the subsequent import error will be
    # informative if this fallback path is ever actually hit.
    return start


repo_root = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from engine.hydrobot import (  # noqa: E402
    CropNotFoundError,
    InsufficientDataError,
    get_cached_analysis,
    run_analysis,
)

# ─────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="HydroBot API", version="4.0.0")

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
class TrainingRow(BaseModel):
    """
    One labelled trial row. Matches engine.hydrobot.ALL_FEATURES plus
    the two targets. All feature fields are optional since a given
    crop sheet may not have every column — the engine only trains on
    whichever features are actually present across the rows it
    receives. `Biofertilizer innoculation` is the human-readable
    string ("Water" / "Trichoderma"), not pre-encoded — the engine
    does that mapping itself, same as it always has.
    """

    Day: Optional[float] = None
    Seed_density: Optional[float] = Field(default=None, alias="Seed density")
    Seed_soaking_time: Optional[float] = Field(default=None, alias="Seed soaking time")
    Biofertilizer_innoculation: Optional[str] = Field(default=None, alias="Biofertilizer innoculation")
    Cocopeat: Optional[float] = None
    Harvest_time: Optional[float] = Field(default=None, alias="Harvest time")
    Blackout_duration: Optional[float] = Field(default=None, alias="Blackout duration")
    Nutrient_EC: Optional[float] = Field(default=None, alias="Nutrient EC")
    nutrient_spray_start_day: Optional[float] = Field(default=None, alias="nutrient spray start day")
    media_thickness: Optional[float] = Field(default=None, alias="media thickness")
    Weight: Optional[float] = None
    Height: Optional[float] = None

    model_config = {"populate_by_name": True}


class AnalyseRequest(BaseModel):
    user_id: str
    crop: str
    scope: Literal["private", "global"]
    rows: list[TrainingRow]


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
    height_excluded_pre_emergence_rows: int
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


class AnalyseJobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]


class AnalyseJobStatusResponse(BaseModel):
    status: Literal["pending", "running", "done", "error"]
    result: Optional[AnalyseResponse] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


jobs: dict[str, dict[str, Any]] = {}


def _rows_to_dicts(rows: list[TrainingRow]) -> list[dict[str, Any]]:
    """
    Converts TrainingRow models to plain dicts using the original
    column names (e.g. "Seed density", not "Seed_density") via their
    Pydantic aliases, since that's what engine.hydrobot expects. Drops
    any field left as None so the engine's own per-feature handling
    (whichever columns are actually present) behaves the same as it
    did when reading straight from an Excel sheet with missing columns.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        as_dict = row.model_dump(by_alias=True, exclude_none=True)
        out.append(as_dict)
    return out


def _make_job_id() -> str:
    return hashlib.sha256(f"{time.time()}-{os.urandom(8).hex()}".encode("utf-8")).hexdigest()[:24]


def _save_job(job_id: str, state: str, result: Optional[dict[str, Any]] = None, error: Optional[str] = None) -> None:
    jobs[job_id] = {
        "status": state,
        "result": result,
        "error": error,
    }


def _build_job_result(body: AnalyseRequest, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": body.user_id,
        "crop": body.crop,
        "scope": body.scope,
        **payload,
    }


def _train_and_store_job(job_id: str, body: AnalyseRequest) -> None:
    _save_job(job_id, "running")
    try:
        result = run_analysis(
            crop=body.crop,
            scope=body.scope,
            rows=_rows_to_dicts(body.rows),
        )
        _save_job(job_id, "done", result=_build_job_result(body, result))
    except Exception as exc:
        _save_job(job_id, "error", error=str(exc))


@app.post(
    "/analyse",
    response_model=AnalyseJobResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        404: {"model": ErrorResponse, "description": "Crop not found"},
        422: {"model": ErrorResponse, "description": "Insufficient training data for this crop"},
    },
    dependencies=[Depends(verify_api_key)],
)
def analyse(body: AnalyseRequest, background_tasks: BackgroundTasks) -> AnalyseJobResponse:
    job_id = _make_job_id()
    _save_job(job_id, "pending")

    try:
        cached_result = get_cached_analysis(
            crop=body.crop, scope=body.scope, rows=_rows_to_dicts(body.rows)
        )
    except CropNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if cached_result is not None:
        _save_job(job_id, "done", result=_build_job_result(body, cached_result))
        return AnalyseJobResponse(job_id=job_id, status="done")

    background_tasks.add_task(_train_and_store_job, job_id, body)
    return AnalyseJobResponse(job_id=job_id, status="pending")


@app.get(
    "/analyse/{job_id}",
    response_model=AnalyseJobStatusResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
    dependencies=[Depends(verify_api_key)],
)
def analyse_status(job_id: str) -> AnalyseJobStatusResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    status = job["status"]
    result = job.get("result")
    error = job.get("error")
    if status == "error" and error is None:
        error = "Unknown error"
    return AnalyseJobStatusResponse(status=status, result=result, error=error)