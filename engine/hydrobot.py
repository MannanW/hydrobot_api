# =====================================================================
# HydroBot Engine — core ML logic, extracted from hydrobotv4.py
#
# This module contains NO print()/input() calls and is safe to call
# from a web request. It mirrors hydrobotv4.py's pipeline exactly:
#   STEP 3  data loading & cleaning
#   STEP 4  train/test split
#   STEP 5  hyperparameter tuning (GridSearchCV)
#   STEP 6  model evaluation
#   STEP 9  golden recipe optimizer
#   STEP 10 top-5 alternative recipes
#
# DATA SOURCE (v4):
#   `run_analysis()` / `get_cached_analysis()` take training rows
#   directly as a parameter — a list of dicts, one per labelled trial,
#   with feature columns matching ALL_FEATURES plus "Weight"/"Height".
#   The caller (api/main.py) is responsible for fetching the right
#   rows for the requested user_id + scope (private vs private+global)
#   before calling in — this module has no knowledge of where rows
#   came from or how access control was applied. The only file-based
#   path left is `_load_local_fallback_rows()`, used by local
#   scripts/tests run directly against this module, never by the API.
# =====================================================================

from __future__ import annotations

import hashlib
import itertools
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Hashable, Optional, cast

import joblib  # type: ignore[import]
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.model_selection import train_test_split as _train_test_split  # type: ignore


def train_test_split(
    X: NDArray[np.float64],
    y_w: NDArray[np.float64],
    y_h: NDArray[np.float64],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    return cast(
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        _train_test_split(X, y_w, y_h, test_size=test_size, random_state=random_state),
    )

# ─────────────────────────────────────────────────────────────────────
# Constants
#
# NOTE: As of v4, training rows arrive directly in the API request
# (sent by the Lovable backend, which owns the database) rather than
# being read from a local Excel file. HYDROBOT_DATA_FILE / the bundled
# data/Microgreens_dataa.xlsx are kept ONLY as an optional local
# fallback for running/testing this engine standalone, outside the API
# — see `_load_local_fallback_rows()` below. The live API path never
# touches this file.
# ─────────────────────────────────────────────────────────────────────
DEFAULT_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "Microgreens_dataa.xlsx"
ROOT_DATA_FILE = Path(__file__).resolve().parent.parent / "Microgreens_dataa.xlsx"


def _resolve_data_file() -> Path:
    data_file_env = os.environ.get("HYDROBOT_DATA_FILE", "")
    if data_file_env:
        return Path(data_file_env).expanduser()

    if DEFAULT_DATA_FILE.exists():
        return DEFAULT_DATA_FILE

    if ROOT_DATA_FILE.exists():
        return ROOT_DATA_FILE

    return DEFAULT_DATA_FILE


@lru_cache(maxsize=1)
def _load_workbook_sheets() -> list[str]:
    xl = pd.ExcelFile(DATA_FILE, engine="openpyxl")
    return [str(s) for s in xl.sheet_names]


@lru_cache(maxsize=32)
def _read_sheet(sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name, engine="openpyxl")  # type: ignore[call-overload]
    return df.copy(deep=True)


def _load_local_fallback_rows(crop: str) -> pd.DataFrame:
    """
    Local-file fallback used only by standalone scripts/tests, never
    by the live API. Looks up `crop` against sheet names in the
    bundled Excel workbook using fuzzy matching.
    """
    if not os.path.exists(DATA_FILE):
        raise CropNotFoundError(f"Local fallback data file not found at {DATA_FILE}")

    sheets = _load_workbook_sheets()
    exact = next((s for s in sheets if s.lower() == crop.lower()), None)
    sheet_name = exact or _closest_sheet(crop, sheets)
    if sheet_name is None:
        raise CropNotFoundError(
            f"No crop matching '{crop}' found. Available crops: {', '.join(sheets)}"
        )
    return _read_sheet(sheet_name)


# Keep this local fallback path available for standalone engine scripts/tests,
# even though the live API route never calls it.
if False:
    _load_local_fallback_rows  # type: ignore[unused]


def _ensure_cache_dir() -> None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_identifier(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value.lower())[:48]


def _training_data_fingerprint(df: pd.DataFrame, feature_cols: list[str]) -> str:
    payload = df[feature_cols + [TARGET_WEIGHT, TARGET_HEIGHT]].astype(str)
    digest = hashlib.sha256(payload.to_csv(index=False).encode("utf-8")).hexdigest()
    return f"{len(df)}-{digest}"


def _cache_file_path(crop: str, scope: str, fingerprint: str) -> Path:
    _ensure_cache_dir()
    file_name = f"{_safe_identifier(crop)}_{_safe_identifier(scope)}_{fingerprint[:16]}.joblib"
    return MODEL_CACHE_DIR / file_name


def _load_cached_analysis(crop: str, scope: str, fingerprint: str) -> Optional[dict[str, Any]]:
    path = _cache_file_path(crop, scope, fingerprint)
    if not path.exists():
        return None
    cached = joblib.load(path)  # type: ignore[assignment]
    if isinstance(cached, dict) and "result" in cached:
        return cast(dict[str, Any], cached["result"])
    return None


def _save_cached_analysis(crop: str, scope: str, fingerprint: str, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    path = _cache_file_path(crop, scope, fingerprint)
    joblib.dump({"result": payload, "metadata": metadata}, path)  # type: ignore[call-arg]


def get_cached_analysis(
    crop: str, scope: str, rows: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """
    Cheap pre-check: does a valid cached model already exist for this
    exact (crop, scope, data) combination? Returns the cached result
    dict if so, else None. Raises CropNotFoundError/InsufficientDataError
    if `rows` itself is invalid (e.g. too few labelled rows) — this
    lets the API surface that error immediately, before even
    attempting to start a background training job.
    """
    _require_nonempty_rows(rows, crop)
    df_raw = pd.DataFrame(rows)
    df, feature_cols, _quality = _clean_and_validate(df_raw, crop=crop)
    fingerprint = _training_data_fingerprint(df, feature_cols)
    return _load_cached_analysis(crop, scope, fingerprint)


DATA_FILE = _resolve_data_file()
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
MODEL_CACHE_DIR = CACHE_DIR / "model_cache"

BIO_MAP_IN: dict[str, int] = {"Water": 0, "Trichoderma": 1}
BIO_MAP_OUT: dict[int, str] = {0: "Water", 1: "Trichoderma"}

ALL_FEATURES: list[str] = [
    "Day", "Seed density", "Seed soaking time", "Biofertilizer innoculation",
    "Cocopeat", "Harvest time", "Blackout duration", "Nutrient EC",
    "nutrient spray start day", "media thickness",
]

TARGET_WEIGHT = "Weight"
TARGET_HEIGHT = "Height"
MIN_ROWS = 20

PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 4],
    "max_features": ["sqrt", 0.7],
}


# ─────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────
class CropNotFoundError(Exception):
    """Raised when the requested crop has no matching data source."""


class InsufficientDataError(Exception):
    """Raised when a crop has fewer than MIN_ROWS labelled rows."""

    def __init__(self, crop: str, rows_with_data: int):
        self.crop = crop
        self.rows_with_data = rows_with_data
        super().__init__(
            f"'{crop}' has only {rows_with_data} labelled rows; "
            f"minimum of {MIN_ROWS} required."
        )


def _require_nonempty_rows(rows: list[dict[str, Any]], crop: str) -> None:
    if not rows:
        raise InsufficientDataError(crop, 0)


# ─────────────────────────────────────────────────────────────────────
# Levenshtein fuzzy crop matching (mirrors hydrobotv4.py)
# ─────────────────────────────────────────────────────────────────────
def _levenshtein(s1: str, s2: str) -> int:
    s1, s2 = s1.lower(), s2.lower()
    dp: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        prev = i
        for j, c2 in enumerate(s2):
            temp = dp[j + 1]
            dp[j + 1] = prev if c1 == c2 else 1 + min(prev, dp[j], dp[j + 1])
            prev = temp
    return dp[-1]


def _closest_sheet(name: str, available: list[str]) -> Optional[str]:
    """
    Fuzzy-matches `name` against `available` sheet names, but only
    accepts a match within a threshold that scales with name length.
    A flat threshold (e.g. always <= 4) is too permissive for short
    names — "Spinach" vs "Pakchoi" has edit distance 4 despite being
    completely unrelated crops. Scaling by length keeps short names
    strict while still tolerating a typo or two on longer names.
    """
    if not available:
        return None
    scored = sorted(((_levenshtein(name, a), a) for a in available), key=lambda t: t[0])
    best_dist, best_name = scored[0]
    max_allowed = max(1, min(3, len(name) // 4))
    return best_name if best_dist <= max_allowed else None


# ─────────────────────────────────────────────────────────────────────
# DATA SOURCE
#
# As of v4, training rows are no longer loaded from a local file in
# the live API path — they arrive directly in the request, fetched by
# Lovable's backend (which already enforces row-level security: each
# user's private rows + globally-shared rows for the requested scope).
# `run_analysis()` / `get_cached_analysis()` below take `rows` as a
# parameter for exactly this reason. `_load_local_fallback_rows()`
# above is the only remaining file-based path, kept for local
# scripts/tests run directly against this module.
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Cleaning (mirrors hydrobotv4.py STEP 3)
# ─────────────────────────────────────────────────────────────────────
def _clean_and_validate(df_raw: pd.DataFrame, crop: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if "Sno" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Sno"])

    if "Biofertilizer innoculation" in df_raw.columns:
        df_raw["Biofertilizer innoculation"] = df_raw["Biofertilizer innoculation"].map(BIO_MAP_IN)

    feature_cols = [c for c in ALL_FEATURES if c in df_raw.columns]

    missing_targets = [t for t in [TARGET_WEIGHT, TARGET_HEIGHT] if t not in df_raw.columns]
    if missing_targets:
        raise CropNotFoundError(
            f"Missing target column(s) {missing_targets} for crop '{crop}'."
        )

    total_rows = len(df_raw)
    rows_with_data = int(df_raw[[TARGET_WEIGHT, TARGET_HEIGHT]].dropna().shape[0])
    rows_zero_h = int((df_raw[TARGET_HEIGHT] == 0).sum())

    if rows_with_data < MIN_ROWS:
        raise InsufficientDataError(crop, rows_with_data)

    df = (
        df_raw[feature_cols + [TARGET_WEIGHT, TARGET_HEIGHT]]
        .dropna(subset=[TARGET_WEIGHT, TARGET_HEIGHT])
        .copy()
    )
    for col in feature_cols:
        median_val = float(df[col].median())
        df[col] = df[col].fillna(median_val)

    quality: dict[str, Any] = {
        "total_rows_in_sheet": total_rows,
        "rows_used": rows_with_data,
        "missing_targets": total_rows - rows_with_data,
        "pre_emergence_rows": rows_zero_h,
        "feature_list": feature_cols,
        "min_rows_required": MIN_ROWS,
        "status": "ok",
    }
    return df, feature_cols, quality


# ─────────────────────────────────────────────────────────────────────
# Metrics (mirrors hydrobotv4.py report())
# ─────────────────────────────────────────────────────────────────────
def _metrics(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> dict[str, float | None]:
    r2 = float(metrics.r2_score(cast(Any, y_true), cast(Any, y_pred)))  # type: ignore
    mae = float(metrics.mean_absolute_error(cast(Any, y_true), cast(Any, y_pred)))  # type: ignore
    rmse = float(np.sqrt(metrics.mean_squared_error(cast(Any, y_true), cast(Any, y_pred))))  # type: ignore
    nonzero = y_true != 0
    if nonzero.any():
        pct = float(np.mean(np.abs(y_pred[nonzero] - y_true[nonzero]) / np.abs(y_true[nonzero])) * 100)
    else:
        pct = float("nan")
    return {
        "r2": round(r2, 4),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "pct_error": None if np.isnan(pct) else round(pct, 1),
    }


# ─────────────────────────────────────────────────────────────────────
# Biological validity filter (mirrors hydrobotv4.py is_valid())
# ─────────────────────────────────────────────────────────────────────
def _is_valid(recipe: dict[str, Any]) -> bool:
    day = recipe.get("Day", 0)
    harvest = recipe.get("Harvest time", 99)
    spray_start = recipe.get("nutrient spray start day", 0)
    if day > harvest:
        return False
    if spray_start >= harvest:
        return False
    return True


def _build_optimizer_grid(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for col in feature_cols:
        raw_uniq_values: list[Any] = df[col].dropna().unique().tolist()
        raw_uniq: list[Any] = [v for v in raw_uniq_values if v is not None]
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in raw_uniq):
            raw_uniq = sorted(raw_uniq, key=float)
        else:
            raw_uniq = sorted(raw_uniq, key=lambda v: str(v))
        if len(raw_uniq) > 8:
            step = max(1, len(raw_uniq) // 8)
            raw_uniq = [raw_uniq[i] for i in range(0, len(raw_uniq), step)][:8]
        grid[col] = [round(float(v), 4) if isinstance(v, float) else v for v in raw_uniq]
    return grid


def _format_recipe(row: dict[Hashable, Any] | pd.Series[Any], feature_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feat in feature_cols:
        raw_val = row.get(feat, None) if isinstance(row, dict) else row.get(feat, None)
        if raw_val is None:
            continue
        if feat == "Biofertilizer innoculation":
            out[feat] = BIO_MAP_OUT.get(int(round(float(raw_val))), str(raw_val))
        elif isinstance(raw_val, (int, float)) and float(raw_val).is_integer():
            out[feat] = int(raw_val)
        else:
            out[feat] = round(float(raw_val), 3)
    return out


# ─────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
def run_analysis(crop: str, scope: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Runs the full HydroBot pipeline for a single crop and returns a
    dict matching the API's response shape. `rows` is the training
    data — a list of dicts, each containing the feature columns
    (whichever of ALL_FEATURES are present) plus "Weight" and
    "Height" — already filtered to the right scope (this user's
    private rows, plus globally-shared rows if scope == "global") by
    the caller. Raises CropNotFoundError or InsufficientDataError on
    bad input — callers should catch these and translate to
    appropriate HTTP responses.
    """
    _require_nonempty_rows(rows, crop)
    df_raw = pd.DataFrame(rows)
    df, feature_cols, quality = _clean_and_validate(df_raw, crop=crop)
    quality["crop"] = crop
    quality["scope"] = scope

    fingerprint = _training_data_fingerprint(df, feature_cols)
    cached = _load_cached_analysis(crop, scope, fingerprint)
    if cached is not None:
        return cached

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y_w = df[TARGET_WEIGHT].to_numpy(dtype=np.float64)
    y_h = df[TARGET_HEIGHT].to_numpy(dtype=np.float64)

    X_train, X_test, yw_train, yw_test, yh_train, yh_test = train_test_split(
        X, y_w, y_h, test_size=0.2, random_state=42
    )

    grid_w = GridSearchCV(
        RandomForestRegressor(random_state=42), PARAM_GRID, cv=5, n_jobs=-1, scoring="r2"
    )
    grid_w.fit(X_train, yw_train)  # type: ignore[call-arg]
    best_model_w = grid_w.best_estimator_

    grid_h = GridSearchCV(
        RandomForestRegressor(random_state=42), PARAM_GRID, cv=5, n_jobs=-1, scoring="r2"
    )
    grid_h.fit(X_train, yh_train)  # type: ignore[call-arg]
    best_model_h = grid_h.best_estimator_

    yw_pred = np.asarray(best_model_w.predict(X_test), dtype=np.float64)  # type: ignore
    yh_pred = np.asarray(best_model_h.predict(X_test), dtype=np.float64)  # type: ignore

    model_performance: dict[str, Any] = {
        "weight": _metrics(yw_test, yw_pred),
        "height": _metrics(yh_test, yh_pred),
        "cv_folds": 5,
    }

    # ── Golden recipe optimizer (mirrors hydrobotv4.py STEP 9/10) ──
    optimize_grid = _build_optimizer_grid(df, feature_cols)
    og_keys, og_vals = zip(*optimize_grid.items())

    total_combos = 1
    for v_list in og_vals:
        total_combos *= len(v_list)

    valid_combos: list[dict[str, Any]] = []
    for combo in itertools.product(*og_vals):
        recipe = dict(zip(og_keys, combo))
        if _is_valid(recipe):
            valid_combos.append(recipe)

    if not valid_combos:
        raise InsufficientDataError(crop, len(valid_combos))

    recipes_df = pd.DataFrame(valid_combos)
    recipes_X = recipes_df[feature_cols].to_numpy(dtype=np.float64)
    recipes_df["Predicted_Weight"] = np.asarray(best_model_w.predict(recipes_X), dtype=np.float64)  # type: ignore
    recipes_df["Predicted_Height"] = np.asarray(best_model_h.predict(recipes_X), dtype=np.float64)  # type: ignore

    weight_vals = recipes_df["Predicted_Weight"].to_numpy(dtype=np.float64)
    height_vals = recipes_df["Predicted_Height"].to_numpy(dtype=np.float64)
    pw_min, pw_max = float(weight_vals.min()), float(weight_vals.max())
    ph_min, ph_max = float(height_vals.min()), float(height_vals.max())
    recipes_df["Score"] = (
        0.7 * (weight_vals - pw_min) / (pw_max - pw_min + 1e-9)
        + 0.3 * (height_vals - ph_min) / (ph_max - ph_min + 1e-9)
    )

    ranked = recipes_df.sort_values("Score", ascending=False)
    best_recipe_row = ranked.iloc[0]
    pred_w = float(best_recipe_row["Predicted_Weight"])
    pred_h = float(best_recipe_row["Predicted_Height"])

    target_weight = df[TARGET_WEIGHT].to_numpy(dtype=np.float64)
    target_height = df[TARGET_HEIGHT].to_numpy(dtype=np.float64)
    pct_w = float((target_weight < pred_w).mean() * 100)
    pct_h = float((target_height < pred_h).mean() * 100)

    top5 = ranked.head(5)
    alternatives: list[dict[str, Any]] = []
    for row in top5.itertuples(index=False, name=None):
        recipe_vals = cast(dict[Hashable, Any], dict(zip(top5.columns, row)))
        alternatives.append({
            "recipe": _format_recipe(recipe_vals, feature_cols),
            "predicted_weight_g": round(float(recipe_vals["Predicted_Weight"]), 2),
            "predicted_height_cm": round(float(recipe_vals["Predicted_Height"]), 2),
            "composite_score": round(float(recipe_vals["Score"]), 4),
        })

    golden_recipe_and_optimizer: dict[str, Any] = {
        "best_recipe": _format_recipe(best_recipe_row.to_dict(), feature_cols),
        "expected_weight_g": round(pred_w, 2),
        "expected_weight_percentile": round(pct_w, 1),
        "expected_height_cm": round(pred_h, 2),
        "expected_height_percentile": round(pct_h, 1),
        "composite_score": round(float(best_recipe_row["Score"]), 4),
        "scoring_weights": {"weight": 0.7, "height": 0.3},
        "top_5_alternatives": alternatives,
        "combinations_explored": {
            "total": total_combos,
            "biologically_valid": len(valid_combos),
            "filtered_out": total_combos - len(valid_combos),
        },
    }

    result = {
        "data_quality_summary": quality,
        "model_performance": model_performance,
        "golden_recipe_and_optimizer": golden_recipe_and_optimizer,
    }

    _save_cached_analysis(
        crop,
        scope,
        fingerprint,
        result,
        {
            "cached_at": time.time(),
            "rows_used": len(df),
            "feature_cols": feature_cols,
        },
    )
    return result
