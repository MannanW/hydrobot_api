# HydroBot API v2

## Structure
```
api/
  main.py        # FastAPI app: auth, request/response validation, routing
engine/
  hydrobot.py     # Real ML pipeline (extracted from hydrobotv4.py), no I/O
data/
  Microgreens_dataa.xlsx   # PLACEHOLDER data source — see below
requirements.txt
```

Run locally:
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Deploy on Render:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Env vars:** `HYDROBOT_API_KEY` (your secret), `FRONTEND_ORIGINS` (see below)

## What changed from v1

1. **Pydantic response models** — every endpoint now declares a typed
   `response_model`, so Swagger (`/docs`) shows the exact response
   shape, FastAPI validates outgoing data, and your frontend gets
   autocomplete if it generates types from the OpenAPI schema.
2. **No fake data** — `/analyse` now calls `engine.hydrobot.run_analysis()`,
   which runs the real cleaning → train/test split → GridSearchCV →
   evaluation → golden-recipe-optimizer pipeline from `hydrobotv4.py`,
   just with all `print()`/`input()` calls removed so it's safe to call
   from a web request.
3. **ML logic split out** — `engine/hydrobot.py` has zero FastAPI/HTTP
   awareness. It raises plain Python exceptions (`CropNotFoundError`,
   `InsufficientDataError`) that `api/main.py` translates to HTTP 404 /
   422. You can unit-test the engine without spinning up a server.
4. **Auth as a dependency** — `Depends(verify_api_key)` on the route
   instead of a manual call inside the handler body. Same check, but
   now it shows up correctly in the OpenAPI schema as a required
   header, and FastAPI runs it before the request body is even parsed.
5. **CORS restricted** — reads allowed origins from `FRONTEND_ORIGINS`
   (comma-separated env var) instead of `allow_origins=["*"]`. Falls
   back to localhost dev ports if unset, **not** to wildcard.

## Important: the data source is still a placeholder

`engine/hydrobot.py`'s `_load_training_rows()` currently loads from the
bundled `data/Microgreens_dataa.xlsx` by matching `crop` to a sheet
name — it ignores `user_id` and `scope` entirely. This was a deliberate
stand-in so the pipeline could run end-to-end today. Before this goes
near real users, replace that function with your actual DB query:

```python
# pseudocode for the real version
SELECT features, targets FROM training_rows
WHERE crop = :crop
  AND (user_id = :user_id OR (:scope = 'global' AND share_global = true))
```

The rest of the pipeline (`_clean_and_validate`, training, optimizer)
doesn't need to change — it just needs a DataFrame with the right
columns in, same as today.

## Known issue found while testing: request latency

I ran the real pipeline end-to-end against your uploaded data
(Pakchoi sheet, 176 labelled rows) and it took **~113 seconds** for a
single `/analyse` call. That's `GridSearchCV` (72 param combinations ×
5-fold CV × 2 models) running synchronously inside the HTTP request —
this is inherent to `hydrobotv4.py`'s tuning approach now running
per-request instead of once offline, not something introduced by this
refactor. A two-minute HTTP request will likely hit timeouts on Render,
in browsers, and in Lovable's server functions. Worth deciding on a fix
before this goes live — options, roughly in order of effort:
- Cut down `PARAM_GRID` (fewer values per hyperparameter)
- Switch `GridSearchCV` to `RandomizedSearchCV` with a fixed iteration budget
- Cache/reuse tuned models per crop+scope instead of retraining every call
- Move training to a background job and have `/analyse` return a job ID
  the frontend polls (bigger change, but the "correct" long-term shape)

I didn't make this change since it changes behavior, not just structure —
flagging it so it's a decision you make rather than one I make for you.

## Found during testing: your uploaded Fenugreek sheet has no labelled data

Calling `run_analysis(crop="Fenugreek", ...)` against your uploaded
`Microgreens_dataa.xlsx` raises `InsufficientDataError` — all 219 rows
have empty `Weight`/`Height` columns. `Pakchoi` (176 labelled rows)
works correctly. This is a data gap in the file, not a bug — the engine
is correctly raising the same "not enough data" condition that
`hydrobotv4.py` was designed to handle.
