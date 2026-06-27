# HydroBot API v3

Trains RandomForest models on microgreens trial data and returns a
data quality report, model performance metrics, and an optimized
"golden recipe" with alternatives — via a polling-based API designed
to stay fast after the first request per crop/scope.

## Structure
```
.python-version          # pins Python 3.11.10 — required, see "Deploying" below
api/
  main.py                 # FastAPI app: auth, job queue, request/response validation
engine/
  hydrobot.py              # ML pipeline + disk cache (extracted from hydrobotv4.py)
  __init__.py
data/
  Microgreens_dataa.xlsx   # PLACEHOLDER data source — see below
requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive API docs.

## How `/analyse` works (read this before wiring up a frontend)

This is **not** a single request/response call — training can take
30+ seconds, so the API uses a job + polling pattern:

1. **`POST /analyse`** with `{user_id, crop, scope}` → returns
   immediately with `{job_id, status}`.
   - If a valid cached model already exists for this exact
     `(crop, scope, data)` combination, `status` comes back as
     `"done"` right away — the result is in the next call's response.
   - Otherwise, training starts in the background and `status` comes
     back as `"pending"`.
2. **`GET /analyse/{job_id}`**, polled every ~2-3 seconds, returns
   `{status, result, error}`:
   - `status: "pending"` or `"running"` → keep polling.
   - `status: "done"` → `result` holds the full dashboard JSON
     (data quality summary, model performance, golden recipe).
   - `status: "error"` → `error` holds a human-readable message
     (e.g. unknown crop, or not enough labelled rows).

**Why it's built this way:** the first version of this API ran
training synchronously inside `/analyse` and took ~113 seconds per
call (exhaustive `GridSearchCV` over 72 hyperparameter combinations).
That reliably hit timeouts in Render's proxy, browsers, and Lovable's
server functions (`Error 524`). Fixing it required three changes
together, not one:

- **A smaller hyperparameter grid** — fewer combinations searched per
  training run. Tested against the real Pakchoi data: R² is
  essentially unchanged (0.984 either way) but training time drops
  from ~115s to **~30s**.
- **Disk-backed model caching**, keyed by crop + scope + a fingerprint
  of the training data itself. Once a crop/scope combination has been
  trained, every later request for it returns in well under a
  second — no retraining — until the underlying data actually
  changes, at which point the fingerprint changes and it retrains
  automatically. No manual cache invalidation needed.
- **Background jobs + polling**, so even the ~30s first-time training
  run never blocks an HTTP connection long enough to hit a platform
  timeout, regardless of how slow the grid search ends up being on a
  given crop's dataset size.

Net effect: the *first* analysis for a new crop/scope combination
takes roughly 30 seconds (well under any reasonable timeout); every
analysis after that, for the same crop/scope and unchanged data, is
near-instant.

**Known limitation:** job state and the model cache both live on
local disk / in-memory inside the running process. Render's free tier
wipes local disk and restarts the process on every deploy (and
possibly after long idle periods), so the cache is lost at that point —
the next request after a restart just retrains once, same as a normal
cache miss. There's no data loss risk here, only a one-time ~30s delay
after a deploy. If this ever needs to survive restarts, swap the disk
cache for a small object store (e.g. S3-compatible bucket) using the
same `(crop, scope, fingerprint)` key scheme — the rest of the pipeline
doesn't need to change.

## Auth

Every request (except `/health`) must include:
```
X-API-Key: <value matching the HYDROBOT_API_KEY env var>
```
Enforced via a FastAPI dependency (`Depends(verify_api_key)`), so it
shows up correctly in the `/docs` schema and runs before the request
body is parsed.

## CORS

Reads allowed origins from the `FRONTEND_ORIGINS` env var
(comma-separated), e.g.:
```
FRONTEND_ORIGINS=https://your-app.lovable.app,https://your-domain.com
```
Falls back to `localhost:5173` / `localhost:3000` for local dev if
unset — **never** falls back to allowing all origins.

## Deploying on Render

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Env vars:** `HYDROBOT_API_KEY`, `FRONTEND_ORIGINS`

**The `.python-version` file at the repo root is required**, not
optional. Without it, Render may resolve a Python version for which
`pandas`/`scikit-learn` have no prebuilt wheel, causing pip to fall
back to compiling from source — this can hang for 15-30+ minutes on
Render's free-tier CPU or fail outright. The pinned version
(`3.11.10`) is a verified-working combination with the pinned package
versions in `requirements.txt`. If you ever bump a dependency version,
re-check that a matching wheel exists for whatever Python version
you're pinned to before deploying.

## Important: the data source is still a placeholder

`engine/hydrobot.py`'s `_load_training_rows()` currently loads from
the bundled `data/Microgreens_dataa.xlsx` by matching `crop` to a
sheet name — it ignores `user_id` and `scope` entirely. This is a
deliberate stand-in so the full pipeline (including caching) can run
end-to-end today. Before this goes near real users, replace that one
function with a real query, e.g.:

```sql
SELECT features, targets FROM training_rows
WHERE crop = :crop
  AND (user_id = :user_id OR (:scope = 'global' AND share_global = true))
```

Nothing else in the pipeline needs to change — cleaning, training,
caching, and the optimizer all just need a DataFrame with the right
columns in, same as today. The cache's data-fingerprinting also keeps
working unmodified: once real rows are flowing in, any new upload
changes the fingerprint and triggers a one-time retrain automatically.

## Known data gap

The bundled `Microgreens_dataa.xlsx` has two sheets: `Pakchoi` (176
labelled rows — works) and `Fenugreek` (219 rows, but `Weight`/`Height`
are entirely empty — correctly raises `InsufficientDataError` rather
than crashing). This is a gap in the data file itself, not a bug.

## Crop name matching

Crop names are fuzzy-matched against the available sheet names using
edit distance, so small typos (`"Pakchoy"`, `"Fenugrek"`) still
resolve correctly. The match threshold scales with name length to
avoid false positives — an earlier flat threshold let unrelated short
names (e.g. `"Spinach"`) silently match `"Pakchoi"` and train on the
wrong crop's data instead of raising `CropNotFoundError`. If a typo
test ever returns the wrong crop instead of a 404, that threshold
(`_closest_sheet()` in `engine/hydrobot.py`) is the first place to
check.
