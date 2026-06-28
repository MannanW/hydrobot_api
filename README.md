# HydroBot API v4

Trains RandomForest models on microgreens trial data and returns a
data quality report, model performance metrics, and an optimized
"golden recipe" with alternatives — via a polling-based API designed
to stay fast after the first request per crop/scope.

## Structure
```
.python-version          # pins Python 3.11.10 — required, see "Deploying" below
main.py                   # FastAPI app: auth, job queue, request/response validation
engine/
  hydrobot.py              # ML pipeline + disk cache (extracted from hydrobotv4.py)
  __init__.py
requirements.txt
```
(`data/Microgreens_dataa.xlsx`, if present, is only used by local
fallback scripts/tests — see "Data source" below. The live API never
reads it.)

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive API docs.

## How `/analyse` works (read this before wiring up a frontend)

This is **not** a single request/response call — training can take
30+ seconds, so the API uses a job + polling pattern:

1. **`POST /analyse`** with `{user_id, crop, scope, rows}` — `rows`
   is the actual training data (a list of trial records; see "Data
   source" below for the exact shape) — returns immediately with
   `{job_id, status}`.
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
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
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

## Data source

`run_analysis()` / `get_cached_analysis()` take training rows directly
as a parameter — a list of dicts, one per labelled trial. **This API
holds no database credential and reads no local data file in the live
path.** The caller (e.g. a backend that owns the database and its
row-level security) is responsible for fetching the correct rows for
the requested `user_id` + `scope` before calling `POST /analyse`:

- `scope: "private"` → only that user's own rows for the crop.
- `scope: "global"` → that user's rows plus any rows from other users
  marked as shared, for the crop.

Each row must contain the feature columns the model trains on (a
subset of `Day`, `Seed density`, `Seed soaking time`,
`Biofertilizer innoculation`, `Cocopeat`, `Harvest time`,
`Blackout duration`, `Nutrient EC`, `nutrient spray start day`,
`media thickness` — whichever are present in the data) plus the two
targets, `Weight` and `Height`. `Biofertilizer innoculation` is sent
as the human-readable string (`"Water"` / `"Trichoderma"`), not a
pre-encoded number — the engine does that mapping itself, same as it
always did when reading straight from a spreadsheet.

The cache's data-fingerprinting works the same regardless of where
rows came from: any change in the actual row contents — a new upload,
an edited value — changes the fingerprint and triggers a one-time
retrain automatically, with no manual cache invalidation needed.

A file-based fallback (`_load_local_fallback_rows()`, reading from a
local `data/Microgreens_dataa.xlsx` if present) exists only for
running this engine standalone in scripts/tests — the live API never
calls it.

## Height model excludes pre-emergence rows

The Weight model trains on every labelled row. The Height model
trains only on rows where `Height > 0` — rows where the plant hadn't
visibly emerged yet at measurement time. This was a real fix, not a
cosmetic one: on a combined dataset where roughly two-thirds of rows
were pre-emergence (Height = 0), training Height on all rows let the
model get "free" accuracy by trivially predicting zero most of the
time — R² looked great (0.86–0.999 depending on the dataset) without
the model actually learning what drives height among plants that did
grow. Verified directly against real data: excluding pre-emergence
rows dropped Height's R² to a more honest ~0.6, with the model now
being evaluated on (and the golden recipe's height prediction now
based on) the harder, more relevant subset.

`model_performance.height_excluded_pre_emergence_rows` reports how
many rows were excluded this way, so this is visible in the API
response rather than a silent internal detail. If a crop+scope
combination has so few "emerged" rows that excluding pre-emergence
rows drops below the 20-row minimum (even though the *total* row
count clears that minimum), `InsufficientDataError` is raised with a
message that specifically calls out the height-model exclusion, so
it's distinguishable from a generic "not enough data" error.

Weight is unaffected by this — confirmed directly that Weight's
range and mean don't differ materially between pre-emergence and
post-emergence rows, so excluding rows for Weight's sake would have
thrown away real signal for no benefit.

## Known data gap (local fallback / test data only)

If you're using the bundled `Microgreens_dataa.xlsx` for local testing,
note it has two sheets: `Pakchoi` (176 labelled rows — works) and
`Fenugreek` (219 rows, but `Weight`/`Height` are entirely empty —
correctly raises `InsufficientDataError` rather than crashing). This
is a gap in that specific file, not a bug, and is unrelated to the
live API path.

## Crop name matching (local fallback only)

In the live API path, `crop` is just a label attached to the rows you
send — there's no sheet-name concept, so there's nothing to fuzzy-match
against. This section applies only to `_load_local_fallback_rows()`,
used for local scripts/tests: there, crop names are fuzzy-matched
against the available sheet names in the local Excel file using edit
distance, so small typos (`"Pakchoy"`, `"Fenugrek"`) still resolve
correctly. The match threshold scales with name length to avoid false
positives — an earlier flat threshold let unrelated short names (e.g.
`"Spinach"`) silently match `"Pakchoi"` and train on the wrong crop's
data instead of raising `CropNotFoundError`. If a typo test in local
scripts ever returns the wrong crop instead of a 404, that threshold
(`_closest_sheet()` in `engine/hydrobot.py`) is the first place to
check.
