# Hybrid IF-SVM — Laravel Prototype

Drag-and-drop web front end for the thesis experiment. Upload CSVs (or a
whole folder) of network flow records; the app scores them with the Shukla
Isolation Forest baseline, a standalone SVM, and the proposed hybrid, and
shows the three side by side.

## How it fits together

```
browser  ──drag/drop──▶  Laravel  ──Symfony Process──▶  ml/predict.py
                            │                                │
                            │                          inference_bundle.joblib
                            ◀────────── JSON ────────────────┘
```

Laravel never loads a model. It stores the upload, shells out to a small
Python worker, and renders the JSON that comes back. The worker needs only
numpy, pandas and joblib — no sklearn, no cuML, no GPU.

## Why a bundle instead of the .joblib models

`joblib.dump()` of the trained pipelines stores references to classes that
exist only inside the training script (`RBFSVMClassifier`,
`KeepColumnSelector`, `CustomIsolationForest`) plus cuML objects that need a
GPU to unpickle. Loading those from a web app would mean shipping the whole
training script and a GPU.

`ml/export_inference_bundle.py` exports the mathematics instead — support
vectors, dual coefficients, intercept, gamma, scaler statistics, the flat
Isolation Forest node arrays, and both tuned thresholds. An RBF decision
function rebuilt this way matches sklearn's to 3e-13.

## Install

```bash
composer create-project laravel/laravel hybrid-ids
cd hybrid-ids
```

Copy these files in:

```
routes/web.php
app/Http/Controllers/AnalysisController.php
app/Services/PredictionRunner.php
config/ml.php
resources/views/analysis/index.blade.php
ml/predict.py
ml/export_inference_bundle.py
```

Python side:

```bash
python3 -m venv ml-venv
./ml-venv/bin/pip install numpy pandas joblib
```

`.env`:

```
ML_PYTHON=/full/path/to/hybrid-ids/ml-venv/bin/python
ML_BUNDLE="${APP_DIR}/storage/app/ml/inference_bundle.joblib"
ML_TIMEOUT=180
ML_MAX_FILES=25
ML_MAX_FILE_KB=262144
```

Upload limits in `php.ini` — Laravel validation runs *after* PHP has already
rejected an oversized POST, so raise these or large drops fail silently:

```ini
upload_max_filesize = 256M
post_max_size = 512M
max_file_uploads = 30
max_execution_time = 300
```

## Get the bundle

1. At the end of a Kaggle run of `hybrid_if_svm_final.py`, paste the contents
   of `ml/export_inference_bundle.py` into a new cell and run it. It needs the
   training session's variables, so it must run in the same session.
2. Download `inference_bundle.joblib` from the Kaggle output.
3. `mkdir -p storage/app/ml && mv inference_bundle.joblib storage/app/ml/`

Then `php artisan serve`.

## Test the worker without Laravel

```bash
./ml-venv/bin/python ml/predict.py \
    --bundle storage/app/ml/inference_bundle.joblib \
    --input  some_traffic.csv
```

Exit codes: 0 ok, 2 bad input, 3 bad bundle, 1 unhandled.

## Uploaded CSV requirements

Must contain these nine columns (case-insensitive, any order, extra columns
ignored):

    sttl, ct_state_ttl, dload, rate, dmean, dttl, ackdat, sload, sbytes

If a `label` column is present (0 normal, 1 attack) the app reports accuracy,
precision, recall, specificity, F1 and MCC. Without it, only the predicted
attack rate — there is nothing to score against.

## Known limits of the prototype

- Files are processed **sequentially inside the request**. Fine for a demo,
  wrong for production: move the loop into a queued job and broadcast
  progress. The 180s timeout is the practical ceiling.
- The worker reloads the bundle on every file. A persistent worker (FastAPI,
  or a Laravel Octane sidecar) would amortise that.
- `predict.py` caps input at 200,000 rows per file and says so in the
  response when it truncates.
- Uploads are deleted after scoring unless `ML_KEEP_UPLOADS=true`.
