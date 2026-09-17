# .env keys for the ML chain

Replace the ML_ block in your .env with this. The two that matter are
ML_PYTHON and ML_BUNDLE.

```dotenv
# Windows — "python3" is a Microsoft Store alias stub that exits silently.
# Forward slashes are fine; quote the value.
ML_PYTHON="C:/full/path/to/your/project/ml-venv/Scripts/python.exe"

# macOS / Linux
# ML_PYTHON="/full/path/to/your/project/ml-venv/bin/python"

# Relative is resolved against the project root, not the CWD. Absolute is safer.
ML_BUNDLE="storage/app/ml/inference_bundle.joblib"

ML_TIMEOUT=180
ML_MAX_FILES=25
ML_MAX_FILE_KB=262144
ML_KEEP_UPLOADS=false
```

Create the venv first:

```bash
# Windows
python -m venv ml-venv
ml-venv\Scripts\pip install numpy pandas joblib

# macOS / Linux
python3 -m venv ml-venv
./ml-venv/bin/pip install numpy pandas joblib
```

Then:

```bash
php artisan config:clear
php artisan ml:doctor --fresh
```

`ml:doctor` walks the whole chain — interpreter, packages, worker script,
bundle load, and a real prediction on a synthetic CSV — and stops at the
first thing that is actually broken, with the fix printed underneath.
