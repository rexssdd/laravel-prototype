import os
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import Client, create_client


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

import predict  # noqa: E402


APP_NAME = "Hybrid IF-SVM inference API"
BUNDLE_PATH = Path(os.getenv("BUNDLE_PATH", ROOT / "storage" / "app" / "ml" / "inference_bundle.joblib"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]

app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

bundle = None
supabase: Client | None = None


def get_supabase() -> Client | None:
    global supabase
    if supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            supabase = create_client(url, key)
    return supabase


def load_bundle() -> dict:
    global bundle
    if bundle is not None:
        return bundle

    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    storage = get_supabase()
    if not BUNDLE_PATH.exists() and storage:
        bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "models")
        remote_path = os.getenv("SUPABASE_BUNDLE_PATH", "inference_bundle.joblib")
        data = storage.storage.from_(bucket).download(remote_path)
        BUNDLE_PATH.write_bytes(data)

    if not BUNDLE_PATH.is_file():
        raise RuntimeError("Model bundle is not available on Render or Supabase Storage.")

    bundle = predict.joblib.load(BUNDLE_PATH)
    return bundle


@app.on_event("startup")
def startup() -> None:
    load_bundle()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "bundle_loaded": bundle is not None}


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    bundle_data = load_bundle()
    analysis_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temporary:
            temp_path = Path(temporary.name)
            total = 0
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="CSV exceeds the upload limit.")
                temporary.write(chunk)

        completed = subprocess.run(
            [sys.executable, str(ROOT / "ml" / "predict.py"), "--bundle", str(BUNDLE_PATH), "--input", str(temp_path), "--name", safe_name],
            capture_output=True,
            text=True,
            timeout=int(os.getenv("PREDICT_TIMEOUT", "180")),
            check=False,
        )
        if completed.returncode != 0:
            raise HTTPException(status_code=422, detail=completed.stdout or completed.stderr or "Inference failed.")
        result = json.loads(completed.stdout)
        if not result.get("ok"):
            raise HTTPException(status_code=422, detail=result.get("error", "Inference failed."))
        result["analysis_id"] = analysis_id
        persist_upload(analysis_id, safe_name, temp_path)
        persist_result(analysis_id, safe_name, result)
        return JSONResponse(result)
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def persist_result(analysis_id: str, filename: str, result: dict) -> None:
    client = get_supabase()
    if not client:
        return

    client.table(os.getenv("SUPABASE_RESULTS_TABLE", "analysis_results")).insert({
        "id": analysis_id,
        "filename": filename,
        "result": result,
    }).execute()


def persist_upload(analysis_id: str, filename: str, path: Path) -> None:
    client = get_supabase()
    bucket = os.getenv("SUPABASE_UPLOADS_BUCKET")
    if not client or not bucket:
        return

    client.storage.from_(bucket).upload(
        f"{analysis_id}/{filename}",
        path.read_bytes(),
        {"content-type": "text/csv", "upsert": "false"},
    )
