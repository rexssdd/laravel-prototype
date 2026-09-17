# Deployment

The deployable split is:

- Vercel hosts `frontend/` as a static site.
- Render runs `render_api/api.py` and the numerical worker in `ml/`.
- Supabase stores `inference_bundle.joblib` in a private `models` bucket and stores JSON results in `analysis_results`.

## Supabase

1. Create a private Storage bucket named `models`.
2. Upload `inference_bundle.joblib` to that bucket.
3. Create a private Storage bucket named `uploads`.
4. Run `supabase/schema.sql` in the SQL editor.
5. Keep the service-role key only in Render environment variables.

## Render

Create a Web Service from this repository. Render can use `render.yaml`, or configure:

```text
Build command: pip install -r render_api/requirements.txt
Start command: uvicorn render_api.api:app --host 0.0.0.0 --port $PORT
Health check: /health
```

Set these environment variables:

```text
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_STORAGE_BUCKET=models
SUPABASE_UPLOADS_BUCKET=uploads
SUPABASE_BUNDLE_PATH=inference_bundle.joblib
SUPABASE_RESULTS_TABLE=analysis_results
CORS_ORIGINS=https://your-vercel-domain.vercel.app
```

The service downloads the bundle from Supabase on startup. It does not commit the model to GitHub.

## Vercel

Import the same repository and set the Vercel Root Directory to `frontend`. No build command is required. After Render is deployed, edit `frontend/index.html` and replace the default `API_URL` with the Render service URL, then redeploy Vercel.

The frontend sends a multipart request with the field name `file` to `/analyse` and renders the JSON response.