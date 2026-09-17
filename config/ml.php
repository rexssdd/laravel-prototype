<?php

return [

    /*
    |--------------------------------------------------------------------------
    | Python interpreter
    |--------------------------------------------------------------------------
    | Needs numpy, pandas and joblib. Nothing else.
    |
    | On Windows do NOT use "python3" — that name is a Microsoft Store alias
    | stub which exits silently and prints nothing, which looks exactly like
    | a crashed worker. Use "python", "py -3", or a full venv path:
    |
    |   ML_PYTHON="C:/projects/hybrid-ids/ml-venv/Scripts/python.exe"
    |   ML_PYTHON="/var/www/hybrid-ids/ml-venv/bin/python"
    |
    | The runner probes python3 / python / py -3 as fallbacks, so leaving
    | this wrong usually still works — but set it explicitly and skip the
    | probe. `php artisan ml:doctor` reports what it resolved to.
    */
    'python' => env('ML_PYTHON', 'python3'),

    /*
    |--------------------------------------------------------------------------
    | Inference worker
    |--------------------------------------------------------------------------
    */
    'script' => env('ML_SCRIPT', base_path('ml/predict.py')),

    /*
    |--------------------------------------------------------------------------
    | Model bundle
    |--------------------------------------------------------------------------
    | Produced by ml/export_inference_bundle.py at the end of a training run.
    | Download it from Kaggle and drop it here.
    |
    | A relative value is resolved against the project root by
    | PredictionRunner::bundlePath(), not against the CWD — the CWD differs
    | between `artisan serve`, php-fpm and a queue worker.
    */
    'bundle' => env('ML_BUNDLE', storage_path('app/ml/inference_bundle.joblib')),

    /*
    |--------------------------------------------------------------------------
    | Limits
    |--------------------------------------------------------------------------
    */
    'timeout' => env('ML_TIMEOUT', 180),   // seconds per file
    'max_files' => env('ML_MAX_FILES', 25),
    'max_file_kb' => env('ML_MAX_FILE_KB', 262144),  // 256 MB
    'keep_uploads' => env('ML_KEEP_UPLOADS', false),

];
