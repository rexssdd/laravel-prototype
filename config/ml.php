<?php

return [

    /*
    |--------------------------------------------------------------------------
    | Python interpreter
    |--------------------------------------------------------------------------
    | Needs numpy, pandas and joblib. Nothing else.
    | A virtualenv path works fine: /var/www/ml-venv/bin/python
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
