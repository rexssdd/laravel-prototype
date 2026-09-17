CORRECTED PROTOTYPE OVERLAY

Copy these files into the same paths in your Laravel project.

Corrections:
1. Baseline 1 is explicitly Shukla et al. (2023) close-replication IF.
2. Baseline decision uses only the 9 Shukla features, train-fitted MinMax scaling, S=256 forest, train-selected contamination threshold, and label switching.
3. Multi-scale IF [64,128,256,512] and engineered features remain hybrid-only.
4. Standalone SVM is explicitly Baseline 2 / supervised reference.
5. UI defaults to the 82,332-row official held-out test when uploaded.
6. Combined train+test metrics are marked diagnostic only and NOT thesis test metrics.
7. Exported bundle metadata now records baseline provenance and separation from hybrid enhancements.

After replacing files:
php artisan optimize:clear
php artisan ml:doctor
php artisan serve

IMPORTANT: Re-run ml/export_inference_bundle.py in Kaggle after your final training run and replace storage/app/ml/inference_bundle.joblib so the new metadata and final Kaggle metrics are embedded.
