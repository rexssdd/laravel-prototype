<?php

use App\Http\Controllers\AnalysisController;
use Illuminate\Support\Facades\Route;

Route::get('/', [AnalysisController::class, 'index'])->name('analysis.index');

Route::post('/analyse', [AnalysisController::class, 'analyse'])
    ->name('analysis.analyse');
