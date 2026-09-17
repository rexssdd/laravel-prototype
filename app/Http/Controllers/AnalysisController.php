<?php

namespace App\Http\Controllers;

use App\Services\PredictionRunner;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use Illuminate\View\View;

class AnalysisController extends Controller
{
    public function __construct(private PredictionRunner $runner)
    {
    }

    public function index(): View
    {
        return view('analysis.index', [
            'bundleReady'  => $this->runner->bundleExists(),
            'maxFiles'     => (int) config('ml.max_files'),
            'maxFileMb'    => round(((int) config('ml.max_file_kb')) / 1024),
        ]);
    }

    /**
     * Accepts one or more CSVs and returns one result object per file.
     *
     * Files are processed sequentially on purpose: the worker loads the
     * whole bundle per call, and running several in parallel on a shared
     * box just thrashes memory. For a real deployment this loop belongs
     * in a queued job with a broadcast channel for progress.
     */
    public function analyse(Request $request): JsonResponse
    {
        $maxFiles = (int) config('ml.max_files');

        $request->validate([
            'files'   => ['required', 'array', 'max:'.$maxFiles],
            'files.*' => [
                'required',
                'file',
                'max:'.config('ml.max_file_kb'),
                // extension-based: real-world CSVs get sniffed as
                // text/plain, application/octet-stream and worse, so a
                // strict mimetypes rule rejects valid uploads.
                'mimes:csv,txt',
            ],
        ], [
            'files.max'     => "Please upload at most {$maxFiles} files at a time.",
            'files.*.max'   => 'One of the files exceeds the size limit.',
            'files.*.mimes' => 'Only CSV files can be analysed.',
        ]);

        $results = [];
        $stored  = [];

        foreach ($request->file('files') as $file) {

            $original = $file->getClientOriginalName();

            // A folder drop sends paths like "batch/day1/traffic.csv".
            // Keep the relative path for display, never for the filesystem.
            $display = $request->input(
                'paths.'.count($results),
                $original
            );

            // Never trust the client filename on disk. Laravel 11 roots
            // the local disk at storage/app/private, so resolve the real
            // path through the disk rather than concatenating.
            $path = $file->storeAs('uploads', Str::uuid().'.csv', 'local');

            $absolute = Storage::disk('local')->path($path);
            $stored[] = $absolute;

            $results[] = $this->runner->run($absolute, $display);
        }

        if (! config('ml.keep_uploads')) {
            foreach ($stored as $path) {
                @unlink($path);
            }
        }

        return response()->json([
            'results' => $results,
            'summary' => $this->aggregate($results),
        ]);
    }

    /**
     * Roll the per-file confusion matrices into one. Only meaningful when
     * the uploads carried a label column — otherwise we sum predictions.
     */
    private function aggregate(array $results): array
    {
        $ok = array_values(array_filter(
            $results,
            fn ($r) => ($r['ok'] ?? false) === true
        ));

        if (empty($ok)) {
            return ['files' => 0, 'rows' => 0, 'labelled' => false];
        }

        $labelled = collect($ok)->every(fn ($r) => $r['labelled'] ?? false);
        $rows     = array_sum(array_column($ok, 'rows'));

        $summary = [
            'files'    => count($ok),
            'failed'   => count($results) - count($ok),
            'rows'     => $rows,
            'labelled' => $labelled,
        ];

        foreach (['baseline', 'svm', 'hybrid'] as $model) {

            if ($labelled) {
                $tp = $fp = $fn = $tn = 0;

                foreach ($ok as $r) {
                    $tp += $r[$model]['tp'] ?? 0;
                    $fp += $r[$model]['fp'] ?? 0;
                    $fn += $r[$model]['fn'] ?? 0;
                    $tn += $r[$model]['tn'] ?? 0;
                }

                $summary[$model] = $this->confusion($tp, $fp, $fn, $tn);
            } else {
                $attacks = 0;
                foreach ($ok as $r) {
                    $attacks += $r[$model]['attacks'] ?? 0;
                }

                $summary[$model] = [
                    'rows'        => $rows,
                    'attacks'     => $attacks,
                    'normal'      => $rows - $attacks,
                    'attack_rate' => $rows ? $attacks / $rows : 0,
                ];
            }
        }

        return $summary;
    }

    private function confusion(int $tp, int $fp, int $fn, int $tn): array
    {
        $total     = $tp + $fp + $fn + $tn;
        $precision = ($tp + $fp) ? $tp / ($tp + $fp) : 0.0;
        $recall    = ($tp + $fn) ? $tp / ($tp + $fn) : 0.0;
        $spec      = ($tn + $fp) ? $tn / ($tn + $fp) : 0.0;

        $denom = sqrt(
            (float) ($tp + $fp) * ($tp + $fn) * ($tn + $fp) * ($tn + $fn)
        );

        return [
            'accuracy'          => $total ? ($tp + $tn) / $total : 0.0,
            'precision'         => $precision,
            'recall'            => $recall,
            'specificity'       => $spec,
            'balanced_accuracy' => 0.5 * ($recall + $spec),
            'f1'                => ($precision + $recall)
                ? 2 * $precision * $recall / ($precision + $recall)
                : 0.0,
            'mcc'               => $denom ? (($tp * $tn) - ($fp * $fn)) / $denom : 0.0,
            'tp' => $tp, 'fp' => $fp, 'fn' => $fn, 'tn' => $tn,
        ];
    }
}
