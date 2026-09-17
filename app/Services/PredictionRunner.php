<?php

namespace App\Services;

use Illuminate\Support\Facades\Log;
use Symfony\Component\Process\Exception\ProcessTimedOutException;
use Symfony\Component\Process\Process;

/**
 * Runs ml/predict.py against one CSV and returns the decoded result.
 *
 * The Python side is deliberately dumb: one CSV in, one JSON object out,
 * no state between calls. That keeps this class thin and makes the worker
 * testable from the command line without Laravel.
 */
class PredictionRunner
{
    public function bundleExists(): bool
    {
        return is_readable(config('ml.bundle'));
    }

    /**
     * @param  string  $absolutePath  CSV on local disk
     * @param  string  $displayName   original filename, for the UI
     */
    public function run(string $absolutePath, string $displayName): array
    {
        if (! $this->bundleExists()) {
            return $this->fail(
                $displayName,
                'Model bundle not found at '.config('ml.bundle').
                '. Run ml/export_inference_bundle.py after training and copy '.
                'the file into storage/app/ml/.'
            );
        }

        $process = new Process([
            config('ml.python'),
            config('ml.script'),
            '--bundle', config('ml.bundle'),
            '--input',  $absolutePath,
            '--name',   $displayName,
        ]);

        $process->setTimeout((float) config('ml.timeout'));

        try {
            $process->run();
        } catch (ProcessTimedOutException $e) {
            return $this->fail(
                $displayName,
                'Timed out after '.config('ml.timeout').'s. The file is '.
                'probably too large for a synchronous request — split it, or '.
                'move this call into a queued job.'
            );
        }

        $stdout = trim($process->getOutput());

        if ($stdout === '') {
            Log::error('predict.py produced no output', [
                'file'   => $displayName,
                'stderr' => $process->getErrorOutput(),
                'exit'   => $process->getExitCode(),
            ]);

            return $this->fail(
                $displayName,
                'The inference worker produced no output. Check '.
                'storage/logs/laravel.log.'
            );
        }

        $decoded = json_decode($stdout, true);

        if (! is_array($decoded)) {
            Log::error('predict.py returned malformed JSON', [
                'file'   => $displayName,
                'stdout' => mb_substr($stdout, 0, 2000),
            ]);

            return $this->fail($displayName, 'Malformed response from the worker.');
        }

        $decoded['file'] = $displayName;

        return $decoded;
    }

    private function fail(string $file, string $message): array
    {
        return ['ok' => false, 'file' => $file, 'error' => $message];
    }
}
