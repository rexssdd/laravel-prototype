<?php

namespace App\Services;

use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Log;
use Symfony\Component\Process\Exception\ProcessTimedOutException;
use Symfony\Component\Process\Process;

/**
 * Runs ml/predict.py against one CSV and returns the decoded result.
 *
 * Two things this handles that the first version did not:
 *
 *  1. Interpreter resolution. "python3" does not exist on Windows — the
 *     name is a Microsoft Store alias stub that exits silently, producing
 *     no stdout and no stderr. That looks exactly like a crashed worker.
 *     We probe candidates until one can import numpy, pandas and joblib.
 *
 *  2. Diagnostics. When the worker fails, the exit code and stderr go back
 *     to the browser. Telling someone to go read a log file is how a
 *     five-second fix turns into an afternoon.
 */
class PredictionRunner
{
    private const PROBE = 'import numpy, pandas, joblib, sys; print(sys.executable)';

    private const CACHE_KEY = 'ml.resolved_python';

    /**
     * Memoised for this process. The cache below is a bonus, not a
     * requirement — see rememberBinary().
     *
     * @var array<int, string>|null
     */
    private static ?array $memo = null;

    public function bundleExists(): bool
    {
        return is_readable($this->bundlePath());
    }

    /**
     * Config may hold a path relative to the project root. is_readable()
     * would resolve that against the CWD, which differs between
     * `artisan serve`, php-fpm and a queue worker.
     */
    public function bundlePath(): string
    {
        $path = trim((string) config('ml.bundle'));

        if ($path === '') {
            return storage_path('app/ml/inference_bundle.joblib');
        }

        $isAbsolute = str_starts_with($path, '/')
            || preg_match('/^[A-Za-z]:[\\\\\/]/', $path) === 1;

        return $isAbsolute ? $path : base_path($path);
    }

    /**
     * Candidate interpreters, most specific first.
     *
     * @return array<int, array<int, string>>
     */
    public function candidates(): array
    {
        $configured = trim((string) config('ml.python'));

        $candidates = [];

        if ($configured !== '') {
            // allows "py -3" or "C:\venv\Scripts\python.exe" alike
            $candidates[] = preg_split('/\s+/', $configured);
        }

        foreach ([['python3'], ['python'], ['py', '-3']] as $fallback) {
            if ($fallback !== ($candidates[0] ?? null)) {
                $candidates[] = $fallback;
            }
        }

        return $candidates;
    }

    /**
     * Probe each candidate for a working interpreter with the three
     * required packages. Cached, because probing costs a process spawn.
     *
     * @return array{binary: ?array<int, string>, attempts: array<int, array<string, mixed>>}
     */
    public function resolvePython(bool $fresh = false): array
    {
        if ($fresh) {
            self::$memo = null;
            $this->forgetBinary();
        } else {
            if (self::$memo !== null) {
                return ['binary' => self::$memo, 'attempts' => []];
            }

            $cached = $this->recallBinary();
            if ($cached !== null) {
                self::$memo = $cached;

                return ['binary' => $cached, 'attempts' => []];
            }
        }

        $attempts = [];

        foreach ($this->candidates() as $candidate) {

            $process = new Process([...$candidate, '-c', self::PROBE], null, $this->workerEnvironment());
            $process->setTimeout(30);

            try {
                $process->run();
            } catch (\Throwable $e) {
                $attempts[] = [
                    'command' => implode(' ', $candidate),
                    'ok' => false,
                    'detail' => $e->getMessage(),
                ];

                continue;
            }

            $stdout = trim($process->getOutput());
            $stderr = trim($process->getErrorOutput());

            if ($process->isSuccessful() && $stdout !== '') {
                $attempts[] = [
                    'command' => implode(' ', $candidate),
                    'ok' => true,
                    'detail' => $stdout,
                ];

                self::$memo = $candidate;
                $this->rememberBinary($candidate);

                return ['binary' => $candidate, 'attempts' => $attempts];
            }

            $attempts[] = [
                'command' => implode(' ', $candidate),
                'ok' => false,
                'exit' => $process->getExitCode(),
                'detail' => $this->explainProbeFailure(
                    $process->getExitCode(),
                    $stderr
                ),
            ];
        }

        return ['binary' => null, 'attempts' => $attempts];
    }

    /*
     * The cache is a convenience, never a dependency. With no .env Laravel
     * defaults CACHE_STORE to "database", and if that table has not been
     * migrated every cache call throws — which would take down inference
     * for a reason that has nothing to do with inference. So: try, shrug,
     * carry on. The static memo above already prevents repeat probes
     * within a request.
     */

    private function recallBinary(): ?array
    {
        try {
            $cached = Cache::get(self::CACHE_KEY);
        } catch (\Throwable $e) {
            return null;
        }

        return is_array($cached) ? $cached : null;
    }

    private function rememberBinary(array $candidate): void
    {
        try {
            Cache::put(self::CACHE_KEY, $candidate, now()->addHour());
        } catch (\Throwable $e) {
            // no cache store available; the static memo is enough
        }
    }

    private function forgetBinary(): void
    {
        try {
            Cache::forget(self::CACHE_KEY);
        } catch (\Throwable $e) {
            // nothing to forget
        }
    }

    private function explainProbeFailure(?int $exit, string $stderr): string
    {
        if ($stderr === '') {
            return 'no output at all (exit '.var_export($exit, true).'). '.
                   'On Windows this is usually the Microsoft Store alias '.
                   'stub for "python3" — use "python", "py -3", or the full '.
                   'path to your venv interpreter.';
        }

        if (str_contains($stderr, 'ModuleNotFoundError')) {
            return 'interpreter works but a package is missing — '.
                   mb_substr($stderr, 0, 300);
        }

        return mb_substr($stderr, 0, 300);
    }

    /**
     * @param  string  $absolutePath  CSV on local disk
     * @param  string  $displayName  original filename, for the UI
     */
    public function run(string $absolutePath, string $displayName): array
    {
        if (! $this->bundleExists()) {
            return $this->fail(
                $displayName,
                'Model bundle not found at '.$this->bundlePath().
                '. Export it from the training notebook and copy it there, '.
                'then run `php artisan ml:doctor`.'
            );
        }

        if (! is_readable((string) config('ml.script'))) {
            return $this->fail(
                $displayName,
                'Worker script not found at '.config('ml.script').
                '. Check ML_SCRIPT, or move predict.py to the project root '.
                'under ml/.'
            );
        }

        $resolved = $this->resolvePython();

        if ($resolved['binary'] === null) {
            $tried = collect($resolved['attempts'])
                ->map(fn ($a) => $a['command'].' → '.$a['detail'])
                ->implode('  |  ');

            return $this->fail(
                $displayName,
                'No working Python interpreter. Tried: '.$tried.
                '  —  set ML_PYTHON in .env to a full path, then run '.
                '`php artisan ml:doctor`.'
            );
        }

        $process = new Process([
            ...$resolved['binary'],
            (string) config('ml.script'),
            '--bundle', $this->bundlePath(),
            '--input', $absolutePath,
            '--name', $displayName,
        ], null, $this->workerEnvironment());

        $process->setTimeout((float) config('ml.timeout'));

        try {
            $process->run();
        } catch (ProcessTimedOutException $e) {
            return $this->fail(
                $displayName,
                'Timed out after '.config('ml.timeout').'s. Split the file, '.
                'raise ML_TIMEOUT, or move this into a queued job.'
            );
        }

        $stdout = trim($process->getOutput());
        $stderr = trim($process->getErrorOutput());

        if ($stdout === '') {
            Log::error('predict.py produced no output', [
                'file' => $displayName,
                'command' => $process->getCommandLine(),
                'exit' => $process->getExitCode(),
                'stderr' => $stderr,
            ]);

            return $this->fail(
                $displayName,
                'Worker exited '.var_export($process->getExitCode(), true).
                ' with no output. '.
                ($stderr !== ''
                    ? 'stderr: '.mb_substr($stderr, 0, 500)
                    : 'No stderr either — run `php artisan ml:doctor`.')
            );
        }

        $decoded = json_decode($stdout, true);

        if (! is_array($decoded)) {
            Log::error('predict.py returned malformed JSON', [
                'file' => $displayName,
                'stdout' => mb_substr($stdout, 0, 2000),
                'stderr' => mb_substr($stderr, 0, 2000),
            ]);

            return $this->fail(
                $displayName,
                'Malformed response from the worker. First 300 chars: '.
                mb_substr($stdout, 0, 300)
            );
        }

        $decoded['file'] = $displayName;

        return $decoded;
    }

    /**
     * Preserve the Windows directory needed by Python's Winsock imports.
     * Symfony's default environment can omit it when HTTP server variables
     * contain only a subset of the host environment.
     *
     * @return array{SystemRoot?: string}
     */
    private function workerEnvironment(): array
    {
        if (PHP_OS_FAMILY !== 'Windows') {
            return [];
        }

        $systemRoot = getenv('SystemRoot');

        return is_string($systemRoot) && $systemRoot !== ''
            ? ['SystemRoot' => $systemRoot]
            : [];
    }

    private function fail(string $file, string $message): array
    {
        return ['ok' => false, 'file' => $file, 'error' => $message];
    }
}
