<?php

namespace App\Console\Commands;

use App\Services\PredictionRunner;
use Illuminate\Console\Command;
use Symfony\Component\Process\Process;

/**
 * php artisan ml:doctor
 *
 * Walks the whole chain — interpreter, packages, worker script, bundle,
 * and a real end-to-end prediction on a synthetic CSV — and stops at the
 * first thing that is actually broken.
 */
class MlDoctor extends Command
{
    protected $signature = 'ml:doctor {--fresh : ignore the cached interpreter}';

    protected $description = 'Diagnose the Python inference chain';

    public function handle(PredictionRunner $runner): int
    {
        $this->newLine();
        $this->line('<options=bold>ML inference chain</>');
        $this->newLine();

        // ---------------------------------------------------------- config
        $this->line('<options=bold>1. Configuration</>');
        $this->table(['Key', 'Value'], [
            ['ml.python', config('ml.python')],
            ['ml.script', config('ml.script')],
            ['ml.bundle (raw)', config('ml.bundle')],
            ['ml.bundle (resolved)', $runner->bundlePath()],
            ['ml.timeout', config('ml.timeout').'s'],
        ]);

        // ------------------------------------------------------ interpreter
        $this->line('<options=bold>2. Python interpreter</>');

        $resolved = $runner->resolvePython((bool) $this->option('fresh'));

        foreach ($resolved['attempts'] as $attempt) {
            $mark = $attempt['ok'] ? '<info>OK  </info>' : '<fg=red>FAIL</>';
            $this->line("   {$mark}  {$attempt['command']}");
            $this->line('         '.$attempt['detail']);
        }

        if ($resolved['binary'] === null) {
            if ($resolved['attempts'] === []) {
                $this->line('   <comment>using cached interpreter; '.
                            're-run with --fresh to re-probe</comment>');
            }

            $this->newLine();
            $this->error('No interpreter with numpy, pandas and joblib.');
            $this->newLine();
            $this->line('Fix: create a venv and point ML_PYTHON at it.');
            $this->newLine();
            $this->line('  <comment>Windows</comment>');
            $this->line('    python -m venv ml-venv');
            $this->line('    ml-venv\\Scripts\\pip install numpy pandas joblib');
            $this->line('    # .env  (forward slashes are fine)');
            $this->line('    ML_PYTHON="C:/full/path/to/project/ml-venv/Scripts/python.exe"');
            $this->newLine();
            $this->line('  <comment>macOS / Linux</comment>');
            $this->line('    python3 -m venv ml-venv');
            $this->line('    ./ml-venv/bin/pip install numpy pandas joblib');
            $this->line('    ML_PYTHON="/full/path/to/project/ml-venv/bin/python"');
            $this->newLine();
            $this->line('Then: php artisan config:clear && php artisan ml:doctor --fresh');

            return self::FAILURE;
        }

        $binary = $resolved['binary'];
        $this->line('   using: <info>'.implode(' ', $binary).'</info>');

        $versions = new Process([
            ...$binary, '-c',
            'import numpy,pandas,joblib,sys;'.
            'print(f"python {sys.version.split()[0]} | numpy {numpy.__version__}'.
            ' | pandas {pandas.__version__} | joblib {joblib.__version__}")',
        ]);
        $versions->run();
        $this->line('   '.trim($versions->getOutput()));
        $this->newLine();

        // ----------------------------------------------------------- script
        $this->line('<options=bold>3. Worker script</>');

        $script = (string) config('ml.script');

        if (! is_readable($script)) {
            $this->error("   NOT READABLE: {$script}");
            $this->line('   predict.py must sit at <comment>ml/predict.py</comment> '.
                        'in the project root, or ML_SCRIPT must point at it.');

            return self::FAILURE;
        }

        $this->line('   <info>OK</info>  '.$script.
                    '  ('.number_format(filesize($script)).' bytes)');
        $this->newLine();

        // ----------------------------------------------------------- bundle
        $this->line('<options=bold>4. Model bundle</>');

        $bundle = $runner->bundlePath();

        if (! is_readable($bundle)) {
            $this->error("   NOT READABLE: {$bundle}");
            $this->line('   Export it from the training notebook, then:');
            $this->line('     mkdir -p storage/app/ml');
            $this->line('     mv inference_bundle.joblib storage/app/ml/');

            return self::FAILURE;
        }

        $this->line('   <info>OK</info>  '.$bundle.'  ('.
                    number_format(filesize($bundle) / 1048576, 2).' MB)');

        $inspect = new Process([
            ...$binary, '-c',
            'import joblib,sys;b=joblib.load(sys.argv[1]);'.
            'print("features:",len(b["feature_order"]),'.
            '"| scales:",b["if_scales"],'.
            '"| hybrid SVs:",b["svm_hybrid"]["support_vectors"].shape[0],'.
            '"| thresholds:",round(b["threshold_baseline"],4),'.
            'round(b["threshold_hybrid"],4))',
            $bundle,
        ]);
        $inspect->setTimeout(120);
        $inspect->run();

        if (! $inspect->isSuccessful()) {
            $this->error('   Bundle will not load:');
            $this->line('   '.mb_substr(trim($inspect->getErrorOutput()), 0, 600));

            return self::FAILURE;
        }

        $this->line('   '.trim($inspect->getOutput()));
        $this->newLine();

        // ------------------------------------------------------ end to end
        $this->line('<options=bold>5. End-to-end prediction</>');

        $csv = tempnam(sys_get_temp_dir(), 'mldoc').'.csv';
        file_put_contents($csv, $this->syntheticCsv());

        $run = new Process([
            ...$binary, $script,
            '--bundle', $bundle,
            '--input', $csv,
            '--name', 'ml_doctor_probe.csv',
        ]);
        $run->setTimeout((float) config('ml.timeout'));
        $run->run();

        @unlink($csv);

        $stdout = trim($run->getOutput());

        if ($stdout === '') {
            $this->error('   Worker exited '.var_export($run->getExitCode(), true).
                         ' with no output.');
            $this->line('   stderr: '.mb_substr(trim($run->getErrorOutput()), 0, 800));

            return self::FAILURE;
        }

        $decoded = json_decode($stdout, true);

        if (! is_array($decoded)) {
            $this->error('   Worker returned non-JSON:');
            $this->line('   '.mb_substr($stdout, 0, 400));

            return self::FAILURE;
        }

        if (($decoded['ok'] ?? false) !== true) {
            $this->error('   Worker reported: '.($decoded['error'] ?? 'unknown'));

            return self::FAILURE;
        }

        $this->line('   <info>OK</info>  scored '.$decoded['rows'].' synthetic rows');
        foreach (['baseline', 'svm', 'hybrid'] as $model) {
            $data = $decoded[$model];
            $this->line(sprintf(
                '     %-9s attack rate %.4f',
                $model,
                $data['attack_rate'] ?? 0
            ));
        }

        $this->newLine();
        $this->info('Chain is healthy. The prototype will work.');
        $this->newLine();
        $this->line('<comment>Note:</comment> those rates come from random '.
                    'numbers, not real traffic — they only prove the wiring. '.
                    'Upload the 82,332-row test partition and check the '.
                    'accuracies match your results table.');

        return self::SUCCESS;
    }

    /**
     * Nine columns of random values. Enough to exercise every stage:
     * cleaning, scaling, all four forests, both SVM decision functions.
     */
    private function syntheticCsv(): string
    {
        $columns = [
            'sttl', 'ct_state_ttl', 'dload', 'rate',
            'dmean', 'dttl', 'ackdat', 'sload', 'sbytes',
        ];

        $lines = [implode(',', $columns)];

        for ($row = 0; $row < 200; $row++) {
            $values = [];
            foreach ($columns as $column) {
                $values[] = round(mt_rand(0, 100000) / 100, 4);
            }
            $lines[] = implode(',', $values);
        }

        return implode("\n", $lines)."\n";
    }
}
