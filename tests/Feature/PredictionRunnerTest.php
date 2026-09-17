<?php

namespace Tests\Feature;

use App\Services\PredictionRunner;
use Tests\TestCase;

class PredictionRunnerTest extends TestCase
{
    public function test_probe_and_worker_receive_system_root_with_filtered_server_variables(): void
    {
        if (PHP_OS_FAMILY !== 'Windows') {
            $this->markTestSkipped('Windows subprocess environment regression.');
        }

        $systemRoot = getenv('SystemRoot');
        $server = $_SERVER;
        $environment = $_ENV;
        config(['ml.bundle' => __FILE__, 'ml.script' => __FILE__]);

        $runner = new class extends PredictionRunner
        {
            public function candidates(): array
            {
                return [[PHP_BINARY, '-r', <<<'PHP'
                    if (in_array('-c', $argv, true)) {
                        if (getenv('SystemRoot')) {
                            echo PHP_BINARY;
                        } else {
                            exit(1);
                        }
                    } else {
                        echo json_encode(['ok' => true, 'system_root' => getenv('SystemRoot')]);
                    }
                    PHP, '--']];
            }
        };

        try {
            $_SERVER = ['PATH' => getenv('PATH')];
            $_ENV = [];

            $resolved = $runner->resolvePython(true);
            $result = $runner->run(__FILE__, 'traffic.csv');

            $this->assertNotNull($resolved['binary']);
            $this->assertSame([
                'ok' => true,
                'system_root' => $systemRoot,
                'file' => 'traffic.csv',
            ], $result);
        } finally {
            $_SERVER = $server;
            $_ENV = $environment;
            (new \ReflectionProperty(PredictionRunner::class, 'memo'))->setValue(null, null);
        }
    }
}
